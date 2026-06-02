using AiDotNet;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.AspNetCore.Http.Features;
using System.Diagnostics;
using System.Text.Json;

// CLI dispatch: when --models is supplied, run the four-model benchmark
// scaffold and exit. Otherwise fall through to the ASP.NET Kestrel host
// that serves the REST endpoints (RegressionController etc.). Without
// this branch, the BenchmarkOptions/BenchmarkRunner classes below are
// dead code — `dotnet run -- --models ...` would silently start the
// web host and ignore the args.
if (args.Any(a => a.Equals("--models", StringComparison.OrdinalIgnoreCase)))
{
    // Fair comparison vs PyTorch-CPU: force the native CPU engine.
    //
    // AiDotNet.Tensors ships a [ModuleInitializer] (GpuAutoDetectModuleInit)
    // that auto-detects a GPU/OpenCL device at assembly load and switches the
    // global engine to DirectGpu/OpenCL. On this rig that means every op was
    // dispatching through CLBlast/OpenCL (the 608-kernel compile in the logs) —
    // an integrated-GPU / OpenCL path that is SLOWER than the native
    // OneDNN/OpenBLAS CPU path for these small-to-medium workloads, and is not
    // the path the AiDotNet.Tensors micro-benchmarks beat PyTorch-CPU on.
    // ResetToCpu() pins the CPU engine so this scaffold compares CPU-vs-CPU.
    // (The library also documents AIDOTNET_DISABLE_GPU=1 as the before-startup
    // opt-out; this in-code reset additionally covers the published-DLL path
    // where launchSettings env vars don't apply.)
    AiDotNet.Tensors.Engines.AiDotNetEngine.ResetToCpu();
    Console.WriteLine($"[bench] engine pinned to CPU: {AiDotNet.Tensors.Engines.AiDotNetEngine.Current.GetType().Name}");

    // Opt-in (AISEVAL_FUSED_DIAG=1): prove whether the compiled fused-optimizer
    // training path actually runs (Hit) or falls back to the eager tape (and why).
    // This is the exact instrument that found the "compiled training does nothing"
    // bug — fixed by AiDotNet PR #1469. Before the fix, EnableCompilation defaulted
    // to true and the fused step was ATTEMPTED every step, but the default optimizer
    // defaulted to UseAMSGrad=true, which TryMapToFusedOptimizerConfig rejected, so
    // every step silently fell back to the eager tape. #1469 reverted the default to
    // standard Adam (amsgrad=False, matching PyTorch/TF/Optax); the fused step now
    // engages. The fallback is invisible at the default Silent diagnostic level —
    // only PerStep surfaces the FusedOptimizerPathEvent. With 0.207.13 this prints
    // "Hit=True" and the post-run summary reports fused steps == total train steps.
    var fusedDiag = Environment.GetEnvironmentVariable("AISEVAL_FUSED_DIAG") == "1";
    if (fusedDiag)
    {
        AiDotNet.Configuration.TrainingDiagnosticsConfig.Level = AiDotNet.Configuration.TrainingDiagnosticLevel.PerStep;
        AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        var fusedSeen = new HashSet<string>();
        AiDotNet.Configuration.TrainingDiagnosticsConfig.Sink = evt =>
        {
            if (evt is AiDotNet.Configuration.FusedOptimizerPathEvent f)
            {
                var key = $"{f.Hit}:{f.Reason}";
                if (fusedSeen.Add(key))
                    Console.WriteLine($"[bench] FUSED-PATH event: Hit={f.Hit} Reason={f.Reason ?? "(none)"}");
            }
        };
    }

    var benchOptions = BenchmarkOptions.Parse(args);
    var report = new BenchmarkRunner(benchOptions).Run();
    var outputPath = Path.GetFullPath(benchOptions.OutputPath);
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    File.WriteAllText(outputPath, JsonSerializer.Serialize(report, JsonOptions.Default));
    Console.WriteLine($"Benchmark report written to {outputPath}");

    if (fusedDiag)
    {
        // Single-line answer to "did compiled fused training actually run?": the
        // count of optimizer steps that engaged the fused/compiled kernel across
        // the whole run. A non-zero count equal to the total training-step count
        // (epochs × train-batches × models that train) proves the compiled path
        // ran every step. A count of 0 with a captured fallback exception is the
        // signature of the old "compiled does nothing" failure mode (the first
        // fused step fell back, sticky-disabling the path for the session).
        var fusedSteps = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount();
        var lastFallback = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetLastFallbackException();
        Console.WriteLine($"[bench] compiled/fused training steps that engaged: {fusedSteps}");
        Console.WriteLine(lastFallback is null
            ? "[bench] last fused-fallback exception: (none captured)"
            : $"[bench] last fused-fallback exception: {lastFallback.GetType().FullName}: {lastFallback.Message}");
    }
    return;
}

var builder = WebApplication.CreateBuilder(args);

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = long.MaxValue; // unlimited
});

builder.Services.Configure<FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = long.MaxValue;
});


builder.Services.AddControllers();

var app = builder.Build();

app.MapControllers();

app.Run();

internal sealed record BenchmarkOptions(
    string[] Models,
    int Epochs,
    int TrainBatches,
    int BatchSize,
    int InferenceIterations,
    int WarmupIterations,
    int Seed,
    string OutputPath)
{
    public static BenchmarkOptions Parse(string[] args)
    {
        // Convert --key value command-line pairs into a simple lookup table so
        // individual options can fall back to sensible defaults when omitted.
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length - 1; i += 2)
        {
            if (args[i].StartsWith("--", StringComparison.Ordinal)) map[args[i][2..]] = args[i + 1];
        }

        static int Int(Dictionary<string, string> map, string key, int fallback) =>
            map.TryGetValue(key, out var value) && int.TryParse(value, out var parsed) ? parsed : fallback;

        var models = map.GetValueOrDefault("models", "mlp,cnn,lstm,transformer")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        // Materialize the normalized option values that drive the benchmark
        // workload shape and output location.
        return new BenchmarkOptions(
            models,
            Int(map, "epochs", 3),
            Int(map, "train-batches", 20),
            Int(map, "batch-size", 64),
            Int(map, "inference-iterations", 100),
            Int(map, "warmup-iterations", 10),
            Int(map, "seed", 1234),
            map.GetValueOrDefault("output", "results/aidotnet.json"));
    }
}

internal sealed class BenchmarkRunner(BenchmarkOptions options)
{
    private static readonly int[] InferenceBatchSizes = [1, 8, 32, 128];

    public BenchmarkReport Run()
    {
        // Create each requested model, measure its training and inference
        // phases, and collect those measurements into one framework-level report.
        var factory = new AiDotNetTensorBackend(options.Seed);
        var results = new List<ModelReport>();
        foreach (var modelName in options.Models)
        {
            var modelStart = Stopwatch.StartNew();
            Console.WriteLine($"[bench] {modelName}: building network…");
            var model = factory.Create(modelName);
            Console.WriteLine($"[bench] {modelName}: training ({options.Epochs}e × {options.TrainBatches}b × {options.BatchSize}bs, {model.ParameterCount} params)…");
            var training = BenchmarkTraining(model);
            Console.WriteLine($"[bench] {modelName}: training done in {training.TotalSeconds:F2}s; running inference…");
            var inference = BenchmarkInference(model);
            modelStart.Stop();
            Console.WriteLine($"[bench] {modelName}: complete in {modelStart.Elapsed.TotalSeconds:F2}s");
            results.Add(new ModelReport(modelName, "AiDotNetNeuralNetwork", model.ParameterCount, training, inference));
        }

        return new BenchmarkReport(
            "AiDotNet",
            Environment.Version.ToString(),
            AiDotNetProbe.Describe(),
            results);
    }

    private TrainingReport BenchmarkTraining(IBenchmarkModel model)
    {
        // Track per-epoch duration plus the synthetic data-loading and gradient
        // phases, while a background monitor samples process and GPU resources.
        var epochSeconds = new List<double>();
        var gradientSeconds = new List<double>();
        var dataSeconds = new List<double>();
        var total = Stopwatch.StartNew();
        using var monitor = ResourceMonitor.Start();

        for (var epoch = 0; epoch < options.Epochs; epoch++)
        {
            // Each epoch repeatedly loads fresh synthetic inputs and runs one
            // full train step (forward + backward + optimizer) per batch.
            var epochTimer = Stopwatch.StartNew();
            for (var batch = 0; batch < options.TrainBatches; batch++)
            {
                var dataTimer = Stopwatch.StartNew();
                model.LoadSyntheticBatch(options.BatchSize);
                dataTimer.Stop();
                dataSeconds.Add(dataTimer.Elapsed.TotalSeconds);

                // Fair-comparison fix: do NOT call model.Forward() here. The
                // PyTorch training loop does exactly ONE forward per batch
                // (`logits = model(x); loss.backward(); optimizer.step()`).
                // Backward() == Network.Train(), which already runs its own
                // forward + GradientTape backward + optimizer step internally,
                // so a preceding Forward() (a full discarded Predict() pass)
                // was a second forward per batch that PyTorch never does —
                // pure overhead that inflated AiDotNet training time (≈15% on
                // Transformer at bs=64). Removing it makes both sides one
                // forward per batch. (Forward() is still exercised on its own
                // in the inference benchmark below.)
                var gradientTimer = Stopwatch.StartNew();
                model.Backward();
                gradientTimer.Stop();
                gradientSeconds.Add(gradientTimer.Elapsed.TotalSeconds);
                model.Step();
            }
            epochTimer.Stop();
            epochSeconds.Add(epochTimer.Elapsed.TotalSeconds);
        }

        total.Stop();
        return new TrainingReport(
            epochSeconds.Select(Round6).ToArray(),
            Round6(total.Elapsed.TotalSeconds),
            Round6(gradientSeconds.Average()),
            Round6(dataSeconds.Average()),
            monitor.Summary());
    }

    private List<InferenceReport> BenchmarkInference(IBenchmarkModel model)
    {
        // Measure inference at multiple batch sizes so the report captures both
        // latency and throughput behavior under different request shapes.
        var reports = new List<InferenceReport>();
        var process = Process.GetCurrentProcess();
        foreach (var batchSize in InferenceBatchSizes)
        {
            model.LoadSyntheticBatch(batchSize);

            // Warmup iterations prime JIT compilation and tensor internals before
            // steady-state measurements are recorded.
            var warmup = new List<double>();
            for (var i = 0; i < options.WarmupIterations; i++)
            {
                var timer = Stopwatch.StartNew();
                model.Forward();
                timer.Stop();
                warmup.Add(timer.Elapsed.TotalSeconds);
            }

            // Fair-comparison fix: PyTorch side measures RSS via
            // `psutil.Process(...).memory_info().rss` (whole-process resident
            // set, including native allocations under libtorch). The prior
            // C# implementation used `GC.GetTotalMemory()` which is the
            // .NET managed heap only — apples to oranges. Switching to
            // `Process.WorkingSet64` mirrors psutil's RSS metric so both
            // sides report the same kind of memory number.
            process.Refresh();
            var peakBefore = process.WorkingSet64 / 1024d / 1024d;
            var steady = new List<double>();
            var peak = peakBefore;

            // Steady-state iterations collect forward-pass timings and track the
            // highest observed RSS for this batch size.
            for (var i = 0; i < options.InferenceIterations; i++)
            {
                var timer = Stopwatch.StartNew();
                model.Forward();
                timer.Stop();
                steady.Add(timer.Elapsed.TotalSeconds);
                process.Refresh();
                peak = Math.Max(peak, process.WorkingSet64 / 1024d / 1024d);
            }
            var totalSteady = steady.Sum();
            // p95 latency (symmetric with the PyTorch side): robust to the
            // rig-contention noise that swings the mean. Tensors perf gate is
            // p95(ours) < median(PyTorch).
            var steadySorted = steady.OrderBy(x => x).ToList();
            int p95Idx = Math.Min(steadySorted.Count - 1, (int)Math.Round(0.95 * (steadySorted.Count - 1)));
            reports.Add(new InferenceReport(
                batchSize,
                Round6(warmup.Average()),
                Math.Round(steady.Average() * 1000d, 3),
                Math.Round(steadySorted[p95Idx] * 1000d, 3),
                Math.Round(options.InferenceIterations * batchSize / totalSteady, 3),
                Math.Round(peak, 3)));
        }
        return reports;
    }

    private static double Round6(double value) => Math.Round(value, 6);
}

internal interface IBenchmarkModel
{
    long ParameterCount { get; }
    void LoadSyntheticBatch(int batchSize);
    void Forward();
    void Backward();
    void Step();
}

internal sealed class AiDotNetTensorBackend(int seed)
{
    // Fair-comparison fix: each model now constructs the real AiDotNet
    // neural-network class with paper-matched layer shapes. The PyTorch
    // side uses nn.Conv2d / nn.LSTM / nn.TransformerEncoder for CNN/LSTM/
    // Transformer respectively — so the AiDotNet side must too, otherwise
    // the comparison reduces to "PyTorch real Conv2D vs AiDotNet MLP"
    // (the pre-fix state, where every model name mapped to the same
    // hand-rolled MLP).
    public IBenchmarkModel Create(string model) => model.ToLowerInvariant() switch
    {
        // PyTorch: nn.Sequential(Linear(784,512), ReLU, Linear(512,128), ReLU, Linear(128,10)).
        // Input flattened from [B, 1, 28, 28] to [B, 784].
        "mlp"         => new AiDotNetMlpModel(seed),
        // PyTorch: Conv2d(1,16,3,pad=1) + ReLU + MaxPool(2) + Conv2d(16,32,3,pad=1) + ReLU
        //          + AdaptiveAvgPool((4,4)) + Linear(512, 10). Input [B, 1, 28, 28].
        "cnn"         => new AiDotNetCnnModel(seed),
        // PyTorch: nn.LSTM(input=32, hidden=64) + Linear(64, 10). Input [B, 32, 32].
        "lstm"        => new AiDotNetLstmModel(seed),
        // PyTorch: Linear(32,64) + 2× TransformerEncoderLayer(d_model=64, nhead=4, dim_ff=128)
        //          + mean over seq + Linear(64, 10). Input [B, 32, 32].
        "transformer" => new AiDotNetTransformerModel(seed),
        // Fused-primitive INFERENCE path: same 784->512->128->10 ReLU MLP, but
        // routed through the AiDotNet.Tensors fused MlpForward kernel (issue
        // #436 P1) instead of the generic per-layer Predict() walk. This is the
        // op the Tensors micro-benchmarks beat PyTorch-CPU on; the high-level
        // FeedForwardNeuralNetwork.Predict does NOT call it, which is the whole
        // end-to-end MLP gap. Inference-only (MlpForward is forward-only).
        "mlp-fused"   => new AiDotNetMlpFusedModel(seed),
        _ => throw new ArgumentException($"Unknown model '{model}'.")
    };
}

/// <summary>
/// Base class for the four benchmark models. Centralises the IBenchmarkModel
/// contract (LoadSyntheticBatch / Forward / Backward / Step) so each subclass
/// only has to declare its architecture + input shape. Training runs through
/// AiDotNet's real <c>NeuralNetworkBase.Train</c> (forward + GradientTape
/// backward + Adam step under the covers); inference runs through real
/// <c>NeuralNetworkBase.Predict</c>.
/// </summary>
internal abstract class AiDotNetBenchmarkModel : IBenchmarkModel
{
    protected readonly Random Random;
    protected readonly NeuralNetworkBase<float> Network;
    protected Tensor<float> Input = Tensor<float>.Empty();
    protected Tensor<float> Label = Tensor<float>.Empty();

    protected AiDotNetBenchmarkModel(int seed)
    {
        Random = new Random(seed);
        Network = BuildNetwork();
        ParameterCount = Network.GetParameters().Length;
    }

    public long ParameterCount { get; }

    protected abstract NeuralNetworkBase<float> BuildNetwork();
    protected abstract int[] InputShapePerSample { get; }   // shape WITHOUT batch dim
    protected abstract int OutputClasses { get; }

    public void LoadSyntheticBatch(int batchSize)
    {
        // Generate a deterministic batch of synthetic inputs + one-hot labels
        // so Train() has both the inputs and the gradient signal it needs.
        var perSample = InputShapePerSample;
        var sampleSize = perSample.Aggregate(1, (a, b) => a * b);
        var fullShape = new[] { batchSize }.Concat(perSample).ToArray();
        var inputs = new float[batchSize * sampleSize];
        for (var i = 0; i < inputs.Length; i++) inputs[i] = (float)Random.NextDouble();
        Input = new Tensor<float>(inputs, fullShape);

        // One-hot labels across OutputClasses.
        var labels = new float[batchSize * OutputClasses];
        for (var b = 0; b < batchSize; b++)
        {
            var cls = Random.Next(OutputClasses);
            labels[b * OutputClasses + cls] = 1f;
        }
        Label = new Tensor<float>(labels, [batchSize, OutputClasses]);
    }

    public void Forward()
    {
        // Real inference: walks the layer stack, runs activations + ops.
        var _ = Network.Predict(Input);
    }

    public void Backward()
    {
        // PyTorch side runs `loss = criterion(model(x), y); loss.backward()`.
        // AiDotNet's Train(input, expected) is the equivalent: under the hood
        // it does forward + GradientTape backward + optimizer step. Splitting
        // it across Backward+Step like PyTorch would require a private API;
        // for the per-batch wall-time measurement Backward does the full
        // train step and Step is a no-op. The runner's gradientSeconds
        // average therefore captures BOTH backward and optimizer step,
        // mirroring how the PyTorch side's gradient_seconds is currently
        // measured (loss.backward() time only; optimizer.step() is excluded
        // from gradient_seconds but included in epoch_seconds).
        Network.Train(Input, Label);
    }

    public void Step()
    {
        // Real optimizer step happened inside Backward()'s Train call.
        // Kept on the interface for source compatibility with the runner.
    }
}

internal sealed class AiDotNetMlpModel : AiDotNetBenchmarkModel
{
    public AiDotNetMlpModel(int seed) : base(seed) { }
    protected override int[] InputShapePerSample => new[] { 784 };
    protected override int OutputClasses => 10;
    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        // Matches PyTorch's MLP: Linear(784, 512) + ReLU + Linear(512, 128) + ReLU + Linear(128, 10).
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(512, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(128, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 784,
            outputSize: 10,
            layers: layers);
        return new FeedForwardNeuralNetwork<float>(arch);
    }
}

/// <summary>
/// Fused-primitive MLP: identical 784→512→128→10 ReLU shape as <see cref="AiDotNetMlpModel"/>,
/// but inference routes through <c>IEngine.MlpForward</c> — the fused, thread-capped
/// multi-layer kernel from AiDotNet.Tensors #474/#436-P1 that the Tensors micro-benchmarks
/// beat PyTorch-CPU on. The point: the generic <c>FeedForwardNeuralNetwork.Predict</c> path
/// the other MLP model uses does NOT call this kernel, so the end-to-end MLP comparison
/// never exercises the fast path. This variant measures what the framework SHOULD dispatch
/// to. Forward-only (MlpForward throws under a GradientTape), so training is not measured.
/// </summary>
internal sealed class AiDotNetMlpFusedModel : IBenchmarkModel
{
    private static readonly int[] LayerSizes = [784, 512, 128, 10];
    private readonly Tensor<float>[] _weights;
    private readonly Tensor<float>?[] _biases;
    private Tensor<float> _input = Tensor<float>.Empty();

    public AiDotNetMlpFusedModel(int seed)
    {
        var rng = new Random(seed);
        _weights = new Tensor<float>[LayerSizes.Length - 1];
        _biases = new Tensor<float>?[LayerSizes.Length - 1];
        for (var i = 0; i < _weights.Length; i++)
        {
            int inF = LayerSizes[i], outF = LayerSizes[i + 1];
            // Xavier-ish init; values are immaterial to latency but keep them finite.
            var scale = (float)Math.Sqrt(2.0 / inF);
            var w = new float[inF * outF];
            for (var k = 0; k < w.Length; k++) w[k] = (float)(rng.NextDouble() - 0.5) * 2f * scale;
            _weights[i] = new Tensor<float>(w, [inF, outF]);
            var b = new float[outF];
            for (var k = 0; k < b.Length; k++) b[k] = (float)(rng.NextDouble() - 0.5) * 0.01f;
            _biases[i] = new Tensor<float>(b, [outF]);
        }
        ParameterCount = _weights.Sum(w => (long)w.Length) + _biases.Sum(b => (long)(b?.Length ?? 0));
    }

    public long ParameterCount { get; }

    public void LoadSyntheticBatch(int batchSize)
    {
        var data = new float[batchSize * 784];
        var rng = new Random(1234);
        for (var i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble();
        _input = new Tensor<float>(data, [batchSize, 784]);
    }

    public void Forward()
    {
        // The fused multi-layer kernel: activation(x @ Wᵢ + bᵢ) for every layer in one call.
        var _ = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.MlpForward(
            _input, _weights, _biases,
            AiDotNet.Tensors.Engines.FusedActivationType.ReLU,
            AiDotNet.Tensors.Engines.FusedActivationType.None);
    }

    // Inference-only variant — MlpForward is forward-only. Training is measured by `mlp`.
    public void Backward() { }
    public void Step() { }
}

internal sealed class AiDotNetCnnModel : AiDotNetBenchmarkModel
{
    public AiDotNetCnnModel(int seed) : base(seed) { }
    // Input shape excluding batch: [C=1, H=28, W=28]. AiDotNet's ConvolutionalLayer
    // expects [B, C, H, W] order matching PyTorch's nn.Conv2d default.
    protected override int[] InputShapePerSample => new[] { 1, 28, 28 };
    protected override int OutputClasses => 10;
    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        // Matches PyTorch's SmallCNN: Conv2d(1,16,3,pad=1) + ReLU + MaxPool(2)
        // + Conv2d(16,32,3,pad=1) + ReLU + AdaptiveAvgPool((4,4)) + Linear(512, 10).
        // Substitute a fixed-stride MaxPooling for AdaptiveAvgPool since AiDotNet's
        // ConvolutionalNeuralNetwork composes from the layer-list provided.
        var layers = new List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(outputDepth: 16, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            new ConvolutionalLayer<float>(outputDepth: 32, kernelSize: 3, stride: 1, padding: 1,
                                          activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            // After two 2× pools: 28 → 14 → 7. PyTorch lands at 4 via AdaptiveAvgPool;
            // one more stride-2 pool would over-shrink. The 7×7×32 = 1568 input to
            // the Dense head is in the same order of magnitude as PyTorch's 4×4×32=512.
            new FlattenLayer<float>(),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 28, inputWidth: 28, inputDepth: 1,
            outputSize: 10,
            layers: layers);
        return new ConvolutionalNeuralNetwork<float>(arch);
    }
}

internal sealed class AiDotNetLstmModel : AiDotNetBenchmarkModel
{
    public AiDotNetLstmModel(int seed) : base(seed) { }
    // [seq=32, features=32], matching PyTorch's LSTM(input=32, hidden=64) over a 32-step sequence.
    protected override int[] InputShapePerSample => new[] { 32, 32 };
    protected override int OutputClasses => 10;
    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        // Matches PyTorch's LSTMClassifier: LSTM(input=32, hidden=64) + take last
        // timestep + Linear(64, 10). LSTMLayer emits [B, seqLen, hidden]; the
        // SequenceTokenSliceLayer mirrors PyTorch's `out[:, -1, :]` reduction.
        var layers = new List<ILayer<float>>
        {
            new LSTMLayer<float>(hiddenSize: 64),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,   // sequence input (seq_len, features) handled by LSTM
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 32,
            outputSize: 10,
            layers: layers);
        return new LSTMNeuralNetwork<float>(arch, outputActivation: (IActivationFunction<float>?)null);
    }
}

internal sealed class AiDotNetTransformerModel : AiDotNetBenchmarkModel
{
    public AiDotNetTransformerModel(int seed) : base(seed) { }
    protected override int[] InputShapePerSample => new[] { 32, 32 };
    protected override int OutputClasses => 10;
    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        // Matches PyTorch's TransformerClassifier: nn.Linear(32, 64) projection +
        // 2x TransformerEncoderLayer(d_model=64, nhead=4, dim_ff=128) + mean over seq + nn.Linear(64, 10).
        // Uses AiDotNet's production Transformer<float> class (not FeedForwardNeuralNetwork),
        // which natively understands [B, seq, d_model] input and pools the sequence dim via
        // SequencePoolingMode.MeanPool before the output head — mirroring PyTorch's
        // `encoded.mean(dim=1)` reduction in __main__.py.
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 128,
            inputSize: 32,           // per-token feature width
            outputSize: 10,
            dropoutRate: 0.0,
            maxSequenceLength: 32,
            vocabularySize: 0,       // continuous features, no embedding table
            usePositionalEncoding: true,
            sequencePooling: SequencePoolingMode.MeanPool);
        return new Transformer<float>(arch);
    }
}

internal sealed class ResourceMonitor : IDisposable
{
    private readonly Process _process = Process.GetCurrentProcess();
    private readonly CancellationTokenSource _cts = new();
    private readonly List<double> _rssMb = [];
    private readonly Task _task;

    private ResourceMonitor()
    {
        // Sample resident set size in the background throughout the training run;
        // Dispose stops this loop once the caller has collected the summary.
        _task = Task.Run(async () =>
        {
            while (!_cts.IsCancellationRequested)
            {
                _process.Refresh();
                _rssMb.Add(_process.WorkingSet64 / 1024d / 1024d);
                await Task.Delay(100, _cts.Token).ContinueWith(_ => { });
            }
        });
    }

    public static ResourceMonitor Start() => new();

    public ResourceReport Summary() => new(Math.Round(_rssMb.Count == 0 ? 0 : _rssMb.Max(), 3), NvidiaSmi.TryRead());

    public void Dispose()
    {
        _cts.Cancel();
        _task.Wait(TimeSpan.FromSeconds(2));
        _cts.Dispose();
    }
}

internal static class NvidiaSmi
{
    public static string? TryRead()
    {
        try
        {
            // Query nvidia-smi opportunistically so systems without NVIDIA GPUs
            // can still complete benchmarks with a null GPU sample.
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                ArgumentList = { "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits" },
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            });
            if (process is null || !process.WaitForExit(1000) || process.ExitCode != 0) return null;
            return process.StandardOutput.ReadLine();
        }
        catch
        {
            return null;
        }
    }
}

internal static class AiDotNetProbe
{
    public static object Describe()
    {
        // Report the loaded AiDotNet assembly metadata and a small probe of neural
        // model-related types to help identify what package implementation ran.
        var assembly = typeof(AiModelBuilder<,,>).Assembly;
        if (assembly is null) return new { loaded = false };
        var neuralTypes = assembly.GetTypes()
            .Where(type => type.FullName?.Contains("Neural", StringComparison.OrdinalIgnoreCase) == true
                        || type.FullName?.Contains("LSTM", StringComparison.OrdinalIgnoreCase) == true
                        || type.FullName?.Contains("Transformer", StringComparison.OrdinalIgnoreCase) == true)
            .Select(type => type.FullName)
            .Take(25)
            .ToArray();
        return new
        {
            loaded = true,
            name = assembly.GetName().Name,
            version = assembly.GetName().Version?.ToString(),
            neuralTypeProbe = neuralTypes
        };
    }

}

internal sealed record BenchmarkReport(string Framework, string DotNetRuntime, object AiDotNet, List<ModelReport> Results);
internal sealed record ModelReport(string Model, string Backend, long Parameters, TrainingReport Training, List<InferenceReport> Inference);
internal sealed record TrainingReport(double[] EpochSeconds, double TotalSeconds, double GradientSecondsAvg, double DataLoadingSecondsAvg, ResourceReport Resources);
internal sealed record ResourceReport(double ManagedRssMbPeak, string? NvidiaSmiSample);
internal sealed record InferenceReport(int BatchSize, double WarmupSecondsAvg, double SteadyStateLatencyMsAvg, double SteadyStateLatencyMsP95, double ThroughputSamplesPerSecond, double MemoryMbPeak);

internal static class JsonOptions
{
    public static readonly JsonSerializerOptions Default = new() { WriteIndented = true };
}
