using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetBenchmarks.Benchmarks;

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
    // Fair-comparison contract: each model constructs the real AiDotNet
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

    // Compiled-inference mode (AISEVAL_COMPILED=1): trace+compile the forward once
    // per batch shape via the public CompileForward pre-warm, then run steady-state
    // iterations through PredictCompiled (plan replay) instead of the per-layer
    // eager walk. The benchmark satisfies compiled replay's documented contract —
    // the SAME input tensor reference (and values) is fed every iteration. The
    // value-stable rebind in CompiledModelHost makes replay safe for this pattern;
    // a one-time eager-vs-compiled output check below guards correctness anyway.
    // PredictCompiled is `protected internal`, so it is bound via reflection into a
    // cached open delegate (no per-call reflection cost).
    private static readonly bool UseCompiled = Environment.GetEnvironmentVariable("AISEVAL_COMPILED") == "1";
    private static readonly System.Reflection.MethodInfo? PredictCompiledMi =
        typeof(NeuralNetworkBase<float>).GetMethod(
            "PredictCompiled",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public);
    private Func<Tensor<float>, Tensor<float>>? _predictCompiled;
    private bool _compiledReady;
    private bool _compiledChecked;

    protected AiDotNetBenchmarkModel(int seed)
    {
        Random = new Random(seed);
        Network = BuildNetwork();
        ParameterCount = Network.GetParameters().Length;
        if (UseCompiled && PredictCompiledMi is not null)
        {
            _predictCompiled = (Func<Tensor<float>, Tensor<float>>)Delegate.CreateDelegate(
                typeof(Func<Tensor<float>, Tensor<float>>), Network, PredictCompiledMi);
        }
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

        // Compiled mode: pre-warm the plan for this batch shape (trace + compile
        // happens here, NOT inside the timed steady-state loop), and run a one-time
        // eager-vs-compiled output comparison so a silently-wrong replay can never
        // masquerade as a perf win.
        if (UseCompiled && _predictCompiled is not null)
        {
            _compiledReady = Network.CompileForward(Input);
            if (_compiledReady && !_compiledChecked)
            {
                _compiledChecked = true;
                var eager = Network.Predict(Input);
                var compiled = _predictCompiled(Input);
                var e = eager.AsSpan(); var c = compiled.AsSpan();
                double maxAbs = 0, maxMag = 1e-6;
                for (int i = 0; i < e.Length; i++)
                {
                    maxAbs = Math.Max(maxAbs, Math.Abs(e[i] - c[i]));
                    maxMag = Math.Max(maxMag, Math.Abs(e[i]));
                }
                if (e.Length != c.Length || maxAbs / maxMag > 1e-3)
                {
                    Console.WriteLine($"[bench] WARNING {GetType().Name}: compiled output diverges from eager (relErr={maxAbs / maxMag:E2}) — falling back to eager.");
                    _compiledReady = false;
                }
            }
            if (!_compiledReady)
                Console.WriteLine($"[bench] {GetType().Name} bs={batchSize}: compiled plan unavailable — eager fallback.");
        }
    }

    public void Forward()
    {
        // Real inference. Compiled mode replays the traced plan (same input tensor
        // reference every call — the documented replay contract); default walks the
        // layer stack eagerly via Predict.
        if (_compiledReady && _predictCompiled is not null)
        {
            var _ = _predictCompiled(Input);
            return;
        }
        var __ = Network.Predict(Input);
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
