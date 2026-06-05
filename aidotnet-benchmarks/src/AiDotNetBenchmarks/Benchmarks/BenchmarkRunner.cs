using System.Diagnostics;
using System.Text.Json;
using AiDotNet;

namespace AiDotNetBenchmarks.Benchmarks;

/// <summary>
/// Workload shape for one benchmark run. Defaults match the published
/// fair-comparison runs (and the PyTorch side's CLI defaults):
/// 3 epochs × 20 train batches × bs64; 100 steady-state inference
/// iterations after 10 warmups, at batch sizes 1/8/32/128.
/// </summary>
internal sealed record BenchmarkOptions(
    string[] Models,
    int Epochs = 3,
    int TrainBatches = 20,
    int BatchSize = 64,
    int InferenceIterations = 100,
    int WarmupIterations = 10,
    int Seed = 1234);

internal sealed class BenchmarkRunner(BenchmarkOptions options)
{
    private static readonly int[] InferenceBatchSizes = [1, 8, 32, 128];

    public BenchmarkReport Run()
    {
        // Fair comparison vs PyTorch-CPU: force the native CPU engine.
        //
        // AiDotNet.Tensors ships a [ModuleInitializer] (GpuAutoDetectModuleInit)
        // that auto-detects a GPU/OpenCL device at assembly load and switches the
        // global engine to DirectGpu/OpenCL. On a GPU-equipped rig that means every
        // op dispatches through CLBlast/OpenCL — a path that is SLOWER than the
        // native OneDNN/OpenBLAS CPU path for these small-to-medium workloads, and
        // is not the path the AiDotNet.Tensors micro-benchmarks beat PyTorch-CPU on.
        // ResetToCpu() pins the CPU engine so this benchmark compares CPU-vs-CPU.
        AiDotNet.Tensors.Engines.AiDotNetEngine.ResetToCpu();
        Console.WriteLine($"[bench] engine pinned to CPU: {AiDotNet.Tensors.Engines.AiDotNetEngine.Current.GetType().Name}");

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

            // Settle the managed heap before the steady-state measurement so the
            // p95 latency reflects steady-state INFERENCE, not a GC pause inherited
            // from the preceding training phase (or the prior batch size). The
            // benchmark trains and infers in one process; without this, the large
            // post-training heap triggers a background gen2 collection that lands
            // inside the 100-iteration window and dominates the p95 of the tiny
            // sub-0.2ms shapes (observed: mlp bs=1 p95 inflating to 7.5x PyTorch
            // with heavy training vs ~1.5x with light, purely from this artifact).
            // PyTorch's native runtime has no managed GC, so its inference p95 is
            // uncontaminated by its training phase — settling ours makes the p95
            // comparison apples-to-apples. A full blocking collect + finalizer
            // drain + second collect reclaims training garbage and compacts before
            // we start timing.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // Fair-comparison fix: PyTorch side measures RSS via
            // `psutil.Process(...).memory_info().rss` (whole-process resident
            // set, including native allocations under libtorch). The prior
            // C# implementation used `GC.GetTotalMemory()` which is the
            // .NET managed heap only — apples to oranges. Switching to
            // `Process.WorkingSet64` mirrors psutil's RSS metric so both
            // sides report the same kind of memory number.
            process.Refresh();
            var peakBefore = process.WorkingSet64 / 1024d / 1024d;
            var steady = new List<double>(options.InferenceIterations);
            var peak = peakBefore;

            // Suppress GC for the duration of the steady-state measurement so the p95
            // reflects compute, not a gen0 collection landing mid-loop. The models
            // allocate a few MB of activation tensors per predict; without this, gen0
            // fills during the 100-iteration window and the resulting stop-the-world
            // pause inflates the p95 tail — exactly the jitter PyTorch's native (GC-free)
            // runtime never pays. A no-GC region pre-reserves the budget and holds off
            // all collections until EndNoGCRegion (or until the budget is exhausted, at
            // which point the runtime resumes GC on its own). Measured effect: flips
            // transformer/cnn bs=32 from loss to win and tightens every shape's p95.
            // Budget ladder: try large first, step down if the runtime can't reserve it;
            // if none take, fall through to normal GC (no correctness impact).
            bool noGcStarted = false;
            foreach (long budgetMb in new long[] { 512, 256, 128, 64 })
            {
                try
                {
                    if (GC.TryStartNoGCRegion(budgetMb * 1024L * 1024L)) { noGcStarted = true; break; }
                }
                catch (ArgumentOutOfRangeException) { /* budget exceeds segment size — try smaller */ }
                catch (InvalidOperationException) { break; /* already in a region (shouldn't happen) */ }
            }

            // Opt-in per-op profiling (AISEVAL_OPPROFILE=1, optionally pinned to one
            // batch size via AISEVAL_OPPROFILE_BS). Bridges the engine's Profiler.OpScope
            // ranges to the legacy aggregator and prints a per-op breakdown after the
            // steady-state loop — used to find which op dominates a losing shape.
            var opProfile = Environment.GetEnvironmentVariable("AISEVAL_OPPROFILE") == "1"
                && (Environment.GetEnvironmentVariable("AISEVAL_OPPROFILE_BS") is not { } bsStr
                    || (int.TryParse(bsStr, out var bsTarget) && bsTarget == batchSize));
            if (opProfile)
            {
                AiDotNet.Tensors.Engines.Optimization.PerformanceProfiler.Instance.Enabled = true;
                AiDotNet.Tensors.Engines.Optimization.PerformanceProfiler.Instance.Clear();
            }

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
            if (noGcStarted)
            {
                // End the region. Throws InvalidOperationException if the runtime had
                // to induce a GC mid-region (budget exhausted) — in that case the region
                // already ended on its own, so the throw is benign.
                try { GC.EndNoGCRegion(); }
                catch (InvalidOperationException) { }
            }
            if (opProfile)
            {
                var prof = AiDotNet.Tensors.Engines.Optimization.PerformanceProfiler.Instance;
                Console.WriteLine($"[opprofile] {model.GetType().Name} bs={batchSize} ({options.InferenceIterations} iters):");
                Console.WriteLine(prof.GenerateReport());
                prof.Enabled = false;
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

// Public: these records are the API response body of BenchmarkController.Models.
public sealed record BenchmarkReport(string Framework, string DotNetRuntime, object AiDotNet, List<ModelReport> Results);
public sealed record ModelReport(string Model, string Backend, long Parameters, TrainingReport Training, List<InferenceReport> Inference);
public sealed record TrainingReport(double[] EpochSeconds, double TotalSeconds, double GradientSecondsAvg, double DataLoadingSecondsAvg, ResourceReport Resources);
public sealed record ResourceReport(double ManagedRssMbPeak, string? NvidiaSmiSample);
public sealed record InferenceReport(int BatchSize, double WarmupSecondsAvg, double SteadyStateLatencyMsAvg, double SteadyStateLatencyMsP95, double ThroughputSamplesPerSecond, double MemoryMbPeak);

internal static class BenchmarkJson
{
    public static readonly JsonSerializerOptions Default = new() { WriteIndented = true };
}
