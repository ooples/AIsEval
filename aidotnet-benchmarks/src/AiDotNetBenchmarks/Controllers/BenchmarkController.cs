using AiDotNetBenchmarks.Benchmarks;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNetBenchmarks.Controllers;

/// <summary>
/// Runs the four-model AiDotNet benchmark (MLP / CNN / LSTM / Transformer) and
/// returns the JSON report — the controller-shaped equivalent of the PyTorch
/// side's <c>POST /api/Benchmark/Models</c> (and of the CLI scaffold this repo
/// briefly carried). Same workload knobs as the PyTorch CLI defaults: 3 epochs ×
/// 20 batches × bs64 training; 100 steady-state inference iterations after 10
/// warmups at batch sizes 1/8/32/128.
/// </summary>
/// <remarks>
/// A full four-model run takes minutes — use a generous client timeout. The run
/// executes in-process: for the publication-grade methodology (fresh process per
/// round, interleaved against PyTorch, min-of-N p95 per shape) restart the host
/// between rounds; see Reporting/findings.md.
/// </remarks>
[ApiController]
[Route("api/[controller]")]
public sealed class BenchmarkController : ControllerBase
{
    // One benchmark at a time: the runner pins the global engine, suppresses GC
    // around steady-state loops, and saturates the CPU — two concurrent runs
    // would corrupt each other's measurements (and a no-GC region cannot nest).
    private static readonly SemaphoreSlim RunGate = new(1, 1);

    [HttpGet("Test")]
    public ActionResult<string> Test() => "BenchmarkController is ready. POST api/Benchmark/Models to run.";

    /// <summary>
    /// Run the benchmark for the requested model families and return the report.
    /// </summary>
    /// <param name="models">Comma-separated subset of mlp,cnn,lstm,transformer,mlp-fused.</param>
    /// <param name="epochs">Training epochs per model.</param>
    /// <param name="trainBatches">Train batches per epoch.</param>
    /// <param name="batchSize">Training batch size.</param>
    /// <param name="inferenceIterations">Steady-state inference iterations per batch size.</param>
    /// <param name="warmupIterations">Warmup iterations before the steady-state window.</param>
    /// <param name="seed">Deterministic synthetic-data seed.</param>
    [HttpPost("Models")]
    public async Task<ActionResult<BenchmarkReport>> Models(
        [FromQuery] string models = "mlp,cnn,lstm,transformer",
        [FromQuery] int epochs = 3,
        [FromQuery] int trainBatches = 20,
        [FromQuery] int batchSize = 64,
        [FromQuery] int inferenceIterations = 100,
        [FromQuery] int warmupIterations = 10,
        [FromQuery] int seed = 1234)
    {
        var modelNames = models.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (modelNames.Length == 0)
            return BadRequest(new { error = "Provide at least one model: mlp, cnn, lstm, transformer, mlp-fused." });
        if (epochs < 1 || trainBatches < 1 || batchSize < 1 || inferenceIterations < 1 || warmupIterations < 0)
            return BadRequest(new { error = "Workload parameters must be positive (warmupIterations may be 0)." });

        if (!await RunGate.WaitAsync(TimeSpan.Zero, HttpContext.RequestAborted))
        {
            return StatusCode(StatusCodes.Status409Conflict, new
            {
                error = "A benchmark run is already in progress. Concurrent runs would contend for CPU and corrupt both measurements."
            });
        }

        try
        {
            var options = new BenchmarkOptions(
                modelNames, epochs, trainBatches, batchSize, inferenceIterations, warmupIterations, seed);
            // The runner is CPU-bound for minutes; keep it off the request thread
            // so Kestrel's loop stays responsive.
            var report = await Task.Run(() => new BenchmarkRunner(options).Run(), HttpContext.RequestAborted);
            return Ok(report);
        }
        catch (ArgumentException ex)
        {
            return BadRequest(new { error = ex.Message });
        }
        finally
        {
            RunGate.Release();
        }
    }
}
