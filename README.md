# AIsEval: PyTorch vs AiDotNet Benchmark Projects

This repository contains two separate benchmark projects designed to compare
PyTorch and AiDotNet under the same high-level workload families and metric
schema.

## Projects

| Project | Runtime | Purpose |
| --- | --- | --- |
| `pytorch-benchmarks/` | Native Python | Runs real PyTorch MLP, CNN, LSTM, and Transformer benchmarks. |
| `aidotnet-benchmarks/` | C# / .NET | Evaluates AiDotNet package identity and provides C# benchmark plumbing with a swappable model backend. |

## Benchmark Coverage

### Training

- Training time per epoch
- Total training time
- CPU/GPU utilization samples
- Memory usage / footprint
- Gradient computation time
- Data-loading overhead

### Inference

- Single-sample latency via batch size 1
- Throughput for batch sizes 1, 8, 32, and 128
- Warm-up versus steady-state timing
- Memory footprint

### Model Families

- MLP: baseline sanity check
- CNN: vision workload
- RNN/LSTM: sequential workload
- Transformer: modern attention-style workload

## Quick Start

### One-time setup

```bash
# Python side
cd pytorch-benchmarks
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# or: source .venv/Scripts/activate      # Windows Git Bash
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell; do not use source here
python -m pip install --upgrade pip
python -m pip install -e .

# C# side
cd ../aidotnet-benchmarks
dotnet restore
dotnet build -c Release
```

### Running the four-model benchmark (controllers — the standard workflow)

Everything runs through web endpoints, like the regression benchmarks. Start
both hosts, then one POST runs both frameworks and returns the two reports
side by side:

```bash
# terminal 1 — AiDotNet host (binds launchSettings: https://localhost:7001 + http://localhost:7000)
cd aidotnet-benchmarks
dotnet run -c Release

# terminal 2 — PyTorch host
cd pytorch-benchmarks
python -m uvicorn pytorch_benchmarks.api:app --host 127.0.0.1 --port 8000

# terminal 3 — run BOTH frameworks via the fan-out (sequential, so they never
# contend for CPU; a full four-model run takes several minutes per side)
curl -k -X POST "https://localhost:7001/api/Both/Benchmark?models=mlp,cnn,lstm,transformer" -o both.json
```

Each side is also available individually with the same query parameters
(`models`, `epochs`, `trainBatches`, `batchSize`, `inferenceIterations`,
`warmupIterations`, `seed` — defaults: 3 epochs × 20 batches × bs64 training,
100 steady-state inference iterations after 10 warmups):

```bash
curl -k -X POST "https://localhost:7001/api/Benchmark/Models?models=mlp"   # AiDotNet only
curl    -X POST "http://localhost:8000/api/Benchmark/Models?models=mlp"    # PyTorch only
```

`GET /api/Benchmark/Test` on either host is a readiness probe. Concurrent runs
are rejected with 409 — a benchmark saturates the CPU, so two at once would
corrupt each other's measurements.

### PyTorch CLI (headless alternative)

The Python side also keeps its CLI (same workload, same report JSON):

```bash
cd pytorch-benchmarks
pytorch-bench --models mlp,cnn,lstm,transformer --output ../results/pytorch.json
```

If `pytorch-bench` is not found, confirm that the virtual environment is active
and rerun `python -m pip install -e .` from `pytorch-benchmarks/`. In PowerShell,
you can avoid PATH issues by running `.\.venv\Scripts\pytorch-bench.exe` or
`python -m pytorch_benchmarks` with the same benchmark arguments.

### Publication-grade runs

A single in-process run on a busy machine is noisy. The published numbers in
`Reporting/findings.md` interleave 8 rounds (AiDotNet, then PyTorch, ×8) with a
**fresh host process per round**, and score each shape as the minimum p95 across
its rounds. To reproduce: restart each host between rounds (process restart
resets JIT/allocator/RSS state), POST `/api/Benchmark/Models` once per round,
and save each response JSON.

## Result Files

Both projects emit indented JSON reports under `results/` by default. Keep raw
reports per hardware profile, for example:

```text
results/
  pytorch-rtx4090-cuda.json
  aidotnet-rtx4090-cuda.json
  pytorch-cpu.json
  aidotnet-cpu.json
```

## Fairness Notes

- Use identical hardware, power settings, process isolation, and batch sizes.
- Run release/optimized builds only (`dotnet run -c Release`, no Python debug tooling).
- Discard the first run if you want to eliminate package JIT/import cache effects.
- The C# project's `Program.cs` benchmark constructs real AiDotNet networks
  (`FeedForwardNeuralNetwork`, `ConvolutionalNeuralNetwork`, `LSTMNeuralNetwork`,
  `FeedForwardNeuralNetwork` with `TransformerEncoderLayer`) matching the
  PyTorch counterparts shape-for-shape, and runs them through
  `NeuralNetworkBase.Train` (real autograd + Adam) and `NeuralNetworkBase.Predict`.
  An earlier scaffold used a hand-rolled MLP for every model name with a fake
  backward/optimizer; see `Reporting/PRIOR-FINDINGS-DISCLAIMER.md` for the
  audit of why numbers from that state should not be cited as a
  framework-to-framework comparison.
- The REST API regression endpoints are now framework-symmetric: AiDotNet's
  `/api/Regression/MultipleRegression` runs the full `AiModelBuilder` lifecycle,
  and the PyTorch `/api/Regression/MultipleRegression` route mirrors that
  lifecycle with an `nn.Linear + MSELoss + Adam` training loop. The PyTorch
  `/api/Regression/Predict` (raw `torch.linalg.lstsq`) route is preserved for
  LAPACK-vs-builder profiling but should not be cited as framework-to-framework
  evidence.
- Memory measurement is now symmetric — both sides record peak RSS (Windows
  `Process.WorkingSet64` / Linux `psutil.Process.memory_info().rss`).
- AiDotNet NuGet pin is `0.207.13` with `AiDotNet.Tensors` pinned to `0.91.1`
  (was `0.207.9` / `0.86.4`; originally `0.185.0`).
- **Compiled/fused training engages** on AiDotNet 0.207.13: PR #1469 reverted the
  default optimizer to standard Adam so the fused step is no longer rejected.
  Run with `AISEVAL_FUSED_DIAG=1` to confirm (`Hit=True`, 60/60 fused steps, no
  fallback for every model). The training loop is one forward per batch on both
  sides (the redundant pre-`Train()` `Forward()` on the C# side was removed).
- PyTorch runs in **eager mode** — no `torch.compile` / `torch.jit` /
  TorchDynamo anywhere in `pytorch-benchmarks/`. Eager is the apples-to-apples
  baseline; a compiled graph would fuse kernels ahead of time in a way that
  compares compilation stacks rather than kernels. The emitted report records
  `torch.__version__`. On this CPU rig eager PyTorch is still 2.2–4.0× faster
  than AiDotNet's compiled path on training and faster on most inference shapes —
  see `Reporting/findings.md`.
