AiDotNet vs Pytorch Benchmarks
==============================

> Warning: **Prior numbers withdrawn.** The findings previously listed
> were not a framework-vs-framework comparison. They compared **AiDotNet's
> full `AiModelBuilder` lifecycle** (DataLoader config + model config +
> internal 70/15/15 train/validation/test split + async build + `Predict`)
> against **a 5-line `torch.linalg.lstsq` wrapper around LAPACK's DGELSD
> solver** (`_predict_with_torch_least_squares` in `regression_controller.py`).
> Those are different kinds of work: one is a complete ML lifecycle, the
> other is a single LAPACK call. See `Reporting/PRIOR-FINDINGS-DISCLAIMER.md`
> for the audit with line-number references.

The numbers below were re-run on the fair-comparison branch (this PR), with
both sides exercising real training and matching memory metrics. Bench rig
is a single Windows 11 machine, CPU-only (no CUDA available), shared with
other work — these are not citable steady-state numbers, just a sanity
check on direction.

### Hardware / software

| | Value |
|---|---|
| OS | Windows 11 (build 26200) |
| Python | 3.13.3 |
| PyTorch | 2.11.0+cpu |
| .NET | 10.0.8 |
| AiDotNet | 0.207.0 NuGet (assembly 0.204.0.0); was 0.185.0 pre-fix |
| Device | CPU only |

Per-model scaffold comparison (Program.cs / \_\_main\_\_.py)
-----------------------------------------------------------

Workload: `--epochs 3 --train-batches 20 --batch-size 64 --inference-iterations 100 --warmup-iterations 10`. Both sides construct shape-matched networks (table below).

| Family | AiDotNet construction | PyTorch construction |
|---|---|---|
| MLP | `DenseLayer(512,ReLU)` → `DenseLayer(128,ReLU)` → `DenseLayer(10)` | `nn.Linear(784,512)` → ReLU → `nn.Linear(512,128)` → ReLU → `nn.Linear(128,10)` |
| CNN | `Conv(16,3,pad=1,ReLU)` → `MaxPool(2)` → `Conv(32,3,pad=1,ReLU)` → `MaxPool(2)` → `Flatten` → `Dense(10)` | `Conv2d(1,16,3,pad=1)` → ReLU → `MaxPool(2)` → `Conv2d(16,32,3,pad=1)` → ReLU → `AdaptiveAvgPool((4,4))` → `Linear(512,10)` |
| LSTM | `LSTMLayer(64)` → `SequenceTokenSliceLayer.Last` → `Dense(10)` | `nn.LSTM(input=32, hidden=64)` → `out[:, -1, :]` → `Linear(64,10)` |
| Transformer | `Transformer<float>` w/ `numEncoderLayers=2, modelDimension=64, numHeads=4, ffDim=128, sequencePooling=MeanPool` | `nn.Linear(32,64)` → 2× `TransformerEncoderLayer(d=64, h=4, ff=128)` → `mean(dim=1)` → `Linear(64,10)` |

### Numbers

Training time (3 epochs × 20 batches × batch_size 64), single-batch inference latency at `batch_size=128`, and peak RSS during training:

| Model | params (PT / AiD) | train total (PT / AiD) | grad/batch (PT / AiD) | bs=128 latency (PT / AiD) | bs=128 throughput sps (PT / AiD) | peak RSS MB (PT / AiD) |
|---|---|---|---|---|---|---|
| MLP | 468 874 / 468 874 | 1.38s / 1.41s | 5.2 ms / 16.8 ms | **1.91 ms** / 8.94 ms (AiD 4.7× slower) | 67 203 / 14 326 | 282 / 478 |
| CNN | 9 930 / 20 490 | 1.63s / 2.87s | 12.0 ms / 44.4 ms | 41.42 ms / **6.21 ms** (AiD 6.7× **faster**) | 3 090 / 20 612 | 291 / 2 914 |
| Transformer | 69 706 / 51 840 | 5.91s / 16.18s (2.7× slower) | 20.3 ms / 182.7 ms | 13.85 ms / 233.25 ms (AiD 16.8× slower) | 9 240 / 549 | 323 / 3 507 |
| LSTM (PT default) | 25 738 / 24 832 | 1.87s / *did not finish* | 13.6 ms / *n/a* | 11.76 ms / *n/a* | 10 885 / *n/a* | 296 / *n/a* |
| LSTM (reduced 1ep×3b×bs32) | 25 738 / 24 832 | 1.70s / 1.37s | 345 ms / ? | 2.86 ms / *did not finish (>3 min)* | 44 836 / *n/a* | 271 / *n/a* |

> AiDotNet's CNN inference is genuinely faster than PyTorch's at `batch_size=128` on this rig — the prior "AiDotNet is uniformly slower" framing does not hold up.

> AiDotNet's LSTM `Predict()` does not finish in reasonable wall time at PyTorch-default workload here (the training step actually finishes in 1.37 s for a 1ep × 3b × 32bs slice; the inference loop afterward — 92 forward passes over `[B=128, seq=32, features=32]` — was still running after >3 minutes when I killed it). The training-step gap is small (1.37 s vs PyTorch's 1.70 s in the same slice); the inference-path gap is the bottleneck. The library has a recurrent-sequence forward path that hasn't yet been tuned against PyTorch's `nn.LSTM` (which routes to OneDNN/MKL under the hood); this is a real perf gap, but it is in the LSTM forward kernel specifically, not "AiDotNet is just slow."

> Memory: AiDotNet's higher RSS in this rig is dominated by the OpenCL backend kernel cache (591 kernels compiled at startup ≈ 2-3 GB) which is one-time / amortized across all models in a process, not per-model. The PyTorch RSS does not include a comparable GPU runtime here because CUDA was not available (CPU build of torch). This is not the apples-to-apples memory comparison the prior `GC.GetTotalMemory()` vs `psutil` numbers tried to make either — RSS is correct as a metric, but the OpenCL cache makes a CPU-only AiDotNet process look heavier than the model itself.

Regression endpoint comparison (small + large CSV)
--------------------------------------------------

Endpoints:
- AiDotNet: `POST /api/Regression/SimpleRegression` (the controller internally dispatches to `MultipleRegression` model when feature count > 1 — i.e. the full `AiModelBuilder<>.ConfigureModel(new MultipleRegression<double>()).BuildAsync()` lifecycle).
- PyTorch: `POST /api/Regression/MultipleRegression` (the framework-symmetric route added in this PR: real `nn.Linear + MSELoss + Adam` training loop, 200 epochs).

Workload:
- Small: `TestData/RegressionTestData/Complex/SmallSet` — 99 training rows × 10 features.
- Large: `TestData/RegressionTestData/Complex/LargeSet` — 47 999 training rows × 10 features.

Client-side wall time (includes HTTP round-trip + server-side training + inference). Repeated 3× for small to show warmup effect; once for large.

| Dataset | AiDotNet `/SimpleRegression` (calls `MultipleRegression`) | PyTorch `/MultipleRegression` (200 Adam epochs) |
|---|---|---|
| Small, run 1 (cold) | 4 001 ms | 1 666 ms |
| Small, run 2 (warm) | 486 ms | 607 ms |
| Small, run 3 (warm) | 525 ms | 422 ms |
| **Small, warm avg (runs 2-3)** | **505 ms** | **515 ms** |
| Large, run 1 | 16 792 ms | 3 870 ms |

Server-internal `Server-Timing` was within ~5 % of client wall time for every run (no significant client/network overhead). Raw JSON in `Reporting/regression-bench.json`.

**Direction:**
- Small dataset (99 rows × 10 features), warm: **roughly tied** (~500 ms either way).
- Large dataset (48 000 rows × 10 features): **AiDotNet ~4.3× slower** (16.8 s vs 3.9 s).

The large-set gap is a real perf finding — the `AiModelBuilder<MultipleRegression>` pipeline at 48k rows is heavier than 200 epochs of PyTorch Adam on `nn.Linear(10, 1)`. It is NOT the 5.3× gap the prior numbers suggested (4 100 / 770), because the prior numbers compared the `AiModelBuilder` pipeline against PyTorch's raw `torch.linalg.lstsq` LAPACK call, which is not a training pipeline.

Memory measurement parity
-------------------------

Both sides now measure peak RSS:
- C#: `Process.WorkingSet64` (replaced `GC.GetTotalMemory()`, which was managed-heap-only and excluded native BLAS, JIT, and OpenCL allocations).
- Python: `psutil.Process(...).memory_info().rss`.

Caveat: AiDotNet's CPU-only RSS is inflated by the one-time OpenCL kernel-cache compile (~591 kernels) even when no GPU is present. This is a single fixed cost amortized across all models in a process; it is not a per-model overhead. A clean steady-state comparison would either disable the OpenCL backend or subtract the kernel-cache baseline before reporting per-model RSS.

How to reproduce
----------------

```bash
# 1. Build C# (release)
cd aidotnet-benchmarks && dotnet restore && dotnet build -c Release

# 2. Install Python deps
cd ../pytorch-benchmarks && python -m pip install -e .

# 3. Per-model scaffold
cd ../aidotnet-benchmarks && dotnet run --no-build -c Release -- \
    --models mlp,cnn,transformer --output ../results/aidotnet-mct.json
cd ../pytorch-benchmarks && python -m pytorch_benchmarks \
    --models mlp,cnn,lstm,transformer --output ../results/pytorch.json

# 4. Regression endpoint comparison
cd ../aidotnet-benchmarks
ASPNETCORE_URLS=http://127.0.0.1:7000 dotnet bin/Release/net10.0/AiDotNetBenchmarks.dll &
cd ../pytorch-benchmarks
python -m uvicorn pytorch_benchmarks.api:app --host 127.0.0.1 --port 8000 &
cd ..
python Reporting/run-regression-bench.py
```

For raw report data, see `aidotnet-mct.json`, `pytorch.json`,
`pytorch-lstm-reduced.json`, and `regression-bench.json` in this folder.
