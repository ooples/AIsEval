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

> **Refreshed 2026-06-01** with the latest packages: **AiDotNet 0.207.13 +
> AiDotNet.Tensors 0.91.1** (was 0.207.9 / 0.86.4). Two headline changes:
>
> 1. **Compiled/fused training now actually engages.** The 2026-05-29 deep-dive
>    (root cause #3 below) found that the compiled fused-training step silently
>    fell back to the eager tape on *every* `Train()` step because the default
>    optimizer defaulted to `UseAMSGrad=true`, which the fused mapper rejected.
>    AiDotNet **PR #1469** reverted the default to standard Adam (matching
>    PyTorch/TF/Optax), and Tensors #501/#502/#513 wired the fused optimizer/
>    activation kernels + compiled-inference plan. Re-running with
>    `AISEVAL_FUSED_DIAG=1` now reports `Hit=True` and **60/60 fused training
>    steps engaged per model, zero fallbacks** for all four families — the
>    "compiled training does nothing" bug is fixed.
> 2. **LSTM training dropped ~9×** (26.5 s → 2.82 s) from the fused-recurrence
>    forward+backward kernels (Tensors #503/#505/#523) + the now-engaged
>    compiled training step.
>
> **Net direction is still PyTorch-favored on this rig**, though: eager
> PyTorch-CPU 2.11 remains 2.2–4.0× faster on training and faster on most
> inference shapes (AiDotNet wins only CNN bs=128 and LSTM bs=8). This is
> AiDotNet's *compiled* path vs PyTorch *eager* — AiDotNet's own PR #1469
> reported beating `torch.compile` (TorchInductor) ~6× on an MLP, but against
> plain eager PyTorch on this hardware AiDotNet does not win. Numbers below.
>
> The benchmark training loop was also made symmetric this run: the C# side
> previously ran a redundant `Forward()` (a full discarded `Predict()` pass)
> *before* `Backward()`/`Train()` every batch — a second forward PyTorch never
> does. Removing it (one forward per batch on both sides) cut AiDotNet training
> time ~15% on Transformer (9.42 s → 7.72 s) and a few % elsewhere.

### Why AiDotNet trailed here — three root causes (2026-05-29 deep-dive)

The Tensors micro-benchmarks beat PyTorch-CPU on the raw fused ops, yet this
end-to-end scaffold showed AiDotNet 9–21× slower. Three distinct causes, none
of which is "the math is slow":

1. **Wrong backend (the 9–21× factor).** AiDotNet.Tensors ships a
   `[ModuleInitializer]` (`GpuAutoDetectModuleInit`) that auto-switches the
   global engine to OpenCL/DirectGpu at assembly load. On this rig it bound to a
   discrete **NVIDIA GTX 1660 Ti via OpenCL** (608 kernels compiled) — and for
   these tiny CPU-class workloads the GPU dispatch/transfer overhead is far
   worse than the native CPU path. The benchmark now calls
   `AiDotNetEngine.ResetToCpu()` at startup (library-documented baseline). Effect
   on MLP bs=128 inference: **12.3 ms → 1.86 ms**; LSTM training 26.5 s → 3.1 s.

2. **Fused inference kernels are not wired into the high-level models.** The
   generic `FeedForwardNeuralNetwork.Predict` / transformer attention walk the
   layer stack op-by-op through the tape; they never call the fused
   `IEngine.MlpForward` / fused-MHA kernels that the Tensors PRs optimized.
   `grep` confirms `MlpForward` appears only in an enum in the framework. Only
   `LSTMLayer` is wired to its fused kernel (`LstmSequenceForward`) — and LSTM
   is the one model that's competitive (bs=1 0.31 ms vs PyTorch 0.27 ms). The
   `mlp-fused` benchmark variant calls `MlpForward` directly: bs=128
   1.86 → 1.19 ms. Still ~2× off this run's PyTorch (0.58 ms), because the
   PyTorch baseline this session was unusually fast (see note below).

3. **The compiled fused-training path silently never ran — ✅ FIXED in 0.207.13
   (AiDotNet PR #1469).** `EnableCompilation` defaults to `true`, so
   `CompiledTapeTrainingStep.TryStepWithFusedOptimizer` is *attempted on every
   `Train()` step* — but on 0.207.9 it **fell back to the eager tape every time**
   because the default optimizer defaulted to `UseAMSGrad=true` (a non-standard
   band-aid), and `TryMapToFusedOptimizerConfig` rejected AMSGrad. The fallback
   was **silent** at the default diagnostic level — it only surfaced at
   `TrainingDiagnosticsConfig.Level = PerStep` (run with `AISEVAL_FUSED_DIAG=1`),
   which is how it was found here. **PR #1469 reverted the default to standard
   Adam** (`amsgrad=False`, matching PyTorch/TF/Optax) and re-architected the
   dispatch around self-describing `IFusedOptimizerSpec` / `IFusedActivation`
   interfaces, and Tensors #501/#502 wired the fused kernels. Verified on
   0.207.13: `AISEVAL_FUSED_DIAG=1` now prints `Hit=True` and the post-run summary
   reports **60 fused training steps engaged (== 3 epochs × 20 batches), zero
   fallback exceptions**, for each of MLP/CNN/LSTM/Transformer. The fallback is
   also no longer silent — it emits a one-time per-model warning when it does
   happen. Effect: LSTM training 26.5 s → 2.82 s; Transformer training 8.9 s →
   7.7 s. (Training is still slower than eager PyTorch — see Numbers — but the
   compiled path is now genuinely exercised rather than dead code.)

> **Baseline caveat.** PyTorch's MLP bs=128 latency swung 3.3× between sessions
> (1.91 ms in the prior table → 0.58 ms here) purely from rig contention. The
> Tensors PRs were validated as *p95(ours) < median(PyTorch)* against a ~1.9 ms
> PyTorch; the fused MLP at 1.19 ms beats that, but loses to an unloaded 0.58 ms
> PyTorch. A credible head-to-head needs both sides thread-pinned and isolated,
> reporting p95 — otherwise rig noise dominates the verdict.

### PyTorch fairness: eager, not compiled

The PyTorch side runs **eager mode** — plain `model(x)` forward passes with
`model.eval()` + `torch.no_grad()` for inference and a standard
forward/backward/Adam loop for training. There is **no `torch.compile`, no
`torch.jit.script`, no TorchDynamo/Inductor** anywhere in
`pytorch-benchmarks/` (grep-verified). A compiled PyTorch graph would be the
unfair comparison — it fuses kernels ahead of time in a way the AiDotNet path
does not — so eager is the correct apples-to-apples baseline. The emitted
report records `torch.__version__` so the mode/version is auditable per run.

### Hardware / software

| | Value |
|---|---|
| OS | Windows 11 (build 26200) |
| Python | 3.13.3 |
| PyTorch | 2.11.0+cpu (eager), 8 threads |
| .NET | 10.0.8 |
| AiDotNet | 0.207.13 NuGet (assembly 0.204.0.0); was 0.185.0 pre-fix |
| AiDotNet.Tensors | 0.91.1 NuGet |
| CPU threads | 8 (`AIDOTNET_BLAS_THREADS=8` ↔ PyTorch `--threads 8`) |
| Device | CPU only (`AIDOTNET_DISABLE_GPU=1`) |

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

AiDotNet **0.207.13 / Tensors 0.91.1** (compiled/fused training engaged, CPU-pinned,
GPU disabled) vs PyTorch **2.11.0+cpu eager**, both 8 threads, same rig, same session.
Workload `--epochs 3 --train-batches 20 --batch-size 64 --inference-iterations 100
--warmup-iterations 10`. One forward per batch on both sides (see symmetric-loop note above).

**Training — total wall time (s), lower is better:**

| Model | params (PT / AiD) | PyTorch | AiDotNet | gap |
|---|---|---|---|---|
| MLP | 468 874 / 468 874 | **0.51 s** | 1.17 s | PT 2.3× faster |
| CNN | 9 930 / 20 490 | **0.86 s** | 2.56 s | PT 3.0× faster |
| LSTM | 25 738 / 24 832 | **0.70 s** | 2.82 s | PT 4.0× faster (was 26.5 s pre-#1469 — ~9× better) |
| Transformer | 69 706 / 51 840 | **3.47 s** | 7.72 s | PT 2.2× faster |

**Inference — steady-state mean latency (ms), lower is better (AiDotNet p95 in parens):**

| Model | bs=1 (PT / AiD) | bs=8 (PT / AiD) | bs=32 (PT / AiD) | bs=128 (PT / AiD) |
|---|---|---|---|---|
| MLP | **0.12** / 0.22 (p95 0.28) | **0.18** / 0.86 | **0.37** / 0.85 | **0.71** / 1.24 |
| CNN | **0.28** / 0.40 | **0.88** / 0.93 | **1.84** / 2.09 | 8.00 / **5.69** (AiD wins) |
| LSTM | **0.44** / 0.51 | 0.68 / **0.44** (AiD wins) | **1.20** / 1.69 | **2.80** / 5.35 |
| Transformer | **0.78** / 0.93 | **1.55** / 5.63 | **2.98** / 13.58 | **8.86** / 47.12 |

> **Compiled/fused training is now live.** `AISEVAL_FUSED_DIAG=1` confirms `Hit=True`
> and 60/60 fused steps with no fallback for every model — the root-cause #3 bug is
> fixed. Training is nonetheless 2.2–4.0× slower than eager PyTorch on this rig: the
> remaining gap is dispatch/threading overhead around the per-op compute, not a dead
> compiled path. (AiDotNet PR #1469 reported the compiled training plan beating
> `torch.compile`/TorchInductor ~6× on an MLP; this table is vs *eager* PyTorch,
> which has no compile warmup to amortize and wins here.)

> **LSTM training improved ~9×** (26.5 s → 2.82 s) — the biggest single change this
> run — from the fused-recurrence forward+backward kernels (Tensors #503/#505/#523)
> plus the now-engaged compiled optimizer step. The remaining 4× training gap and
> the bs≥32 inference gap are BPTT/recurrent-step overhead vs PyTorch's fused
> OneDNN/MKL `nn.LSTM`.

> **CNN inference wins at bs=128** (5.69 ms vs PyTorch 8.00 ms) and is within ~15%
> at smaller batches — the "uniformly slower" framing does not hold for CNN.

> **Transformer** is where AiDotNet trails most (5–16× on inference at bs≥8). The
> encoder attention `Predict()` walk does not yet route to the fused-MHA/SDPA path
> the Tensors micro-benchmarks win on; that wiring is the largest remaining gap.

> **`mlp-fused`** (AiDotNet-only variant calling `IEngine.MlpForward` directly) runs
> bs=128 inference at **1.04 ms** vs the high-level `Predict()` MLP's 1.24 ms — the
> generic `FeedForwardNeuralNetwork.Predict` path still does not fully exploit the
> fused multi-layer kernel even on 0.207.13.

> Memory: AiDotNet's RSS still includes the one-time OpenCL kernel-cache compile at
> startup even with `AIDOTNET_DISABLE_GPU=1` setting the engine to CPU; it is a fixed
> per-process cost amortized across all models, not per-model overhead. RSS is the
> correct metric (matches `psutil` on the PyTorch side) but is not a clean
> model-weight comparison.

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

# 3. Per-model scaffold (AiDotNet 0.207.13 / Tensors 0.91.1).
#    Pin CPU + threads on both sides; AISEVAL_FUSED_DIAG=1 prints whether the
#    compiled/fused training step engaged (expect "Hit=True" + 60 fused steps).
cd ../aidotnet-benchmarks
#   PowerShell: $env:AIDOTNET_DISABLE_GPU=1; $env:AIDOTNET_BLAS_THREADS=8; $env:AISEVAL_FUSED_DIAG=1
AIDOTNET_DISABLE_GPU=1 AIDOTNET_BLAS_THREADS=8 AISEVAL_FUSED_DIAG=1 \
  dotnet run --no-build -c Release -- \
    --models mlp,cnn,lstm,transformer,mlp-fused \
    --epochs 3 --train-batches 20 --batch-size 64 \
    --inference-iterations 100 --warmup-iterations 10 \
    --output ../results/aidotnet.json
cd ../pytorch-benchmarks && python -m pytorch_benchmarks \
    --models mlp,cnn,lstm,transformer --device cpu --threads 8 \
    --epochs 3 --train-batches 20 --batch-size 64 \
    --inference-iterations 100 --warmup-iterations 10 \
    --output ../results/pytorch.json

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
