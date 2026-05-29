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

> **Refreshed 2026-05-29** with the latest packages: **AiDotNet 0.207.9 +
> AiDotNet.Tensors 0.86.4** (was 0.207.0 / 0.86.1). The headline change is
> that **AiDotNet's LSTM benchmark now completes** — its inference path used
> to never finish at the PyTorch-default workload (see the prior note below);
> with the 0.86.x fused LSTM-inference wiring it runs end-to-end. Transformer
> inference also dropped ~3.8× (233 ms → 61 ms at bs=128) from the fused-MHA /
> SDPA work in Tensors 0.86.x. PyTorch was re-run on the same rig in the same
> session; it came out faster than the prior table (the machine was less
> contended this run), so the absolute gaps shifted even though the AiDotNet
> side improved — read the two sides as measured together, not against the
> older PyTorch column.

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

3. **The compiled fused-training path silently never runs.** `EnableCompilation`
   defaults to `true`, so `CompiledTapeTrainingStep.TryStepWithFusedOptimizer`
   is *attempted on every `Train()` step* — but it **falls back to the eager tape
   every time** because `TryMapToFusedOptimizerConfig` rejects the model's default
   optimizer (`FusedOptimizerPathEvent: Hit=False, Reason="optimizer
   AdamOptimizer\`3 not compatible with fused kernel"`). The fallback is **silent**
   at the default diagnostic level — it only surfaces at
   `TrainingDiagnosticsConfig.Level = PerStep`, which is how it was found here
   (run with `AISEVAL_FUSED_DIAG=1`). So the env vars `AIDOTNET_COMPILED_BACKWARD`
   / `AIDOTNET_CROSS_LAYER_FUSION` had no effect — the *framework* fused-training
   step was already gated off by optimizer incompatibility, and those env vars
   only toggle an unrelated Tensors-side backward-walk that optimizes graph-walk
   dispatch (negligible vs the GEMM compute that dominates). **This is the
   AiDotNet-side bug to fix**: either make the fused mapper accept the default
   optimizer config, default models to a fused-compatible optimizer, or — at
   minimum — stop the fallback from being silent.

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
| PyTorch | 2.11.0+cpu (eager) |
| .NET | 10.0.8 |
| AiDotNet | 0.207.9 NuGet (assembly 0.204.0.0); was 0.185.0 pre-fix |
| AiDotNet.Tensors | 0.86.4 NuGet |
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

Training time (3 epochs × 20 batches × batch_size 64), steady-state inference latency at `batch_size=128`, and throughput. All four models now run end-to-end on both sides (AiDotNet 0.207.9 / Tensors 0.86.4, PyTorch 2.11.0+cpu eager), same rig, same session:

| Model | params (PT / AiD) | train total (PT / AiD) | grad/batch (PT / AiD) | bs=128 latency (PT / AiD) | bs=128 throughput sps (PT / AiD) |
|---|---|---|---|---|---|
| MLP | 468 874 / 468 874 | 0.35s / 1.66s | 1.1 ms / 21.1 ms | **0.58 ms** / 12.33 ms (AiD ~21× slower) | 219 028 / 10 385 |
| CNN | 9 930 / 20 490 | 0.63s / 1.86s | 4.3 ms / 27.3 ms | 6.09 ms / **5.97 ms** (≈ tied, AiD marginally faster) | 21 026 / 21 451 |
| LSTM | 25 738 / 24 832 | 0.47s / 26.49s | 2.7 ms / 432.4 ms | 1.98 ms / 8.24 ms (AiD 4.2× slower — **now finishes**) | 64 693 / 15 532 |
| Transformer | 69 706 / 51 840 | 2.53s / 8.87s (3.5× slower) | 9.5 ms / 108.5 ms | 6.43 ms / 60.63 ms (AiD 9.4× slower) | 19 917 / 2 111 |

> **LSTM now completes (the headline change).** On 0.207.0 / 0.86.1 AiDotNet's LSTM `Predict()` never finished at this workload — the inference loop (92 forward passes over `[B=128, seq=32, features=32]`) was still running after >3 minutes. With the 0.86.x fused LSTM-inference wiring it runs end-to-end: bs=128 steady-state latency is **8.24 ms** (vs PyTorch 1.98 ms — a real 4.2× gap, but a *measured* one, not an infinite hang). The remaining LSTM gap has moved to the **training** path (26.5 s vs PyTorch 0.47 s — BPTT through the recurrent cell is the bottleneck now; PyTorch's `nn.LSTM` routes to a fused OneDNN/MKL kernel).

> **Transformer improved ~3.8× on inference** (prior 233 ms → 61 ms at bs=128) and ~1.8× on training (16.2 s → 8.9 s) from the fused-QKV / transpose-fused-SDPA work in Tensors 0.86.x. It is still 9.4× slower than eager PyTorch at bs=128 here.

> **CNN inference is ≈ tied** with eager PyTorch at bs=128 (5.97 ms vs 6.09 ms). The prior "AiDotNet is uniformly slower" framing does not hold for CNN inference.

> **MLP** is where AiDotNet trails most on this rig (~21× slower bs=128 inference). PyTorch's `nn.Linear` stack on a 784→512→128→10 net is essentially three OneDNN GEMMs; the gap is the per-op dispatch/threading overhead around small dense matmuls, not the matmul math itself.

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

# 3. Per-model scaffold (LSTM now finishes on 0.207.9 / Tensors 0.86.4)
cd ../aidotnet-benchmarks && dotnet run --no-build -c Release -- \
    --models mlp,cnn,lstm,transformer \
    --epochs 3 --train-batches 20 --batch-size 64 \
    --inference-iterations 100 --warmup-iterations 10 \
    --output ../results/aidotnet.json
cd ../pytorch-benchmarks && python src/pytorch_benchmarks \
    --models mlp,cnn,lstm,transformer \
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
