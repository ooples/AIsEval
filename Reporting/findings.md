AiDotNet vs Pytorch Benchmarks
==============================

> ⚠️ **Prior numbers withdrawn.** The findings previously listed below were
> not a framework-vs-framework comparison. They compared **AiDotNet's full
> `AiModelBuilder` lifecycle** (DataLoader config + model config + internal
> 70/15/15 train/validation/test split + async build + `Predict`) against
> **a 5-line `torch.linalg.lstsq` wrapper around LAPACK's DGELSD solver**
> (`_predict_with_torch_least_squares` in `regression_controller.py`).
> Those are different kinds of work: one is a complete ML lifecycle, the
> other is a single LAPACK call.
>
> The fair-comparison fixes in this branch add a `/MultipleRegression`
> endpoint on the PyTorch side that mirrors AiDotNet's builder ceremony
> with `nn.Linear + MSELoss + Adam` training, so head-to-head timings can
> be apples-to-apples. The `Program.cs` scaffold also now constructs real
> AiDotNet networks (`FeedForwardNeuralNetwork`, `ConvolutionalNeuralNetwork`,
> `LSTMNeuralNetwork`, `Transformer` via `TransformerEncoderLayer`) for
> the four model families instead of using a hand-rolled MLP for every
> name. See `Reporting/PRIOR-FINDINGS-DISCLAIMER.md` for the detailed
> audit of why the original numbers should not be cited.

After re-running with the fair-comparison endpoints + real AiDotNet models,
re-populate the comparison table below.

Datasets used:
- small (100 rows)
- large (48000 rows)

Endpoints for the regression-style comparison:
- AiDotNet: `POST /api/Regression/MultipleRegression`
  (runs `AiModelBuilder<>.ConfigureModel(new MultipleRegression<double>())`)
- PyTorch: `POST /api/Regression/MultipleRegression`
  (runs `nn.Linear + MSELoss + Adam` training loop — matches the builder
  lifecycle the AiDotNet side runs)

The `POST /api/Regression/Predict` (PyTorch raw `torch.linalg.lstsq`) and
`POST /api/Regression/SimpleRegression` routes remain available for the
LAPACK-vs-builder timing point, but they should not be cited as a
framework-to-framework comparison.

Per-model wall-time comparison (MLP / CNN / LSTM / Transformer)
---------------------------------------------------------------

Run from `aidotnet-benchmarks/` and `pytorch-benchmarks/`. Both sides now
construct equivalent real networks. Re-populate after a clean run with
fixed seeds + identical hardware. Reference shapes:

| Family       | AiDotNet construction (Program.cs)        | PyTorch construction (`__main__.py`)             |
|--------------|-------------------------------------------|---------------------------------------------------|
| MLP          | DenseLayer(512,ReLU) → DenseLayer(128,ReLU) → DenseLayer(10) | nn.Linear(784,512) → ReLU → nn.Linear(512,128) → ReLU → nn.Linear(128,10) |
| CNN          | ConvolutionalLayer(16,3,pad=1,ReLU) → MaxPool(2,2) → ConvolutionalLayer(32,3,pad=1,ReLU) → MaxPool(2,2) → FlattenLayer → DenseLayer(10) | Conv2d(1,16,3,pad=1) → ReLU → MaxPool(2) → Conv2d(16,32,3,pad=1) → ReLU → AdaptiveAvgPool((4,4)) → Linear(512,10) |
| LSTM         | LSTMLayer(64) → DenseLayer(10)            | nn.LSTM(input=32, hidden=64) → Linear(64,10)     |
| Transformer  | DenseLayer(64) → TransformerEncoderLayer(heads=4, ff=128) ×2 → DenseLayer(10) | nn.Linear(32,64) → 2× TransformerEncoderLayer(d=64, h=4, ff=128) → mean → Linear(64,10) |

Memory measurement parity
-------------------------

Both sides now measure peak RSS (resident-set size):
- C# `BenchmarkInference`: `Process.WorkingSet64` (was `GC.GetTotalMemory()` — managed heap only)
- Python: `psutil.Process(...).memory_info().rss`

Comparing these two metrics is now apples-to-apples; the pre-fix comparison
was managed-heap-bytes vs full-process-RSS which can't be compared directly.

Software versions
-----------------

- `aidotnet-benchmarks/AiDotNetBenchmarks.csproj` references **AiDotNet 0.207.0**
  (was 0.185.0 — 22 releases behind, missing substantial perf work)
- Latest available NuGet release: track via `https://www.nuget.org/packages/AiDotNet`

For raw report data, see the `.json` files in this folder.
