Prior-findings disclaimer (pre fair-comparison-fixes)
=====================================================

Before the fair-comparison branch, this repo reported:

| Dataset           | AiDotNet | PyTorch |
|-------------------|----------|---------|
| Small (100 rows)  | ~190 ms  | ~650 ms |
| Large (48000 rows)| ~4100 ms | ~770 ms |

…with the conclusion that PyTorch is faster on large datasets and more
memory-efficient overall. **Those numbers should not be cited as a
framework-vs-framework comparison.** This file documents exactly why,
with line-number references to the source.

What the published numbers actually measured
--------------------------------------------

The JSON reports in `Reporting/aidotnet-small-set` and
`Reporting/pytorch-small-set` reveal the underlying paths:

```json
"framework":"AiDotNet",
"model":"MultipleRegression",
"parameter_count": 11,
"flops_estimate": 27023
```

The numbers come from the **REST API regression controllers**, not from
the MLP/CNN/LSTM/Transformer benchmark in `Program.cs` / `__main__.py`.
Specifically:

### AiDotNet side — `aidotnet-benchmarks/.../Controllers/RegressionController.cs:187-192`

```csharp
var multipleRegressionResult = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(new MultipleRegression<double>())
    .BuildAsync();
return multipleRegressionResult.Predict(testData);
```

This runs the full `AiModelBuilder` pipeline, which the in-file comment
(lines 168-174) acknowledges does **an internal 70/15/15 train/
validation/test split**. So every call performs:

1. DataLoader configuration
2. Model configuration ceremony
3. Async `BuildAsync()` (Task-based overhead)
4. Internal 70/15/15 train/validation/test split
5. Validation phase against the 15 % holdout
6. Final `Predict()` against `testData`

### PyTorch side — `pytorch-benchmarks/.../controllers/regression_controller.py:480-495` (pre-fix)

```python
def _predict_with_torch_least_squares(x_train, y_train, x_tests):
    train_bias = torch.ones((x_train.shape[0], 1), ...)
    test_bias  = torch.ones((x_tests.shape[0], 1), ...)
    train_design = torch.cat((train_bias, x_train), dim=1)
    test_design  = torch.cat((test_bias,  x_tests), dim=1)
    solution = torch.linalg.lstsq(train_design, y_train).solution
    return test_design.matmul(solution).squeeze(dim=1)
```

That's a **5-line wrapper around LAPACK's DGELSD solver**. No `nn.Module`.
No optimizer. No epoch loop. No training-step / validation-step. The same
LAPACK that AiDotNet, NumPy, MATLAB, and Eigen all use.

### Side-by-side

| Aspect | AiDotNet side | PyTorch side |
|---|---|---|
| Code path | Full `AiModelBuilder<>` lifecycle | Single LAPACK call |
| Train/test split | 70/15/15 internal | None — uses all rows |
| Validation phase | Yes (15 % holdout) | No |
| Builder ceremony | `ConfigureDataLoader` + `ConfigureModel` + `BuildAsync()` | None |
| Async overhead | Task-based `await` | Synchronous |
| Underlying solver | AiDotNet's solver | `torch.linalg.lstsq` → LAPACK DGELSD |
| What "inference_ms" measures | Builder + split + train + validate + Predict | LAPACK solve + matmul |

This is the equivalent of timing **"Word with full document-creation
pipeline"** against **"raw `printf`"** and concluding Word is slow at
printing characters.

For the small dataset: `parameter_count: 11`, `flops_estimate: 27023`.
That's 27 KFLOPs of actual arithmetic in ~190 ms. ~99.99 % of the wall
time on the AiDotNet side is framework overhead, not compute.

The `Program.cs` / `__main__.py` scaffold also has its own asymmetry
----------------------------------------------------------------------

Even if the published findings had come from this path (they didn't),
the `Program.cs` benchmark was structurally broken too. The pre-fix
`AiDotNetTensorBackend.Create` at `Program.cs:194-201` mapped every
model name to the same hand-rolled MLP:

```csharp
"mlp"         => new AiDotNetTensorModel(seed, 784, 10, [512, 128]),
"cnn"         => new AiDotNetTensorModel(seed, 784, 10, [256, 128]),       // not a CNN
"lstm"        => new AiDotNetTensorModel(seed, 1024, 10, [256, 64]),       // not an LSTM
"transformer" => new AiDotNetTensorModel(seed, 1024, 10, [512, 256, 64]),  // not a Transformer
```

`AiDotNetTensorModel.Forward()` at `Program.cs:242-255` is unconditionally
a stack of `Tensor<float>.MatrixMultiply` with ReLU between layers — no
`Conv2D`, no recurrence, no attention. The PyTorch side (`__main__.py:117-160`)
runs real `nn.Conv2d` / `nn.LSTM` / `nn.TransformerEncoderLayer` models.

Plus the C# `Backward()` at `Program.cs:257-268` does not compute gradients:

```csharp
for (var i = 0; i < weights.Length; i += 4) weights[i] += scale * 0.000001f;
```

And `Step()` at `Program.cs:270-280` does not run an optimizer:

```csharp
for (var i = 0; i < weights.Length; i += 16) weights[i] *= 0.99999f;
_weights[layer] = CreateTensor(weights, _widths[layer], _widths[layer + 1]);
```

The PyTorch side runs `loss.backward()` + `optim.AdamW.step()` for real.

Memory measurement was apples-to-oranges
----------------------------------------

Pre-fix C# `Program.cs:155, 167`:

```csharp
peak = Math.Max(peak, GC.GetTotalMemory(false) / 1024d / 1024d);
```

`GC.GetTotalMemory()` returns **only the .NET managed heap**. Excludes
native memory (BLAS, ArrayPool unused, JIT code, libtorch).

PyTorch side `__main__.py:62-76, 247`:

```python
self.rss_mb.append(self.process.memory_info().rss / 1024 / 1024)
```

`psutil` RSS is the **entire OS-visible process footprint** — Python
interpreter, libtorch `.so` files, all C++ allocations libtorch makes,
plus everything else in the process.

Comparing these two numbers and concluding "X is more memory-efficient"
has no meaning. The fair-comparison fixes change the C# side to use
`Process.WorkingSet64`, which is the Windows/Linux RSS equivalent.

Stale AiDotNet version
----------------------

`aidotnet-benchmarks/AiDotNetBenchmarks.csproj` pinned
`AiDotNet 0.185.0`. Current NuGet release at the time of writing this
disclaimer is `0.207.0` — 22 releases of perf work newer (BlasManaged
GEMM rewrite, FP64 Stage 1-9, INT8 row-scaled matmul, FlashAttention
CPU path, etc.). Fair-comparison branch bumps to 0.207.0.

The repo's own README disclaims publishing claims from this state
-----------------------------------------------------------------

`aidotnet-benchmarks/README.md` "Fairness Notes":

> *"The C# project currently includes a managed reference backend for
> validating benchmark infrastructure. **Add a production AiDotNet
> implementation behind `IBenchmarkModel` before publishing
> framework-to-framework claims.**"*

The original author explicitly noted that these numbers are scaffolding,
not a framework comparison.

What the fair-comparison branch changes
---------------------------------------

| Path | Pre-fix | Fair-comparison fix |
|------|---------|---------------------|
| `Program.cs` model factory | All names → same MLP | `mlp`/`cnn`/`lstm`/`transformer` → `FeedForwardNeuralNetwork` / `ConvolutionalNeuralNetwork` / `LSTMNeuralNetwork` / `FeedForwardNeuralNetwork` with `TransformerEncoderLayer` |
| `Program.cs` Backward | `weights[i] += 0.000001f` | `Network.Train(input, label)` → real autograd + Adam |
| `Program.cs` Step | `weights[i] *= 0.99999f` + tensor realloc | No-op (real optimizer step happens inside Train) |
| `Program.cs` memory metric | `GC.GetTotalMemory()` | `Process.WorkingSet64` (matches psutil RSS) |
| `regression_controller.py` regression endpoint | Single `torch.linalg.lstsq` call | New `/MultipleRegression` route runs `nn.Linear + MSELoss + Adam` training loop matching AiDotNet's builder lifecycle |
| `AiDotNetBenchmarks.csproj` package version | 0.185.0 | 0.207.0 |

After re-running with the fair-comparison branch, the numbers in
`findings.md` can be re-populated as an actual framework comparison.
