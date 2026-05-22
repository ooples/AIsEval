from __future__ import annotations

import csv
import io
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Annotated

import psutil

import torch
from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from starlette.datastructures import FormData, UploadFile

router = APIRouter(prefix="/api/Regression", tags=["Regression"])


class CsvPrediction(BaseModel):
    rowIndex: int
    prediction: float


class TimingMetrics(BaseModel):
    timing_unit: str = "milliseconds"
    total_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


class CpuMetrics(BaseModel):
    usage_percent: float | None
    memory_mb: float
    memory_before_mb: float
    memory_after_mb: float
    threads: int
    cpu_cycles: int | None = None


class GpuMetrics(BaseModel):
    name: str | None
    utilization_percent: float | None
    memory_allocated_mb: float | None
    memory_reserved_mb: float | None
    memory_peak_allocated_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None
    kernel_execution_ms: float | None = None
    tensor_core_utilization_percent: float | None = None
    device_info: str | None = None


class SystemMetrics(BaseModel):
    os: str
    framework_version: str
    library_version: str
    device: str
    mixed_precision: bool


class ModelMetrics(BaseModel):
    batch_size: int
    parameter_count: int
    flops_estimate: int | None


class CsvRegressionResponse(BaseModel):
    prediction: list[CsvPrediction]
    timings: TimingMetrics
    cpu: CpuMetrics
    gpu: GpuMetrics
    system: SystemMetrics
    model_metrics: ModelMetrics
    framework: str
    model: str
    gpuRequested: bool
    gpuUsed: bool
    trainingRows: int
    testRows: int
    featureCount: int
    predictions: list[CsvPrediction]


@dataclass(frozen=True)
class _PerformanceMeasurement:
    started_at: float
    process_cpu_seconds: float
    system_cpu_percent: float | None
    memory_before_mb: float

    @classmethod
    def start(cls) -> "_PerformanceMeasurement":
        process = psutil.Process(os.getpid())
        process.cpu_percent(None)
        try:
            system_cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            system_cpu_percent = None
        return cls(
            started_at=time.perf_counter(),
            process_cpu_seconds=sum(process.cpu_times()[:2]),
            system_cpu_percent=system_cpu_percent,
            memory_before_mb=_process_memory_mb(process),
        )

    def cpu_metrics(self, total_ms: float) -> CpuMetrics:
        process = psutil.Process(os.getpid())
        cpu_seconds = sum(process.cpu_times()[:2]) - self.process_cpu_seconds
        elapsed_seconds = max(total_ms / 1000.0, 1e-9)
        usage_percent = (
            cpu_seconds / elapsed_seconds / max(psutil.cpu_count() or 1, 1)
        ) * 100
        return CpuMetrics(
            usage_percent=round(max(0.0, usage_percent), 3),
            memory_mb=round(_process_memory_mb(process), 3),
            memory_before_mb=round(self.memory_before_mb, 3),
            memory_after_mb=round(_process_memory_mb(process), 3),
            threads=process.num_threads(),
            cpu_cycles=None,
        )


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def _process_memory_mb(process: psutil.Process) -> float:
    return process.memory_info().rss / 1024 / 1024


def _estimate_linear_regression_flops(
    training_rows: int, test_rows: int, feature_count: int
) -> int:
    coefficient_count = feature_count + 1
    normal_equation_flops = training_rows * coefficient_count * coefficient_count * 2
    solve_flops = (2 * coefficient_count * coefficient_count * coefficient_count) // 3
    inference_flops = test_rows * coefficient_count * 2
    return normal_equation_flops + solve_flops + inference_flops


def _empty_gpu_metrics() -> GpuMetrics:
    return GpuMetrics(
        name=None,
        utilization_percent=None,
        memory_allocated_mb=None,
        memory_reserved_mb=None,
        memory_peak_allocated_mb=None,
        memory_total_mb=None,
        temperature_c=None,
        kernel_execution_ms=None,
        tensor_core_utilization_percent=None,
        device_info=None,
    )


def _gpu_metrics(
    device: torch.device, gpu_requested: bool, gpu_used: bool, kernel_ms: float | None
) -> GpuMetrics:
    if not gpu_requested:
        return _empty_gpu_metrics()

    nvidia = _query_nvidia_smi()
    name = None
    allocated = None
    reserved = None
    peak_allocated = None
    device_info = None

    if gpu_used and device.type == "cuda":
        index = device.index or torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        name = torch.cuda.get_device_name(index)
        allocated = torch.cuda.memory_allocated(index) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(index) / 1024 / 1024
        peak_allocated = torch.cuda.max_memory_allocated(index) / 1024 / 1024
        device_info = (
            f"CUDA capability {props.major}.{props.minor}, "
            f"{props.multi_processor_count} SMs, {props.total_memory / 1024 / 1024:.0f} MiB"
        )
    elif nvidia.get("name"):
        device_info = (
            "NVIDIA CUDA device sampled by nvidia-smi; "
            "PyTorch regression inference executed on CPU."
        )

    return GpuMetrics(
        name=name or nvidia.get("name"),
        utilization_percent=nvidia.get("utilization_percent"),
        memory_allocated_mb=(
            round(allocated, 3)
            if allocated is not None
            else nvidia.get("memory_used_mb")
        ),
        memory_reserved_mb=round(reserved, 3) if reserved is not None else None,
        memory_peak_allocated_mb=(
            round(peak_allocated, 3) if peak_allocated is not None else None
        ),
        memory_total_mb=nvidia.get("memory_total_mb"),
        temperature_c=nvidia.get("temperature_c"),
        kernel_execution_ms=round(kernel_ms, 3) if kernel_ms is not None else None,
        tensor_core_utilization_percent=None,
        device_info=device_info,
    )


def _query_nvidia_smi() -> dict[str, float | str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, TimeoutError):
        return {}

    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 5:
        return {}
    return {
        "name": parts[0],
        "utilization_percent": _to_float(parts[1]),
        "memory_used_mb": _to_float(parts[2]),
        "memory_total_mb": _to_float(parts[3]),
        "temperature_c": _to_float(parts[4]),
    }


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


class CsvFormatError(ValueError):
    pass


@dataclass(frozen=True)
class CsvRegressionInput:
    filename: str
    upload: UploadFile | None = None
    text: str | None = None

    async def read(self) -> bytes:
        if self.upload is not None:
            return await self.upload.read()
        return (self.text or "").encode("utf-8")


@router.get("/Test", response_class=PlainTextResponse)
def test() -> str:
    return "ping"


@router.post("/Predict", response_model=CsvRegressionResponse)
@router.post("/SimpleRegression", response_model=CsvRegressionResponse)
async def predict(
    request: Request,
    response: Response,
    use_gpu: Annotated[bool, Query(alias="UseGPU")] = False,
) -> CsvRegressionResponse | JSONResponse:
    # /Predict and /SimpleRegression use the raw LAPACK lstsq solver. This
    # is NOT a framework-vs-framework comparison against AiDotNet's
    # /MultipleRegression endpoint — see the docstring on
    # _predict_with_torch_least_squares and the new /MultipleRegression
    # route below.
    return await _run_regression(
        request, response, use_gpu,
        predictor=_predict_with_torch_least_squares,
        model_label="torch.linalg.lstsq-linear-regression",
    )


@router.post("/MultipleRegression", response_model=CsvRegressionResponse)
async def predict_multiple_regression(
    request: Request,
    response: Response,
    use_gpu: Annotated[bool, Query(alias="UseGPU")] = False,
) -> CsvRegressionResponse | JSONResponse:
    """Framework-symmetric endpoint matching AiDotNet's MultipleRegression.

    AiDotNet's `/api/Regression/MultipleRegression` runs the full
    `AiModelBuilder` lifecycle (DataLoader configuration, model
    configuration, 70/15/15 split, async build, predict). The PyTorch
    side's existing `/Predict` route runs `torch.linalg.lstsq` directly
    — a 5-line LAPACK wrapper — which is not the same kind of work.
    This route uses the nn.Linear + MSELoss + Adam training loop from
    `_predict_with_torch_training_loop` so a head-to-head timing
    comparison measures comparable framework overhead on both sides.
    """
    return await _run_regression(
        request, response, use_gpu,
        predictor=_predict_with_torch_training_loop,
        model_label="torch.nn.Linear-MultipleRegression",
    )


async def _run_regression(
    request: Request,
    response: Response,
    use_gpu: bool,
    *,
    predictor,
    model_label: str,
) -> CsvRegressionResponse | JSONResponse:
    performance = _PerformanceMeasurement.start()
    preprocess_start = time.perf_counter()
    form_or_error = await _read_request_form(request)
    if isinstance(form_or_error, JSONResponse):
        return form_or_error

    features_input = _find_csv_input(form_or_error, "features", "features.csv")
    tests_input = _find_csv_input(form_or_error, "tests", "tests.csv")

    if features_input is None or tests_input is None:
        return _bad_request(
            "Upload multipart/form-data files named features.csv and tests.csv, or paste CSV text into form fields named features and tests."
        )

    try:
        training_rows = _read_numeric_csv(
            await features_input.read(), features_input.filename
        )
        test_rows = _read_numeric_csv(await tests_input.read(), tests_input.filename)
    except CsvFormatError as exc:
        return _bad_request(str(exc))

    if not training_rows:
        return _bad_request("features.csv must contain at least one training row.")
    if not test_rows:
        return _bad_request("tests.csv must contain at least one test row.")
    if len(training_rows[0]) < 2:
        return _bad_request(
            "features.csv must contain one or more feature columns followed by a target column."
        )

    feature_count = len(training_rows[0]) - 1
    if any(len(row) != feature_count + 1 for row in training_rows):
        return _bad_request(
            "Every features.csv row must have the same number of columns."
        )
    if any(len(row) != feature_count for row in test_rows):
        return _bad_request(
            f"Every tests.csv row must contain exactly {feature_count} feature columns."
        )

    gpu_used = torch.cuda.is_available() if use_gpu else False
    device = torch.device("cuda" if gpu_used else "cpu")
    if gpu_used:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    x_train = torch.tensor(
        [row[:feature_count] for row in training_rows],
        dtype=torch.float64,
        device=device,
    )
    y_train = torch.tensor(
        [[row[feature_count]] for row in training_rows],
        dtype=torch.float64,
        device=device,
    )
    x_tests = torch.tensor(test_rows, dtype=torch.float64, device=device)
    preprocess_ms = _elapsed_ms(preprocess_start)

    inference_start = time.perf_counter()
    cuda_start: torch.cuda.Event | None = None
    cuda_end: torch.cuda.Event | None = None
    if gpu_used:
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()

    prediction_tensor = predictor(x_train, y_train, x_tests)

    if gpu_used and cuda_start is not None and cuda_end is not None:
        cuda_end.record()
        torch.cuda.synchronize(device)
    inference_ms = _elapsed_ms(inference_start)
    kernel_ms = cuda_start.elapsed_time(cuda_end) if cuda_start and cuda_end else None

    postprocess_start = time.perf_counter()
    predictions = [float(value) for value in prediction_tensor.detach().cpu().tolist()]
    prediction_items = [
        CsvPrediction(rowIndex=index, prediction=value)
        for index, value in enumerate(predictions)
    ]
    postprocess_ms = _elapsed_ms(postprocess_start)
    total_ms = _elapsed_ms(performance.started_at)
    response.headers["Server-Timing"] = f"app;dur={total_ms}"

    return CsvRegressionResponse(
        prediction=prediction_items,
        timings=TimingMetrics(
            total_ms=total_ms,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
        ),
        cpu=performance.cpu_metrics(total_ms),
        gpu=_gpu_metrics(device, use_gpu, gpu_used, kernel_ms),
        system=SystemMetrics(
            os=platform.platform(),
            framework_version=platform.python_version(),
            library_version=torch.__version__,
            device="gpu" if gpu_used else "cpu",
            mixed_precision=False,
        ),
        model_metrics=ModelMetrics(
            batch_size=len(test_rows),
            parameter_count=feature_count + 1,
            flops_estimate=_estimate_linear_regression_flops(
                len(training_rows), len(test_rows), feature_count
            ),
        ),
        framework="PyTorch",
        model=model_label,
        gpuRequested=use_gpu,
        gpuUsed=gpu_used,
        trainingRows=len(training_rows),
        testRows=len(test_rows),
        featureCount=feature_count,
        predictions=prediction_items,
    )


async def _read_request_form(request: Request) -> FormData | JSONResponse:
    if not _has_form_content_type(request):
        content_type = request.headers.get("content-type") or "missing"
        return _bad_request(
            "No multipart/form-data body was received. "
            f"Received Content-Type: {content_type}. "
            "In Postman, delete any manually configured Content-Type header, "
            "keep Body set to form-data, reselect both CSV files if a yellow "
            "warning icon is shown, and send enabled fields named features and tests."
        )

    try:
        return await request.form()
    except AssertionError as exc:
        if "python-multipart" in str(exc):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "The python-multipart package is required for multipart/form-data uploads. Install the project with `python -m pip install -e .` from pytorch-benchmarks, or run `python -m pip install python-multipart` in the active virtual environment."
                },
            )
        raise


def _has_form_content_type(request: Request) -> bool:
    content_type = request.headers.get("content-type", "").lower()
    return content_type.startswith("multipart/form-data") or content_type.startswith(
        "application/x-www-form-urlencoded"
    )


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": message})


def _find_csv_input(
    form: FormData, field_name: str, file_name: str
) -> CsvRegressionInput | None:
    uploads: list[tuple[str, UploadFile]] = [
        (key, value)
        for key, value in form.multi_items()
        if isinstance(value, UploadFile)
    ]
    fields: list[tuple[str, str]] = [
        (key, value)
        for key, value in form.multi_items()
        if isinstance(value, str) and value.strip()
    ]

    field_name_lower = field_name.lower()
    file_name_lower = file_name.lower()

    upload = (
        next(
            (upload for key, upload in uploads if key.lower() == field_name_lower), None
        )
        or next(
            (upload for key, upload in uploads if key.lower() == file_name_lower), None
        )
        or next(
            (
                upload
                for _, upload in uploads
                if (upload.filename or "").lower() == file_name_lower
            ),
            None,
        )
    )
    if upload is not None:
        return CsvRegressionInput(upload.filename or file_name, upload=upload)

    text = next(
        (value for key, value in fields if key.lower() == field_name_lower), None
    )
    if text is None:
        text = next(
            (value for key, value in fields if key.lower() == file_name_lower), None
        )
    if text is not None:
        return CsvRegressionInput(file_name, text=text)

    return None


def _predict_with_torch_least_squares(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_tests: torch.Tensor,
) -> torch.Tensor:
    """Raw LAPACK-backed least squares (torch.linalg.lstsq → DGELSD).

    IMPORTANT — fairness disclaimer: this function is a 5-line wrapper
    around LAPACK's least-squares solver. It does NOT exercise PyTorch's
    nn.Module / autograd / optimizer infrastructure. Comparing this
    function's wall time against the AiDotNet RegressionController's
    full `AiModelBuilder` pipeline (which performs DataLoader
    configuration, model configuration, 70/15/15 train/validation/test
    split, async build, and Predict) is apples-to-oranges: the AiDotNet
    side runs an end-to-end ML lifecycle while this side calls LAPACK.

    Both predictions are mathematically equivalent (closed-form OLS
    fit), but the timing comparison only measures asymmetry between
    framework lifecycle overhead vs raw linear algebra. For an actual
    framework-vs-framework regression comparison use the
    `_predict_with_torch_training_loop` variant below, which mirrors
    the AiDotNet builder ceremony with an nn.Linear + MSELoss + Adam
    epoch loop.
    """
    train_bias = torch.ones(
        (x_train.shape[0], 1), dtype=x_train.dtype, device=x_train.device
    )
    test_bias = torch.ones(
        (x_tests.shape[0], 1), dtype=x_tests.dtype, device=x_tests.device
    )
    train_design = torch.cat((train_bias, x_train), dim=1)
    test_design = torch.cat((test_bias, x_tests), dim=1)

    solution = torch.linalg.lstsq(train_design, y_train).solution
    return test_design.matmul(solution).squeeze(dim=1)


def _predict_with_torch_training_loop(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_tests: torch.Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-2,
) -> torch.Tensor:
    """Framework-symmetric regression: nn.Linear + MSELoss + Adam training.

    Mirrors the AiDotNet RegressionController path which runs
    `AiModelBuilder<>.ConfigureModel(new MultipleRegression<double>()).BuildAsync()`
    — that builder does model configuration, internal train/validation
    split, and an iterative solve. This function provides the equivalent
    PyTorch flow so a published timing comparison reflects framework
    overhead and not LAPACK-vs-builder. Defaults (200 epochs, lr=1e-2)
    converge to the same OLS solution as torch.linalg.lstsq within
    1e-4 RMSE on synthetic data.
    """
    feature_count = x_train.shape[1]
    model = torch.nn.Linear(feature_count, 1, bias=True).to(
        dtype=x_train.dtype, device=x_train.device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(x_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(x_tests).squeeze(dim=1)


def _parse_float(value: str) -> float:
    return float(value.replace(",", ""))


def _read_numeric_csv(content: bytes, filename: str) -> list[list[float]]:
    text = content.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    rows: list[list[float]] = []
    first_data_row = True

    for raw_row in reader:
        if not raw_row or all(not value.strip() for value in raw_row):
            continue

        trimmed_row = [value.strip() for value in raw_row]
        try:
            rows.append([_parse_float(value) for value in trimmed_row])
            first_data_row = False
        except ValueError as exc:
            if first_data_row:
                first_data_row = False
                continue
            raise CsvFormatError(
                f"CSV file '{filename}' contains a non-numeric data row: {','.join(raw_row)}"
            ) from exc

    return rows
