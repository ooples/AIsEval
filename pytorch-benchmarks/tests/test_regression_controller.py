from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import Response
from starlette.datastructures import FormData

import torch

from pytorch_benchmarks.controllers.regression_controller import (
    CsvRegressionInput,
    RegressionDidNotConvergeError,
    _find_csv_input,
    _gpu_metrics,
    _predict_with_torch_least_squares,
    _predict_with_torch_training_loop,
    _read_request_form,
)


class _MissingMultipartRequest:
    headers = {"content-type": "multipart/form-data; boundary=test"}

    async def form(self) -> FormData:
        raise AssertionError(
            "The `python-multipart` library must be installed to use form parsing."
        )


def test_find_csv_input_accepts_text_form_fields() -> None:
    form = FormData(
        [
            ("features", "x,y\n1,2\n"),
            ("tests", "x\n3\n"),
        ]
    )

    features = _find_csv_input(form, "features", "features.csv")
    tests = _find_csv_input(form, "tests", "tests.csv")

    assert isinstance(features, CsvRegressionInput)
    assert isinstance(tests, CsvRegressionInput)
    assert asyncio.run(features.read()) == b"x,y\n1,2\n"
    assert asyncio.run(tests.read()) == b"x\n3\n"


class _JsonContentTypeRequest:
    headers = {"content-type": "application/json"}

    async def form(self) -> FormData:
        raise AssertionError("form parsing should not be attempted")


def test_read_request_form_reports_received_content_type_for_postman_misconfiguration() -> (
    None
):
    response = asyncio.run(_read_request_form(_JsonContentTypeRequest()))

    assert response.status_code == 400
    body = json.loads(response.body)
    assert "Received Content-Type: application/json" in body["error"]
    assert "delete any manually configured Content-Type header" in body["error"]


def test_read_request_form_reports_missing_python_multipart() -> None:
    response = asyncio.run(_read_request_form(_MissingMultipartRequest()))

    assert response.status_code == 500
    body = json.loads(response.body)
    assert "python-multipart" in body["error"]
    assert "python -m pip install -e ." in body["error"]


class _TextFormRequest:
    headers = {"content-type": "application/x-www-form-urlencoded"}

    async def form(self) -> FormData:
        return FormData(
            [
                ("features", "x,y\n1,3\n2,5\n3,7\n"),
                ("tests", "x\n4\n"),
            ]
        )


def test_predict_accepts_text_form_fields() -> None:
    from pytorch_benchmarks.controllers.regression_controller import (
        CsvRegressionResponse,
        predict,
    )

    fastapi_response = Response()
    response = asyncio.run(predict(_TextFormRequest(), fastapi_response, use_gpu=False))

    assert isinstance(response, CsvRegressionResponse)
    assert response.trainingRows == 3
    assert response.testRows == 1
    assert response.featureCount == 1
    assert response.timings.timing_unit == "milliseconds"
    assert response.timings.total_ms > 0
    assert fastapi_response.headers["Server-Timing"].startswith("app;dur=")
    assert response.predictions[0].prediction == pytest.approx(9.0)


def test_gpu_metrics_does_not_probe_gpu_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import pytorch_benchmarks.controllers.regression_controller as controller

    def fail_query() -> dict[str, float | str]:
        raise AssertionError("nvidia-smi should not be queried when UseGPU=false")

    monkeypatch.setattr(controller, "_query_nvidia_smi", fail_query)

    metrics = _gpu_metrics(
        torch.device("cpu"), gpu_requested=False, gpu_used=False, kernel_ms=None
    )

    assert metrics.name is None
    assert metrics.utilization_percent is None
    assert metrics.memory_allocated_mb is None


def _unscaled_regression() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Columns on the benchmark's scales: square feet in the thousands, a rating near 5, a price
    # in the hundreds of thousands.
    generator = torch.Generator().manual_seed(7)
    x = torch.stack(
        [
            torch.rand(200, generator=generator, dtype=torch.float64) * 3000 + 500,
            torch.rand(200, generator=generator, dtype=torch.float64) * 10,
        ],
        dim=1,
    )
    noise = torch.randn(200, generator=generator, dtype=torch.float64) * 5000
    y = (150 * x[:, 0] + 20000 * x[:, 1] + 50000 + noise).unsqueeze(1)
    return x, y, x[:20]


def test_training_loop_matches_least_squares_on_unscaled_columns() -> None:
    x, y, x_tests = _unscaled_regression()

    trained = _predict_with_torch_training_loop(x, y, x_tests)
    solved = _predict_with_torch_least_squares(x, y, x_tests)

    relative_rmse = ((trained - solved) ** 2).mean().sqrt() / solved.abs().mean()
    assert float(relative_rmse) < 1e-4


def test_training_loop_rejects_an_unconverged_fit() -> None:
    x, y, x_tests = _unscaled_regression()

    with pytest.raises(RegressionDidNotConvergeError):
        _predict_with_torch_training_loop(x, y, x_tests, epochs=5)
