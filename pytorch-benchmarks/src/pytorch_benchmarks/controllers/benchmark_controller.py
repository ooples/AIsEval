"""Controller route for the four-model PyTorch benchmark.

Mirrors the AiDotNet side's ``POST /api/Benchmark/Models`` so the
``BothController`` fan-out (and any developer with two running hosts) can
trigger both frameworks' benchmarks over HTTP and compare the JSON reports —
the same controller-shaped workflow as the regression endpoints.

The implementation delegates to the exact same ``run(...)`` used by the
``python -m pytorch_benchmarks`` CLI, so endpoint mode and CLI mode measure
identical workloads.
"""

from __future__ import annotations

import argparse
import threading

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pytorch_benchmarks.__main__ import run

router = APIRouter(prefix="/api/Benchmark", tags=["Benchmark"])

# One benchmark at a time: a run saturates the CPU for minutes; two concurrent
# runs would each be measuring the other's contention. Mirrors the AiDotNet
# BenchmarkController's run gate.
_run_gate = threading.Lock()


@router.get("/Test")
def test() -> str:
    return "Benchmark controller is ready. POST api/Benchmark/Models to run."


@router.post("/Models")
def models(
    models: str = "mlp,cnn,lstm,transformer",
    epochs: int = 3,
    trainBatches: int = 20,
    batchSize: int = 64,
    inferenceIterations: int = 100,
    warmupIterations: int = 10,
    seed: int = 1234,
    device: str = "cpu",
    threads: int = 8,
) -> JSONResponse:
    """Run the benchmark for the requested model families and return the report.

    Query parameter names match the AiDotNet ``BenchmarkController`` so the
    ``/api/Both/Benchmark`` fan-out can forward one query string to both hosts.
    ``device`` defaults to ``cpu`` and ``threads`` to 8 — the fair-comparison
    defaults (the AiDotNet side pins its CPU engine and
    ``AIDOTNET_BLAS_THREADS=8``).
    """
    names = [item.strip() for item in models.split(",") if item.strip()]
    if not names:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide at least one model: mlp, cnn, lstm, transformer."},
        )
    if epochs < 1 or trainBatches < 1 or batchSize < 1 or inferenceIterations < 1 or warmupIterations < 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Workload parameters must be positive (warmupIterations may be 0)."},
        )

    if not _run_gate.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={
                "error": "A benchmark run is already in progress. Concurrent runs would "
                         "contend for CPU and corrupt both measurements."
            },
        )
    try:
        args = argparse.Namespace(
            models=",".join(names),
            device=device,
            epochs=epochs,
            train_batches=trainBatches,
            batch_size=batchSize,
            inference_iterations=inferenceIterations,
            warmup_iterations=warmupIterations,
            seed=seed,
            threads=threads,
        )
        try:
            report = run(args)
        except ValueError as exc:  # unknown model name from make_model
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return JSONResponse(status_code=200, content=report)
    finally:
        _run_gate.release()
