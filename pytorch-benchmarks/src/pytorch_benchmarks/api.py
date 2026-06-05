from __future__ import annotations

from fastapi import FastAPI

from pytorch_benchmarks.controllers.benchmark_controller import router as benchmark_router
from pytorch_benchmarks.controllers.regression_controller import router as regression_router


def create_app() -> FastAPI:
    app = FastAPI(title="PyTorch Benchmarks API")
    app.include_router(regression_router)
    app.include_router(benchmark_router)
    return app


app = create_app()
