"""Runs the small (100 rows) + large (48000 rows) regression CSVs against
both REST APIs and writes a summary JSON.

Endpoints:
  AiDotNet: POST /api/Regression/SimpleRegression  (internally dispatches to
            MultipleRegression when feature count > 1 — full AiModelBuilder
            lifecycle: ConfigureDataLoader + ConfigureModel + BuildAsync +
            Predict, with internal 70/15/15 split.)
  PyTorch:  POST /api/Regression/MultipleRegression (real nn.Linear+MSELoss+
            Adam training loop — the fair-comparison endpoint added in this
            branch. The /Predict raw-LAPACK lstsq route is preserved for
            LAPACK-vs-builder timing but is not used here.)

Reports both client-wall-time (this process clock) and the server's
Server-Timing header (server-internal duration).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def post_csv(url: str, features: Path, tests: Path, timeout: float = 600.0) -> tuple[float, str, int, dict]:
    with features.open("rb") as f_feat, tests.open("rb") as f_test:
        files = {
            "features": ("features.csv", f_feat, "text/csv"),
            "tests":    ("tests.csv",    f_test, "text/csv"),
        }
        t0 = time.perf_counter()
        resp = requests.post(url, files=files, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    server_timing = resp.headers.get("Server-Timing", "")
    try:
        body = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
    except Exception:
        body = {}
    return elapsed_ms, server_timing, resp.status_code, body


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aidotnet-url", default="http://127.0.0.1:7000")
    p.add_argument("--pytorch-url",  default="http://127.0.0.1:8000")
    p.add_argument("--repeat-small", type=int, default=3)
    p.add_argument("--repeat-large", type=int, default=1)
    p.add_argument("--output", default="Reporting/regression-bench.json")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / "TestData" / "RegressionTestData" / "Complex"

    plan = [
        ("Small", "SmallSet", args.repeat_small),
        ("Large", "LargeSet", args.repeat_large),
    ]
    endpoints = [
        ("AiDotNet", f"{args.aidotnet_url}/api/Regression/SimpleRegression"),
        ("PyTorch",  f"{args.pytorch_url}/api/Regression/MultipleRegression"),
    ]

    results: list[dict] = []
    for size_label, subdir, repeat in plan:
        features = data_root / subdir / "features.csv"
        tests    = data_root / subdir / "tests.csv"
        for framework, url in endpoints:
            for i in range(1, repeat + 1):
                print(f"[bench] {framework} / {size_label} / run {i}/{repeat} … ", end="", flush=True)
                try:
                    ms, st, code, body = post_csv(url, features, tests)
                    print(f"{ms:.1f}ms (status={code}, ST={st!r})")
                    timings = body.get("timings") or {}
                    metrics = body.get("model_metrics") or {}
                    results.append({
                        "framework": framework, "size": size_label, "run": i,
                        "url": url, "client_ms": round(ms, 2),
                        "server_timing": st, "status": code,
                        "server_inference_ms": timings.get("inference_ms"),
                        "server_total_ms": timings.get("total_ms"),
                        "parameter_count": metrics.get("parameter_count"),
                        "flops_estimate": metrics.get("flops_estimate"),
                        "model": body.get("model"),
                    })
                except Exception as exc:
                    print(f"FAILED: {exc!r}")
                    results.append({
                        "framework": framework, "size": size_label, "run": i,
                        "url": url, "error": repr(exc),
                    })

    summary = {}
    for size_label, _, _ in plan:
        for framework, _ in endpoints:
            xs = [r["client_ms"] for r in results
                  if r.get("framework") == framework and r.get("size") == size_label and "client_ms" in r]
            if xs:
                summary[f"{framework}/{size_label}"] = {
                    "runs": len(xs),
                    "client_ms_min":  round(min(xs), 2),
                    "client_ms_mean": round(statistics.mean(xs), 2),
                    "client_ms_max":  round(max(xs), 2),
                }

    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "runs": results}, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
