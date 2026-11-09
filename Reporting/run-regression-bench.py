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
Server-Timing header (server-internal duration). Only a validated response (HTTP 200
with one finite prediction per test row) is timed; any other outcome is recorded as a
failure, left out of the summary, and makes the script exit nonzero.

Needs `requests`: install the harness with `python -m pip install -e ".[bench]"` from
pytorch-benchmarks/.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
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


def test_row_count(tests: Path) -> int:
    """Data rows in a tests CSV: every non-blank line after the header."""
    with tests.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip()) - 1


def validation_error(code: int, body: dict, expected_rows: int) -> str | None:
    """Why a response is not a usable prediction, or None when it is."""
    if code != 200:
        detail = body.get("error")
        return f"HTTP {code}: {detail}" if detail else f"HTTP {code}"
    predictions = body.get("prediction")
    if not isinstance(predictions, list) or len(predictions) != expected_rows:
        count = len(predictions) if isinstance(predictions, list) else "no"
        return f"expected {expected_rows} predictions, got {count}"
    for item in predictions:
        value = item.get("prediction") if isinstance(item, dict) else None
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"a prediction is not a finite number: {item!r}"
    return None


def main() -> int:
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
        expected_rows = test_row_count(tests)
        for framework, url in endpoints:
            for i in range(1, repeat + 1):
                print(f"[bench] {framework} / {size_label} / run {i}/{repeat} … ", end="", flush=True)
                try:
                    ms, st, code, body = post_csv(url, features, tests)
                    error = validation_error(code, body, expected_rows)
                    if error is not None:
                        print(f"FAILED: {error}")
                        results.append({
                            "framework": framework, "size": size_label, "run": i,
                            "url": url, "status": code, "error": error,
                        })
                        continue
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
                except (requests.RequestException, OSError, ValueError) as exc:
                    print(f"FAILED: {exc!r}")
                    results.append({
                        "framework": framework, "size": size_label, "run": i,
                        "url": url, "error": repr(exc),
                    })

    summary = {}
    for size_label, _, _ in plan:
        for framework, _ in endpoints:
            xs = [r["client_ms"] for r in results
                  if r.get("framework") == framework and r.get("size") == size_label and "error" not in r]
            if xs:
                summary[f"{framework}/{size_label}"] = {
                    "runs": len(xs),
                    "client_ms_min":  round(min(xs), 2),
                    "client_ms_mean": round(statistics.mean(xs), 2),
                    "client_ms_max":  round(max(xs), 2),
                }

    failures = [r for r in results if "error" in r]
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "runs": results, "failures": len(failures)}, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(summary, indent=2))
    if failures:
        print(f"{len(failures)} run(s) failed and are excluded from the summary.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
