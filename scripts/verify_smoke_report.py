"""Validate execution evidence; smoke timings are not performance claims."""
import json
import math
import sys


def verify(path):
    with open(path, encoding="utf-8") as source:
        report = json.load(source)
    results = report["Results"]
    assert {row["Model"] for row in results} == {"mlp", "cnn", "lstm", "transformer"}
    assert len(results) == 4
    metrics = ("WarmupSecondsAvg", "SteadyStateLatencyMsAvg", "SteadyStateLatencyMsP95",
               "ThroughputSamplesPerSecond", "MemoryMbPeak")
    # Include lazily allocated classifier heads; constructor-time counts omitted them.
    parameters = {"mlp": 468874, "cnn": 20490, "lstm": 25482, "transformer": 69322}
    for model in results:
        assert model["Parameters"] == parameters[model["Model"]], model["Model"]
        assert len(model["Inference"]) == 4
        assert {row["BatchSize"] for row in model["Inference"]} == {1, 8, 32, 128}
        for row in model["Inference"]:
            assert row["ExecutionMode"] in {"eager", "compiled", "fused"}
            assert isinstance(row["NoGcRegionCompleted"], bool)
            assert all(math.isfinite(row[key]) and row[key] >= 0 for key in metrics)
    print(f"{path}: four models, sixteen valid inference configurations")


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        verify(filename)
