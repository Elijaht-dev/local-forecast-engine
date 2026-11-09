"""Exercise both pinned model families against an already running engine."""

import json
import math
import os
import sys
from time import monotonic, sleep, time
from urllib.request import Request, urlopen

BASE = os.environ.get("LF_TEST_URL", "http://127.0.0.1:18099")
TOKEN = os.environ["LF_TOKEN"]


def api(method, path, body=None):
    request = Request(
        BASE + "/v1" + path,
        method=method,
        headers={"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None,
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def finish(job, timeout=900):
    deadline = monotonic() + timeout
    while job["state"] in {"queued", "running", "cancelling"} and monotonic() < deadline:
        sleep(1)
        job = api("GET", "/jobs/" + job["id"])
    if job["state"] != "succeeded":
        raise RuntimeError(f"{job['kind']} ended as {job['state']}: {job.get('error')}")
    return job["result"]


def main():
    assert api("GET", "/health")["api"] == 1
    cutoff = int(time()) // 3600 * 3600
    target = [50 + 20 * math.sin(index / 24 * math.tau) for index in range(28 * 24)]
    dataset = {
        "project_id": "smoke-test",
        "series_key": "temperature",
        "unit": "°C",
        "timezone": "Europe/Paris",
        "step_minutes": 60,
        "cutoff": cutoff,
        "horizon": 48,
        "target": target,
        "covariates": [
            {
                "key": "outdoor",
                "history": [value / 2 for value in target],
                "future": [value / 2 for value in target[:48]],
                "received_at": int(time()),
            },
            {
                "key": "state",
                "history": [None if index % 17 == 0 else index % 2 for index in range(len(target))],
            },
        ],
    }
    for model in sys.argv[1:] or ["timestral-5m", "chronos-2"]:
        if model not in {"persistence", "previous-day", "previous-week"}:
            finish(api("POST", f"/models/{model}/install"))
        result = finish(
            api("POST", "/forecasts", {"dataset": dataset, "model": model, "publish": True})
        )
        assert len(result["points"]) == 48 and result["points"][0]["start"] == cutoff
        for point in result["points"]:
            assert math.isfinite(point["value"]), point
            if model not in {"persistence", "previous-day", "previous-week"}:
                assert point["p10"] <= point["value"] <= point["p90"], point
        print(
            f"{model}: 48 valid points, {result['runtime_seconds']} s, {result['peak_rss_mb']} MiB",
            flush=True,
        )
    result = finish(
        api(
            "POST",
            "/backtests",
            {
                "dataset": dataset,
                "variants": [{"model": "previous-day"}, {"model": "previous-week"}],
            },
        )
    )
    assert result["common_count"] and result["archived_input_folds"] == 0
    print(
        f"Backtest: {result['folds']} historical cutoffs, {result['common_count']} common observations",
        flush=True,
    )


if __name__ == "__main__":
    main()
