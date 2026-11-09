"""Forecast execution, solar conversions, and evaluation."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from math import isfinite
from threading import Event, Thread
from time import perf_counter
from zoneinfo import ZoneInfo

import psutil
from astral import Observer
from astral.sun import elevation

from .contracts import BacktestRequest, Dataset, Number
from .models import BASELINES, MODELS, ModelRunner


def iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def seasonal_value(dataset: Dataset, stamp: int, days: int) -> Number:
    zone = ZoneInfo(dataset.timezone)
    local = datetime.fromtimestamp(stamp, zone)
    while True:
        local = local - timedelta(days=days)
        previous = int(local.timestamp())
        if previous < dataset.cutoff:
            break
    if (
        previous < dataset.start
        or (previous - dataset.start) % dataset.step
        or datetime.fromtimestamp(previous, zone).replace(tzinfo=None) != local.replace(tzinfo=None)
    ):
        return None
    return dataset.target[(previous - dataset.start) // dataset.step]


def solar_elevations(dataset: Dataset) -> list[float]:
    assert dataset.solar
    observer = Observer(dataset.solar.latitude, dataset.solar.longitude)
    return [
        elevation(
            observer, datetime.fromtimestamp(stamp + dataset.step // 2, UTC), with_refraction=False
        )
        for stamp in range(
            dataset.start, dataset.cutoff + dataset.horizon * dataset.step, dataset.step
        )
    ]


def daylight(dataset: Dataset, stamp: int) -> bool:
    assert dataset.solar
    observer = Observer(dataset.solar.latitude, dataset.solar.longitude)
    return any(
        elevation(observer, datetime.fromtimestamp(stamp + offset, UTC), with_refraction=False) > 0
        for offset in (0, dataset.step // 2, dataset.step - 1)
    )


def predict(dataset: Dataset, model: str, use_inputs: bool, runner: ModelRunner) -> list[dict]:
    if not any(x is not None for x in dataset.target):
        raise ValueError("No observed target intervals are available")
    stamps = [dataset.cutoff + index * dataset.step for index in range(dataset.horizon)]
    if model == "persistence":
        value = next(value for value in reversed(dataset.target) if value is not None)
        predictions = [(None, value, None)] * dataset.horizon
    elif model in BASELINES:
        days = 1 if model == "previous-day" else 7
        predictions = [(None, seasonal_value(dataset, stamp, days), None) for stamp in stamps]
    else:
        past, known = {}, {}
        if use_inputs:
            for covariate in dataset.covariates:
                if any(x is not None for x in covariate.history):
                    past[covariate.key] = covariate.history
                    if covariate.future and any(x is not None for x in covariate.future):
                        known[covariate.key] = covariate.future
        if dataset.solar and use_inputs:
            sun = solar_elevations(dataset)
            past["solar_elevation"] = sun[: len(dataset.target)]
            known["solar_elevation"] = sun[len(dataset.target) :]
        predictions = runner.predict(model, dataset.target, past, known, dataset.horizon)
    result = []
    for stamp, (lower, value, upper) in zip(stamps, predictions, strict=True):
        if not all(x is None or isfinite(x) for x in (lower, value, upper)):
            lower, value, upper = None, None, None
        if dataset.solar and value is not None:
            lit = daylight(dataset, stamp)
            lower, value, upper = (
                None if x is None else max(0.0, x) if lit else 0.0 for x in (lower, value, upper)
            )
        if lower is not None and upper is not None and value is not None:
            lower, value, upper = sorted((lower, value, upper))
        result.append({"start": stamp, "value": value, "p10": lower, "p90": upper})
    return result


def solar_hours(dataset: Dataset, points: list[dict]) -> dict[str, float]:
    assert dataset.solar
    scale = dataset.step / 3600 if dataset.solar.kind == "power" else 1
    energy = {p["start"]: p["value"] * scale for p in points if p["value"] is not None}
    hours = {}
    for hour in sorted({start // 3600 * 3600 for start in energy}):
        stamps = range(hour, hour + 3600, dataset.step)
        if all(stamp in energy for stamp in stamps):
            hours[iso(hour)] = round(sum(energy[stamp] for stamp in stamps), 3)
    return hours


class Resources:
    """Measure job duration and sampled peak resident memory."""

    def __enter__(self):
        self.process = psutil.Process()
        self.started = perf_counter()
        self.peak = self.process.memory_info().rss
        self.done = Event()

        def sample():
            while not self.done.wait(0.05):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.done.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    def result(self) -> dict:
        return {
            "runtime_seconds": round(perf_counter() - self.started, 3),
            "peak_rss_mb": round(self.peak / 2**20, 1),
        }


def metrics(pairs: list[tuple[float, dict]]) -> dict:
    errors = [abs(actual - point["value"]) for actual, point in pairs]
    total = sum(abs(actual) for actual, _ in pairs)
    covered = [
        point["p10"] <= actual <= point["p90"]
        for actual, point in pairs
        if point["p10"] is not None and point["p90"] is not None
    ]
    return {
        "count": len(pairs),
        "mae": sum(errors) / len(errors) if errors else None,
        "wape": sum(errors) / total if total else None,
        "coverage": sum(covered) / len(covered) if covered else None,
        "interval_count": len(covered),
    }


def forecast_result(dataset: Dataset, model: str, points: list[dict], issued_at: int) -> dict:
    if not any(point["value"] is not None for point in points):
        raise ValueError("This model has no predictions for the requested intervals")
    return {
        "project_id": dataset.project_id,
        "series_key": dataset.series_key,
        "config_key": dataset.config_key,
        "model": model,
        "revision": MODELS[model].revision if model in MODELS else "baseline-v1",
        "cutoff": dataset.cutoff,
        "issued_at": issued_at,
        "step_minutes": dataset.step_minutes,
        "unit": dataset.unit,
        "timezone": dataset.timezone,
        "points": points,
        "actual": [
            {"start": dataset.start + index * dataset.step, "value": dataset.target[index]}
            for index in range(
                max(0, len(dataset.target) - 2880 // dataset.step_minutes), len(dataset.target)
            )
        ],
        "coverage": {
            "available": sum(x is not None for x in dataset.target),
            "requested": len(dataset.target),
        },
        "solar": {"wh_hours": solar_hours(dataset, points)} if dataset.solar else None,
    }


def backtest(
    request: BacktestRequest,
    runner: ModelRunner,
    lookup: Callable[[Dataset, str, int], list[Number] | None],
    progress: Callable[[str], None],
) -> dict:
    dataset = request.dataset
    last = len(dataset.target) - dataset.horizon
    stride = max(dataset.horizon, 1440 // dataset.step_minutes)
    indices = [
        last - index * stride for index in range(request.folds) if last - index * stride >= 32
    ]
    folds = []
    truth = {}
    for index in reversed(indices):
        origin = dataset.start + index * dataset.step
        if not any(x is not None for x in dataset.target[index : index + dataset.horizon]):
            continue
        inputs = [
            item.model_copy(
                update={
                    "history": item.history[:index],
                    "future": lookup(dataset, item.key, origin),
                    "received_at": origin,
                }
            )
            for item in dataset.covariates
        ]
        folds.append(
            dataset.model_copy(
                update={"target": dataset.target[:index], "cutoff": origin, "covariates": inputs}
            )
        )
        truth.update(
            {
                (origin, offset): actual
                for offset, actual in enumerate(dataset.target[index : index + dataset.horizon])
                if actual is not None
            }
        )
    if not folds:
        raise ValueError("Not enough recorded history for a backtest at this horizon")
    rows = []
    for variant in request.variants:
        progress(f"Evaluating {variant.model}")
        predictions = {}
        error = None
        with Resources() as resources:
            for fold in folds:
                try:
                    points = predict(fold, variant.model, variant.use_inputs, runner)
                except ValueError as exc:
                    error = str(exc)
                    continue
                predictions.update(
                    {
                        (fold.cutoff, index): point
                        for index, point in enumerate(points)
                        if point["value"] is not None and (fold.cutoff, index) in truth
                    }
                )
        rows.append(
            (
                {
                    "model": variant.model,
                    "use_inputs": variant.use_inputs,
                    "error": None if predictions else error or "No matching predictions",
                    "available": len(predictions),
                    **resources.result(),
                },
                predictions,
            )
        )
    scored = [set(predictions) for _, predictions in rows if predictions]
    common = set.intersection(*scored) if scored else set()
    results = [
        {
            **row,
            **metrics(
                [
                    (truth[key], predictions[key])
                    for key in sorted(common or predictions.keys())
                    if key in predictions
                ]
            ),
        }
        for row, predictions in rows
    ]
    return {
        "unit": dataset.unit,
        "folds": len(folds),
        "observations": len(truth),
        "common_count": len(common),
        "archived_input_folds": sum(
            any(
                item.future
                and any(x is not None for x in item.future)
                and any(x is not None for x in item.history)
                for item in fold.covariates
            )
            for fold in folds
        ),
        "variants": results,
    }
