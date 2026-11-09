from datetime import UTC, datetime
from time import time

import pytest
from pydantic import ValidationError

from local_forecast_engine.contracts import BacktestRequest, Dataset
from local_forecast_engine.forecast import backtest, metrics, predict, solar_hours
from local_forecast_engine.models import ModelRunner
from local_forecast_engine.store import Store


def sample(**updates):
    return Dataset.model_validate(
        {
            "project_id": "pv",
            "series_key": "watts",
            "unit": "W",
            "timezone": "Europe/Paris",
            "step_minutes": 60,
            "cutoff": int(time()) // 3600 * 3600,
            "horizon": 24,
            "target": [float(index) for index in range(24 * 10)],
            **updates,
        }
    )


def test_future_inputs_require_aligned_history_and_receipt():
    with pytest.raises(ValidationError, match="align with the target"):
        sample(covariates=[{"key": "weather", "history": [20]}])
    with pytest.raises(ValidationError, match="receipt timestamp"):
        sample(covariates=[{"key": "weather", "history": [None] * 240, "future": [20] * 24}])


def test_seasonal_baseline_never_wraps_through_missing_observations(tmp_path):
    dataset = sample(
        cutoff=int(datetime(2026, 1, 15, tzinfo=UTC).timestamp()),
        target=[4] * 216 + [None] + [8] * 23,
    )
    points = predict(dataset, "previous-day", True, ModelRunner(tmp_path, 1))
    assert [point["value"] for point in points] == [None] + [8] * 23
    assert points[0]["p10"] is None


def test_solar_aggregates_only_complete_hours():
    cutoff = int(datetime(2026, 10, 24, 14, tzinfo=UTC).timestamp())
    dataset = sample(
        cutoff=cutoff,
        step_minutes=15,
        horizon=192,
        unit="Wh",
        solar={"kind": "energy", "latitude": 48.85, "longitude": 2.35},
    )
    points = [{"start": cutoff + i * 900, "value": 100} for i in range(192)]
    hourly = solar_hours(dataset, points)
    assert len(hourly) == 48
    assert set(hourly.values()) == {400}
    points[1]["value"] = None
    assert len(solar_hours(dataset, points)) == 47


def test_solar_night_is_zero_and_day_keeps_prediction(tmp_path):
    cutoff = int(datetime(2026, 6, 1, tzinfo=UTC).timestamp())
    dataset = sample(
        cutoff=cutoff,
        target=[300] * 48,
        solar={"kind": "power", "latitude": 48.85, "longitude": 2.35},
    )
    points = predict(dataset, "persistence", True, ModelRunner(tmp_path, 1))
    assert points[0]["value"] == 0
    assert points[12]["value"] == 300


def test_rolling_origin_uses_only_archived_future_inputs():
    dataset = sample(
        covariates=[
            {
                "key": "weather",
                "history": [10] * 240,
                "future": [999] * 24,
                "received_at": int(time()),
            }
        ]
    )
    contexts = []

    class RecordingRunner:
        def predict(self, model, target, past, known, horizon):
            contexts.append((target[-1], known.get("weather")))
            return [(0, 1, 2)] * horizon

    request = BacktestRequest.model_validate(
        {
            "dataset": dataset,
            "variants": [{"model": "timestral-5m"}, {"model": "timestral-5m", "use_inputs": False}],
            "folds": 2,
        }
    )
    result = backtest(request, RecordingRunner(), lambda _, _key, _origin: [7] * 24, lambda _: None)
    assert contexts == [(191, [7] * 24), (215, [7] * 24), (191, None), (215, None)]
    assert result["common_count"] == 48
    assert result["archived_input_folds"] == 2
    assert result["variants"][0]["coverage"] == 0


def test_archive_does_not_return_future_snapshot(tmp_path):
    store = Store(tmp_path)
    store.migrate()
    dataset = sample(
        covariates=[
            {
                "key": "weather",
                "history": [10] * 240,
                "future": [12] * 24,
                "received_at": int(time()),
            }
        ]
    )
    store.merge_dataset(dataset, capture_future=True)
    assert store.known_at(dataset, "weather", dataset.cutoff - 3600) is None
    assert store.known_at(dataset, "weather", dataset.cutoff + 3600) == [12] * 23 + [None]


def test_zero_actuals_do_not_produce_infinite_percentage():
    assert metrics([(0, {"value": 1, "p10": None, "p90": None})]) == {
        "count": 1,
        "mae": 1,
        "wape": None,
        "coverage": None,
        "interval_count": 0,
    }
