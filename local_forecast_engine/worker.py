"""Run inference and downloads outside the API process."""

import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from time import sleep, time

import psutil

from .contracts import BacktestRequest, ForecastRequest
from .forecast import Resources, backtest, forecast_result, predict
from .models import ModelRunner, download
from .store import Store

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("retention", type=int)
    parser.add_argument("threads", type=int)
    parser.add_argument("parent", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parent = psutil.Process(args.parent)
    store = Store(args.data_dir, args.retention)
    runner = ModelRunner(args.data_dir, args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    while parent.is_running():
        claimed = store.claim(os.getpid())
        if claimed is None:
            sleep(0.25)
            continue
        job, kind, payload = claimed
        publish = False
        try:
            if kind == "install":
                store.progress(job, "Downloading pinned model files")
                download(args.data_dir, payload["model"])
                result = {"model": payload["model"], "installed": True}
            elif kind == "backtest":
                request = BacktestRequest.model_validate(payload)
                result = backtest(
                    request, runner, store.known_at, lambda message: store.progress(job, message)
                )
                result["config_key"] = request.dataset.config_key
            else:
                request = ForecastRequest.model_validate(payload)
                store.progress(job, f"Running {request.model}")
                with Resources() as resources:
                    points = predict(request.dataset, request.model, request.use_inputs, runner)
                    result = forecast_result(request.dataset, request.model, points, int(time()))
                result["use_inputs"] = request.use_inputs
                result["inputs"] = [
                    {
                        "key": item.key,
                        "received_at": item.received_at,
                        "history_count": sum(value is not None for value in item.history),
                        "future_count": sum(value is not None for value in (item.future or [])),
                    }
                    for item in request.dataset.covariates
                    if request.use_inputs
                ]
                result.update(resources.result())
                publish = request.publish
            if not parent.is_running() or not store.succeed(job, result, publish=publish):
                break
        except ValueError as error:
            if not store.fail(job, str(error)):
                break
        except Exception:
            LOGGER.exception("Job %s (%s) failed", job, kind)
            if not store.fail(job, "The engine could not complete this job. Check the engine log."):
                break


if __name__ == "__main__":
    main()
