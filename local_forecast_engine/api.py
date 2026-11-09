"""Authenticated, local-only engine API."""

import logging
import os
import secrets
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Thread
from time import time
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .contracts import BacktestRequest, ForecastRequest, Slug
from .models import BASELINES, MODELS, catalog, installed
from .store import QueueFull, Store

LOGGER = logging.getLogger(__name__)
TIMEOUTS = {"forecast": 300, "backtest": 1800, "install": 1800}


def secret_file(data_dir: Path, name: str, default: str) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / name
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return path.read_text(encoding="utf-8").strip()
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(default + "\n")
    return default


class Supervisor:
    def __init__(self, store: Store, data_dir: Path, threads: int) -> None:
        self.store = store
        self.data_dir = data_dir
        self.threads = threads
        self.stop = Event()
        self.thread = Thread(target=self.run, name="worker-supervisor", daemon=True)

    @staticmethod
    def terminate(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def run(self) -> None:
        process = None
        next_cleanup = 0
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"LF_TOKEN", "SUPERVISOR_TOKEN"}
        }
        try:
            while not self.stop.is_set():
                if process is None:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "local_forecast_engine.worker",
                            str(self.data_dir),
                            str(self.store.retention_days),
                            str(self.threads),
                            str(os.getpid()),
                        ],
                        env=env,
                    )
                if process.poll() is not None:
                    self.store.recover("The worker exited before completing this job", process.pid)
                    process = None
                    self.stop.wait(0.75)
                    continue
                now = int(time())
                active = self.store.active(process.pid)
                if active and (active[0] == "cancelling" or now - active[2] > TIMEOUTS[active[1]]):
                    self.terminate(process)
                    self.store.recover("The worker timed out", process.pid)
                    process = None
                if now >= next_cleanup:
                    self.store.purge(now)
                    next_cleanup = now + 3600
                self.stop.wait(0.25)
        except Exception:
            LOGGER.exception("The inference supervisor stopped unexpectedly")
        finally:
            if process is not None:
                self.terminate(process)
                self.store.recover("The engine stopped before completing this job", process.pid)


def create_app(data_dir: Path, token: str, retention_days: int = 90, threads: int = 2) -> FastAPI:
    if len(token) < 24 or not token.isascii() or any(character.isspace() for character in token):
        raise ValueError("Engine token must contain at least 24 ASCII characters and no whitespace")
    if not 1 <= retention_days <= 365 or not 1 <= threads <= 8:
        raise ValueError("Invalid retention or worker thread count")
    data_dir = data_dir.resolve()
    store = Store(data_dir, retention_days)
    installation_id = secret_file(data_dir, "installation-id", uuid4().hex)
    supervisor = Supervisor(store, data_dir, threads)

    @asynccontextmanager
    async def lifespan(_app):
        store.migrate()
        store.recover("The engine restarted before completing this job")
        supervisor.thread.start()
        try:
            yield
        finally:
            supervisor.stop.set()
            supervisor.thread.join(timeout=7)

    app = FastAPI(
        title="Local Forecast Engine",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def body_limit(request: Request, call_next):
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > 4_000_000):
            return JSONResponse({"detail": "Request body is too large"}, status_code=413)
        return await call_next(request)

    def auth(authorization: str = Header(default="")) -> None:
        if not secrets.compare_digest(authorization.encode(), ("Bearer " + token).encode()):
            raise HTTPException(401, "Invalid engine token", headers={"WWW-Authenticate": "Bearer"})

    authorized = [Depends(auth)]

    def enqueue(kind: str, payload: dict, project: str | None = None):
        if not supervisor.thread.is_alive():
            raise HTTPException(503, "The inference worker is unavailable")
        try:
            return store.enqueue(kind, payload, project)
        except QueueFull as exc:
            raise HTTPException(429, "The job queue is full") from exc

    @app.get("/healthz")
    def healthz():
        if not supervisor.thread.is_alive():
            raise HTTPException(503, "Worker supervisor unavailable")
        return {"ready": True}

    @app.get("/v1/health", dependencies=authorized)
    def health():
        return {"api": 1, "version": __version__, "id": installation_id, **healthz()}

    @app.get("/v1/models", dependencies=authorized)
    def models():
        return catalog(data_dir)

    @app.post("/v1/models/{model}/install", dependencies=authorized, status_code=202)
    def install(model: Slug):
        if model not in MODELS:
            raise HTTPException(404, "Unknown downloadable model")
        if installed(data_dir, model):
            return {"state": "succeeded", "result": {"model": model, "installed": True}}
        return enqueue("install", {"model": model})

    @app.post("/v1/forecasts", dependencies=authorized, status_code=202)
    def forecast(request: ForecastRequest):
        if request.model not in BASELINES and request.model not in MODELS:
            raise HTTPException(422, "Unknown model")
        if not installed(data_dir, request.model):
            raise HTTPException(409, "Install the selected model before running it")
        if request.dataset.cutoff > int(time()):
            raise HTTPException(422, "The historical cutoff cannot be in the future")
        dataset = store.merge_dataset(request.dataset, capture_future=request.use_inputs)
        return enqueue(
            "forecast",
            request.model_copy(update={"dataset": dataset}).model_dump(),
            dataset.project_id,
        )

    @app.post("/v1/backtests", dependencies=authorized, status_code=202)
    def evaluate(request: BacktestRequest):
        if any(
            variant.model not in BASELINES and variant.model not in MODELS
            for variant in request.variants
        ):
            raise HTTPException(422, "Unknown evaluation model")
        if request.dataset.cutoff > int(time()):
            raise HTTPException(422, "The historical cutoff cannot be in the future")
        dataset = store.merge_dataset(request.dataset, capture_future=False)
        return enqueue(
            "backtest",
            request.model_copy(update={"dataset": dataset}).model_dump(),
            dataset.project_id,
        )

    @app.get("/v1/jobs/{job}", dependencies=authorized)
    def job_state(job: Slug):
        result = store.job(job)
        if result is None:
            raise HTTPException(404, "Job not found")
        return result

    @app.delete("/v1/jobs/{job}", dependencies=authorized)
    def cancel(job: Slug):
        result = store.cancel(job)
        if result is None:
            raise HTTPException(404, "Job not found")
        return result

    @app.get("/v1/projects/{project}/forecast", dependencies=authorized)
    def latest(project: Slug):
        return store.latest(project)

    @app.get("/v1/projects/{project}/accuracy", dependencies=authorized)
    def accuracy(project: Slug):
        return store.score(project)

    @app.get("/v1/projects/{project}/backtest", dependencies=authorized)
    def latest_backtest(project: Slug):
        return store.latest_backtest(project)

    return app
