"""Allowlisted checkpoints and a common adapter for the two model families."""

from dataclasses import dataclass
from pathlib import Path

from .contracts import Number

BASELINES = {
    "persistence": "Persistence",
    "previous-day": "Previous day",
    "previous-week": "Previous week",
}


@dataclass(frozen=True)
class Model:
    label: str
    repo: str
    revision: str
    bytes_mb: int
    max_context: int
    max_horizon: int
    license: str


MODELS = {
    "timestral-5m": Model(
        "Timestral-5M",
        "mistralai/Timestral-5M",
        "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
        21,
        32768,
        1024,
        "Modified MIT",
    ),
    "timestral-28m": Model(
        "Timestral-28M",
        "mistralai/Timestral-28M",
        "49b26409ac87b5935539a5d8ffe7f397b720db47",
        111,
        32768,
        1024,
        "Modified MIT",
    ),
    "timestral-123m": Model(
        "Timestral-123M",
        "mistralai/Timestral-123M",
        "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
        490,
        32768,
        1024,
        "Modified MIT",
    ),
    "timestral-209m": Model(
        "Timestral-209M",
        "mistralai/Timestral-209M",
        "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
        837,
        32768,
        1024,
        "Modified MIT",
    ),
    "chronos-2": Model(
        "Chronos-2",
        "amazon/chronos-2",
        "29ec3766d36d6f73f0696f85560a422f50e8498c",
        478,
        8192,
        1024,
        "Apache-2.0",
    ),
}


def cache_dir(data_dir: Path) -> Path:
    return data_dir / "models"


def checkpoint_path(data_dir: Path, model: str) -> Path:
    spec = MODELS[model]
    return (
        cache_dir(data_dir)
        / ("models--" + spec.repo.replace("/", "--"))
        / "snapshots"
        / spec.revision
    )


def installed(data_dir: Path, model: str) -> bool:
    if model in BASELINES:
        return True
    path = checkpoint_path(data_dir, model)
    return all((path / filename).is_file() for filename in ("config.json", "model.safetensors"))


def catalog(data_dir: Path) -> list[dict]:
    return [
        {"id": name, "label": label, "installed": True, "baseline": True}
        for name, label in BASELINES.items()
    ] + [
        {
            "id": name,
            "label": spec.label,
            "installed": installed(data_dir, name),
            "repo": spec.repo,
            "revision": spec.revision,
            "download_mb": spec.bytes_mb,
            "license": spec.license,
            "max_horizon": spec.max_horizon,
            "baseline": False,
        }
        for name, spec in MODELS.items()
    ]


def download(data_dir: Path, model: str) -> None:
    from huggingface_hub import snapshot_download

    spec = MODELS[model]
    snapshot_download(
        repo_id=spec.repo,
        revision=spec.revision,
        cache_dir=cache_dir(data_dir),
        endpoint="https://huggingface.co",
        token=False,
        allow_patterns=["config.json", "model.safetensors", "LICENSE", "LICENSE.*", "README.md"],
        max_workers=2,
    )
    if not installed(data_dir, model):
        raise RuntimeError("Checkpoint download did not contain the expected files")


class ModelRunner:
    """Keep only one model in memory in the inference process."""

    def __init__(self, data_dir: Path, threads: int) -> None:
        self.data_dir = data_dir
        self.threads = threads
        self.name: str | None = None
        self.pipeline = None

    def load(self, model: str):
        if self.name == model:
            return self.pipeline
        if not installed(self.data_dir, model):
            raise ValueError(f"Install {MODELS[model].label} before running it")
        import gc

        import torch

        self.pipeline = None
        self.name = None
        gc.collect()
        torch.set_num_threads(self.threads)
        path = str(checkpoint_path(self.data_dir, model))
        if model == "chronos-2":
            from chronos import Chronos2Pipeline

            self.pipeline = Chronos2Pipeline.from_pretrained(
                path,
                device_map="cpu",
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
            )
        else:
            from timestral import TimestralForecaster

            self.pipeline = TimestralForecaster.from_pretrained(
                path, device="cpu", local_files_only=True
            )
        self.name = model
        return self.pipeline

    def predict(
        self,
        model: str,
        target: list[Number],
        past: dict[str, list[Number]],
        known: dict[str, list[Number]],
        horizon: int,
    ) -> list[tuple[float, float, float]]:
        import numpy as np

        spec = MODELS[model]
        if horizon > spec.max_horizon:
            raise ValueError(f"{spec.label} supports at most {spec.max_horizon} steps")
        if sum(x is not None for x in target[-spec.max_context :]) < 32:
            raise ValueError("At least 32 observed target intervals are required for this model")
        pipeline = self.load(model)
        context = np.asarray(target[-spec.max_context :], dtype=np.float32)
        length = len(context)
        if model == "chronos-2":
            inputs = {
                "target": context,
                "past_covariates": {
                    key: np.asarray(values[-length:], dtype=np.float32)
                    for key, values in past.items()
                },
                "future_covariates": {
                    key: np.asarray(values, dtype=np.float32) for key, values in known.items()
                },
            }
            quantiles, _ = pipeline.predict_quantiles(
                [inputs],
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
                context_length=length,
            )
            return [tuple(map(float, row)) for row in quantiles[0][0].cpu().numpy()]
        historical = [past[key][-length:] for key in past if key not in known]
        future = [past[key][-length:] + values for key, values in known.items()]
        forecast = pipeline.forecast(
            context,
            prediction_length=horizon,
            past_covariates=np.asarray([historical], dtype=np.float32) if historical else None,
            known_covariates=np.asarray([future], dtype=np.float32) if future else None,
        )
        return [
            tuple(map(float, row))
            for row in zip(*(forecast.quantile(q)[0, 0] for q in (0.1, 0.5, 0.9)), strict=True)
        ]
