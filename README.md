# Local Forecast Engine

Run forecasts and backtests locally with Timestral, Chronos-2 or seasonal baselines. Send your time series to the standalone HTTP API, or use Forecast Lab in Home Assistant to select recorded entities. A SQLite queue stores jobs while a CPU worker runs inference.

## Home Assistant app

In **Settings → Apps → App store → Repositories**, add `https://github.com/Elijaht-dev/local-forecast-engine#ha-beta`, including the branch suffix. Install **Local Forecast Engine** and start it. Supervisor builds the app on the first install. Install the [Local Forecast integration](https://github.com/Elijaht-dev/local-forecast#install-the-public-beta), accept the discovered engine in **Settings → Devices & services**, and open **Forecast Lab**. The API port is internal to the Supervisor network by default.

For a manual app install, extract the released source bundle into the local apps share (`/local_apps` in Terminal & SSH on current HA OS), then reload the app store. On Raspberry Pi 4, use the built-in baselines with engine v0.1.0. [Source bundle and compatibility notes](https://github.com/Elijaht-dev/local-forecast-engine/releases/tag/v0.1.0)

## Standalone Docker

```sh
docker compose up --build -d
docker compose exec local-forecast cat /data/api-token
```

The second command displays the generated API token. Send it in the `Authorization: Bearer <token>` header when calling `/v1` endpoints, or enter it in the Home Assistant pairing form. The sample Compose file binds the API to loopback. If HA is in Docker on the same machine, put both services on a shared Docker network and use the engine service's internal URL. For a separate machine, expose the port on a private interface or use an HTTPS reverse proxy and enter that URL when pairing.

The `/data` volume contains the database, installation ID, token, and model cache. Keep it across upgrades. Stop the engine before making a file-level copy. In HA OS, include the app in HA backups.

| Option | Default | Purpose |
| --- | --- | --- |
| `LF_RETENTION_DAYS` | 90 | Retain forecast results and selected observations for this many days |
| `LF_THREADS` | 2 | Maximum PyTorch inference threads |
| `LF_TOKEN` | Generated | Optional deployment token, at least 24 ASCII characters |

HA app users set equivalent options in the app Configuration tab.

## Models

| Model | Approximate checkpoint download | License |
| --- | ---: | --- |
| [Timestral-5M](https://huggingface.co/mistralai/Timestral-5M) | 21 MB | Modified MIT |
| [Timestral-28M](https://huggingface.co/mistralai/Timestral-28M) | 111 MB | Modified MIT |
| [Timestral-123M](https://huggingface.co/mistralai/Timestral-123M) | 490 MB | Modified MIT |
| [Timestral-209M](https://huggingface.co/mistralai/Timestral-209M) | 837 MB | Modified MIT |
| [Chronos-2](https://huggingface.co/amazon/chronos-2) | 478 MB | Apache-2.0 |

Install checkpoints explicitly in Forecast Lab. Inference reads pinned safetensors from the local cache. Persistence, previous-day, and previous-week baselines are built in and work before any weights are installed.

Start with 5M on small ARM64 hosts. Weight size differs from runtime memory. Use the comparison table to see sampled worker RSS on your device. The worker RSS includes loaded libraries and allocator memory.

### Measured CPU inference

One run per checkpoint from a cold worker, with weights already on disk, two inference threads, 672 hourly history intervals, and 48 forecast intervals. Runtime includes model loading. RSS measures the worker process and excludes HA, Supervisor, and the engine API process.

| Environment | Timestral-5M time / RSS | Chronos-2 time / RSS |
| --- | ---: | ---: |
| HA OS AMD64 VM, recorded target | 1.155 s / 300.5 MiB | 2.379 s / 875.9 MiB |
| Linux ARM64 container, synthetic target with covariates | 2.359 s / 313.5 MiB | 3.903 s / 897.0 MiB |

In a separate ARM64 run that switched checkpoints in the same worker, Timestral-28M, 123M, and 209M took 1.354 s, 1.506 s, and 2.392 s. Sampled worker RSS was 450.8 MiB, 992.3 MiB, and 1615.6 MiB. These measurements are for the tested configurations. Raspberry Pi 4 and 5 measurements are still pending.

## API v1

All `/v1` requests use a Bearer token. `/healthz` reports process readiness for health checks.

| Operation | Request |
| --- | --- |
| Pair and list models | `GET /v1/health`, `GET /v1/models` |
| Queue an explicit download | `POST /v1/models/{id}/install` |
| Queue forecast or backtest | `POST /v1/forecasts`, `POST /v1/backtests` |
| Inspect or cancel a job | `GET /v1/jobs/{id}`, `DELETE /v1/jobs/{id}` |
| Read latest published result and scores | `GET /v1/projects/{id}/forecast`, `GET /v1/projects/{id}/accuracy`, `GET /v1/projects/{id}/backtest` |

UTC interval starts are Unix seconds. The target and each historical input end just before the cutoff, while the first prediction starts at the cutoff. Null is a missing observation. Forecast results expose point values and model P10/P90 when available. Native solar output contains complete hourly intervals in Wh. `local_forecast_engine/contracts.py` is the request schema.

## Build and test

```sh
python -m pip install ".[test]"
python -m pytest -q
python -m ruff check .
docker build -t local-forecast-engine:local .
```

Run `tests/smoke_container.py` with `LF_TOKEN` against a test engine on loopback port 18099. It installs the pinned checkpoints and verifies actual inference for both model families and the backtest path. CI has native Linux AMD64 and ARM64 jobs.

Original code is MIT licensed. The inference libraries and model weights retain their upstream licenses. The model download retains upstream license notices available in the checkpoint repositories.
