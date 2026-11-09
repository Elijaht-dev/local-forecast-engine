"""Start Local Forecast standalone or as a Home Assistant app."""

import json
import logging
import os
import secrets
import socket
from pathlib import Path
from threading import Thread
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

import uvicorn

from .api import create_app, secret_file

LOGGER = logging.getLogger(__name__)


def advertise(port: int, token: str, supervisor_token: str) -> None:
    for _ in range(25):
        try:
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2):
                break
        except (URLError, TimeoutError):
            sleep(0.5)
    request = Request(
        "http://supervisor/discovery",
        data=json.dumps(
            {
                "service": "local_forecast",
                "config": {"url": f"http://{socket.gethostname()}:{port}", "token": token},
            }
        ).encode(),
        headers={"Authorization": "Bearer " + supervisor_token, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10):
            LOGGER.info("Announced Local Forecast to Home Assistant")
    except (URLError, TimeoutError):
        LOGGER.warning(
            "App discovery failed. Verify the app configuration and restart it to retry."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_dir = Path(
        os.environ.get("LF_DATA_DIR", "/data" if Path("/data").exists() else "data")
    ).resolve()
    is_app = (data_dir / "options.json").is_file() and bool(os.environ.get("SUPERVISOR_TOKEN"))
    options = json.loads((data_dir / "options.json").read_text()) if is_app else {}
    token = (
        options.get("api_token")
        or os.environ.get("LF_TOKEN")
        or secret_file(data_dir, "api-token", secrets.token_urlsafe(32))
    )
    port = int(os.environ.get("LF_PORT", "8099"))
    retention = int(options.get("retention_days", os.environ.get("LF_RETENTION_DAYS", "90")))
    threads = int(options.get("threads", os.environ.get("LF_THREADS", "2")))
    app = create_app(data_dir, token, retention, threads)
    if is_app:
        Thread(
            target=advertise, args=(port, token, os.environ["SUPERVISOR_TOKEN"]), daemon=True
        ).start()
    uvicorn.run(app, host=os.environ.get("LF_HOST", "127.0.0.1"), port=port, access_log=False)


if __name__ == "__main__":
    main()
