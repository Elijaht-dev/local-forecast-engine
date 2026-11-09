from time import monotonic, sleep, time

from fastapi.testclient import TestClient

from local_forecast_engine.api import create_app
from local_forecast_engine.store import Store

TOKEN = "test-token-with-more-than-24-characters"


def wait(client, job):
    deadline = monotonic() + 15
    while job["state"] not in {"succeeded", "failed", "cancelled"} and monotonic() < deadline:
        sleep(0.15)
        job = client.get(f"/v1/jobs/{job['id']}").json()
    assert job["state"] == "succeeded", job
    return job


def test_authenticated_job_persists_and_survives_restart(tmp_path):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(create_app(tmp_path, TOKEN)) as anonymous:
        assert anonymous.get("/v1/models").status_code == 401
        anonymous.headers.update(headers)
        payload = {
            "model": "persistence",
            "publish": True,
            "dataset": {
                "project_id": "test",
                "series_key": "power",
                "config_key": "version1",
                "unit": "W",
                "step_minutes": 60,
                "cutoff": int(time()) // 3600 * 3600,
                "horizon": 6,
                "target": [None, 10, 12, None],
            },
        }
        response = anonymous.post("/v1/forecasts", json=payload)
        assert response.status_code == 202, response.text
        job = wait(anonymous, response.json())
        assert [point["value"] for point in job["result"]["points"]] == [12] * 6
        assert job["result"]["coverage"] == {"available": 2, "requested": 4}
    with TestClient(create_app(tmp_path, TOKEN), headers=headers) as restarted:
        archived = restarted.get("/v1/projects/test/forecast").json()
        assert archived["id"] == job["id"]
        assert archived["config_key"] == "version1"
        assert restarted.get("/v1/projects/test/accuracy").json()["count"] == 0


def test_cancellation_and_worker_failure_do_not_publish_partial_results(tmp_path):
    store = Store(tmp_path)
    store.migrate()
    running = store.enqueue("forecast", {}, "pv")
    assert store.claim(51)[0] == running["id"]
    assert store.cancel(running["id"])["state"] == "cancelling"
    assert not store.succeed(running["id"], {}, publish=True)
    store.recover("Worker was stopped", 51)
    assert store.job(running["id"])["state"] == "cancelled"
    assert store.latest("pv") is None

    interrupted = store.enqueue("forecast", {}, "pv")
    store.claim(52)
    queued = store.enqueue("forecast", {}, "pv")
    store.recover("Worker exited", 52)
    assert store.job(interrupted["id"])["error"] == "Worker exited"
    assert store.job(queued["id"])["state"] == "queued"
    assert store.claim(53)[0] == queued["id"]
