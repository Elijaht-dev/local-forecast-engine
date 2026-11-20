"""Durable queue, forecast archive, and versioned local storage."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from time import time
from uuid import uuid4

from .contracts import Dataset, Number
from .forecast import metrics

SCHEMA = """
CREATE TABLE jobs (
    id TEXT PRIMARY KEY, project TEXT, kind TEXT NOT NULL, state TEXT NOT NULL,
    created INTEGER NOT NULL, updated INTEGER NOT NULL, request TEXT,
    worker INTEGER, message TEXT, result TEXT, error TEXT
);
CREATE INDEX job_queue ON jobs(state, created);
CREATE INDEX project_jobs ON jobs(project, updated DESC);
CREATE TABLE forecasts (
    id TEXT PRIMARY KEY, project TEXT NOT NULL, series TEXT NOT NULL,
    issued INTEGER NOT NULL, result TEXT NOT NULL
);
CREATE INDEX project_forecasts ON forecasts(project, issued DESC);
CREATE TABLE samples (
    project TEXT NOT NULL, series TEXT NOT NULL, step INTEGER NOT NULL,
    start INTEGER NOT NULL, value REAL NOT NULL,
    PRIMARY KEY (project, series, step, start)
) WITHOUT ROWID;
CREATE INDEX sample_retention ON samples(start);
CREATE TABLE inputs (
    project TEXT NOT NULL, key TEXT NOT NULL, step INTEGER NOT NULL,
    first INTEGER NOT NULL, received INTEGER NOT NULL, values_json TEXT NOT NULL,
    PRIMARY KEY (project, key, step, received, first)
) WITHOUT ROWID;
CREATE INDEX input_retention ON inputs(received);
PRAGMA user_version=1;
"""


class QueueFull(Exception):
    """All queue slots are occupied."""


class Store:
    def __init__(self, data_dir: Path, retention_days: int = 90) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "forecast.sqlite3"
        self.retention_days = retention_days

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def migrate(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                db.executescript("BEGIN IMMEDIATE;\n" + SCHEMA + "\nCOMMIT;")
            elif version != 1:
                raise RuntimeError(f"Unsupported database version {version}")

    def purge(self, now: int) -> None:
        oldest = now - self.retention_days * 86400
        with self.connect() as db:
            db.execute("DELETE FROM forecasts WHERE issued < ?", (oldest,))
            db.execute("DELETE FROM inputs WHERE received < ?", (oldest,))
            db.execute("DELETE FROM samples WHERE start < ?", (oldest,))
            db.execute(
                "DELETE FROM jobs WHERE updated < ? AND state IN ('succeeded', 'failed', 'cancelled')",
                (oldest,),
            )

    def merge_dataset(self, dataset: Dataset, *, capture_future: bool) -> Dataset:
        """Fill purged intervals from local cache and retain observations/inputs."""
        oldest = int(time()) - self.retention_days * 86400
        with self.connect() as db:

            def merge(key: str, values: list[Number]) -> list[Number]:
                existing = dict(
                    db.execute(
                        "SELECT start,value FROM samples WHERE project=? AND series=? AND step=? "
                        "AND start>=? AND start<?",
                        (dataset.project_id, key, dataset.step, dataset.start, dataset.cutoff),
                    ).fetchall()
                )
                db.executemany(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?) ON CONFLICT DO UPDATE SET value=excluded.value",
                    (
                        (dataset.project_id, key, dataset.step, stamp, value)
                        for index, value in enumerate(values)
                        if value is not None
                        and (stamp := dataset.start + index * dataset.step) >= oldest
                    ),
                )
                return [
                    value
                    if value is not None
                    else existing.get(dataset.start + index * dataset.step)
                    for index, value in enumerate(values)
                ]

            target = merge(dataset.series_key, dataset.target)
            inputs = []
            for item in dataset.covariates:
                history = merge("input-" + item.key, item.history)
                inputs.append(item.model_copy(update={"history": history}))
                if capture_future and item.future is not None and item.received_at is not None:
                    # The engine's clock is a lower bound on when it first saw this snapshot.
                    receipt = max(item.received_at, int(time()))
                    db.execute(
                        "INSERT OR IGNORE INTO inputs VALUES (?,?,?,?,?,?)",
                        (
                            dataset.project_id,
                            item.key,
                            dataset.step,
                            dataset.cutoff,
                            receipt,
                            json.dumps(item.future),
                        ),
                    )
        return dataset.model_copy(update={"target": target, "covariates": inputs})

    def known_at(self, dataset: Dataset, key: str, origin: int) -> list[Number] | None:
        with self.connect() as db:
            rows = db.execute(
                "SELECT first,values_json FROM inputs WHERE project=? AND key=? AND step=? "
                "AND received<=? AND first<=? AND first>? ORDER BY received DESC, first DESC LIMIT 1",
                (dataset.project_id, key, dataset.step, origin, origin, origin - 7 * 86400),
            )
            row = rows.fetchone()
        if row is None:
            return None
        values = json.loads(row["values_json"])
        offset = (origin - row["first"]) // dataset.step
        aligned = values[offset : offset + dataset.horizon]
        return (
            aligned + [None] * (dataset.horizon - len(aligned))
            if any(x is not None for x in aligned)
            else None
        )

    def enqueue(self, kind: str, request: dict, project: str | None = None) -> dict:
        job = uuid4().hex
        now = int(time())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute(
                    "SELECT count(*) FROM jobs WHERE state IN ('queued','running','cancelling')"
                ).fetchone()[0]
                >= 24
            ):
                raise QueueFull
            db.execute(
                "INSERT INTO jobs(id,project,kind,state,created,updated,request) VALUES (?,?,?,'queued',?,?,?)",
                (job, project, kind, now, now, json.dumps(request, allow_nan=False)),
            )
        return self.job(job)

    def job(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT id,project,kind,state,created,updated,message,result,error FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["result"] = json.loads(item["result"]) if item["result"] else None
        return item

    def claim(self, pid: int) -> tuple[str, str, dict] | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id,kind,request FROM jobs WHERE state='queued' ORDER BY created,rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET state='running',worker=?,updated=? WHERE id=?",
                (pid, int(time()), row["id"]),
            )
        return row["id"], row["kind"], json.loads(row["request"])

    def active(self, pid: int) -> tuple[str, str, int] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT state,kind,updated FROM jobs WHERE worker=? AND state IN ('running','cancelling')",
                (pid,),
            ).fetchone()
        return tuple(row) if row else None

    def progress(self, job: str, message: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET message=? WHERE id=? AND state='running'", (message, job))

    def succeed(self, job: str, result: dict, *, publish: bool = False) -> bool:
        body = json.dumps(result, allow_nan=False)
        with self.connect() as db:
            updated = db.execute(
                "UPDATE jobs SET state='succeeded',updated=?,result=?,request=NULL,worker=NULL,message=NULL "
                "WHERE id=? AND state='running'",
                (int(time()), body, job),
            )
            if updated.rowcount and publish:
                db.execute(
                    "INSERT INTO forecasts VALUES (?,?,?,?,?)",
                    (job, result["project_id"], result["series_key"], result["issued_at"], body),
                )
        return bool(updated.rowcount)

    def fail(self, job: str, error: str) -> bool:
        with self.connect() as db:
            updated = db.execute(
                "UPDATE jobs SET state='failed',updated=?,error=?,request=NULL,worker=NULL,message=NULL "
                "WHERE id=? AND state='running'",
                (int(time()), error, job),
            )
        return bool(updated.rowcount)

    def cancel(self, job: str) -> dict | None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET state=CASE state WHEN 'running' THEN 'cancelling' ELSE 'cancelled' END, "
                "updated=?, request=NULL WHERE id=? AND state IN ('queued','running')",
                (int(time()), job),
            )
        return self.job(job)

    def recover(self, error: str, pid: int | None = None) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET state=CASE state WHEN 'cancelling' THEN 'cancelled' ELSE 'failed' END, "
                "updated=?, error=CASE state WHEN 'cancelling' THEN NULL ELSE ? END, "
                "worker=NULL,request=NULL,message=NULL WHERE state IN ('running','cancelling') "
                "AND (? IS NULL OR worker=?)",
                (int(time()), error, pid, pid),
            )

    def latest(self, project: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT id,result FROM forecasts WHERE project=? ORDER BY issued DESC,rowid DESC LIMIT 1",
                (project,),
            ).fetchone()
        return {**json.loads(row["result"]), "id": row["id"]} if row else None

    def latest_backtest(self, project: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT result FROM jobs WHERE project=? AND kind='backtest' AND state='succeeded' "
                "ORDER BY updated DESC,rowid DESC LIMIT 1",
                (project,),
            ).fetchone()
        return json.loads(row["result"]) if row else None

    def score(self, project: str) -> dict:
        """Score the latest 50 issued runs against matching, subsequently observed intervals."""
        pairs = []
        with self.connect() as db:
            rows = db.execute(
                "SELECT result FROM forecasts WHERE project=? ORDER BY issued DESC,rowid DESC LIMIT 50",
                (project,),
            ).fetchall()
            runs = [json.loads(row["result"]) for row in rows]
            if not runs:
                return {"runs": 0, **metrics([])}
            latest = runs[0]
            for run in runs:
                if (
                    run["series_key"] != latest["series_key"]
                    or run["step_minutes"] != latest["step_minutes"]
                ):
                    continue
                step = run["step_minutes"] * 60
                actual = dict(
                    db.execute(
                        "SELECT start,value FROM samples WHERE project=? AND series=? AND step=? AND start>=? AND start<?",
                        (
                            project,
                            run["series_key"],
                            step,
                            run["issued_at"],
                            run["cutoff"] + step * len(run["points"]),
                        ),
                    ).fetchall()
                )
                pairs.extend(
                    (actual[p["start"]], p)
                    for p in run["points"]
                    if p["value"] is not None and p["start"] in actual
                )
        return {"runs": len(runs), "unit": latest["unit"], **metrics(pairs)}
