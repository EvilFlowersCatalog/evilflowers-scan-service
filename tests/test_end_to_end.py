"""A real prefork worker (``evilflowers_worker.testing.WorkerProcess``) against Redis, as the orchestrator sends.

What the orchestrator sees on the wire: ``link`` with a compact summary after the outcome file, ``link_error`` after a
failed outcome, ``link_error`` and no outcome for an invalid message, and its headers on the job's log/span fields.

Needs ``TEST_REDIS_URL`` (default ``redis://localhost:6379/15``); skipped when no Redis answers, except in CI (``CI``
set), where a missing Redis is a failure.
"""

import json
import os
import uuid
from pathlib import Path

import pytest
from evilflowers_worker.runtime import standalone_callbacks
from evilflowers_worker.testing import WorkerProcess, redis_client, wait_for

from conftest import write_frame

REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
HERE = Path(__file__).parent
OUT = "digitized/book/book_studio/jobs/17-a1"
FRAME = "raw-scans/fei/book/0002.tif"
HEADERS = {
    "digitalizer_run_id": "42",
    "digitalizer_job_id": "17",
    "digitalizer_stage": "clean",
    "digitalizer_lane": "bulk",
}


@pytest.fixture
def broker():
    if redis_client(REDIS_URL) is None:
        if os.environ.get("CI"):
            pytest.fail(f"no Redis at {REDIS_URL}: the real-worker tests must run in CI")
        pytest.skip(f"no Redis at {REDIS_URL}")
    return REDIS_URL


@pytest.fixture
def storage(tmp_path) -> Path:
    write_frame(tmp_path / FRAME)
    return tmp_path


@pytest.fixture
def worker(broker, storage):
    env = {"STORAGE_PATH": str(storage), "JUDGES_ENABLED": "0", "ARTEFACTS_ENABLED": "0"}
    with WorkerProcess("scan_test_app", redis_url=broker, pythonpath=[str(HERE)], env=env) as process:
        process.wait_ready()
        yield process


@pytest.fixture
def callbacks(broker):
    queue = f"test-callbacks-{uuid.uuid4().hex[:8]}"
    yield queue
    redis_client(broker).delete(queue)


def message(**overrides) -> dict:
    return {
        "run_id": "42",
        "job": "clean-0003-a1",
        "token": "tok",
        "inputs": [FRAME],
        "output_dir": OUT,
        "outcome": f"{OUT}/outcome.json",
        "leaves": [{"leaf": 3, "frame": FRAME, "side": 0, "crop_box": [0, 0, 600, 800]}],
        **overrides,
    }


def send(worker: WorkerProcess, callback_queue: str, kwargs: dict) -> str:
    app = worker.producer()
    options = standalone_callbacks(app, done="digitalizer.job_done", failed="digitalizer.job_failed",
                                   queue=callback_queue, job_id="17")  # fmt: skip
    return worker.send_task("evilflowers_scan.clean", kwargs, headers=HEADERS, **options)


def callback(worker: WorkerProcess, queue: str) -> dict:
    assert wait_for(lambda: worker.queued(queue), 60), worker.output
    (sent,) = worker.queued(queue)
    return sent


def test_a_clean_job_writes_its_outcome_then_fires_link(worker, storage, callbacks):
    send(worker, callbacks, message())
    sent = callback(worker, callbacks)

    assert sent["headers"]["task"] == "digitalizer.job_done"
    assert worker.args(sent) == [{"job": "clean-0003-a1", "leaves": 1, "marks": 0}]
    assert worker.kwargs(sent) == {"job_id": "17"}
    outcome = json.loads((storage / OUT / "outcome.json").read_text())
    assert (outcome["status"], outcome["token"]) == ("done", "tok")
    assert (storage / outcome["leaves"][0]["image"]).is_file()

    bound = [json.loads(line) for line in (storage / "bound.jsonl").read_text().splitlines()]
    assert bound == [{"run_id": "42", "job": "clean-0003-a1", **HEADERS}]


def test_a_failing_job_writes_a_failed_outcome_then_fires_link_error(worker, storage, callbacks):
    bad = message(leaves=[{"leaf": 3, "frame": FRAME, "crop_box": [5000, 0, 100, 100]}])
    task_id = send(worker, callbacks, bad)
    sent = callback(worker, callbacks)

    assert sent["headers"]["task"] == "digitalizer.job_failed"
    assert worker.args(sent) == [task_id]
    outcome = json.loads((storage / OUT / "outcome.json").read_text())
    assert outcome["status"] == "failed"
    assert "outside" in outcome["error"]


def test_an_invalid_message_fires_link_error_and_writes_nothing(worker, storage, callbacks):
    send(worker, callbacks, message(leaves=[]))
    sent = callback(worker, callbacks)
    assert sent["headers"]["task"] == "digitalizer.job_failed"
    assert not (storage / OUT).exists()
