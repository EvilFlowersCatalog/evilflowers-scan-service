"""The orchestrator's message headers on spans and log lines, so a trace crosses the boundary (#1, comment 2).

Each message carries the Celery headers ``digitalizer_run_id``, ``digitalizer_job_id``, ``digitalizer_stage`` and
``digitalizer_lane``. ``bound(task, message)`` binds them, with the message's ``run_id`` and ``job``, through the kit's
``observability.bind``: span attributes under ``ATTRIBUTES``, JSON log fields under their short names.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

from celery import Task
from evilflowers_worker.contract import StandaloneJob
from evilflowers_worker.observability import CATALOG_JOB_ATTRIBUTES, bind

HEADERS = ("digitalizer_run_id", "digitalizer_job_id", "digitalizer_stage", "digitalizer_lane")

#: Short (log) name → span attribute, passed to ``create_app(attributes=...)``.
ATTRIBUTES: Mapping[str, str] = MappingProxyType(
    {
        **CATALOG_JOB_ATTRIBUTES,
        "job": "evilflowers.job.name",
        "digitalizer_run_id": "digitalizer.run.id",
        "digitalizer_job_id": "digitalizer.job.id",
        "digitalizer_stage": "digitalizer.stage",
        "digitalizer_lane": "digitalizer.lane",
    }
)


def headers(task: Task) -> dict[str, Any]:
    """The digitalizer's headers of the running task's message (Celery puts custom headers on the request, and, in
    some versions, under ``request.headers``)."""
    request = task.request
    nested = getattr(request, "headers", None) or {}
    found = {}
    for name in HEADERS:
        value = getattr(request, name, None)
        if value is None and isinstance(nested, Mapping):
            value = nested.get(name)
        if value is not None:
            found[name] = str(value)
    return found


@contextmanager
def bound(task: Task, message: StandaloneJob) -> Iterator[None]:
    with bind(run_id=message.run_id, job=message.job, **headers(task)):
        yield
