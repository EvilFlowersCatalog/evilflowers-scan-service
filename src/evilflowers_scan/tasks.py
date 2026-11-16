"""``evilflowers_scan.clean`` and ``evilflowers_scan.erase``: standalone jobs sent by the digitalizer's orchestrator.

``run_task`` validates (an invalid message raises at once: ``link_error``, no outcome), counts the delivery and calls
``give_up`` instead of the job past ``MAX_DELIVERIES``. The return value is the ``link``'s first argument: compact.
"""

from functools import cache
from typing import Any

from celery import Task, shared_task
from evilflowers_worker.runtime import JOB_TASK_OPTIONS, run_task, runtime

# The module that calls `create_app`: a pool child started by spawn imports only this one.
import evilflowers_scan.worker
from evilflowers_scan import jobs
from evilflowers_scan.contract import CleanJob, EraseJob
from evilflowers_scan.settings import Settings
from evilflowers_scan.telemetry import bound

LIMITS = {
    "time_limit": evilflowers_scan.worker.settings.task_time_limit,
    "soft_time_limit": evilflowers_scan.worker.settings.task_soft_time_limit,
}


@cache
def services() -> jobs.Services:
    """Built once per process: the ScanTailor configuration and the judges' clients with their breakers."""
    settings = runtime().settings
    assert isinstance(settings, Settings)
    return jobs.Services.for_settings(settings)


@shared_task(name="evilflowers_scan.clean", bind=True, **LIMITS, **JOB_TASK_OPTIONS)
def clean(self: Task, **kwargs: Any) -> dict[str, Any]:
    """Raw frames → cleaned leaves, WebP previews, Deep Zoom pyramids and marks; the outcome file last."""

    def run(message: CleanJob, delivery: int) -> dict[str, Any]:
        with bound(self, message):
            return jobs.clean(services(), message)

    def give_up(message: CleanJob, delivery: int) -> dict[str, Any]:
        with bound(self, message):
            return jobs.give_up(services(), message, delivery, "clean")

    return run_task(self, run, CleanJob, kwargs, on_capped=give_up)


@shared_task(name="evilflowers_scan.erase", bind=True, **LIMITS, **JOB_TASK_OPTIONS)
def erase(self: Task, **kwargs: Any) -> dict[str, Any]:
    """Confirmed ``remove`` boxes painted white on a copy of each page, then the same renditions as ``clean``."""

    def run(message: EraseJob, delivery: int) -> dict[str, Any]:
        with bound(self, message):
            return jobs.erase(services(), message)

    def give_up(message: EraseJob, delivery: int) -> dict[str, Any]:
        with bound(self, message):
            return jobs.give_up(services(), message, delivery, "erase")

    return run_task(self, run, EraseJob, kwargs, on_capped=give_up)
