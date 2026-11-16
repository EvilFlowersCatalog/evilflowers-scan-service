"""The Celery application: ``celery -A evilflowers_scan.worker worker --queues=<one of the two>``.

The task names, queues and kwargs are a contract with the books digitalizer's orchestrator, which sends them:

==========================  =========================================  ==============================================
Task                        Queues                                     Message
==========================  =========================================  ==============================================
evilflowers_scan.clean      ``SCAN_QUEUE`` (bulk, ≤ 32 leaves),        ``contract.CleanJob``
                            ``SCAN_INTERACTIVE_QUEUE`` (one leaf)
evilflowers_scan.erase      the same two                               ``contract.EraseJob``
==========================  =========================================  ==============================================

Both queues are declared; a Deployment consumes one (``--queues``). Everything else is evilflowers-worker's
``create_app``: late acknowledgement, prefetch 1, the db-wide visibility timeout, the scale-in protocol on SIGTERM, the
delivery cap, telemetry and JSON logs. No result backend: the orchestrator learns the end from ``link`` /
``link_error``.
"""

from evilflowers_worker.runtime import create_app

from evilflowers_scan.settings import get_settings
from evilflowers_scan.telemetry import ATTRIBUTES

settings = get_settings()
app = create_app(
    "scan",
    settings,
    include=["evilflowers_scan.tasks"],
    queues=[settings.queue, settings.interactive_queue],
    routes={"evilflowers_scan.*": settings.queue},
    attributes=ATTRIBUTES,
)
