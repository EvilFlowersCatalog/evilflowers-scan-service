"""The real worker app with two test seams, for ``test_end_to_end.py`` (``celery -A scan_test_app worker``).

- ScanTailor is the ``FakeScanTailor`` (CI has no binary; ``test_cleaning.py`` runs the real one where it exists);
- what ``telemetry.bound`` binds is also appended to ``$STORAGE_PATH/bound.jsonl``, so the test can see the
  digitalizer headers arrive from a real message.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import evilflowers_scan.tasks as tasks
import evilflowers_scan.telemetry as telemetry
from conftest import FakeScanTailor
from evilflowers_scan import jobs
from evilflowers_scan.imaging import ImageOps, ScanTailorCleaner
from evilflowers_scan.worker import app, settings

__all__ = ["app"]

_real_bind = telemetry.bind


@contextmanager
def _recording_bind(**fields):
    with (Path(os.environ["STORAGE_PATH"]) / "bound.jsonl").open("a") as file:
        file.write(json.dumps(fields) + "\n")
    with _real_bind(**fields):
        yield


telemetry.bind = _recording_bind
_services = jobs.Services(settings, ScanTailorCleaner(FakeScanTailor(output_dpi=600)), ImageOps(), None)
tasks.services = lambda: _services
# A pool child started by spawn (macOS) imports only the app's `include`: list this module so the seams apply there.
app.conf.include = [*app.conf.include, "scan_test_app"]
