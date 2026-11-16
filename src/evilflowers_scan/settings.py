"""Configuration of the worker: the kit's ``RuntimeSettings`` and ``JudgeSettings`` plus the scan service's own fields.

The broker, delivery semantics (``WORKER_SOFT_SHUTDOWN_TIMEOUT``, ``WORKER_CANCEL_TIMEOUT``, ``MAX_DELIVERIES``,
``BROKER_VISIBILITY_TIMEOUT``), storage and logs come from evilflowers-worker, the decider-vision client from its
``judges`` extra (``JUDGES_ENABLED``, ``DECIDER_VISION_URL``, ``JUDGE_THRESHOLDS`` …). No catalog URL and no API key:
a standalone job reports through its outcome file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from evilflowers_worker.judges import JudgeSettings
from evilflowers_worker.settings import RuntimeSettings
from pydantic import AliasChoices, Field

ColorMode = Literal["mixed", "color_grayscale", "black_and_white"]
Despeckle = Literal["off", "cautious", "normal", "aggressive"]

#: The bulk lane (KEDA 0..N) and the interactive lane (one replica, always on), evilflowers-scan-service#1.
BULK_QUEUE = "evilflowers_scan.standalone"
INTERACTIVE_QUEUE = "evilflowers_scan.standalone.interactive"


class Settings(RuntimeSettings, JudgeSettings):
    # -- Broker -------------------------------------------------------------------------------------------------------
    #: The digitalizer's Redis db (db 8); the callback queue must live on the same db.
    redis_url: str = "redis://redis:6379/8"
    #: The bulk queue. ``QUEUE`` is what the kit's test harness sets.
    queue: str = Field(BULK_QUEUE, validation_alias=AliasChoices("SCAN_QUEUE", "QUEUE"))
    #: The interactive queue (one leaf, within seconds). Both are declared; ``--queues`` picks what a pod consumes.
    interactive_queue: str = Field(INTERACTIVE_QUEUE, validation_alias=AliasChoices("SCAN_INTERACTIVE_QUEUE"))

    # -- Limits -------------------------------------------------------------------------------------------------------
    #: Seconds a job may run: 32 leaves of ScanTailor + DZI, under the orchestrator's deadline (clean: 3600 s).
    #: Past it the job writes a ``failed`` outcome (``task_time_limit`` hard-kills ``soft_margin`` seconds later).
    task_soft_time_limit: int = Field(3240, ge=60)
    task_time_limit: int = Field(3300, ge=60)

    # -- ScanTailor (``scantailor-deviant-cli``) ----------------------------------------------------------------------
    scantailor_binary: str = "scantailor-deviant-cli"
    #: ``mixed``: text binarized, pictures keep tones; ``color_grayscale``: tones everywhere (pair with DocRes).
    color_mode: ColorMode = "mixed"
    despeckle: Despeckle = "cautious"
    margins_mm: float = 8.0
    dewarping: Literal["off", "auto", "marginal"] = "auto"
    normalize_illumination: bool = True
    #: Resolution of the frames; ``output_dpi`` supersamples the binarization (600 from 300 smooths glyph edges).
    dpi: int = Field(300, ge=50)
    output_dpi: int | None = Field(None, ge=50)

    # -- DocRes (optional; the weights on the /models PVC) ------------------------------------------------------------
    docres_enabled: bool = False
    docres_repo: Path = Path("/models/DocRes")
    docres_python: Path = Path("/models/venv-docres/bin/python")
    docres_task: str = "appearance"
    #: Seconds per DocRes run (one group of leaves).
    docres_timeout: float = Field(1800.0, gt=0)

    # -- Renditions ---------------------------------------------------------------------------------------------------
    preview_px: int = Field(1600, ge=64)
    webp_quality: int = Field(80, ge=1, le=100)
    dzi_tile: int = Field(254, ge=64)
    dzi_quality: int = Field(85, ge=1, le=100)

    # -- Artefact check (evilflowers-scan-service#2) ------------------------------------------------------------------
    artefacts_enabled: bool = True
    #: A mark at or above this ``p`` from the signals is banded without asking a judge; below ``artefact_floor`` it is
    #: dropped as noise; in between it is uncertain (decider-vision, or ``review`` without one).
    artefact_sure: float = Field(0.85, ge=0.0, le=1.0)
    artefact_floor: float = Field(0.35, ge=0.0, le=1.0)
    #: The same mark (perceptual hash within ``artefact_hash_distance`` bits) on this many leaves of a job is
    #: furniture.
    artefact_min_repeats: int = Field(3, ge=2)
    artefact_hash_distance: int = Field(6, ge=0, le=32)
    #: Marks per leaf at most (largest first): a speckled page must not flood the outcome or the judge budget.
    artefact_max_marks: int = Field(40, ge=1)

    @property
    def effective_output_dpi(self) -> int:
        return self.output_dpi or self.dpi


@lru_cache
def get_settings() -> Settings:
    return Settings()
