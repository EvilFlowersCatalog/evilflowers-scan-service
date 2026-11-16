"""What ``evilflowers_scan.clean`` and ``evilflowers_scan.erase`` do, from a validated message to the outcome file.

The executable spec is evilflowers-books-digitalizer ``studio/testing/standalone.py`` (``scan_clean``,
``scan_erase``); the rules are the kit's (``evilflowers_worker.outcome``) and the orchestrator's (#1 comments):

- the frames are only read; every file is written under ``output_dir`` (``StandaloneJob.output_file`` refuses any
  other path), and every path reported is relative to ``STORAGE_PATH``;
- the outcome file is written **last** and atomically: its presence tells the orchestrator's ``reconcile`` that the
  attempt's files are complete; ``run_id`` and ``token`` are echoed untouched;
- a failure writes ``status: failed`` with ``error``, then raises, so ``link_error`` fires and the run shows the error;
- a deliberate requeue (SIGTERM) never gets here: the kit kills the child, nothing is written, the redelivery carries
  the same token.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded
from evilflowers_worker.contract import StandaloneJob
from evilflowers_worker.judges import Judges
from evilflowers_worker.observability import metrics
from evilflowers_worker.outcome import write_outcome
from PIL import Image

from evilflowers_scan import artefacts
from evilflowers_scan.contract import CleanedLeaf, CleanJob, CleanOutcome, EraseJob, Mark
from evilflowers_scan.geometry import BBox, PageOverride
from evilflowers_scan.imaging import CleanedPage, ImageOps, LeafPlan, ScanTailorCleaner
from evilflowers_scan.imaging.docres import DocRes
from evilflowers_scan.imaging.scantailor import ScanTailor
from evilflowers_scan.settings import Settings

logger = logging.getLogger(__name__)


class Services:
    """What a job needs besides its message; built once per process (``for_settings``), replaced in tests."""

    def __init__(self, settings: Settings, cleaner: ScanTailorCleaner, ops: ImageOps, judges: Judges | None):
        self.settings = settings
        self.cleaner = cleaner
        self.ops = ops
        self.judges = judges

    @classmethod
    def for_settings(cls, settings: Settings) -> Services:
        scantailor = ScanTailor(
            binary=settings.scantailor_binary,
            color_mode=settings.color_mode,
            despeckle=settings.despeckle,
            margins_mm=settings.margins_mm,
            dewarping=settings.dewarping,
            normalize_illumination=settings.normalize_illumination,
            dpi=settings.dpi,
            output_dpi=settings.effective_output_dpi,
        )
        docres = (
            DocRes(settings.docres_repo, settings.docres_python, settings.docres_task, settings.docres_timeout)
            if settings.docres_enabled
            else None
        )
        ops = ImageOps(webp_quality=settings.webp_quality, tile=settings.dzi_tile, dzi_quality=settings.dzi_quality)
        judges = Judges.from_settings(settings) if settings.artefacts_enabled else None
        return cls(settings, ScanTailorCleaner(scantailor, docres=docres), ops, judges)


def relative(settings: Settings, path: Path) -> str:
    """``path`` relative to ``STORAGE_PATH`` (POSIX); it must lie under it."""
    return path.resolve().relative_to(settings.storage_path.resolve()).as_posix()


def _check_inputs(message: StandaloneJob, settings: Settings, files: list[str]) -> list[Path]:
    """Every file the job reads is one of its declared ``inputs`` and stays under ``STORAGE_PATH``."""
    declared = set(message.inputs)
    undeclared = [name for name in files if name not in declared]
    if undeclared:
        raise ValueError(f"not in the job's inputs: {undeclared}")
    paths = [settings.storage_file(name) for name in files]
    missing = [name for name, path in zip(files, paths, strict=True) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input does not exist: {missing}")
    return paths


def _check_output_dir(message: StandaloneJob, settings: Settings) -> Path:
    """The job's directory, created and path-checked; no input may be the outcome or lie inside the directory (the
    orchestrator deletes the directory when it refuses the attempt)."""
    directory = message.prepare_output(settings)
    if message.outcome in message.inputs:
        raise ValueError("the outcome path is one of the inputs")
    for name in message.inputs:
        if settings.storage_file(name).is_relative_to(directory):
            raise ValueError(f"input {name!r} lies inside the output directory")
    # A delivery killed mid-run (OOM, the hard time limit) leaves its ScanTailor staging behind: the directory is this
    # job's alone, so the redelivery sweeps it.
    for stale in directory.glob(".st-*"):
        shutil.rmtree(stale, ignore_errors=True)
    return directory


def _renditions(services: Services, page: Path, directory: Path) -> tuple[str, str]:
    """``leaves/<leaf>.webp`` and ``dzi/<leaf>.dzi``, checked to lie inside the job's directory."""
    renditions = services.ops.previews(page, directory, max_px=services.settings.preview_px)
    for path in (renditions.preview, renditions.dzi):
        if not path.resolve().is_relative_to(directory.resolve()):
            raise RuntimeError(f"rendition outside the job's directory: {path}")
    return relative(services.settings, renditions.preview), relative(services.settings, renditions.dzi)


def _marks(
    services: Services, message: CleanJob, pages: list[CleanedPage]
) -> tuple[dict[int, list[Mark]], str | None]:
    """The marks per leaf, or none and a warning: the artefact check never fails the clean (only the time limit
    does)."""
    if not services.settings.artefacts_enabled:
        return {}, None
    sides = {leaf.leaf: leaf.side for leaf in message.leaves}
    panel = services.judges.panel() if services.judges is not None else None
    try:
        found = artefacts.marks(
            [artefacts.LeafImage(page.leaf, page.image, sides.get(page.leaf)) for page in pages],
            artefacts.from_settings(services.settings),
            panel=panel,
            judges=services.judges,
        )
    except SoftTimeLimitExceeded:
        raise
    except Exception as error:
        logger.warning("Artefact check of job %s failed; leaves reported without marks", message.job, exc_info=True)
        return {}, f"artefact check failed: {type(error).__name__}: {error}"
    return found, None


def failed(services: Services, message: StandaloneJob, error: BaseException) -> None:
    """A ``failed`` outcome with the error; the caller raises (``link_error``)."""
    text = f"{type(error).__name__}: {error}"[:2000]
    outcome = CleanOutcome(run_id=message.run_id, token=message.token, status="failed", error=text)
    write_outcome(services.settings, message.outcome, outcome)


def clean(services: Services, message: CleanJob, kind: str = "clean") -> dict[str, Any]:
    settings = services.settings
    try:
        frames = _check_inputs(message, settings, [leaf.frame for leaf in message.leaves])
        directory = _check_output_dir(message, settings)
        plans = [
            LeafPlan(
                leaf.leaf,
                frame,
                leaf.side,
                BBox.from_xywh(leaf.crop_box),
                leaf.rotate_degree,
                PageOverride.from_dict(leaf.override),
            )
            for leaf, frame in zip(message.leaves, frames, strict=True)
        ]
        pages = services.cleaner.clean(plans, directory)
        found, marks_warning = _marks(services, message, pages)
        rotations = {leaf.leaf: leaf.rotate_degree for leaf in message.leaves}
        leaves = []
        for page in pages:
            leaf_marks = found.get(page.leaf, [])
            hidden = [BBox.of(mark.bbox) for mark in leaf_marks if mark.band == "hide"]
            if message.erase_hidden and hidden:
                services.ops.erase(page.image, hidden, page.image)
            preview, dzi = _renditions(services, page.image, directory)
            leaves.append(
                CleanedLeaf(
                    leaf=page.leaf,
                    image=relative(settings, page.image),
                    width=page.width,
                    height=page.height,
                    dpi=page.dpi,
                    skew_deg=page.skew_deg,
                    preview=preview,
                    dzi=dzi,
                    crop_box=tuple(page.crop.to_xywh()),  # type: ignore[arg-type]
                    rotate_degree=rotations[page.leaf],
                    warnings=[*page.warnings, *([marks_warning] if marks_warning else [])],
                    marks=leaf_marks,
                )
            )
    except Exception as error:
        logger.exception("Job %s of run %s failed", message.job, message.run_id)
        failed(services, message, error)
        metrics.record_job(kind, "failed")
        raise
    write_outcome(settings, message.outcome, CleanOutcome(run_id=message.run_id, token=message.token,
                                                          status="done", leaves=leaves))  # fmt: skip
    metrics.record_job(kind, "done")
    return {"job": message.job, "leaves": len(leaves), "marks": sum(len(leaf.marks) for leaf in leaves)}


def erase(services: Services, message: EraseJob) -> dict[str, Any]:
    settings = services.settings
    try:
        images = _check_inputs(message, settings, [leaf.image for leaf in message.leaves])
        directory = _check_output_dir(message, settings)
        leaves = []
        for leaf, source in zip(message.leaves, images, strict=True):
            with Image.open(source) as im:
                size = im.size
            target = message.output_file(settings, source.name)
            services.ops.erase(source, [BBox.of(box) for box in leaf.boxes], target)
            with Image.open(target) as im:
                if im.size != size:  # the orchestrator refuses an outcome that changes the size
                    raise RuntimeError(f"leaf {leaf.leaf}: erase changed the size {size} -> {im.size}")
            preview, dzi = _renditions(services, target, directory)
            leaves.append(CleanedLeaf(leaf=leaf.leaf, image=relative(settings, target), width=size[0],
                                      height=size[1], dpi=leaf.dpi, preview=preview, dzi=dzi))  # fmt: skip
    except Exception as error:
        logger.exception("Job %s of run %s failed", message.job, message.run_id)
        failed(services, message, error)
        metrics.record_job("erase", "failed")
        raise
    write_outcome(settings, message.outcome, CleanOutcome(run_id=message.run_id, token=message.token,
                                                          status="done", leaves=leaves))  # fmt: skip
    metrics.record_job("erase", "done")
    return {"job": message.job, "leaves": len(leaves)}


def give_up(services: Services, message: StandaloneJob, delivery: int, kind: str) -> dict[str, Any]:
    """Past ``MAX_DELIVERIES``: the job killed its worker every time. A failed outcome, then raise."""
    error = RuntimeError(f"gave up after {delivery - 1} deliveries: the worker died during each of them")
    failed(services, message, error)
    metrics.record_job(kind, "failed")
    raise error
