"""The page cleaner: planned leaves → cut, ScanTailor, optional DocRes → ``<leaf>.tif`` in the job's directory.

Ported from evilflowers-books-digitalizer ``studio/adapters/imaging/scantailor.py`` (``ScanTailorCleaner.clean``,
``cut_leaf``). The planning half (spread detection, ``plan``) stays with the orchestrator: it fixes every leaf's frame,
side and crop once per book, so leaf numbers are stable and overrides can be keyed by them.

Each leaf is cut out of its frame (rotation first, then the crop: IA scandata order), the operator's override applied,
and the leaves are grouped by their effective ``(color_mode, despeckle)``, because ScanTailor's flags are per run.
Every group works in a private staging directory **inside** the output directory, which is removed afterwards, so the
job never writes outside ``output_dir`` and never into ``/tmp``.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from evilflowers_scan.geometry import BBox, PageOverride
from evilflowers_scan.imaging.docres import DocRes
from evilflowers_scan.imaging.scantailor import ScanTailor, ScanTailorError

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # 600 dpi pages and uncompressed spreads are large

type GroupKey = tuple[str, str]


class CleanError(RuntimeError):
    """A leaf cannot be cleaned (an unreadable frame, a crop outside it, a tool failure)."""


@dataclass(frozen=True, slots=True)
class LeafPlan:
    """Where one leaf comes from (the orchestrator's plan)."""

    leaf: int
    frame: Path
    #: 0 left / 1 right half of a spread, ``None`` for a whole frame.
    side: int | None
    #: Frame pixels after rotation.
    crop: BBox
    rotate_degree: int = 0
    override: PageOverride = field(default_factory=PageOverride)


@dataclass(frozen=True, slots=True)
class CleanedPage:
    leaf: int
    image: Path
    width: int
    height: int
    dpi: int
    skew_deg: float
    #: The crop actually cut, in rotated frame pixels.
    crop: BBox
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Cut:
    """A leaf cut out of its frame, and what had to give."""

    pixels: np.ndarray
    crop: BBox
    warnings: tuple[str, ...]


def load_bgr(path: Path) -> np.ndarray:
    """An image file as an 8-bit BGR array; ``CleanError`` if OpenCV cannot read it."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise CleanError(f"cannot read frame {path.name}")
    return img


def rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    """IA ``rotateDegree`` (clockwise, multiples of 90) applied to a frame."""
    return np.ascontiguousarray(np.rot90(img, k=-(degrees // 90) % 4)) if degrees % 360 else img


def deskew(img: np.ndarray, degrees: float) -> np.ndarray:
    """The operator's deskew: rotate about the centre, uncovered corners white."""
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))  # fmt: skip


def effective_crop(plan: LeafPlan) -> BBox:
    """The override's crop wins; else ``split_x`` moves the gutter edge of a spread half; else the planned crop."""
    override = plan.override
    if override.crop is not None:
        return override.crop
    crop = plan.crop
    if override.split_x is not None and plan.side in (0, 1):
        if plan.side == 0:
            return BBox(crop.x0, crop.y0, override.split_x, crop.y1)
        return BBox(override.split_x, crop.y0, crop.x1, crop.y1)
    return crop


def cut_leaf(frame: np.ndarray, plan: LeafPlan) -> Cut:
    """Rotation, then the crop (clamped to the frame, with a warning), then the operator's deskew."""
    img = rotate(frame, plan.rotate_degree)
    h, w = img.shape[:2]
    try:
        wanted = effective_crop(plan)
    except ValueError as error:  # split_x on the wrong side of the crop
        raise CleanError(f"leaf {plan.leaf}: {error}") from error
    x0, y0, x1, y1 = max(0, wanted.x0), max(0, wanted.y0), min(w, wanted.x1), min(h, wanted.y1)
    if x1 <= x0 or y1 <= y0:
        raise CleanError(f"leaf {plan.leaf}: crop {wanted.as_list()} is outside the {w}x{h} frame")
    warnings: list[str] = []
    crop = BBox(x0, y0, x1, y1)
    if crop != wanted:
        warnings.append(f"crop {wanted.as_list()} clamped to the {w}x{h} frame")
    leaf = img[y0:y1, x0:x1]
    if plan.override.deskew_deg:
        leaf = deskew(leaf, plan.override.deskew_deg)
    return Cut(leaf, crop, tuple(warnings))


class ScanTailorCleaner:
    """Cleans planned leaves into ``out_dir/<leaf:04d>.tif`` (lossless)."""

    def __init__(self, scantailor: ScanTailor, *, docres: DocRes | None = None,
                 docres_modes: Sequence[str] = ("color_grayscale",)) -> None:  # fmt: skip
        self.scantailor = scantailor
        self.docres = docres
        self.docres_modes = frozenset(docres_modes)

    def with_options(self, changes: Mapping[str, Any]) -> ScanTailorCleaner:
        """This cleaner with the job's ScanTailor settings (the project's, #3); itself when there are none."""
        if not changes:
            return self
        return ScanTailorCleaner(replace(self.scantailor, **changes), docres=self.docres,
                                 docres_modes=tuple(self.docres_modes))  # fmt: skip

    def tool(self, color_mode: str, despeckle: str) -> ScanTailor:
        """The ScanTailor run of one group (the override's flags)."""
        return replace(self.scantailor, color_mode=color_mode, despeckle=despeckle)

    def clean(self, plans: Sequence[LeafPlan], out_dir: Path,
              progress: Callable[[int], None] | None = None) -> list[CleanedPage]:  # fmt: skip
        """Clean every leaf; ``progress(n)`` after each group. Raises ``CleanError`` on the first failure."""
        out_dir = out_dir.resolve()  # Leptonica/ScanTailor choke on macOS /tmp symlinks
        out_dir.mkdir(parents=True, exist_ok=True)
        groups: dict[GroupKey, list[LeafPlan]] = {}
        for plan in sorted(plans, key=lambda p: p.leaf):
            groups.setdefault(self._group_key(plan.override), []).append(plan)
        pages: list[CleanedPage] = []
        for key, group in groups.items():
            pages += self._clean_group(key, group, out_dir)
            if progress is not None:
                progress(len(pages))
        return sorted(pages, key=lambda p: p.leaf)

    def _group_key(self, override: PageOverride) -> GroupKey:
        return (override.color_mode or self.scantailor.color_mode, override.despeckle or self.scantailor.despeckle)

    def _clean_group(self, key: GroupKey, plans: list[LeafPlan], out_dir: Path) -> list[CleanedPage]:
        stage = Path(tempfile.mkdtemp(prefix=".st-", dir=out_dir))
        try:
            cuts = self._stage_leaves(plans, stage / "in")
            tool = self.tool(*key)
            try:
                pages = tool.run([stage / "in" / f"{plan.leaf:04d}.tif" for plan in plans], stage / "out")
            except (ScanTailorError, ValueError) as error:
                raise CleanError(str(error)) from error
            if self.docres is not None and key[0] in self.docres_modes:
                pages = self.docres.run(pages, stage / "docres")
            return [self._collect(plan, cut, page, out_dir, tool.output_dpi)
                    for plan, cut, page in zip(plans, cuts, pages, strict=True)]  # fmt: skip
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def _stage_leaves(plans: list[LeafPlan], stage: Path) -> list[Cut]:
        """Cut every leaf into ``stage/<leaf>.tif`` (name order = leaf order); a spread is read once."""
        stage.mkdir(parents=True)
        cache: tuple[Path, np.ndarray] | None = None
        cuts = []
        for plan in plans:
            if cache is None or cache[0] != plan.frame:
                cache = (plan.frame, load_bgr(plan.frame))
            cut = cut_leaf(cache[1], plan)
            if not cv2.imwrite(str(stage / f"{plan.leaf:04d}.tif"), cut.pixels):
                raise CleanError(f"leaf {plan.leaf}: cannot stage the cut leaf")
            cuts.append(cut)
        return cuts

    @staticmethod
    def _collect(plan: LeafPlan, cut: Cut, page: Path, out_dir: Path, dpi: int) -> CleanedPage:
        target = out_dir / f"{plan.leaf:04d}.tif"
        shutil.move(page, target)
        with Image.open(target) as im:
            width, height = im.size
        skew = plan.override.deskew_deg or 0.0
        return CleanedPage(plan.leaf, target, width, height, dpi, skew, cut.crop, cut.warnings)
