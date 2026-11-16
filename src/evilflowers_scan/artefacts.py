"""Is this mark real or an artefact? Stamps, stickers, bleed-through and scan edges on cleaned leaves (#2).

**Signals first, no model.** Each leaf yields candidates from four cheap detectors:

- ``stamp``: saturated blue/violet/red ink (printed text here is black), the digitalizer's ``stamp_mask`` widened to
  red stamps;
- ``edge``: a dark band touching the page border (scanner bed, a finger shadow, the neighbouring page);
- ``sticker``: a large, flat, non-white rectangle (call-number labels, library stickers);
- ``bleed_through``: faint grey ink away from any dark ink (the other side of the leaf shining through);
- colour-free ink clusters in a margin or the gutter (``stamp``, or ``sticker`` when dense): ScanTailor's default
  ``mixed`` mode binarizes a stamp, so colour alone would miss it there. Signals alone never take such a mark past
  ``COLOURLESS_CAP`` (it may be a running head or an ornament): it goes to the judge, or to ``review``.

Each candidate gets a ``p`` from its own signals (saturation, darkness, fill, contrast), raised by its position (a
margin or the gutter) and by repetition: a perceptual hash (dHash, 64 bits) equal within a few bits on several
leaves of the job is **furniture** (the same stamp on every leaf), as image-service#6 treats repeated images.

**Bands.** ``p >= artefact_sure``: ``hide`` (no model asked). ``p < artefact_floor``: dropped as noise. In between the
mark is uncertain, and only then is ``decider-vision`` asked a ``choice`` question through the kit's judges, within the
job's budget. Its verdict decides the category and, with ``JUDGE_THRESHOLDS["scan_mark"]``, the band. Without a judge
(disabled, unreachable, budget spent) an uncertain mark is ``review``: a disputed region in the page studio.

Nothing is erased here. ``jobs`` paints ``hide`` marks white only when the message says the project enables it.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from evilflowers_worker.judges import Choice, Judges, Panel, Verdict
from evilflowers_worker.observability.metrics import Instrument
from PIL import Image

from evilflowers_scan.contract import Band, Mark, MarkCategory
from evilflowers_scan.geometry import BBox

logger = logging.getLogger(__name__)

#: ``scan_marks_total{category, band}``: every mark reported, by its final category and band. The judge calls
#: themselves are the kit's ``scan_judge_total{question, outcome, band}``.
MARKS = Instrument("counter", "{service}.marks", "{mark}", "Marks reported on cleaned leaves, by category and band")

ARTEFACTS: frozenset[str] = frozenset({"stamp", "sticker", "bleed_through", "edge"})

QUESTION = Choice(
    id="scan_mark",
    instructions=(
        "A region of a scanned library book page, shown with a little context. Is it part of the printed page, or "
        "something added to or around it?"
    ),
    criteria={
        "stamp": "an ink stamp: a library or ownership stamp, a date stamp, a signature",
        "sticker": "a sticker or label pasted on the page: call number, barcode, library label",
        "bleed_through": "faint text or images showing through from the other side of the leaf",
        "edge": "the edge of the scan: scanner bed, a finger, a shadow, the neighbouring page",
        "content": "part of the printed page itself: text, a figure, an ornament, a printed rule",
    },
    reject=ARTEFACTS,
)

#: The detection runs on the page scaled to this longest side; boxes go back to page pixels.
WORK_PX = 1600
#: Mean grey of paper next to a scan edge.
PAPER = 150.0
#: The highest ``p`` signals alone give a mark found without colour: it may be a running head, a page number or an
#: ornament, so it goes to the judge (or ``review``) and is never hidden on signals alone.
COLOURLESS_CAP = 0.8


@dataclass(frozen=True, slots=True)
class MarkSettings:
    sure: float = 0.85
    floor: float = 0.35
    min_repeats: int = 3
    hash_distance: int = 6
    max_marks: int = 40
    #: The smallest candidate, as a share of the page area.
    min_area: float = 0.0004


@dataclass(slots=True)
class Candidate:
    leaf: int
    bbox: BBox
    category: MarkCategory
    p: float
    signals: dict[str, Any] = field(default_factory=dict)
    hash: int = 0
    crop: bytes = b""


@dataclass(frozen=True, slots=True)
class LeafImage:
    leaf: int
    image: Path
    #: 0 left / 1 right half of a spread (the gutter is right of side 0, left of side 1), ``None`` if unknown.
    side: int | None = None


# -- Pixels -------------------------------------------------------------------------------------------------------


def read_bgr(path: Path) -> np.ndarray:
    """8-bit BGR whatever the TIFF flavour (bitonal, grey, 16-bit, alpha)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read page {path.name}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, int(img.max())))
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash: robust to scale and small tone changes, so one stamp on two leaves hashes alike."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, bit in enumerate(bits) if bit))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _components(mask: np.ndarray, min_area: float) -> list[tuple[int, int, int, int, int]]:
    """``(x, y, w, h, area)`` of the connected components of ``mask`` no smaller than ``min_area`` pixels."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [tuple(int(v) for v in stats[i]) for i in range(1, n) if stats[i][4] >= min_area]  # type: ignore[misc]


type Box4 = tuple[int, int, int, int]


def text_block(gray: np.ndarray) -> Box4 | None:
    """The bounding box of the page's text lines (long thin ink runs), or ``None`` on a page without any.

    ScanTailor adds margins and centres the content, so "the margin" is where the text block is not, not a fixed
    share of the page.
    """
    width = gray.shape[1]
    ink = (gray < 128).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 25), np.uint8))
    rows = [c for c in _components(lines, 50) if c[2] > 4 * c[3] and c[2] > 0.2 * width]
    if len(rows) < 3:
        return None
    return (min(c[0] for c in rows), min(c[1] for c in rows), max(c[0] + c[2] for c in rows),
            max(c[1] + c[3] for c in rows))  # fmt: skip


def position(box: Box4, width: int, height: int, side: int | None, block: Box4 | None = None) -> str:
    """``gutter``, ``margin`` or ``body``: outside the text block (or, without one, in the outer 12 %) is a margin; the
    margin on the binding side of a spread half is the gutter."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if block is not None:
        inside = block[0] <= cx <= block[2] and block[1] <= cy <= block[3]
        left, right = cx < block[0], cx > block[2]
    else:
        inside = 0.12 * width <= cx <= 0.88 * width and 0.12 * height <= cy <= 0.88 * height
        left, right = cx < 0.12 * width, cx > 0.88 * width
    if inside:
        return "body"
    if (side == 0 and right) or (side == 1 and left):
        return "gutter"
    return "margin"


# -- Detectors (on the work-scale page) ---------------------------------------------------------------------------


def stamp_boxes(bgr: np.ndarray, min_area: float) -> list[tuple[tuple[int, int, int, int, int], float]]:
    """Saturated blue/violet or red ink; each with its mean saturation (0..255)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = ((h >= 95) & (h <= 160)) | (h <= 8) | (h >= 170)
    ink = ((s > 70) & (v < 235) & hue).astype(np.uint8)
    mask = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    found = []
    for comp in _components(mask, min_area):
        x, y, w, hh, _ = comp
        found.append((comp, float(s[y : y + hh, x : x + w][mask[y : y + hh, x : x + w] > 0].mean())))
    return found


def edge_boxes(gray: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    """Dark bands touching a border: ``(x0, y0, x1, y1)`` and their darkness (0..1)."""
    h, w = gray.shape
    found = []
    limit_x, limit_y = max(1, w // 6), max(1, h // 6)
    cols, rows = gray.mean(axis=0), gray.mean(axis=1)
    dark = 90.0

    def run(values: np.ndarray) -> int:
        n = 0
        while n < len(values) and values[n] < dark:
            n += 1
        return n

    for side, values, limit in (("left", cols, limit_x), ("right", cols[::-1], limit_x),
                                ("top", rows, limit_y), ("bottom", rows[::-1], limit_y)):  # fmt: skip
        n = run(values[:limit])
        # A band ends inside the outer sixth and paper follows it; a dark plate or a full-bleed photo does neither.
        if n < 2 or n >= limit or values[n : n + max(2, limit // 4)].mean() < PAPER:
            continue
        darkness = float(1 - values[:n].mean() / 255)
        box = {"left": (0, 0, n, h), "right": (w - n, 0, w, h), "top": (0, 0, w, n), "bottom": (0, h - n, w, h)}[side]
        found.append((box, darkness))
    return found


def ink_boxes(gray: np.ndarray, min_side: int, max_area: float) -> list[tuple[tuple[int, int, int, int, int], float]]:
    """Compact clusters of ink, colour or not (a mixed-mode page is binarized, so a stamp is black there too).

    Glyphs are merged by a closing; text lines stay long and thin and are dropped by the aspect bound, the body text
    block by ``max_area``. Each with its ink fill ratio inside the box.
    """
    ink = (gray < 128).astype(np.uint8)
    merged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    found = []
    for comp in _components(merged, min_side * min_side):
        x, y, w, h, area = comp
        if min(w, h) < min_side or area > max_area or not 1 / 3 <= w / h <= 3:
            continue
        found.append((comp, float(ink[y : y + h, x : x + w].mean())))
    return found


def sticker_boxes(gray: np.ndarray, min_area: float) -> list[tuple[tuple[int, int, int, int, int], float]]:
    """Large flat non-white patches that fill their bounding box; each with its fill ratio."""
    flat = cv2.blur(cv2.absdiff(gray, cv2.blur(gray, (7, 7))), (7, 7)) < 6
    patch = ((gray > 110) & (gray < 228) & flat).astype(np.uint8)
    mask = cv2.morphologyEx(patch, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    found = []
    for comp in _components(mask, max(min_area * 4, 400)):
        _x, _y, w, h, area = comp
        fill = area / float(w * h)
        if fill > 0.8 and 0.15 < w / max(h, 1) < 8:
            found.append((comp, fill))
    return found


def bleed_boxes(gray: np.ndarray, min_area: float) -> list[tuple[tuple[int, int, int, int, int], float]]:
    """Faint grey ink well away from dark ink; each with its contrast to the paper (0..1, low = faint)."""
    ink = cv2.dilate((gray < 110).astype(np.uint8), np.ones((15, 15), np.uint8))
    faint = ((gray > 150) & (gray < 215) & (ink == 0)).astype(np.uint8)
    faint = cv2.morphologyEx(faint, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    paper = float(np.percentile(gray, 90))
    found = []
    for comp in _components(faint, min_area * 2):
        x, y, w, h, _ = comp
        region = gray[y : y + h, x : x + w]
        contrast = max(0.0, (paper - float(region.min())) / 255)
        found.append((comp, contrast))
    return found


# -- Candidates of one leaf ---------------------------------------------------------------------------------------


def _crop_png(bgr: np.ndarray, box: tuple[int, int, int, int], max_side: int = 512) -> bytes:
    """The region with a little context, PNG, at most ``max_side`` px: what a judge looks at."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box
    pad_x, pad_y = max(8, (x1 - x0) // 4), max(8, (y1 - y0) // 4)
    region = bgr[max(0, y0 - pad_y) : min(h, y1 + pad_y), max(0, x0 - pad_x) : min(w, x1 + pad_x)]
    image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def leaf_candidates(leaf: LeafImage, settings: MarkSettings) -> list[Candidate]:
    """Every candidate mark of one leaf, in page pixels, largest first, at most ``max_marks``."""
    page = read_bgr(leaf.image)
    height, width = page.shape[:2]
    scale = min(1.0, WORK_PX / max(height, width))
    work = cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else page
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    min_area = settings.min_area * gray.size
    block = text_block(gray)
    work_h, work_w = gray.shape
    found: list[tuple[Box4, MarkCategory, float, dict[str, Any]]] = []

    for (x, y, w, h, area), saturation in stamp_boxes(work, min_area):
        p = 0.5 + 0.35 * min(1.0, saturation / 160)
        found.append(((x, y, x + w, y + h), "stamp", p, {"saturation": round(saturation, 1), "area": area}))
    for band, darkness in edge_boxes(gray):
        found.append((band, "edge", 0.55 + 0.4 * darkness, {"darkness": round(darkness, 3)}))
    # Colour-free: ink clusters outside the body text. A stamp in mixed mode lands here, binarized.
    stamped = [(x0, y0, x1, y1) for (x0, y0, x1, y1), category, *_ in found if category == "stamp"]
    min_side = max(8, int(0.04 * min(gray.shape)))
    for (x, y, w, h, area), fill in ink_boxes(gray, min_side, 0.06 * gray.size):
        cluster = (x, y, x + w, y + h)
        if any(_overlaps(cluster, other) for other in stamped):
            continue
        kind: MarkCategory = "sticker" if fill > 0.6 else "stamp"
        found.append((cluster, kind, 0.45, {"fill": round(fill, 3), "area": area, "colour": False}))
    for (x, y, w, h, area), fill in sticker_boxes(gray, min_area):
        found.append(((x, y, x + w, y + h), "sticker", 0.4 + 0.35 * fill, {"fill": round(fill, 3), "area": area}))
    for (x, y, w, h, area), contrast in bleed_boxes(gray, min_area):
        p = 0.35 + 0.3 * max(0.0, 1 - contrast / 0.35)
        found.append(((x, y, x + w, y + h), "bleed_through", p, {"contrast": round(contrast, 3), "area": area}))

    candidates = []
    for (x0, y0, x1, y1), category, p, signals in found:
        try:
            box = BBox(int(x0 / scale), int(y0 / scale), min(width, int(np.ceil(x1 / scale))),
                       min(height, int(np.ceil(y1 / scale))))  # fmt: skip
        except ValueError:
            continue
        where = position((x0, y0, x1, y1), work_w, work_h, leaf.side, block)
        colourless = signals.get("colour") is False
        if colourless and where == "body":
            continue  # ink inside the text block is the text
        if where != "body" and category != "edge":
            p += 0.1
        roi = gray[y0:y1, x0:x1]
        ink = float((roi < 128).mean()) if roi.size else 0.0
        candidates.append(
            Candidate(
                leaf=leaf.leaf,
                bbox=box,
                category=category,
                p=min(p, _cap(signals)),
                signals={
                    **signals,
                    "position": where,
                    "ink_density": round(ink, 3),
                    "aspect": round(box.width / box.height, 3),
                },
                hash=dhash(roi) if roi.size and min(roi.shape) >= 2 else 0,
                crop=_crop_png(work, (x0, y0, x1, y1)),
            )
        )
    candidates.sort(key=lambda c: c.bbox.area, reverse=True)
    return candidates[: settings.max_marks]


def count_repeats(candidates: Sequence[Candidate], settings: MarkSettings) -> None:
    """On each candidate: on how many leaves of the job a mark of the same category and hash appears."""
    hashed = [c for c in candidates if c.hash and c.category != "edge"]
    for candidate in hashed:
        leaves = {other.leaf for other in hashed if other.category == candidate.category
                  and hamming(candidate.hash, other.hash) <= settings.hash_distance}  # fmt: skip
        candidate.signals["repeats"] = len(leaves)
        candidate.signals["dhash"] = f"{candidate.hash:016x}"
        if len(leaves) >= settings.min_repeats:
            candidate.p = min(_cap(candidate.signals), candidate.p + 0.15)


def _cap(signals: dict[str, Any]) -> float:
    return COLOURLESS_CAP if signals.get("colour") is False else 0.97


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# -- Verdicts -----------------------------------------------------------------------------------------------------


def _signal_band(category: MarkCategory) -> Band:
    return "hide" if category in ARTEFACTS else "keep"


def judge(candidate: Candidate, panel: Panel | None, judges: Judges | None) -> Mark:
    """The mark of one uncertain candidate: the judge's verdict when one answers, else ``review`` on signals."""
    verdict: Verdict | None = None
    if panel is not None and judges is not None:
        state = f"Page region, signals: {candidate.category}, position {candidate.signals.get('position')}."
        verdict = panel.ask(judges.decider_vision, QUESTION, state=state, images=[candidate.crop])
    if verdict is None:
        return _mark(candidate, candidate.category, candidate.p, "review", "signals")
    category: MarkCategory = verdict.verdict if verdict.verdict in QUESTION.criteria else candidate.category  # type: ignore[assignment]
    return _mark(candidate, category, verdict.p, verdict.band or "review", verdict.model_ref, verdict.reason)


def _mark(candidate: Candidate, category: MarkCategory, p: float, band: Band, model: str,
          reason: str | None = None) -> Mark:  # fmt: skip
    MARKS.add(1, {"category": category, "band": band})
    return Mark(bbox=tuple(candidate.bbox.as_list()), category=category, p=round(p, 3), band=band, model=model,
                signals=candidate.signals, reason=reason)  # type: ignore[arg-type]  # fmt: skip


def marks(leaves: Sequence[LeafImage], settings: MarkSettings, *, panel: Panel | None = None,
          judges: Judges | None = None) -> dict[int, list[Mark]]:  # fmt: skip
    """The marks of every leaf of a job, keyed by leaf. A leaf whose page cannot be read gets none (logged)."""
    candidates: list[Candidate] = []
    for leaf in leaves:
        try:
            candidates += leaf_candidates(leaf, settings)
        except Exception:
            logger.warning("Artefact check skipped leaf %d", leaf.leaf, exc_info=True)
    count_repeats(candidates, settings)
    result: dict[int, list[Mark]] = {leaf.leaf: [] for leaf in leaves}
    for candidate in candidates:
        if candidate.p < settings.floor:
            continue
        if candidate.p >= settings.sure and candidate.signals.get("colour") is not False:
            mark = _mark(candidate, candidate.category, candidate.p, _signal_band(candidate.category), "signals")
        else:
            mark = judge(candidate, panel, judges)
        result[candidate.leaf].append(mark)
    return result


def from_settings(settings: Any) -> MarkSettings:
    """``MarkSettings`` from the service's ``Settings``."""
    return replace(
        MarkSettings(),
        sure=settings.artefact_sure,
        floor=settings.artefact_floor,
        min_repeats=settings.artefact_min_repeats,
        hash_distance=settings.artefact_hash_distance,
        max_marks=settings.artefact_max_marks,
    )
