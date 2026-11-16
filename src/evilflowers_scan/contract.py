"""The messages the digitalizer's orchestrator sends and the outcome it reads (evilflowers-scan-service#1).

The generic half is the kit's ``StandaloneJob`` (``run_id``, ``job``, ``token``, ``inputs``, ``output_dir``,
``outcome``); the per-task halves mirror evilflowers-books-digitalizer ``studio.orchestration.contract``
(``clean_inputs``, ``erase_inputs``, ``CleanOutcome``). Every path is relative to ``STORAGE_PATH``.

The orchestrator reads the outcome with ``extra="ignore"``: the fields it does not know yet (``crop_box``,
``rotate_degree``, ``warnings``, ``marks``) cost nothing and are there for it to adopt.
"""

from typing import Any, Literal

from evilflowers_worker.contract import StandaloneJob
from evilflowers_worker.outcome import OutcomeFile
from pydantic import BaseModel, ConfigDict, Field, field_validator

from evilflowers_scan.geometry import BBox

Band = Literal["keep", "review", "hide"]
#: What a mark can be (#2). ``content`` is a real part of the page (a figure, an ornament).
MarkCategory = Literal["stamp", "sticker", "bleed_through", "edge", "content"]


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


# -- Messages -----------------------------------------------------------------------------------------------------


class CleanLeaf(_Row):
    """One leaf of the orchestrator's plan (``clean_inputs``)."""

    leaf: int = Field(ge=0)
    #: The raw frame, relative to ``STORAGE_PATH`` (``raw-scans/…``); only ever read.
    frame: str = Field(min_length=1)
    #: 0 left / 1 right half of a spread, ``None`` for a whole frame.
    side: int | None = None
    #: Frame pixels after rotation, IA ``x, y, w, h``.
    crop_box: tuple[int, int, int, int]
    #: IA ``rotateDegree``: clockwise, a multiple of 90.
    rotate_degree: int = 0
    hand_side: str | None = None
    #: The operator's override (``PageOverride.as_dict()``): crop, deskew_deg, split_x, despeckle, color_mode.
    override: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rotate_degree")
    @classmethod
    def _quarter_turns(cls, value: int) -> int:
        if value % 90:
            raise ValueError(f"rotate_degree must be a multiple of 90, not {value}")
        return value

    @field_validator("crop_box")
    @classmethod
    def _not_empty(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        BBox.from_xywh(value)
        return value


class ScanTailorOptions(_Row):
    """The project's ScanTailor settings (``[projects.X.pipeline.scantailor]``, #3), keyed as the digitalizer's
    ``ScanTailorCleaner.from_settings``. Every key is optional: a missing one keeps the worker's env value. Unknown
    keys (``split``) and ``binary`` are ignored: a message never picks the executable."""

    color_mode: Literal["mixed", "color_grayscale", "black_and_white"] | None = None
    despeckle: Literal["off", "cautious", "normal", "aggressive"] | None = None
    margins_mm: float | None = Field(None, ge=0)
    dewarping: Literal["off", "auto", "marginal"] | None = None
    normalize_illumination: bool | None = None
    dpi: int | None = Field(None, ge=50)
    output_dpi: int | None = Field(None, ge=50)

    def changes(self, env_output_dpi: int | None) -> dict[str, Any]:
        """The ``ScanTailor`` fields this job changes. ``output_dpi`` defaults to ``dpi`` as in the digitalizer:
        a message ``dpi`` without ``output_dpi`` sets both, unless the env pins ``OUTPUT_DPI``."""
        changes = self.model_dump(exclude_none=True)
        if self.output_dpi is None and self.dpi is not None and env_output_dpi is None:
            changes["output_dpi"] = self.dpi
        return changes


class CleanJob(StandaloneJob):
    """``evilflowers_scan.clean``: up to 32 leaves (bulk) or one (interactive)."""

    leaves: list[CleanLeaf] = Field(min_length=1)
    #: The project's ScanTailor settings for the whole job (#3); a leaf's ``override`` still wins.
    scantailor: ScanTailorOptions | None = None
    #: The project lets ``hide``-band marks be painted white on the cleaned leaf (#2). Off: nothing is erased.
    erase_hidden: bool = False

    @field_validator("leaves")
    @classmethod
    def _unique(cls, leaves: list[CleanLeaf]) -> list[CleanLeaf]:
        numbers = [leaf.leaf for leaf in leaves]
        if len(set(numbers)) != len(numbers):
            raise ValueError("a leaf appears twice in one job")
        return leaves


class EraseLeaf(_Row):
    """One leaf of ``erase_inputs``: the cleaned page and the operator's confirmed ``remove`` boxes."""

    leaf: int = Field(ge=0)
    image: str = Field(min_length=1)
    dpi: int = Field(ge=1)
    #: ``[x0, y0, x1, y1]`` in page pixels, x1 and y1 exclusive.
    boxes: list[tuple[int, int, int, int]]

    @field_validator("boxes")
    @classmethod
    def _not_empty(cls, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        for box in boxes:
            BBox.of(box)
        return boxes


class EraseJob(StandaloneJob):
    """``evilflowers_scan.erase``: paint confirmed boxes white on a copy, then the same renditions as ``clean``."""

    leaves: list[EraseLeaf] = Field(min_length=1)


# -- Outcome ------------------------------------------------------------------------------------------------------


class Mark(_Row):
    """A mark on a leaf (#2): a stamp, a sticker, bleed-through, a scan edge, or content.

    ``band``: ``keep`` (content), ``review`` (a disputed region in the page studio) or ``hide`` (an artefact; erased
    only when the project enables it). ``model`` is ``signals`` when no judge was asked, else the judge's
    ``model@revision``.
    """

    bbox: tuple[int, int, int, int]
    category: MarkCategory
    p: float = Field(ge=0.0, le=1.0)
    band: Band
    model: str
    #: The cheap signals behind the verdict: hash, position, density, aspect, repeats.
    signals: dict[str, Any] = Field(default_factory=dict)
    #: The judge's reason, when one answered.
    reason: str | None = None


class CleanedLeaf(_Row):
    """``CleanedLeaf`` of the orchestrator's contract, plus what this service applied and found."""

    leaf: int
    #: The cleaned page (lossless TIFF), relative to ``STORAGE_PATH``.
    image: str
    width: int
    height: int
    dpi: int
    skew_deg: float = 0.0
    preview: str | None = None
    dzi: str
    #: The crop actually cut from the rotated frame, IA ``x, y, w, h`` (the override's, when it had one).
    crop_box: tuple[int, int, int, int] | None = None
    rotate_degree: int | None = None
    warnings: list[str] = Field(default_factory=list)
    marks: list[Mark] = Field(default_factory=list)


class CleanOutcome(OutcomeFile):
    """The outcome of ``clean`` and ``erase`` (the orchestrator reads both with its ``CleanOutcome``)."""

    status: Literal["done", "failed"]
    error: str | None = None
    leaves: list[CleanedLeaf] = Field(default_factory=list)
