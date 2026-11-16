"""Boxes and the operator's per-leaf override, ported from evilflowers-books-digitalizer ``studio.domain``.

``BBox``: x0, y0 inclusive, x1, y1 exclusive, in pixels. The digitalizer stores crops as IA ``x, y, w, h``
(``from_xywh``); boxes in a message (an erase, a mark) are ``[x0, y0, x1, y1]``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"empty bbox {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    def to_xywh(self) -> list[int]:
        return [self.x0, self.y0, self.width, self.height]

    @classmethod
    def of(cls, values: Sequence[float]) -> BBox:
        x0, y0, x1, y1 = (round(v) for v in values)
        return cls(x0, y0, x1, y1)

    @classmethod
    def from_xywh(cls, values: Sequence[float]) -> BBox:
        x, y, w, h = (round(v) for v in values)
        return cls(x, y, x + w, y + h)


@dataclass(frozen=True, slots=True)
class PageOverride:
    """Operator settings pinned to one leaf (the digitalizer's ``PageOverride``); ``None`` means automatic.

    Geometry is in frame pixels after rotation, like the planned crop.
    """

    crop: BBox | None = None
    deskew_deg: float | None = None
    #: The gutter of a spread: side 0 is cut left of it, side 1 right of it.
    split_x: int | None = None
    despeckle: str | None = None
    color_mode: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PageOverride:
        """The digitalizer's ``PageOverride.as_dict()``; unknown keys (``ocr_language``) are ignored."""
        d = dict(data or {})
        crop = d.get("crop")
        return cls(
            crop=BBox.of(crop) if isinstance(crop, list | tuple) else None,
            deskew_deg=float(d["deskew_deg"]) if d.get("deskew_deg") is not None else None,
            split_x=int(d["split_x"]) if d.get("split_x") is not None else None,
            despeckle=d.get("despeckle") or None,
            color_mode=d.get("color_mode") or None,
            extra=dict(d.get("extra") or {}),
        )
