"""Deep Zoom (DZI) pyramids for OpenSeadragon, ported from evilflowers-books-digitalizer
``studio/adapters/imaging/dzi.py`` (the grid and the batch writer; the orchestrator keeps the on-demand tile cache).

Pyramid convention (Microsoft Deep Zoom, as OpenSeadragon reads it): level ``max_level = ceil(log2(max(w, h)))`` is
full resolution, each level below halves it (rounding up), level 0 is 1x1. Tiles are ``tile_size`` px plus ``overlap``
px shared with each neighbour. The grid must stay identical to the orchestrator's, or its tiles and these disagree.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TILE_SIZE = 254
OVERLAP = 1
DZI_NS = "http://schemas.microsoft.com/deepzoom/2008"

type Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class DeepZoomGrid:
    width: int
    height: int
    tile_size: int = TILE_SIZE
    overlap: int = OVERLAP

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError(f"empty image {self.width}x{self.height}")

    @property
    def max_level(self) -> int:
        return math.ceil(math.log2(max(self.width, self.height)))

    def scale(self, level: int) -> int:
        """Full-resolution pixels per level pixel (a power of two)."""
        if not 0 <= level <= self.max_level:
            raise ValueError(f"level {level} outside 0..{self.max_level}")
        return 2 ** (self.max_level - level)

    def level_size(self, level: int) -> tuple[int, int]:
        s = self.scale(level)
        return max(1, math.ceil(self.width / s)), max(1, math.ceil(self.height / s))

    def tile_count(self, level: int) -> tuple[int, int]:
        w, h = self.level_size(level)
        return math.ceil(w / self.tile_size), math.ceil(h / self.tile_size)

    def tile_box(self, level: int, col: int, row: int) -> Box:
        """The tile's pixels at ``level`` (x0, y0, x1, y1), overlap included."""
        cols, rows = self.tile_count(level)
        if not (0 <= col < cols and 0 <= row < rows):
            raise ValueError(f"tile {col},{row} outside level {level} ({cols}x{rows})")
        w, h = self.level_size(level)
        (x0, x1), (y0, y1) = self._span(col, w), self._span(row, h)
        return x0, y0, x1, y1

    def descriptor(self, fmt: str = "jpg") -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<Image xmlns="{DZI_NS}" Format="{fmt}" Overlap="{self.overlap}" TileSize="{self.tile_size}">'
            f'<Size Width="{self.width}" Height="{self.height}"/></Image>\n'
        )

    def _span(self, index: int, extent: int) -> tuple[int, int]:
        start = index * self.tile_size - (self.overlap if index else 0)
        end = min(extent, (index + 1) * self.tile_size + self.overlap)
        return start, end


def browser_mode(im: Image.Image) -> Image.Image:
    return im.convert("L") if im.mode in ("1", "L", "I;16", "I", "LA") else im.convert("RGB")


def write_dzi(image: Image.Image, dzi: Path, *, tile: int = TILE_SIZE, overlap: int = OVERLAP, fmt: str = "jpg",
              quality: int = 85) -> Path:  # fmt: skip
    """Write ``dzi`` and its ``<stem>_files`` pyramid; returns ``dzi``.

    Each level is the previous one halved (not the full page resampled each time), which keeps the cost close to one
    full-size pass; halving with rounding up gives exactly ``DeepZoomGrid.level_size``.
    """
    base = browser_mode(image)
    grid = DeepZoomGrid(base.width, base.height, tile, overlap)
    files = dzi.with_name(f"{dzi.stem}_files")
    if files.exists():
        shutil.rmtree(files)
    level_img = base
    for level in range(grid.max_level, -1, -1):
        out = files / str(level)
        out.mkdir(parents=True, exist_ok=True)
        cols, rows = grid.tile_count(level)
        for col in range(cols):
            for row in range(rows):
                level_img.crop(grid.tile_box(level, col, row)).save(out / f"{col}_{row}.{fmt}", quality=quality)
        if level:
            level_img = level_img.resize(grid.level_size(level - 1), Image.Resampling.LANCZOS)
    dzi.parent.mkdir(parents=True, exist_ok=True)
    dzi.write_text(grid.descriptor(fmt))
    return dzi
