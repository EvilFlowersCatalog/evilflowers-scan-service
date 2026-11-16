"""Erasures and browser renditions of a cleaned page, ported from evilflowers-books-digitalizer
``studio/adapters/imaging/pillow_ops.py``.

Layout inside the job's directory (free, per the orchestrator): ``<leaf>.tif`` (the page), ``leaves/<leaf>.webp``
(preview) and ``dzi/<leaf>.dzi`` with ``dzi/<leaf>_files/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from evilflowers_scan.geometry import BBox
from evilflowers_scan.imaging.dzi import TILE_SIZE, write_dzi

Image.MAX_IMAGE_PIXELS = None


def paper_white(mode: str) -> int | tuple[int, ...]:
    """Paper white in the image's own mode (a bitonal page stays bitonal)."""
    match mode:
        case "1":
            return 1
        case "L" | "P" | "I" | "F":
            return 255
        case "I;16":
            return 65535
        case "RGBA":
            return (255, 255, 255, 255)
        case "LA":
            return (255, 255)
        case _:
            return (255, 255, 255)


@dataclass(frozen=True, slots=True)
class Renditions:
    preview: Path
    dzi: Path


@dataclass(frozen=True, slots=True)
class ImageOps:
    webp_quality: int = 80
    tile: int = TILE_SIZE
    dzi_quality: int = 85

    def erase(self, image: Path, boxes: Sequence[BBox], out: Path) -> Path:
        """Paint ``boxes`` paper-white (x1, y1 exclusive) on a copy written to ``out``; size, mode, dpi and the TIFF
        compression rule (group4 for bitonal, deflate otherwise) are kept. ``out`` may equal ``image``."""
        with Image.open(image) as im:
            im.load()
            dpi = im.info.get("dpi")
            page = im.copy()
        draw = ImageDraw.Draw(page)
        for box in boxes:
            x0, y0 = max(0, box.x0), max(0, box.y0)
            x1, y1 = min(page.width, box.x1), min(page.height, box.y1)
            if x1 > x0 and y1 > y0:
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=paper_white(page.mode))
        out.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, object] = {"dpi": dpi} if dpi else {}
        if out.suffix.lower() in (".tif", ".tiff"):
            kwargs["compression"] = "group4" if page.mode == "1" else "tiff_deflate"
        page.save(out, **kwargs)  # type: ignore[arg-type]
        return out

    def previews(self, image: Path, out_dir: Path, *, max_px: int) -> Renditions:
        """``leaves/<stem>.webp`` (bounded by ``max_px``) and ``dzi/<stem>.dzi`` + ``_files/``."""
        with Image.open(image) as im:
            im.load()
            page = im.copy()
        webp = out_dir / "leaves" / f"{image.stem}.webp"
        webp.parent.mkdir(parents=True, exist_ok=True)
        small = page.convert("L") if page.mode in ("1", "L", "I;16") else page.convert("RGB")
        small.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        small.save(webp, "WEBP", quality=self.webp_quality, method=4)
        dzi = write_dzi(page, out_dir / "dzi" / f"{image.stem}.dzi", tile=self.tile, quality=self.dzi_quality)
        return Renditions(webp, dzi)
