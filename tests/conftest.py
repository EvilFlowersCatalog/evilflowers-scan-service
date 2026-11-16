import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evilflowers_scan.imaging.scantailor import ScanTailor
from evilflowers_scan.settings import Settings

HAS_SCANTAILOR = shutil.which("scantailor-deviant-cli") is not None
needs_scantailor = pytest.mark.skipif(not HAS_SCANTAILOR, reason="scantailor-deviant-cli is not on PATH")

#: A spread: 1200 x 800 frame, gutter at x = 600.
FRAME_W, FRAME_H = 1200, 800


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Defaults only (no `.env`), storage in a temporary directory, no judges."""
    return Settings(_env_file=None, storage_path=tmp_path, image_tag="test", judges_enabled=False)


def page_pixels(
    width: int, height: int, *, stamp: tuple[int, int] | None = None, seed: int = 0, radius: int = 34
) -> np.ndarray:
    """An RGB page: white paper, black "text" lines, optionally a blue stamp (a ring of ``radius``) at ``stamp``."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 250, np.uint8)
    for y in range(int(height * 0.2), int(height * 0.8), 28):
        x0 = int(width * 0.18) + int(rng.integers(0, 10))
        img[y : y + 9, x0 : int(width * 0.8)] = 20
    if stamp is not None:
        cx, cy = stamp
        yy, xx = np.ogrid[:height, :width]
        ring = np.abs(np.hypot(xx - cx, yy - cy) - radius) < max(7, radius // 5)
        img[ring] = (40, 60, 200)  # RGB: blue ink
    return img


def write_frame(path: Path, *, stamp: tuple[int, int] | None = None) -> Path:
    """A spread frame as an uncompressed RGB TIFF (what the scanner leaves)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    left = page_pixels(FRAME_W // 2, FRAME_H, stamp=stamp)
    right = page_pixels(FRAME_W // 2, FRAME_H, seed=1)
    Image.fromarray(np.hstack([left, right])).save(path, dpi=(300, 300))
    return path


def write_page(path: Path, *, stamp: tuple[int, int] | None = None, mode: str = "RGB", size=(800, 1000)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(page_pixels(*size, stamp=stamp)).convert(mode).save(path, dpi=(600, 600),
                                                                         compression="tiff_deflate")  # fmt: skip
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class FakeScanTailor(ScanTailor):
    """Stands in for the binary: each staged leaf comes back as a deflate TIFF of the same name, in grey."""

    def run(self, inputs: list[Path], out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        pages = []
        for path in inputs:
            target = out_dir / path.name
            with Image.open(path) as im:
                im.convert("L" if self.color_mode != "color_grayscale" else "RGB").save(
                    target, compression="tiff_deflate", dpi=(self.output_dpi, self.output_dpi)
                )
            pages.append(target)
        RUNS.append((self.color_mode, self.despeckle, [p.name for p in inputs]))
        return pages


#: ``(color_mode, despeckle, staged names)`` of every FakeScanTailor run.
RUNS: list[tuple[str, str, list[str]]] = []


@pytest.fixture(autouse=True)
def _reset_runs():
    RUNS.clear()
    yield
    RUNS.clear()
