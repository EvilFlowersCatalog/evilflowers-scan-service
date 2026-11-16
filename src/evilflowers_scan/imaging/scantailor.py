"""ScanTailor (``scantailor-deviant-cli``) over a directory of already cut leaves.

Ported from evilflowers-books-digitalizer ``pipeline/steps/scantailor.py``. Two gotchas noted there are kept:

- **The split bug.** ``scantailor-deviant-cli``'s single-pass batch mode finds the gutter of a spread but does not
  carry the cut into content detection (the right page gets the whole spread, the left page comes out blank). So it
  never splits here: the orchestrator plans the leaves, ``cleaner.cut_leaf`` cuts them, ScanTailor runs with
  ``--layout=1``.
- **The /tmp symlink.** Leptonica / ScanTailor choke on macOS's ``/tmp`` → ``/private/tmp`` symlink: every path handed
  to the binary is resolved first. The service also stages inside the job's ``output_dir``, never in ``/tmp``.

ScanTailor keeps the input names; the result is checked to hold one page per input.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

COLOR_MODES = ("mixed", "color_grayscale", "black_and_white")
DESPECKLE = ("off", "cautious", "normal", "aggressive")


class ScanTailorError(RuntimeError):
    """The binary is missing, failed, or did not return one page per input."""


@dataclass(frozen=True, slots=True)
class ScanTailor:
    binary: str = "scantailor-deviant-cli"
    color_mode: str = "mixed"
    despeckle: str = "cautious"
    margins_mm: float = 8.0
    dewarping: str = "auto"
    normalize_illumination: bool = True
    dpi: int = 300
    output_dpi: int = 300
    #: Seconds per run; ``None``: the task's time limit is the bound.
    timeout: float | None = None

    def __post_init__(self) -> None:
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"unknown color_mode: {self.color_mode!r}")
        if self.despeckle not in DESPECKLE:
            raise ValueError(f"unknown despeckle: {self.despeckle!r}")

    def resolve_binary(self) -> str:
        path = shutil.which(self.binary)
        if path is None:
            raise ScanTailorError(f"ScanTailor binary not found: {self.binary!r} (SCANTAILOR_BINARY)")
        return path

    def command(self, binary: str, in_dir: Path, out_dir: Path) -> list[str]:
        cmd = [
            binary,
            "--layout=1",  # leaves are pre-cut: see the split bug above
            "--deskew=auto",
            "--content-detection=normal",
            f"--margins={self.margins_mm}",
            "--alignment=center",
            f"--dpi={self.dpi}",
            f"--output-dpi={self.output_dpi}",
            f"--color-mode={self.color_mode}",
            "--white-margins",  # mandatory: the margins show the raw scanner bed otherwise
            f"--despeckle={self.despeckle}",
            f"--dewarping={self.dewarping}",
            "--start-filter=1",
            "--end-filter=6",
            "--tiff-compression=deflate",
        ]
        if self.normalize_illumination:
            cmd.append("--normalize-illumination")
        return [*cmd, str(in_dir.resolve()), str(out_dir.resolve())]

    def run(self, inputs: list[Path], out_dir: Path) -> list[Path]:
        """Clean ``inputs`` (all in one directory, name order = leaf order) into ``out_dir``; the pages, in order."""
        if not inputs:
            raise ValueError("no leaves to clean")
        in_dir = inputs[0].parent
        if any(path.parent != in_dir for path in inputs):
            raise ValueError("ScanTailor takes one input directory")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.command(self.resolve_binary(), in_dir, out_dir)
        logger.debug("ScanTailor: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise ScanTailorError(f"scantailor timed out after {error.timeout:.0f} s") from error
        if result.returncode != 0:
            raise ScanTailorError(f"scantailor exited {result.returncode}: {result.stderr[-2000:].strip()}")
        by_stem = {page.stem: page for page in out_dir.glob("*.tif")}
        pages = [by_stem.get(path.stem) for path in inputs]
        if any(page is None for page in pages) or len(by_stem) != len(inputs):
            raise ScanTailorError(f"scantailor returned {len(by_stem)} pages for {len(inputs)} leaves")
        return [page for page in pages if page is not None]
