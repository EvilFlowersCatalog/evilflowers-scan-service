"""DocRes appearance enhancement of cleaned pages (optional), ported from evilflowers-books-digitalizer
``pipeline/steps/docres.py``.

DocRes (CVPR 2024) whitens paper and removes show-through while keeping tones (photos, stamps): the complement to
ScanTailor's ``color_grayscale`` output, where binarization would destroy content. The ``appearance`` task divides a
shadow map out of the original-resolution image, so pixels do not move and the page geometry is unchanged.

It runs out of process, in its own venv with torch, from the weights on the ``/models`` PVC
(``DOCRES_REPO``, ``DOCRES_PYTHON``). Off unless ``DOCRES_ENABLED``.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class DocResError(RuntimeError):
    """DocRes is missing, failed, or left a page out."""


@dataclass(frozen=True, slots=True)
class DocRes:
    repo: Path
    python: Path
    task: str = "appearance"
    timeout: float | None = None

    def available(self) -> bool:
        return self.python.exists() and (self.repo / "inference.py").exists()

    def run(self, pages: list[Path], out_dir: Path) -> list[Path]:
        """Enhance ``pages`` (one directory) into ``out_dir``; the enhanced pages, in the same order."""
        if not pages:
            return []
        if not self.available():
            raise DocResError(f"DocRes not found (repo={self.repo}, python={self.python}); or set DOCRES_ENABLED=0")
        in_dir = pages[0].parent.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.python), "inference.py", "--task", self.task, "--im_path", str(in_dir),
               "--out_folder", str(out_dir.resolve())]  # fmt: skip
        try:
            result = subprocess.run(
                cmd, cwd=self.repo, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except subprocess.TimeoutExpired as error:
            raise DocResError(f"DocRes timed out after {error.timeout:.0f} s") from error
        if result.returncode != 0:
            raise DocResError(f"DocRes exited {result.returncode}: {result.stderr[-2000:].strip()}")
        # inference.py writes <stem>_<task>.<ext> for each input.
        enhanced = []
        for page in pages:
            out = out_dir / f"{page.stem}_{self.task}{page.suffix}"
            if not out.exists():
                raise DocResError(f"DocRes output missing: {out.name}")
            enhanced.append(out)
        logger.info("DocRes %s on %d pages", self.task, len(enhanced))
        return enhanced
