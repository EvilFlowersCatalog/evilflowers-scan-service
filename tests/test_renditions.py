"""Erasures, WebP previews and Deep Zoom pyramids."""

import numpy as np
import pytest
from PIL import Image

from evilflowers_scan.geometry import BBox
from evilflowers_scan.imaging.dzi import DeepZoomGrid, write_dzi
from evilflowers_scan.imaging.ops import ImageOps


def test_the_grid_follows_deep_zoom():
    grid = DeepZoomGrid(1000, 600)
    assert grid.max_level == 10
    assert grid.level_size(10) == (1000, 600)
    assert grid.level_size(9) == (500, 300)
    assert grid.level_size(0) == (1, 1)
    assert grid.tile_count(10) == (4, 3)
    assert grid.tile_box(10, 1, 0) == (253, 0, 509, 255)  # overlap on both sides


def test_write_dzi_writes_every_level(tmp_path):
    dzi = write_dzi(Image.new("L", (600, 300), 200), tmp_path / "dzi" / "0003.dzi")
    assert 'Width="600" Height="300"' in dzi.read_text()
    files = tmp_path / "dzi" / "0003_files"
    assert sorted(int(p.name) for p in files.iterdir()) == list(range(DeepZoomGrid(600, 300).max_level + 1))
    # 600 x 300 at 254 px tiles: 3 columns, 2 rows.
    assert sorted(p.name for p in (files / "10").iterdir()) == [f"{c}_{r}.jpg" for c in range(3) for r in range(2)]


@pytest.mark.parametrize("mode", ["RGB", "L", "1"])
def test_erase_paints_white_keeps_size_mode_and_dpi(tmp_path, mode):
    source = tmp_path / "page.tif"
    Image.new(mode, (100, 80), 0).save(source, dpi=(600, 600))
    out = tmp_path / "out" / "page.tif"
    ImageOps().erase(source, [BBox(10, 20, 30, 40)], out)
    with Image.open(out) as im:
        assert (im.size, im.mode) == ((100, 80), mode)
        assert round(im.info["dpi"][0]) == 600
        # Pillow reads deflate back as `tiff_adobe_deflate` (the same codec, the newer tag value).
        assert im.info["compression"] in ({"group4"} if mode == "1" else {"tiff_deflate", "tiff_adobe_deflate"})
        pixels = np.asarray(im.convert("L"))
    assert pixels[20:40, 10:30].min() == 255
    assert pixels[40, 10] == 0 and pixels[20, 30] == 0  # x1 and y1 are exclusive
    with Image.open(source) as im:
        assert np.asarray(im.convert("L")).max() == 0  # the source is untouched


def test_previews_lay_out_webp_and_dzi(tmp_path):
    page = tmp_path / "0003.tif"
    Image.new("1", (2000, 3000), 1).save(page)
    renditions = ImageOps().previews(page, tmp_path, max_px=1600)
    assert renditions.preview == tmp_path / "leaves" / "0003.webp"
    assert renditions.dzi == tmp_path / "dzi" / "0003.dzi"
    with Image.open(renditions.preview) as im:
        assert max(im.size) == 1600
