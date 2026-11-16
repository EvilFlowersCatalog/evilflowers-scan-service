"""The cleaner: cutting a planned leaf, overrides, groups per ScanTailor run, staging inside the output directory."""

import numpy as np
import pytest
from PIL import Image

from conftest import FRAME_H, FRAME_W, RUNS, FakeScanTailor, digest, needs_scantailor, write_frame
from evilflowers_scan.geometry import BBox, PageOverride
from evilflowers_scan.imaging.cleaner import CleanError, LeafPlan, ScanTailorCleaner, cut_leaf, effective_crop
from evilflowers_scan.imaging.scantailor import ScanTailor


def plan(frame, leaf=0, side=0, crop=(0, 0, 600, 800), rotate=0, **override) -> LeafPlan:
    return LeafPlan(leaf, frame, side, BBox(*crop), rotate, PageOverride.from_dict(override))


def test_geometry_round_trips_ia_crops():
    box = BBox.from_xywh([10, 20, 100, 50])
    assert box.as_list() == [10, 20, 110, 70]
    assert box.to_xywh() == [10, 20, 100, 50]
    with pytest.raises(ValueError):
        BBox(5, 5, 5, 10)


def test_override_keys_are_the_digitalizers():
    override = PageOverride.from_dict(
        {
            "crop": [1, 2, 30, 40],
            "deskew_deg": 0.5,
            "split_x": 590,
            "despeckle": "off",
            "color_mode": "mixed",
            "ocr_language": "slk",
        }
    )
    assert override.crop == BBox(1, 2, 30, 40)
    assert (override.deskew_deg, override.split_x, override.despeckle, override.color_mode) == (0.5, 590, "off",
                                                                                                "mixed")  # fmt: skip


def test_the_override_crop_wins_and_split_x_moves_the_gutter_edge(tmp_path):
    frame = tmp_path / "f.tif"
    assert effective_crop(plan(frame, crop=(0, 0, 600, 800))) == BBox(0, 0, 600, 800)
    assert effective_crop(plan(frame, side=0, split_x=580)) == BBox(0, 0, 580, 800)
    assert effective_crop(plan(frame, side=1, crop=(600, 0, 1200, 800), split_x=580)) == BBox(580, 0, 1200, 800)
    with_crop = LeafPlan(0, frame, 0, BBox(0, 0, 600, 800), 0,
                         PageOverride.from_dict({"split_x": 580, "crop": [10, 10, 50, 50]}))  # fmt: skip
    assert effective_crop(with_crop) == BBox(10, 10, 50, 50)


def test_rotation_comes_before_the_crop():
    frame = np.zeros((100, 200, 3), np.uint8)  # 200 wide, 100 high
    cut = cut_leaf(frame, plan(None, crop=(0, 0, 100, 200), rotate=90))
    assert cut.pixels.shape[:2] == (200, 100)  # rotated to 100 x 200, then the whole of it
    assert cut.warnings == ()


def test_a_crop_past_the_frame_is_clamped_with_a_warning():
    frame = np.zeros((100, 200, 3), np.uint8)
    cut = cut_leaf(frame, plan(None, crop=(150, 0, 260, 100)))
    assert cut.crop == BBox(150, 0, 200, 100)
    assert "clamped" in cut.warnings[0]


def test_a_crop_outside_the_frame_fails():
    with pytest.raises(CleanError, match="outside"):
        cut_leaf(np.zeros((100, 200, 3), np.uint8), plan(None, crop=(300, 0, 400, 100)))


def test_leaves_are_grouped_by_their_effective_flags(tmp_path):
    frame = write_frame(tmp_path / "raw" / "0001.tif")
    out = tmp_path / "jobs" / "1-a1"
    cleaner = ScanTailorCleaner(FakeScanTailor())
    pages = cleaner.clean(
        [
            plan(frame, leaf=0, side=0, crop=(0, 0, 600, 800)),
            plan(frame, leaf=1, side=1, crop=(600, 0, 1200, 800), color_mode="color_grayscale"),
            plan(frame, leaf=2, side=0, crop=(0, 0, 600, 800), despeckle="off"),
        ],
        out,
    )
    assert sorted(RUNS) == [
        ("color_grayscale", "cautious", ["0001.tif"]),
        ("mixed", "cautious", ["0000.tif"]),
        ("mixed", "off", ["0002.tif"]),
    ]
    assert [p.leaf for p in pages] == [0, 1, 2]
    assert sorted(path.name for path in out.iterdir()) == ["0000.tif", "0001.tif", "0002.tif"]  # staging removed
    assert pages[1].width == 600 and pages[1].height == 800


def test_the_frame_is_only_read(tmp_path):
    frame = write_frame(tmp_path / "raw" / "0001.tif")
    before = digest(frame)
    ScanTailorCleaner(FakeScanTailor()).clean([plan(frame)], tmp_path / "out")
    assert digest(frame) == before


def test_scantailor_is_called_with_layout_1_and_resolved_paths(tmp_path):
    cmd = ScanTailor().command("/bin/scantailor", tmp_path / "in", tmp_path / "out")
    assert "--layout=1" in cmd
    assert cmd[-2:] == [str((tmp_path / "in").resolve()), str((tmp_path / "out").resolve())]


def test_a_missing_binary_is_a_clean_error(tmp_path):
    frame = write_frame(tmp_path / "raw" / "0001.tif")
    cleaner = ScanTailorCleaner(ScanTailor(binary="no-such-scantailor"))
    with pytest.raises(CleanError, match="not found"):
        cleaner.clean([plan(frame)], tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []  # the staging is gone


@needs_scantailor
def test_real_scantailor_cleans_both_halves_of_a_spread(tmp_path):
    frame = write_frame(tmp_path / "raw" / "0001.tif")
    cleaner = ScanTailorCleaner(ScanTailor(output_dpi=300))
    pages = cleaner.clean(
        [plan(frame, leaf=0, side=0, crop=(0, 0, FRAME_W // 2, FRAME_H)),
         plan(frame, leaf=1, side=1, crop=(FRAME_W // 2, 0, FRAME_W, FRAME_H))],
        tmp_path / "out",
    )  # fmt: skip
    assert [p.leaf for p in pages] == [0, 1]
    for page in pages:
        with Image.open(page.image) as im:
            assert im.size == (page.width, page.height)
            # Not blank: the split bug would leave one half white.
            assert np.asarray(im.convert("L")).min() < 128
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == ["0000.tif", "0001.tif"]
