"""The artefact check (#2): signals, repetition, bands, and the judge only for the uncertain band."""

from dataclasses import dataclass, field

import numpy as np
from evilflowers_worker.judges import Budget, CircuitBreaker, Panel, Thresholds, Verdict
from PIL import Image

from conftest import needs_scantailor, page_pixels, write_frame, write_page
from evilflowers_scan import artefacts
from evilflowers_scan.artefacts import LeafImage, MarkSettings, dhash, hamming
from evilflowers_scan.geometry import BBox
from evilflowers_scan.imaging.cleaner import LeafPlan, ScanTailorCleaner
from evilflowers_scan.imaging.scantailor import ScanTailor


def test_dhash_is_stable_under_scale_and_differs_between_shapes():
    ring = page_pixels(200, 200, stamp=(100, 100))[..., 2]
    small = np.asarray(Image.fromarray(ring).resize((120, 120)))
    lines = page_pixels(200, 200)[..., 2]
    assert hamming(dhash(ring), dhash(small)) <= 6
    assert hamming(dhash(ring), dhash(lines)) > 6


def test_a_blue_stamp_in_the_margin_is_a_stamp(tmp_path):
    page = write_page(tmp_path / "0003.tif", stamp=(70, 70))
    found = artefacts.leaf_candidates(LeafImage(3, page), MarkSettings())
    stamps = [c for c in found if c.category == "stamp"]
    assert len(stamps) == 1
    x0, y0, x1, y1 = stamps[0].bbox.as_list()
    assert x0 <= 70 <= x1 and y0 <= 70 <= y1
    assert stamps[0].signals["position"] == "margin"
    assert stamps[0].p > 0.6


def test_black_text_alone_proposes_no_stamp(tmp_path):
    page = write_page(tmp_path / "0003.tif")
    assert not [c for c in artefacts.leaf_candidates(LeafImage(3, page), MarkSettings()) if c.category == "stamp"]


def test_a_dark_border_is_an_edge(tmp_path):
    pixels = page_pixels(800, 1000)
    pixels[:, :40] = 15  # the scanner bed along the left edge
    path = tmp_path / "0001.tif"
    Image.fromarray(pixels).save(path)
    edges = [c for c in artefacts.leaf_candidates(LeafImage(1, path), MarkSettings()) if c.category == "edge"]
    assert len(edges) == 1
    assert edges[0].bbox.x0 == 0 and 30 <= edges[0].bbox.x1 <= 50
    assert edges[0].p >= 0.85  # sure: no judge needed


def test_the_same_stamp_on_many_leaves_is_furniture(tmp_path):
    leaves = [LeafImage(n, write_page(tmp_path / f"{n:04d}.tif", stamp=(70, 70))) for n in range(4)]
    alone = artefacts.marks(leaves[:1], MarkSettings(sure=0.99, floor=0.1))
    many = artefacts.marks(leaves, MarkSettings(sure=0.99, floor=0.1))
    p_alone = next(m.p for m in alone[0] if m.category == "stamp")
    stamp = next(m for m in many[0] if m.category == "stamp")
    assert stamp.signals["repeats"] == 4
    assert stamp.p > p_alone


def test_without_a_judge_an_uncertain_mark_is_review(tmp_path):
    page = write_page(tmp_path / "0003.tif", stamp=(70, 70))
    result = artefacts.marks([LeafImage(3, page)], MarkSettings(sure=0.99, floor=0.1))
    stamp = next(m for m in result[3] if m.category == "stamp")
    assert (stamp.band, stamp.model) == ("review", "signals")


def test_a_sure_mark_is_banded_without_a_judge(tmp_path):
    page = write_page(tmp_path / "0003.tif", stamp=(70, 70))
    judge = RecordingJudge()
    result = artefacts.marks([LeafImage(3, page)], MarkSettings(sure=0.5), panel=panel(), judges=Judges(judge))
    stamp = next(m for m in result[3] if m.category == "stamp")
    assert (stamp.band, stamp.model) == ("hide", "signals")
    assert judge.asked == []


# -- With a judge -------------------------------------------------------------------------------------------------


@dataclass
class RecordingJudge:
    """A decider-vision stand-in: answers ``answer`` with ``p`` and records the questions."""

    answer: str = "content"
    p: float = 0.95
    name: str = "decider-vision"
    timeout: float = 5.0
    breaker: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(3, 60))
    asked: list = field(default_factory=list)

    def ask(self, question, *, state="", images=(), timeout=None):
        self.asked.append((question.id, len(images)))
        return Verdict(question=question.id, verdict=self.answer, p=self.p, model="strands-decider-2b",
                       revision="r1", reason="printed ornament")  # fmt: skip


@dataclass
class Judges:
    decider_vision: RecordingJudge


def panel() -> Panel:
    return Panel(Budget(10, 0), {"scan_mark": Thresholds(keep=0.8, hide=0.9)})


def test_an_uncertain_mark_is_asked_and_the_verdict_decides(tmp_path):
    page = write_page(tmp_path / "0003.tif", stamp=(70, 70))
    judge = RecordingJudge(answer="content", p=0.95)
    result = artefacts.marks([LeafImage(3, page)], MarkSettings(sure=0.99, floor=0.1), panel=panel(),
                             judges=Judges(judge))  # fmt: skip
    mark = next(m for m in result[3] if m.signals.get("saturation"))
    assert ("scan_mark", 1) in judge.asked
    assert (mark.category, mark.band, mark.model) == ("content", "keep", "strands-decider-2b@r1")
    assert mark.reason == "printed ornament"


def test_a_judged_artefact_below_the_hide_threshold_is_review(tmp_path):
    page = write_page(tmp_path / "0003.tif", stamp=(70, 70))
    judge = RecordingJudge(answer="stamp", p=0.7)
    result = artefacts.marks([LeafImage(3, page)], MarkSettings(sure=0.99, floor=0.1), panel=panel(),
                             judges=Judges(judge))  # fmt: skip
    mark = next(m for m in result[3] if m.category == "stamp")
    assert (mark.band, mark.p) == ("review", 0.7)


def test_a_spent_budget_leaves_the_mark_unjudged(tmp_path):
    page = write_page(tmp_path / "0003.tif", stamp=(70, 70))
    judge = RecordingJudge()
    spent = Panel(Budget(1, 0, used_calls=1))
    result = artefacts.marks([LeafImage(3, page)], MarkSettings(sure=0.99, floor=0.1), panel=spent,
                             judges=Judges(judge))  # fmt: skip
    assert judge.asked == []
    assert {m.band for m in result[3]} == {"review"}


def test_an_unreadable_page_gets_no_marks(tmp_path):
    bad = tmp_path / "0003.tif"
    bad.write_bytes(b"not a tiff")
    assert artefacts.marks([LeafImage(3, bad)], MarkSettings()) == {3: []}


# -- Without colour (ScanTailor's default mixed mode binarizes a stamp) -------------------------------------------


def test_a_binarized_stamp_in_the_margin_is_found_without_colour(tmp_path):
    page = write_page(tmp_path / "0003.tif", stamp=(70, 70), mode="L")
    (mark,) = [m for m in artefacts.marks([LeafImage(3, page)], MarkSettings())[3] if m.category == "stamp"]
    assert mark.signals["colour"] is False
    assert mark.signals["position"] == "margin"
    assert (mark.band, mark.model) == ("review", "signals")


def test_signals_alone_never_hide_a_mark_found_without_colour(tmp_path):
    leaves = [LeafImage(n, write_page(tmp_path / f"{n:04d}.tif", stamp=(70, 70), mode="L")) for n in range(4)]
    result = artefacts.marks(leaves, MarkSettings(sure=0.5))
    marks = [m for leaf in result.values() for m in leaf if m.signals.get("colour") is False]
    assert marks and all(m.band == "review" and m.p <= artefacts.COLOURLESS_CAP for m in marks)
    assert all(m.signals["repeats"] == 4 for m in marks)


def test_body_text_is_not_a_mark(tmp_path):
    page = write_page(tmp_path / "0003.tif", mode="L")
    assert artefacts.marks([LeafImage(3, page)], MarkSettings()) == {3: []}


def test_a_dark_full_bleed_page_has_no_scan_edge(tmp_path):
    path = tmp_path / "0001.tif"
    Image.new("L", (800, 1000), 30).save(path)  # a dark plate: no paper after any "band"
    found = artefacts.leaf_candidates(LeafImage(1, path), MarkSettings())
    assert not [c for c in found if c.category == "edge"]


@needs_scantailor
def test_real_mixed_mode_output_still_yields_the_stamp(tmp_path):
    frame = write_frame(tmp_path / "frame.tif", stamp=(70, 70))
    cleaner = ScanTailorCleaner(ScanTailor(color_mode="mixed", output_dpi=600))
    (page,) = cleaner.clean([LeafPlan(0, frame, 0, BBox(0, 0, 600, 800))], tmp_path / "out")
    result = artefacts.marks([LeafImage(0, page.image, 0)], MarkSettings())
    stamps = [m for m in result[0] if m.category == "stamp"]
    assert stamps, result
    assert stamps[0].band == "review" and stamps[0].signals["position"] in ("margin", "gutter")
