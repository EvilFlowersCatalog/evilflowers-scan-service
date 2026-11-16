"""The jobs in-process against the orchestrator's contract (#1 comments, digitalizer ``studio/testing/standalone.py``).

``OrchestratorCleanOutcome`` mirrors evilflowers-books-digitalizer ``studio.orchestration.contract.CleanOutcome`` (the
model it reads the file with): an outcome that fails it would be refused.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from conftest import RUNS, FakeScanTailor, digest, write_frame, write_page
from evilflowers_scan import jobs
from evilflowers_scan.contract import CleanJob, EraseJob
from evilflowers_scan.imaging import ImageOps, ScanTailorCleaner
from evilflowers_scan.imaging.scantailor import ScanTailor


class _Leaf(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    leaf: int
    image: str
    width: int
    height: int
    dpi: int
    skew_deg: float = 0.0
    preview: str | None = None
    dzi: str


class OrchestratorCleanOutcome(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    run_id: str
    token: str
    status: Literal["done", "skipped", "failed"]
    error: str | None = None
    leaves: list[_Leaf] = []


OUT = "digitized/book/book_studio/jobs/17-a1"
FRAME = "raw-scans/fei/book/0002.tif"


@pytest.fixture
def services(settings) -> jobs.Services:
    return jobs.Services(settings, ScanTailorCleaner(FakeScanTailor(output_dpi=600)), ImageOps(), None)


@pytest.fixture
def frame(settings) -> Path:
    return write_frame(settings.storage_path / FRAME, stamp=(60, 60))


def clean_message(**overrides) -> CleanJob:
    message = {
        "run_id": "42",
        "job": "clean-0003-a1",
        "token": "secret-token",
        "inputs": [FRAME],
        "output_dir": OUT,
        "outcome": f"{OUT}/outcome.json",
        "leaves": [
            {
                "leaf": 3,
                "frame": FRAME,
                "side": 0,
                "crop_box": [0, 0, 600, 800],
                "rotate_degree": 0,
                "hand_side": "LEFT",
                "override": {},
            },
            {
                "leaf": 4,
                "frame": FRAME,
                "side": 1,
                "crop_box": [600, 0, 600, 800],
                "rotate_degree": 0,
                "hand_side": "RIGHT",
                "override": {"deskew_deg": 0.4},
            },
        ],
        **overrides,
    }
    return CleanJob.model_validate(message)


def outcome(settings, relative=f"{OUT}/outcome.json") -> dict:
    return json.loads((settings.storage_path / relative).read_text())


def test_clean_writes_leaves_previews_dzi_and_the_outcome(settings, services, frame):
    before = digest(frame)
    summary = jobs.clean(services, clean_message())

    assert summary == {"job": "clean-0003-a1", "leaves": 2, "marks": summary["marks"]}
    data = outcome(settings)
    parsed = OrchestratorCleanOutcome.model_validate(data)
    assert (parsed.run_id, parsed.token, parsed.status, parsed.error) == ("42", "secret-token", "done", None)
    assert [leaf.leaf for leaf in parsed.leaves] == [3, 4]
    for leaf in parsed.leaves:
        for path in (leaf.image, leaf.preview, leaf.dzi):
            assert path is not None and path.startswith(f"{OUT}/")  # relative to STORAGE_PATH, in the job's dir
            assert (settings.storage_path / path).is_file()
        assert leaf.dpi == 600
        assert (leaf.width, leaf.height) == (600, 800)
    assert data["leaves"][0]["crop_box"] == [0, 0, 600, 800]
    assert data["leaves"][1]["skew_deg"] == 0.4
    assert digest(frame) == before  # raw-scans read only


def test_nothing_is_written_outside_the_output_directory(settings, services, frame):
    before = {p for p in settings.storage_path.rglob("*")}
    jobs.clean(services, clean_message())
    new = {p for p in settings.storage_path.rglob("*")} - before
    out = (settings.storage_path / OUT).resolve()
    assert new and all(p.resolve().is_relative_to(out) or out.is_relative_to(p.resolve()) for p in new)
    assert not [p for p in (settings.storage_path / OUT).iterdir() if p.name.startswith(".st-")]


def test_the_outcome_is_written_last(settings, services, frame, monkeypatch):
    seen = []
    real = jobs.write_outcome

    def spy(s, relative, model):
        files = [p for p in (settings.storage_path / OUT).rglob("*") if p.is_file()]
        seen.append(sorted(p.name for p in files))
        return real(s, relative, model)

    monkeypatch.setattr(jobs, "write_outcome", spy)
    jobs.clean(services, clean_message())
    assert {"0003.tif", "0004.tif", "0003.webp", "0004.webp", "0003.dzi", "0004.dzi"} <= set(seen[0])
    assert "outcome.json" not in seen[0]
    assert not list((settings.storage_path / OUT).glob("*.partial"))


class _Broken(FakeScanTailor):
    def run(self, inputs, out_dir):
        raise RuntimeError("scantailor exited 3")


def test_a_failure_writes_a_failed_outcome_then_raises(settings, frame):
    services = jobs.Services(settings, ScanTailorCleaner(_Broken()), ImageOps(), None)
    with pytest.raises(RuntimeError, match="scantailor exited 3"):
        jobs.clean(services, clean_message())
    data = outcome(settings)
    assert (data["status"], data["run_id"], data["token"]) == ("failed", "42", "secret-token")
    assert "scantailor exited 3" in data["error"]
    OrchestratorCleanOutcome.model_validate(data)


def test_a_missing_binary_fails_the_job_with_its_error(settings, frame):
    services = jobs.Services(settings, ScanTailorCleaner(ScanTailor(binary="no-such-binary")), ImageOps(), None)
    with pytest.raises(Exception, match="not found"):
        jobs.clean(services, clean_message())
    assert outcome(settings)["status"] == "failed"


def test_a_frame_that_is_not_an_input_is_refused(settings, services, frame):
    message = clean_message(inputs=["raw-scans/other.tif"])
    with pytest.raises(ValueError, match="not in the job's inputs"):
        jobs.clean(services, message)
    assert outcome(settings)["status"] == "failed"


def test_a_frame_outside_the_storage_is_refused(settings, services, frame):
    message = clean_message(
        inputs=["../etc/passwd"],
        leaves=[{"leaf": 0, "frame": "../etc/passwd", "crop_box": [0, 0, 10, 10]}],
    )
    with pytest.raises(ValueError, match="escapes"):
        jobs.clean(services, message)


@pytest.mark.parametrize(
    "change",
    [
        {"leaves": []},
        {"leaves": [{"leaf": 1, "frame": FRAME, "crop_box": [0, 0, 0, 10]}]},
        {"leaves": [{"leaf": 1, "frame": FRAME, "crop_box": [0, 0, 5, 5], "rotate_degree": 45}]},
        {"leaves": [{"leaf": 1, "frame": FRAME, "crop_box": [0, 0, 5, 5]}] * 2},
        {"token": ""},
    ],
)
def test_an_invalid_message_does_not_validate(change):
    with pytest.raises(ValidationError):
        clean_message(**change)


def test_erase_hidden_paints_hide_marks_white(settings, services, frame):
    # Colour kept (a mixed-mode page would be binarized, and the stamp's colour with it); a low "sure" bar.
    settings_on = settings.model_copy(update={"artefact_sure": 0.6})
    cleaner = ScanTailorCleaner(FakeScanTailor(color_mode="color_grayscale", output_dpi=600))
    services = jobs.Services(settings_on, cleaner, services.ops, None)
    jobs.clean(services, clean_message(erase_hidden=True))
    data = outcome(settings)
    hidden = [m for m in data["leaves"][0]["marks"] if m["band"] == "hide" and m["category"] == "stamp"]
    assert hidden, data["leaves"][0]["marks"]
    from PIL import Image

    with Image.open(settings.storage_path / data["leaves"][0]["image"]) as im:
        x0, y0, x1, y1 = hidden[0]["bbox"]
        assert np.asarray(im.convert("L"))[y0:y1, x0:x1].min() == 255


# -- erase --------------------------------------------------------------------------------------------------------

PAGE = "digitized/book/book_studio/jobs/9-a1/0003.tif"
ERASE_OUT = "digitized/book/book_studio/jobs/18-a1"


def erase_message(**overrides) -> EraseJob:
    return EraseJob.model_validate(
        {"run_id": "42", "job": "analyze-erase-0003-a1", "token": "t", "inputs": [PAGE], "output_dir": ERASE_OUT,
         "outcome": f"{ERASE_OUT}/outcome.json",
         "leaves": [{"leaf": 3, "image": PAGE, "dpi": 600, "boxes": [[10, 210, 50, 250]]}], **overrides}
    )  # fmt: skip


def test_erase_paints_the_boxes_on_a_copy_and_keeps_the_size(settings, services):
    page = write_page(settings.storage_path / PAGE, mode="L")
    before = digest(page)
    assert jobs.erase(services, erase_message()) == {"job": "analyze-erase-0003-a1", "leaves": 1}

    data = outcome(settings, f"{ERASE_OUT}/outcome.json")
    parsed = OrchestratorCleanOutcome.model_validate(data)
    (leaf,) = parsed.leaves
    assert (leaf.leaf, leaf.width, leaf.height, leaf.dpi) == (3, 800, 1000, 600)
    assert leaf.image == f"{ERASE_OUT}/0003.tif"
    assert leaf.preview == f"{ERASE_OUT}/leaves/0003.webp" and leaf.dzi == f"{ERASE_OUT}/dzi/0003.dzi"
    from PIL import Image

    with Image.open(settings.storage_path / leaf.image) as im:
        pixels = np.asarray(im)
    assert pixels[210:250, 10:50].min() == 255
    assert digest(page) == before  # the earlier attempt's page is untouched


def test_erase_of_a_missing_page_fails_with_an_outcome(settings, services):
    with pytest.raises(FileNotFoundError):
        jobs.erase(services, erase_message())
    assert outcome(settings, f"{ERASE_OUT}/outcome.json")["status"] == "failed"


def test_give_up_writes_a_failed_outcome_and_raises(settings, services, frame):
    with pytest.raises(RuntimeError, match="gave up after 5 deliveries"):
        jobs.give_up(services, clean_message(), 6, "clean")
    assert outcome(settings)["status"] == "failed"


def test_a_failing_artefact_check_is_a_warning_not_a_failed_job(settings, services, frame, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("a huge page")

    monkeypatch.setattr(jobs.artefacts, "marks", broken)
    jobs.clean(services, clean_message())
    data = outcome(settings)
    assert data["status"] == "done"
    assert all("artefact check failed: RuntimeError: a huge page" in leaf["warnings"] for leaf in data["leaves"])


def test_a_redelivery_sweeps_the_staging_of_a_killed_delivery(settings, services, frame):
    stale = settings.storage_path / OUT / ".st-killed" / "in"
    stale.mkdir(parents=True)
    (stale / "0003.tif").write_bytes(b"half a leaf")
    jobs.clean(services, clean_message())
    assert not (settings.storage_path / OUT / ".st-killed").exists()


# -- The project's ScanTailor settings (#3) ------------------------------------------------------------------------


@pytest.fixture
def env_services(settings) -> jobs.Services:
    """ScanTailor as the env builds it: dpi 300, output_dpi = dpi, mixed / cautious."""
    return jobs.Services(settings, ScanTailorCleaner(FakeScanTailor(dpi=300, output_dpi=300)), ImageOps(), None)


def test_without_project_settings_the_env_ones_apply(settings, env_services, frame):
    jobs.clean(env_services, clean_message())
    assert {leaf["dpi"] for leaf in outcome(settings)["leaves"]} == {300}
    assert {(mode, despeckle) for mode, despeckle, _ in RUNS} == {("mixed", "cautious")}


def test_the_project_settings_apply_to_the_whole_job(settings, env_services, frame):
    project = {"color_mode": "color_grayscale", "despeckle": "off", "margins_mm": 5.0, "output_dpi": 600}
    jobs.clean(env_services, clean_message(scantailor=project))
    assert {leaf["dpi"] for leaf in outcome(settings)["leaves"]} == {600}
    assert {(mode, despeckle) for mode, despeckle, _ in RUNS} == {("color_grayscale", "off")}


def test_a_leaf_override_still_wins_over_the_project(settings, env_services, frame):
    message = clean_message(scantailor={"color_mode": "color_grayscale"})
    leaves = [leaf.model_dump() for leaf in message.leaves]
    leaves[1]["override"] = {"color_mode": "black_and_white"}
    jobs.clean(env_services, clean_message(scantailor={"color_mode": "color_grayscale"}, leaves=leaves))
    assert sorted((mode, names) for mode, _, names in RUNS) == [
        ("black_and_white", ["0004.tif"]),
        ("color_grayscale", ["0003.tif"]),
    ]


def test_a_project_dpi_alone_also_sets_the_output_dpi(settings, env_services, frame):
    jobs.clean(env_services, clean_message(scantailor={"dpi": 400}))
    assert {leaf["dpi"] for leaf in outcome(settings)["leaves"]} == {400}


def test_the_message_never_picks_the_binary_and_ignores_unknown_keys():
    options = clean_message(scantailor={"binary": "/bin/sh", "split": True, "output_dpi": 600}).scantailor
    assert options is not None
    assert options.changes(None) == {"output_dpi": 600}


@pytest.mark.parametrize("bad", [{"color_mode": "sepia"}, {"despeckle": "max"}, {"output_dpi": 10}])
def test_invalid_project_settings_do_not_validate(bad):
    with pytest.raises(ValidationError):
        clean_message(scantailor=bad)


# -- A judge misconfiguration never fails the clean (#3) -----------------------------------------------------------


class _BrokenJudges:
    def panel(self):
        raise ValueError("DECIDER_VISION_URL is not a URL")


def test_a_failing_judge_panel_is_a_warning_not_a_failed_job(settings, services, frame):
    broken = jobs.Services(settings, services.cleaner, services.ops, _BrokenJudges())  # type: ignore[arg-type]
    jobs.clean(broken, clean_message())
    data = outcome(settings)
    assert data["status"] == "done"
    expected = "artefact check failed: ValueError: DECIDER_VISION_URL is not a URL"
    assert all(expected in leaf["warnings"] for leaf in data["leaves"])
