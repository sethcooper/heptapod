"""
# test_convert.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for ConvertEventsTool, and for HepMC3 written straight from generation.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("pythia8mc", reason="pythia8mc not installed")
ak = pytest.importorskip("awkward", reason="awkward not installed")
pytest.importorskip("uproot", reason="uproot not installed")
pyhepmc = pytest.importorskip("pyhepmc", reason="pyhepmc not installed")

from tools import event_io
from tools.events.convert import ConvertEventsTool
from tools.pythia.generate import PythiaFromRunCardTool

CARD = (
    "Beams:eCM = 13600.\nHardQCD:all = on\nPhaseSpace:pTHatMin = 50.\n"
    "UncertaintyBands:doVariations = on\n"
    "UncertaintyBands:List = {fsr:murfac=0.5 fsr:murfac=2.0}\n"
)


def _generate(base, **kwargs):
    base.mkdir(parents=True, exist_ok=True)
    (base / "run.cmnd").write_text(CARD)
    result = json.loads(
        PythiaFromRunCardTool(base_directory=str(base)).execute(
            data_dir="run", cmnd_path="run.cmnd", n_events=5, seed=3, **kwargs
        )
    )
    assert result["status"] == "ok", result
    return result


def _convert(base, **kwargs):
    out = ConvertEventsTool(base_directory=str(base)).execute(**kwargs)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"status": "error", "text": out}


def _read_hepmc(path):
    with pyhepmc.open(str(path)) as handle:
        return list(handle)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A small ROOT sample with weight variations, and its manifest."""
    base = tmp_path_factory.mktemp("convert")
    _generate(base)
    return base


def test_root_to_hepmc_takes_beams_weights_and_cross_section_from_the_manifest(
    run,
):
    result = _convert(
        run, input_path="run/events.root", output_path="out/events.hepmc"
    )
    assert result["status"] == "ok" and result["beams_from"] == "manifest"
    manifest = json.loads((run / "run" / "manifest.json").read_text())
    events = _read_hepmc(run / "out" / "events.hepmc")

    assert len(events) == 5
    event = events[0]
    assert [(p.pid, p.status) for p in event.beams] == [(2212, 4), (2212, 4)]
    assert set(event.numpy.particles.status.tolist()) == {1, 4}
    assert event.weight_names == manifest["outputs"]["weights"]["names"]
    assert event.cross_section.xsec(0) == pytest.approx(
        manifest["outputs"]["xsec"]["sigmaGen_mb"] * 1e9
    )


def test_the_final_state_survives_the_round_trip(run):
    _convert(run, input_path="run/events.root", output_path="rt/events.hepmc")
    _convert(run, input_path="rt/events.hepmc", output_path="rt/events.jsonl")
    original = event_io.load_events(str(run / "run" / "events.root"))
    back = event_io.load_events(str(run / "rt" / "events.jsonl"))
    assert ak.to_list(back.prt.id) == ak.to_list(original.prt.id)
    assert ak.all(abs(back.prt.e - original.prt.e) < 1e-2)
    assert ak.all(
        abs(ak.flatten(back.weights) - ak.flatten(original.weights)) < 1e-12
    )


def test_a_full_record_reaches_hepmc_with_its_genealogy(tmp_path):
    _generate(tmp_path, store="full")
    result = _convert(
        tmp_path, input_path="run/events.root", output_path="events.hepmc"
    )
    assert result["status"] == "ok", result
    full = event_io.load_events(str(tmp_path / "run" / "events.root"))
    final = ak.sum(full.prt.status > 0, axis=1)
    events = _read_hepmc(tmp_path / "events.hepmc")
    assert [
        int((e.numpy.particles.status == 1).sum()) for e in events
    ] == ak.to_list(final)
    # Decayed hadrons (status 2) and the shower history are there, in vertices.
    assert all((e.numpy.particles.status == 2).any() for e in events)
    assert all(len(e.vertices) > 100 for e in events)
    # Unless the final state alone is asked for.
    result = _convert(
        tmp_path,
        input_path="run/events.root",
        output_path="flat.hepmc",
        hepmc_record="final-state",
    )
    assert result["status"] == "ok", result
    assert all(
        len(e.vertices) == 1 for e in _read_hepmc(tmp_path / "flat.hepmc")
    )


def test_beams_can_be_given_without_a_manifest(tmp_path):
    (tmp_path / "e.jsonl").write_text(
        json.dumps(
            {
                "event_id": 0,
                "data": {
                    "particles": [
                        {
                            "id": 11,
                            "status": 1,
                            "px": 60.0,
                            "py": 1.0,
                            "pz": 12.0,
                            "E": 61.2,
                            "m": 0.000511,
                        },
                        {
                            "id": 23,
                            "status": 2,
                            "px": 0.0,
                            "py": 0.0,
                            "pz": 0.0,
                            "E": 91.2,
                            "m": 91.2,
                        },
                    ]
                },
            }
        )
        + "\n"
    )
    result = _convert(
        tmp_path,
        input_path="e.jsonl",
        output_path="e.hepmc",
        beam_ids=[2212, 2212],
        beam_energies_gev=[6800.0, 6800.0],
    )
    assert result["status"] == "ok" and result["beams_from"] == "arguments"
    [event] = _read_hepmc(tmp_path / "e.hepmc")
    # The intermediate Z (status 2) is not final state and is not written.
    assert [(p.pid, p.status) for p in event.particles] == [
        (2212, 4),
        (2212, 4),
        (11, 1),
    ]


def test_hepmc_output_without_beams_is_refused(tmp_path):
    (tmp_path / "e.jsonl").write_text(
        json.dumps({"data": {"particles": []}}) + "\n"
    )
    result = _convert(tmp_path, input_path="e.jsonl", output_path="e.hepmc")
    assert "Missing Beams" in result["text"]


def test_paths_outside_the_sandbox_are_refused(tmp_path):
    result = _convert(tmp_path, input_path="../x.root", output_path="y.hepmc")
    assert "Access Denied" in result["text"]


def test_generation_writes_hepmc_directly(tmp_path):
    result = _generate(tmp_path, output_format="hepmc3")
    assert result["events"].endswith("events.hepmc")
    events = _read_hepmc(tmp_path / result["events"])
    assert len(events) == 5
    assert events[-1].cross_section.xsec(0) == pytest.approx(
        result["xsec"]["sigmaGen_mb"] * 1e9
    )
    assert len(events[0].weights) == 3


def test_generation_writes_the_full_record_as_hepmc(tmp_path):
    result = _generate(tmp_path, output_format="hepmc3", store="full")
    assert result["hepmc3_record"] == "full"
    events = _read_hepmc(tmp_path / result["events"])
    assert len(events) == 5
    assert all((e.numpy.particles.status == 2).any() for e in events)
    assert events[0].pdf_info is not None
    manifest = json.loads((tmp_path / result["manifest_json"]).read_text())
    assert manifest["inputs"]["hepmc3_record"] == "full"
    flat = _generate(tmp_path / "flat", output_format="hepmc3")
    assert flat["hepmc3_record"] == "final-state"


def run_all_tests() -> bool:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    return pytest.main([__file__, "-v" if verbose else "-q"]) == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
