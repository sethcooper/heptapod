"""
# test_generation.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Regression tests for PythiaFromRunCardTool and JetClusterSlowJetTool.

Each test pins a result that was once silently wrong: a cross section that
never reached the manifest, a tool seed the card overrode, an LHE path that
was not applied, and jet constituents reported with a neighbour's momentum.
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("pythia8mc", reason="pythia8mc not installed")
pytest.importorskip("awkward", reason="awkward not installed")
pytest.importorskip("uproot", reason="uproot not installed")

from tools import event_io
from tools.pythia.generate import PythiaFromRunCardTool
from tools.pythia.jets import JetClusterSlowJetTool

# Hard QCD above 50 GeV: jetty, and a few ms per event.
DIJET_CARD = """\
Beams:idA = 2212
Beams:idB = 2212
Beams:eCM = 13600.
HardQCD:all = on
PhaseSpace:pTHatMin = 50.
"""

# Three e+e- -> u ubar events at the Z pole, the smallest LHE Pythia accepts.
LHE_EVENT = """<event>
 4 1 1.0 91.19 0.0078 0.118
 11 -1 0 0 0 0 0. 0. 45.595 45.595 0. 0. 9.
 -11 -1 0 0 0 0 0. 0. -45.595 45.595 0. 0. 9.
 2 1 1 2 501 0 {px} {py} {pz} 45.595 0. 0. 9.
 -2 1 1 2 0 501 {mpx} {mpy} {mpz} 45.595 0. 0. 9.
</event>
"""


def _lhe(n_events: int) -> str:
    header = """<LesHouchesEvents version="1.0">
<init>
 11 -11 45.595 45.595 0 0 0 0 3 1
 1.0 0.0 1.0 1
</init>
"""
    events = []
    for k in range(n_events):
        theta = 0.4 + 0.7 * k
        px, pz = 45.595 * math.sin(theta), 45.595 * math.cos(theta)
        events.append(
            LHE_EVENT.format(px=px, py=0.0, pz=pz, mpx=-px, mpy=-0.0, mpz=-pz)
        )
    return header + "".join(events) + "</LesHouchesEvents>\n"


def _generate(tmp_path, card=DIJET_CARD, card_name="run.cmnd", **kwargs):
    (tmp_path / card_name).write_text(card)
    tool = PythiaFromRunCardTool(base_directory=str(tmp_path))
    kwargs.setdefault("data_dir", "out")
    kwargs.setdefault("n_events", 20)
    out = tool.execute(cmnd_path=card_name, **kwargs)
    # format_error returns plain text; a success is JSON.
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"status": "error", "text": out}


# ------------------------------------------------------------------------- #
# Generation
# ------------------------------------------------------------------------- #


def test_cross_section_reaches_result_and_manifest(tmp_path):
    result = _generate(tmp_path, seed=5)
    assert result["status"] == "ok", result
    assert result["xsec"]["sigmaGen_mb"] > 0
    assert result["xsec"]["sigmaErr_mb"] >= 0

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["outputs"]["xsec"] == result["xsec"]
    assert manifest["pythia_version"].startswith("8.")
    # stat() writes the cross-section table into the run's log.
    assert (
        "PYTHIA Event and Cross Section Statistics"
        in (tmp_path / "out" / "pythia.log").read_text()
    )


def test_tool_seed_overrides_the_cards_seed(tmp_path):
    card = DIJET_CARD + "Random:setSeed = on\nRandom:seed = 1\n"

    def multiplicities(seed, name):
        _generate(tmp_path, card=card, data_dir=name, n_events=5, seed=seed)
        return event_io.load_events(str(tmp_path / name / "events.root"))[
            "prt"
        ].id.layout.offsets.data.tolist()

    first = multiplicities(111, "a")
    assert multiplicities(111, "b") == first
    assert multiplicities(222, "c") != first
    assert (
        (tmp_path / "c" / "run.cmnd")
        .read_text()
        .rstrip()
        .endswith("Random:seed = 222")
    )


def test_seed_outside_pythias_range_is_rejected(tmp_path):
    result = _generate(tmp_path, seed=900_000_001)
    assert result.get("status") != "ok"
    assert "seed" in json.dumps(result)


def test_init_failure_reports_pythias_diagnostics(tmp_path):
    # No process switched on: Pythia refuses to initialise and says why.
    card = "Beams:eCM = 13600.\n"
    result = json.dumps(_generate(tmp_path, card=card))
    assert "Pythia Init Failed" in result
    assert "no process switched on" in result


def test_select_pids_keeps_only_those_species(tmp_path):
    result = _generate(tmp_path, seed=5, select_pids=[211])
    assert result["status"] == "ok", result
    ids = event_io.load_events(str(tmp_path / "out" / "events.root"))["prt"].id
    import awkward as ak

    flat = set(abs(int(i)) for i in ak.flatten(ids).tolist())
    assert flat == {211}
    assert [s["id"] for s in result["summary"]["top_species"]] in (
        [211, -211],
        [-211, 211],
    )


@pytest.mark.parametrize(
    "lhef_line", ["", "Beams:LHEF=stale.lhe\n", "beams:lhef = stale.lhe\n"]
)
def test_lhe_path_is_applied_however_the_card_spells_it(tmp_path, lhef_line):
    (tmp_path / "events.lhe").write_text(_lhe(3))
    # No frameType in the card: the tool must set it, or the LHE is ignored.
    card = lhef_line + "PartonLevel:ISR = off\n"
    result = _generate(
        tmp_path,
        card=card,
        n_events=10,
        shower_lhe=True,
        lhe_path="events.lhe",
        output_format="jsonl",
    )
    assert result["status"] == "ok", result
    # The file holds three events: all are showered, and the shortfall is
    # the input running out, not failed generation.
    assert result["accepted"] == 3
    assert result["failed"] == 0
    assert result["lhe_exhausted"] is True


def test_a_relative_include_is_found_next_to_the_card_and_inlined(tmp_path):
    (tmp_path / "cards").mkdir()
    (tmp_path / "cards" / "common").mkdir()
    (tmp_path / "cards" / "common" / "cuts.cmnd").write_text(
        "PhaseSpace:pTHatMin = 50.\n"
    )
    (tmp_path / "cards" / "beams.cmnd").write_text(
        "Beams:eCM = 13600.\ninclude common/cuts.cmnd\n"
    )
    card = "include beams.cmnd\nHardQCD:all = on\n"
    result = _generate(
        tmp_path, card=card, card_name="cards/run.cmnd", seed=5, n_events=2
    )
    assert result["status"] == "ok", result

    # run.cmnd stands alone: both levels of include are inlined, none left.
    recorded = (tmp_path / "out" / "run.cmnd").read_text()
    assert (
        "PhaseSpace:pTHatMin = 50." in recorded
        and "Beams:eCM = 13600." in recorded
    )
    assert not any(
        line.startswith("include") for line in recorded.splitlines()
    )


def test_an_include_outside_the_sandbox_is_refused(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside.cmnd"
    outside.write_text("Beams:eCM = 13600.\n")
    try:
        (tmp_path / "base").mkdir()
        result = _generate(
            tmp_path / "base", card=f"include {outside}\nHardQCD:all = on\n"
        )
        assert result.get("status") != "ok"
        assert "outside the allowed directories" in result["text"]
    finally:
        outside.unlink()


def test_a_missing_include_is_reported_before_generating(tmp_path):
    result = _generate(
        tmp_path, card="include nowhere.cmnd\nHardQCD:all = on\n"
    )
    assert (
        "Include Error" in result["text"] and "nowhere.cmnd" in result["text"]
    )
    assert not (tmp_path / "out" / "pythia.log").exists()


def test_the_validator_notes_an_include_found_next_to_the_card(tmp_path):
    from tools.pythia.validate import ValidatePythiaRunCardTool

    (tmp_path / "cuts.cmnd").write_text("PhaseSpace:pTHatMin = 50.\n")
    (tmp_path / "run.cmnd").write_text(
        DIJET_CARD.replace("PhaseSpace:pTHatMin = 50.\n", "")
        + "include cuts.cmnd\n"
    )
    report = json.loads(
        ValidatePythiaRunCardTool(base_directory=str(tmp_path)).execute(
            cmnd_path="run.cmnd"
        )
    )
    assert report["valid"] is True
    assert any(
        "inlines it into run.cmnd" in f["message"] for f in report["findings"]
    )


def test_schema_marks_only_real_inputs_required():
    assert sorted(
        PythiaFromRunCardTool.get_input_schema().get("required", [])
    ) == [
        "cmnd_path",
        "data_dir",
        "n_events",
    ]
    assert JetClusterSlowJetTool.get_input_schema().get("required", []) == []


# ------------------------------------------------------------------------- #
# Jet clustering
# ------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def dijet_sample(tmp_path_factory):
    base = tmp_path_factory.mktemp("jets")
    result = _generate(base, seed=7, n_events=10)
    assert result["status"] == "ok", result
    return base


def _jets(base, **kwargs):
    tool = JetClusterSlowJetTool(base_directory=str(base))
    out = json.loads(
        tool.execute(events_path="out/events.root", event_index=0, **kwargs)
    )
    return out["results"][0]["data"]["jets"]


def test_constituents_are_the_particles_that_make_the_jet(dijet_sample):
    particles = event_io.load_events(
        str(dijet_sample / "out" / "events.root")
    )["prt"][0]
    jets = _jets(dijet_sample)
    assert jets
    for jet in jets:
        for c in jet["constituents"]:
            stored = particles[c["event_index"]]
            assert c["px"] == pytest.approx(float(stored.px))
            assert c["id"] == int(stored.id)
        px = sum(c["px"] for c in jet["constituents"])
        py = sum(c["py"] for c in jet["constituents"])
        assert math.hypot(px, py) == pytest.approx(jet["pT"], rel=1e-4)


def test_invisible_particles_stay_out_of_jets_by_default(dijet_sample):
    neutrinos = {12, 14, 16}
    visible = [
        c["id"]
        for j in _jets(dijet_sample, ptmin=5)
        for c in j["constituents"]
    ]
    everything = [
        c["id"]
        for j in _jets(dijet_sample, ptmin=5, include_invisible=True)
        for c in j["constituents"]
    ]
    assert not neutrinos & {abs(i) for i in visible}
    assert len(everything) >= len(visible)


def test_eta_is_pseudorapidity_and_y_is_rapidity(dijet_sample):
    for jet in _jets(dijet_sample):
        p = math.sqrt(jet["px"] ** 2 + jet["py"] ** 2 + jet["pz"] ** 2)
        eta = 0.5 * math.log((p + jet["pz"]) / (p - jet["pz"]))
        y = 0.5 * math.log((jet["E"] + jet["pz"]) / (jet["E"] - jet["pz"]))
        assert jet["eta"] == pytest.approx(eta, abs=1e-6)
        assert jet["y"] == pytest.approx(y, abs=1e-6)


def run_all_tests() -> bool:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    return pytest.main([__file__, "-v" if verbose else "-q"]) == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
