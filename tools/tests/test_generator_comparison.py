"""
# test_generator_comparison.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Pythia and Sherpa on the same process, through the same tools.

The two run-card tools write one dataset layout through one loop, and the
claim that follows -- a Pythia sample and a Sherpa sample of one process
compare directly, in the dataset and in Rivet -- is what these tests check.
They need both generators importable (pythia8mc and sherpa-mc), and Rivet
for the last one.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import event_io  # noqa: E402

HAVE_PYTHIA = importlib.util.find_spec("pythia8mc") is not None
HAVE_SHERPA = importlib.util.find_spec("Sherpa") is not None
HAVE_IO = all(
    importlib.util.find_spec(m) is not None for m in ("awkward", "uproot")
)
_REAL_RIVET = os.environ.get("RIVET_PATH", "")
HAVE_REAL_RIVET = bool(_REAL_RIVET) and os.path.isdir(_REAL_RIVET)

pytestmark = pytest.mark.skipif(
    not (HAVE_PYTHIA and HAVE_SHERPA and HAVE_IO),
    reason="pythia8mc, sherpa-mc, awkward and uproot needed",
)

#: e+e- -> hadrons at the Z pole, the process both generators do with no
#: external input and no PDF.
PYTHIA_CARD = """\
Beams:idA = 11
Beams:idB = -11
Beams:eCM = 91.2
WeakSingleBoson:ffbar2gmZ = on
23:onMode = off
23:onIfAny = 1 2 3 4 5
PDF:lepton = off
PartonLevel:MPI = off
"""
SHERPA_CARD = """\
BEAMS: [11, -11]
BEAM_ENERGIES: 45.6
PDF_LIBRARY: None
PROCESSES:
- 11 -11 -> 93 93: {}
"""
N_EVENTS = 150


def _json(text):
    out = json.loads(text)
    assert isinstance(out, dict), text
    return out


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    """One sample from each generator, as the tools write them."""
    from tools.pythia.generate import PythiaFromRunCardTool
    from tools.sherpa.generate import SherpaFromRunCardTool

    base = tmp_path_factory.mktemp("compare")
    (base / "ee.cmnd").write_text(PYTHIA_CARD)
    (base / "ee.yaml").write_text(SHERPA_CARD)
    pythia = _json(
        PythiaFromRunCardTool(base_directory=str(base)).execute(
            data_dir="pythia",
            cmnd_path="ee.cmnd",
            n_events=N_EVENTS,
            seed=11,
            store="finals",
        )
    )
    sherpa = _json(
        SherpaFromRunCardTool(base_directory=str(base)).execute(
            data_dir="sherpa",
            card_path="ee.yaml",
            n_events=N_EVENTS,
            seed=11,
            store="finals",
        )
    )
    assert pythia["status"] == "ok", pythia
    assert sherpa["status"] == "ok", sherpa
    return base, pythia, sherpa


def test_the_results_have_the_same_shape(samples):
    _, pythia, sherpa = samples
    for key in (
        "events",
        "format",
        "store",
        "accepted",
        "xsec",
        "summary",
        "weighted",
        "manifest_json",
    ):
        assert key in pythia and key in sherpa, key
    for key in ("sigmaGen_mb", "sigmaErr_mb", "sigmaGen_pb", "weightSum"):
        assert key in pythia["xsec"] and key in sherpa["xsec"], key
    assert pythia["xsec"]["sigmaGen_pb"] == pytest.approx(
        pythia["xsec"]["sigmaGen_mb"] * 1e9
    )


def test_the_datasets_read_back_identically(samples):
    import awkward as ak

    base, _, _ = samples
    pythia = event_io.load_events(str(base / "pythia" / "events.root"))
    sherpa = event_io.load_events(str(base / "sherpa" / "events.root"))
    assert len(pythia) == len(sherpa) == N_EVENTS
    # The fields every dataset shares; each generator adds its own lineage.
    shared = set(event_io._KINEMATIC_FIELDS) | {"vx", "vy", "vz", "vt"}
    assert shared <= set(pythia.prt.fields)
    assert shared <= set(sherpa.prt.fields)
    for events in (pythia, sherpa):
        # The finals store: every stored particle is final-state.
        assert bool(ak.all(events.prt.status > 0))
        assert "weight" in events.fields


def test_the_physics_agrees_where_it_must(samples):
    """e+e- -> hadrons at the Z pole: the cross section and the charged
    multiplicity are the same physics in both generators, to the precision
    a small sample allows."""
    import awkward as ak
    import numpy as np

    base, pythia, sherpa = samples
    # The hadronic Z cross section at the pole is about 41 nb; both report pb.
    for result in (pythia, sherpa):
        assert 2.5e4 < result["xsec"]["sigmaGen_pb"] < 6.0e4, result["xsec"]
    # Mean charged multiplicity, from the stored ids; the measured value is
    # 20.9 and both tunes sit within a few per cent of it.
    charged = {211, 321, 2212, 11, 13}

    def mean_charged(events):
        ids = np.abs(ak.to_numpy(ak.flatten(events.prt.id)))
        counts = ak.to_numpy(ak.num(events.prt.id))
        keep = np.isin(ids, list(charged))
        per_event = np.add.reduceat(
            keep.astype(int), np.r_[0, np.cumsum(counts)[:-1]]
        )
        return float(per_event.mean())

    n_pythia = mean_charged(
        event_io.load_events(str(base / "pythia" / "events.root"))
    )
    n_sherpa = mean_charged(
        event_io.load_events(str(base / "sherpa" / "events.root"))
    )
    assert 17.0 < n_pythia < 25.0, n_pythia
    assert 17.0 < n_sherpa < 25.0, n_sherpa
    assert abs(n_pythia - n_sherpa) / n_pythia < 0.15, (n_pythia, n_sherpa)


@pytest.mark.skipif(not HAVE_REAL_RIVET, reason="RIVET_PATH needed")
def test_both_generators_meet_in_rivet(tmp_path):
    """The same analysis on both, compared side by side against the data."""
    from tools.pythia.generate import PythiaFromRunCardTool
    from tools.rivet.compare import RivetCompareTool
    from tools.sherpa.generate import SherpaFromRunCardTool

    base = tmp_path
    (base / "ee.cmnd").write_text(PYTHIA_CARD)
    (base / "ee.yaml").write_text(SHERPA_CARD)
    analysis = "ALEPH_1996_I428072"
    for tool, kwargs in (
        (PythiaFromRunCardTool, {"data_dir": "p", "cmnd_path": "ee.cmnd"}),
        (SherpaFromRunCardTool, {"data_dir": "s", "card_path": "ee.yaml"}),
    ):
        out = _json(
            tool(base_directory=str(base), rivet_path=_REAL_RIVET).execute(
                n_events=60,
                seed=5,
                store="full",
                rivet_analyses=[analysis],
                **kwargs,
            )
        )
        assert out["status"] == "ok", out
        assert out["rivet"]["analyses"][analysis]["filled"] > 0
    compared = _json(
        RivetCompareTool(
            base_directory=str(base), rivet_path=_REAL_RIVET
        ).execute(
            yoda_paths=["p/rivet.yoda", "s/rivet.yoda"],
            labels=["Pythia", "Sherpa"],
            analyses=[analysis],
            output_path="compare.json",
        )
    )
    assert compared["status"] == "ok", compared
    assert [f["label"] for f in compared["files"]] == ["Pythia", "Sherpa"]
    for entry in compared["files"]:
        assert entry["analyses"][analysis]["histograms_compared"] > 0
