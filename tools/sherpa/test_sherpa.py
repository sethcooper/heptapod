"""
# test_sherpa.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests of the Sherpa tools.

The card grammar, the validator, the record conversion and the dataset
layout are tested without Sherpa, on cards and on an event built from a
stand-in for Sherpa's blob list. The real chain -- generation through the
Python interface and through the driver, the dry run, HepMC3 with the
genealogy, Rivet -- runs when the sherpa-mc wheel is importable (and a C++
compiler, and RIVET_PATH, are available).
"""

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import event_io  # noqa: E402
from tools.sherpa import card as cardmod  # noqa: E402
from tools.sherpa import record  # noqa: E402
from tools.sherpa.validate import (  # noqa: E402
    ValidateSherpaRunCardTool,
    children_of,
    known_settings,
    lint_card,
    top_level_keys,
)

HAVE_SHERPA = importlib.util.find_spec("Sherpa") is not None
HAVE_CXX = bool(os.environ.get("CXX") or shutil.which("c++"))
HAVE_AWKWARD = importlib.util.find_spec("awkward") is not None
HAVE_PYHEPMC = importlib.util.find_spec("pyhepmc") is not None
HAVE_UPROOT = importlib.util.find_spec("uproot") is not None
_REAL_RIVET = os.environ.get("RIVET_PATH", "")
HAVE_REAL_RIVET = bool(_REAL_RIVET) and os.path.isdir(_REAL_RIVET)

LEP_CARD = """\
BEAMS: [11, -11]
BEAM_ENERGIES: 45.6
PDF_LIBRARY: None
PROCESSES:
- 11 -11 -> 93 93: {}
"""

LEP_VARIATIONS_CARD = (
    LEP_CARD
    + """\
SCALE_VARIATIONS: [[0.25, 0.25], [4.0, 4.0]]
"""
)


def _json(text):
    out = json.loads(text)
    assert isinstance(out, dict), text
    return out


# --------------------------------------------------------------------- #
# The card
# --------------------------------------------------------------------- #


def test_card_parses_to_a_mapping():
    card = cardmod.parse_card(LEP_CARD)
    assert card["BEAMS"] == [11, -11]
    assert card["PROCESSES"][0] == {"11 -11 -> 93 93": {}}
    assert cardmod.key_lines(LEP_CARD)["PROCESSES"] == 4


def test_card_syntax_errors_name_the_line():
    with pytest.raises(cardmod.CardError) as info:
        cardmod.parse_card("BEAMS: [11, -11\nBEAM_ENERGIES: 45.6\n")
    assert info.value.line is not None
    with pytest.raises(cardmod.CardError):
        cardmod.parse_card("- just\n- a list\n")


def test_process_grammar():
    incoming, outgoing = cardmod.parse_process("93 93 -> 11 -11 93{2}")
    assert incoming == [93, 93]
    assert outgoing == [(11, 0), (-11, 0), (93, 2)]
    with pytest.raises(ValueError):
        cardmod.parse_process("93 93 > 11 -11")
    with pytest.raises(ValueError):
        cardmod.parse_process("p p -> e+ e-")


def test_prepared_card_adds_only_the_ufo_lines():
    assert cardmod.prepare_card(LEP_CARD) == LEP_CARD
    prepared = cardmod.prepare_card(LEP_CARD, ufo_model="/models/MyModel")
    assert prepared.startswith(LEP_CARD)
    assert "MODEL: MyModel" in prepared
    assert "UFO_PARAM_CARD: param_MyModel.dat" in prepared


def test_beams_from_card():
    beams = record.beams_from_card({"BEAMS": 2212, "BEAM_ENERGIES": 6500})
    assert [b["id"] for b in beams] == [2212, 2212]
    assert beams[0]["pz"] > 0 > beams[1]["pz"]
    assert beams[0]["e"] == 6500 and beams[0]["m"] == pytest.approx(
        0.938, abs=1e-3
    )
    lep = record.beams_from_card(
        {"BEAMS": [11, -11], "BEAM_ENERGIES": [45.6, 45.6]},
        mass_of=lambda kf: 0.000511,
    )
    assert [b["id"] for b in lep] == [11, -11]
    with pytest.raises(ValueError):
        record.beams_from_card({"BEAMS": 2212})


# --------------------------------------------------------------------- #
# The validator
# --------------------------------------------------------------------- #


def test_snapshot_knows_the_settings_every_card_uses():
    tops = top_level_keys()
    for key in (
        "BEAMS",
        "BEAM_ENERGIES",
        "PROCESSES",
        "SELECTORS",
        "SHOWER",
        "SCALE_VARIATIONS",
        "RANDOM_SEED",
        "MODEL",
        "RIVET",
    ):
        assert key in tops, key
    assert "Order" in children_of("PROCESSES:*")
    assert "PROCESSES:*:NLO_Mode" in known_settings()
    assert "PARTICLE_DATA:*:Mass" in known_settings()


def test_a_good_card_has_no_findings():
    assert lint_card(LEP_CARD) == []
    assert lint_card(LEP_VARIATIONS_CARD) == []


def test_misspelt_keys_are_errors_with_a_suggestion():
    findings = lint_card(LEP_CARD.replace("BEAM_ENERGIES", "BEAM_ENERGIEZ"))
    errors = [f for f in findings if f["severity"] == "error"]
    assert any(
        f.get("key") == "BEAM_ENERGIEZ" and "BEAM_ENERGIES" in f["suggestion"]
        for f in errors
    )
    # The required key is reported missing too.
    assert any("BEAM_ENERGIES is not set" in f["message"] for f in errors)


def test_a_misspelt_block_entry_is_caught():
    findings = lint_card(LEP_CARD + "SHOWER:\n  FS_PT2MINN: 1.0\n")
    assert any(
        f["severity"] == "error"
        and f["key"] == "SHOWER:FS_PT2MINN"
        and "FS_PT2MIN" in f["suggestion"]
        for f in findings
    )


def test_pythia_settings_are_recognised_as_such():
    findings = lint_card(LEP_CARD + "Beams:eCM: 91.2\n")
    assert any(
        "Pythia setting" in f["message"] and f["severity"] == "error"
        for f in findings
    )


def test_shapes_of_the_core_settings():
    findings = lint_card(
        "BEAMS: proton\nBEAM_ENERGIES: -1\nEVENTS: many\n"
        "EVENT_GENERATION_MODE: Sometimes\nPROCESSES:\n- 93 93 -> 11 -11:\n"
        "    Ordre: {QCD: 0}\n"
    )
    messages = " | ".join(
        f["message"] + " " + f.get("suggestion", "") for f in findings
    )
    assert "BEAMS is one PDG id" in messages
    assert "BEAM_ENERGIES is one positive" in messages
    assert "EVENTS takes an integer" in messages
    assert "EVENT_GENERATION_MODE" in messages
    assert "Ordre" in messages and "Order" in messages


def test_selectors_and_generators_are_checked_against_sherpas_lists():
    findings = lint_card(
        LEP_CARD
        + "SELECTORS:\n- [NoSuchSelector, 11, 0, 1]\n"
        + "- [Masss, 11, -11, 60, 120]\n- [PT, 93, 20, E_CMS]\n"
        + "ME_GENERATORS: [Comix, OpenLoops]\nSHOWER_GENERATOR: Vincia\n"
    )
    errors = {f["message"]: f for f in findings if f["severity"] == "error"}
    assert any(
        "'NoSuchSelector' is not a Sherpa selector" in m for m in errors
    )
    assert any(
        "Masss" in m and "Mass" in f.get("suggestion", "")
        for m, f in errors.items()
    )
    assert not any("'PT'" in m for m in errors)
    assert any("ME_GENERATORS: 'OpenLoops'" in m for m in errors)
    assert any("SHOWER_GENERATOR: 'Vincia'" in m for m in errors)


def test_unknown_but_plausible_keys_are_warnings_not_errors():
    findings = lint_card(LEP_CARD + "MY_OWN_PLUGIN_SETTING: 1\n")
    assert [f["severity"] for f in findings] == ["warning"]


def test_validator_tool_reads_a_card_and_reports(tmp_path):
    (tmp_path / "lep.yaml").write_text(
        LEP_CARD.replace("PROCESSES", "PROCESES")
    )
    out = _json(
        ValidateSherpaRunCardTool(base_directory=str(tmp_path)).execute(
            card_path="lep.yaml"
        )
    )
    assert out["valid"] is False and out["counts"]["error"] >= 1
    assert any(f.get("key") == "PROCESES" for f in out["findings"])
    assert out["variations"] is False
    inline = _json(
        ValidateSherpaRunCardTool(base_directory=str(tmp_path)).execute(
            card_text=LEP_VARIATIONS_CARD
        )
    )
    assert inline["valid"] is True and inline["variations"] is True
    assert "Access Denied" in ValidateSherpaRunCardTool(
        base_directory=str(tmp_path)
    ).execute(card_path="../lep.yaml")


# --------------------------------------------------------------------- #
# The record: a stand-in for Sherpa's blob list
# --------------------------------------------------------------------- #


class _Vec:
    def __init__(self, e, px, py, pz):
        self.v = [e, px, py, pz]

    def __getitem__(self, k):
        return self.v[k]

    def Mass(self):
        e, px, py, pz = self.v
        return max(e * e - px * px - py * py - pz * pz, 0.0) ** 0.5


class _Flav:
    def __init__(self, kf, hadron=False):
        self.kf, self.hadron = kf, hadron

    def IsQuark(self):
        return abs(self.kf) <= 6

    def IsGluon(self):
        return self.kf == 21

    def IsDiQuark(self):
        return 1100 <= abs(self.kf) <= 5505 and not self.hadron

    def Kfcode(self):
        return abs(self.kf)

    def IsAnti(self):
        return self.kf < 0

    def IsHadron(self):
        return self.hadron


class _Particle:
    _next = 1

    def __init__(self, kf, mom, stat=1, hadron=False, info="F", x=None):
        self.flav = _Flav(kf, hadron)
        self.mom = mom
        self.stat = stat
        self.info = info
        self.x = x or _Vec(0.0, 0.0, 0.0, 0.0)
        self.prod = None
        self.dec = None
        self.this = _Particle._next
        _Particle._next += 1

    def Flav(self):
        return self.flav

    def Momentum(self):
        return self.mom

    def XProd(self):
        return self.x

    def Stat(self):
        return self.stat

    def Info(self):
        return self.info

    def HasProdBlob(self):
        return self.prod is not None

    def ProdBlob(self):
        return self.prod

    def HasDecBlob(self):
        return self.dec is not None

    def DecBlob(self):
        return self.dec


class _Blob:
    def __init__(self, ident, incoming, outgoing, data=None):
        self.ident = ident
        self.incoming = incoming
        self.outgoing = outgoing
        self.data = data or {}
        for p in incoming:
            p.dec = self
        for p in outgoing:
            p.prod = self

    def Id(self):
        return self.ident

    def NInP(self):
        return len(self.incoming)

    def NOutP(self):
        return len(self.outgoing)

    def InPart(self, j):
        return self.incoming[j]

    def OutPart(self, j):
        return self.outgoing[j]

    def __getitem__(self, name):
        return self.data[name]


class _BlobList:
    def __init__(self, blobs, weight):
        self.blobs = blobs
        self.weight = weight

    def size(self):
        return len(self.blobs)

    def __getitem__(self, k):
        return self.blobs[k]

    def GetFirst(self, _type):
        return self.blobs[1]

    def Weight(self):
        return self.weight


def _fake_event():
    """e+ e- -> q qbar, the quarks into a hadronisation blob, one hadron
    decaying to two pions: beams, a hard process, fragmentation, a decay."""
    em = _Particle(11, _Vec(45.6, 0, 0, 45.6), stat=2, info="a")
    ep = _Particle(-11, _Vec(45.6, 0, 0, -45.6), stat=2, info="a")
    q = _Particle(2, _Vec(45.6, 30.0, 0, 34.3), stat=4, info="H")
    qb = _Particle(-2, _Vec(45.6, -30.0, 0, -34.3), stat=4, info="H")
    rho = _Particle(
        113, _Vec(20.0, 10.0, 0, 15.0), stat=2, hadron=True, info="P"
    )
    pi1 = _Particle(
        211, _Vec(30.0, 20.0, 0, 20.0), stat=1, hadron=True, info="P"
    )
    pi2 = _Particle(
        -211, _Vec(41.2, -30.0, 0, -35.0), stat=1, hadron=True, info="P"
    )
    pia = _Particle(
        211,
        _Vec(10.0, 5.0, 0, 7.0),
        stat=1,
        hadron=True,
        info="D",
        x=_Vec(1.0, 0.1, 0.2, 0.3),
    )
    pib = _Particle(
        -211,
        _Vec(10.0, 5.0, 0, 8.0),
        stat=1,
        hadron=True,
        info="D",
        x=_Vec(1.0, 0.1, 0.2, 0.3),
    )
    blobs = [
        _Blob(
            10,
            [em, ep],
            [q, qb],
            {
                "Trials": 1.0,
                "MEWeight": 2.5,
                "Renormalization_Scale": 8317.44,
                "Factorization_Scale": 8317.44,
            },
        ),
        _Blob(11, [q, qb], [rho, pi1, pi2]),
        _Blob(12, [rho], [pia, pib]),
    ]
    # GetFirst(Signal_Process) is the hard-process blob.
    blobs = [blobs[1], blobs[0], blobs[2]]
    return _BlobList(blobs, 43552.7)


def test_event_rows_follow_the_blob_graph():
    ev = record.event_rows(_fake_event())
    rows = {r[0] * 1000 + r[2]: r for r in ev["p"]}  # id and info as key
    ids = [r[0] for r in ev["p"]]
    assert sorted(ids) == sorted([11, -11, 2, -2, 113, 211, -211, 211, -211])
    by_id = {}
    for r in ev["p"]:
        by_id.setdefault(r[0], []).append(r)
    beam = by_id[11][0]
    assert beam[1] == -record.BEAM and beam[3] == -1 and beam[4] == 1
    quark = by_id[2][0]
    assert quark[1] == -record.INTERMEDIATE and quark[3] == 1 and quark[4] == 0
    rho = by_id[113][0]
    assert rho[1] == -record.DECAYED and rho[3] == 0 and rho[4] == 2
    finals = [r for r in ev["p"] if r[1] == record.FINAL]
    assert len(finals) == 4
    decayed_pion = [r for r in finals if r[2] == ord("D")][0]
    assert decayed_pion[10:14] == [0.1, 0.2, 0.3, 1.0]
    assert ev["i"] == [1.0, 2.5, 8317.44, 8317.44]
    assert ev["w"] == [43552.7]
    assert rows  # the (id, info) pairs are distinct in this event


@pytest.mark.skipif(not HAVE_AWKWARD, reason="awkward needed")
def test_batch_layout_and_store_specs():
    import awkward as ak

    batch = record.batch_from_events([record.event_rows(_fake_event())] * 2)
    assert batch.prt.fields == [
        "id",
        "status",
        "info",
        "prod_vertex",
        "end_vertex",
        "p",
        "m",
        "vProd",
    ]
    assert ak.to_list(ak.num(batch.prt)) == [9, 9]
    assert ak.to_list(batch.info.weights) == [[43552.7], [43552.7]]
    assert batch.info.fields == list(record.INFO_FIELDS) + ["weights"]
    full = event_io.apply_store(batch, store="full", precision="f64")
    assert "prod_vertex" in full.prt.fields and "info" in full.prt.fields
    assert event_io.has_genealogy(full)
    finals = event_io.apply_store(batch, store="finals+kinematics")
    assert ak.to_list(ak.num(finals.prt)) == [4, 4]
    assert finals.prt.fields == list(event_io._KINEMATIC_FIELDS)
    assert ak.to_list(finals.weight) == [43552.7, 43552.7]
    assert len(record.batch_from_events([])) == 0


@pytest.mark.skipif(
    not (HAVE_AWKWARD and HAVE_PYHEPMC), reason="awkward and pyhepmc needed"
)
def test_hepmc3_from_the_vertex_graph(tmp_path):
    import pyhepmc

    batch = record.batch_from_events([record.event_rows(_fake_event())])
    beams = record.beams_from_card({"BEAMS": [11, -11], "BEAM_ENERGIES": 45.6})
    path = str(tmp_path / "ev.hepmc")
    with event_io.EventWriter(
        path, store="full", fmt="hepmc3", beams=beams
    ) as w:
        w.write(batch)
    with pyhepmc.open(path) as handle:
        event = handle.read()
    statuses = sorted(p.status for p in event.particles)
    assert statuses == [1, 1, 1, 1, 2, 4, 4, 11, 11]
    assert len(event.vertices) == 3
    decay = [v for v in event.vertices if len(v.particles_in) == 1][0]
    assert decay.particles_in[0].pid == 113
    assert [p.pid for p in decay.particles_out] == [211, -211]
    assert decay.position.x == pytest.approx(0.1)
    hard = [
        v
        for v in event.vertices
        if len(v.particles_in) == 2
        and {p.pid for p in v.particles_in} == {11, -11}
    ][0]
    assert {p.pid for p in hard.particles_out} == {2, -2}


def _cyclic_event():
    """Sherpa's actual shower structure: one shower blob takes the beam
    leptons in and gives out the copies that enter the hard process, and
    takes the hard process's quarks in and gives out their showered
    products. The hard-process vertex and the shower vertex feed each
    other."""
    em0 = _Particle(11, _Vec(45.6, 0, 0, 45.6), stat=2, info="a")
    ep0 = _Particle(-11, _Vec(45.6, 0, 0, -45.6), stat=2, info="a")
    em1 = _Particle(11, _Vec(45.6, 0, 0, 45.6), stat=2, info="I")
    ep1 = _Particle(-11, _Vec(45.6, 0, 0, -45.6), stat=2, info="I")
    em2 = _Particle(11, _Vec(45.6, 0, 0, 45.6), stat=2, info="G")
    ep2 = _Particle(-11, _Vec(45.6, 0, 0, -45.6), stat=2, info="G")
    q = _Particle(1, _Vec(45.6, 30.0, 0, 34.3), stat=2, info="H")
    qb = _Particle(-1, _Vec(45.6, -30.0, 0, -34.3), stat=2, info="H")
    q2 = _Particle(1, _Vec(40.0, 26.0, 0, 30.0), stat=1, info="F")
    g = _Particle(21, _Vec(11.2, 4.0, 0, 4.3), stat=1, info="F")
    qb2 = _Particle(-1, _Vec(40.0, -30.0, 0, -34.3), stat=1, info="F")
    beam_a = _Blob(1, [em0], [em1])
    beam_b = _Blob(2, [ep0], [ep1])
    hard = _Blob(
        3,
        [em2, ep2],
        [q, qb],
        {
            "Trials": 1.0,
            "MEWeight": 1.0,
            "Renormalization_Scale": 8317.0,
            "Factorization_Scale": 8317.0,
        },
    )
    shower = _Blob(4, [em1, ep1, q, qb], [em2, ep2, q2, g, qb2])
    return _BlobList([beam_a, hard, beam_b, shower], 43552.7)


def test_a_beam_hadron_breaking_up_is_not_a_decay():
    """The proton's copy goes into a blob of its own that gives out partons;
    Rivet reads a status-2 hadron ancestor as "from a hadron decay" and
    would call every particle of the event non-prompt."""
    p0 = _Particle(
        2212, _Vec(3500.0, 0, 0, 3500.0), stat=2, info="a", hadron=True
    )
    p1 = _Particle(
        2212, _Vec(3500.0, 0, 0, 3500.0), stat=2, info="I", hadron=True
    )
    u = _Particle(2, _Vec(700.0, 0, 0, 700.0), stat=1, info="I")
    remnant = _Particle(2101, _Vec(2800.0, 0, 0, 2800.0), stat=1, info="I")
    u.flav.IsQuark = lambda: True
    u.flav.IsGluon = lambda: False
    u.flav.IsDiQuark = lambda: False
    remnant.flav.IsQuark = lambda: False
    remnant.flav.IsGluon = lambda: False
    remnant.flav.IsDiQuark = lambda: True
    for part in (p0, p1):
        part.flav.IsQuark = lambda: False
        part.flav.IsGluon = lambda: False
        part.flav.IsDiQuark = lambda: False
    beam = _Blob(1, [p0], [p1], {"Trials": 1.0})
    bunch = _Blob(2, [p1], [u, remnant])
    rows = record.event_rows(_BlobList([beam, bunch], 1.0))["p"]
    by_info = {chr(r[2]): r for r in rows if r[0] == 2212}
    assert by_info["a"][1] == -record.BEAM
    assert by_info["I"][1] == -record.INTERMEDIATE
    # A hadron that decays to hadrons is still a decay.
    rho = _Particle(
        113, _Vec(20.0, 10.0, 0, 15.0), stat=2, hadron=True, info="P"
    )
    pi1 = _Particle(
        211, _Vec(10.0, 5.0, 0, 7.0), stat=1, hadron=True, info="D"
    )
    pi2 = _Particle(
        -211, _Vec(10.0, 5.0, 0, 8.0), stat=1, hadron=True, info="D"
    )
    for part in (rho, pi1, pi2):
        part.flav.IsQuark = lambda: False
        part.flav.IsGluon = lambda: False
        part.flav.IsDiQuark = lambda: False
    _Blob(3, [p1], [rho])
    _Blob(4, [rho], [pi1, pi2])
    rows = record.event_rows(
        _BlobList(
            [beam, _Blob(5, [p1], [rho]), _Blob(6, [rho], [pi1, pi2])], 1.0
        )
    )["p"]
    assert [r for r in rows if r[0] == 113][0][1] == -record.DECAYED


def test_the_shower_vertex_is_split_so_the_graph_is_acyclic():
    ev = record.event_rows(_cyclic_event())
    rows = ev["p"]
    # Before: the hard process (vertex 1) and the shower (vertex 3) feed
    # each other.
    assert any(r[3] == 3 and r[4] == 1 for r in rows)
    assert any(r[3] == 1 and r[4] == 3 for r in rows)
    record.untangle(rows)
    assert (
        record._find_cycle(
            rows,
            {
                v: [i for i, r in enumerate(rows) if r[3] == v]
                for v in {int(r[3]) for r in rows if r[3] >= 0}
            },
        )
        is None
    )
    # The copies that enter the hard process now come from a vertex of
    # their own, fed by the beam leptons; the showered partons still come
    # from the shower vertex, fed by the hard process's quarks.
    copies = [r for r in rows if r[2] == ord("G")]
    assert {r[3] for r in copies} == {4} and {r[4] for r in copies} == {1}
    initial = [r for r in rows if r[2] == ord("I")]
    assert {r[4] for r in initial} == {4}
    showered = [r for r in rows if r[2] == ord("F")]
    assert {r[3] for r in showered} == {3}
    hard_out = [r for r in rows if r[2] == ord("H")]
    assert {r[3] for r in hard_out} == {1} and {r[4] for r in hard_out} == {3}


@pytest.mark.skipif(
    not (HAVE_AWKWARD and HAVE_PYHEPMC), reason="awkward and pyhepmc needed"
)
def test_hepmc3_of_the_real_shower_structure(tmp_path):
    import pyhepmc

    batch = record.batch_from_events([record.event_rows(_cyclic_event())])
    beams = record.beams_from_card({"BEAMS": [11, -11], "BEAM_ENERGIES": 45.6})
    path = str(tmp_path / "ev.hepmc")
    with event_io.EventWriter(
        path, store="full", fmt="hepmc3", beams=beams
    ) as w:
        w.write(batch)
    with pyhepmc.open(path) as handle:
        event = handle.read()
    assert (
        sorted(p.status for p in event.particles) == [1, 1, 1, 4, 4] + [11] * 6
    )
    assert len(event.vertices) == 5
    hard = [
        v
        for v in event.vertices
        if {p.pid for p in v.particles_out} == {1, -1}
    ]
    assert len(hard) == 1 and {p.pid for p in hard[0].particles_in} == {
        11,
        -11,
    }


# --------------------------------------------------------------------- #
# The real chain
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sherpa_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("sherpa")


@pytest.mark.skipif(
    not (HAVE_SHERPA and HAVE_AWKWARD and HAVE_UPROOT),
    reason="sherpa-mc, awkward and uproot needed",
)
def test_generation_through_the_python_interface(sherpa_dir):
    from tools.sherpa.generate import SherpaFromRunCardTool

    base = sherpa_dir
    (base / "lep.yaml").write_text(LEP_CARD)
    tool = SherpaFromRunCardTool(base_directory=str(base))
    out = _json(
        tool.execute(
            data_dir="py",
            card_path="lep.yaml",
            n_events=30,
            seed=3,
        )
    )
    assert out["status"] == "ok", out
    assert out["accepted"] == 30 and out["backend"] == "sherpa-mc"
    assert out["xsec"]["sigmaGen_pb"] == pytest.approx(4.0e4, rel=0.25)
    assert out["xsec"]["sigmaGen_mb"] == pytest.approx(
        out["xsec"]["sigmaGen_pb"] * 1e-9
    )
    assert out["weighted"] is False
    assert out["summary"]["n_events"] == 30
    assert (base / "py" / "run.yaml").read_text() == LEP_CARD
    manifest = json.loads((base / "py" / "manifest.json").read_text())
    assert manifest["sherpa_version"].startswith("3.")
    assert manifest["beams"][0]["id"] == 11
    events = event_io.load_events(str(base / "py" / "events.root"))
    assert len(events) == 30
    # Unweighted: every event carries the same weight, a cross section in
    # pb of the order of the total; the sum is the weight times the count.
    weight = float(events.weight[0])
    assert weight == pytest.approx(out["xsec"]["sigmaGen_pb"], rel=0.25)
    assert manifest["outputs"]["xsec"]["weightSum"] == pytest.approx(
        30 * weight, rel=1e-6
    )
    # Sherpa's own output went to the log (the citation box ends every
    # run), and nothing of it reached the result.
    assert "References.tex" in (base / "py" / "sherpa.log").read_text()


@pytest.mark.skipif(
    not (HAVE_SHERPA and HAVE_AWKWARD and HAVE_PYHEPMC),
    reason="sherpa-mc, awkward and pyhepmc needed",
)
def test_full_record_reaches_hepmc3(sherpa_dir):
    import pyhepmc

    from tools.sherpa.generate import SherpaFromRunCardTool

    base = sherpa_dir
    (base / "lep.yaml").write_text(LEP_CARD)
    out = _json(
        SherpaFromRunCardTool(base_directory=str(base)).execute(
            data_dir="full",
            card_path="lep.yaml",
            n_events=5,
            seed=4,
            output_format="hepmc3",
            store="full",
        )
    )
    assert out["status"] == "ok", out
    assert out["hepmc3_record"] == "full"
    with pyhepmc.open(str(base / "full" / "events.hepmc")) as handle:
        event = handle.read()
    beams = [p for p in event.particles if p.status == 4]
    assert sorted(p.pid for p in beams) == [-11, 11]
    assert any(p.status == 2 for p in event.particles)
    assert all(p.status == 1 for p in event.particles if not p.end_vertex)
    assert len(event.vertices) > 5
    assert event.cross_section.xsec() == pytest.approx(
        out["xsec"]["sigmaGen_pb"], rel=0.25
    )
    assert event.weights[0] == pytest.approx(
        out["xsec"]["sigmaGen_pb"], rel=0.25
    )


@pytest.mark.skipif(not HAVE_SHERPA, reason="sherpa-mc needed")
def test_dry_run_catches_what_the_static_checks_cannot(sherpa_dir):
    base = sherpa_dir
    tool = ValidateSherpaRunCardTool(base_directory=str(base))
    good = _json(
        tool.execute(
            card_text=LEP_CARD,
            dry_run_init=True,
            work_dir="dry",
            timeout_s=600,
        )
    )
    assert good["valid"] is True, good
    assert good["dry_run"]["ok"] is True and good["dry_run"]["fatal"] == []
    settings = {e["setting"]: e for e in good["dry_run"]["effective_settings"]}
    assert settings["BEAMS"]["final"].replace(" ", "") == "11,-11"
    bad = _json(
        tool.execute(
            card_text=LEP_CARD + "SHOWER: {FS_PT2MIN: abc}\n",
            dry_run_init=True,
            work_dir="dry_bad",
            timeout_s=600,
        )
    )
    assert bad["valid"] is False
    assert bad["dry_run"]["ok"] is False and bad["dry_run"]["fatal"], bad[
        "dry_run"
    ]
    unread = _json(
        tool.execute(
            card_text=LEP_CARD + "MY_OWN_PLUGIN_SETTING: 1\n",
            dry_run_init=True,
            work_dir="dry_unread",
            timeout_s=600,
        )
    )
    assert any(
        "never read" in f["message"]
        and "MY_OWN_PLUGIN_SETTING" in f["message"]
        for f in unread["findings"]
    )


@pytest.mark.skipif(
    not (HAVE_SHERPA and HAVE_CXX and HAVE_AWKWARD and HAVE_UPROOT),
    reason="sherpa-mc, a C++ compiler, awkward and uproot needed",
)
def test_generation_through_the_driver_carries_the_variations(sherpa_dir):
    from tools.sherpa.generate import (
        SherpaCPPFromRunCardTool,
        SherpaFromRunCardTool,
    )

    base = sherpa_dir
    (base / "var.yaml").write_text(LEP_VARIATIONS_CARD)
    # Without sherpa_path the C++ tool builds against the wheel's libraries.
    out = _json(
        SherpaCPPFromRunCardTool(base_directory=str(base)).execute(
            data_dir="drv",
            card_path="var.yaml",
            n_events=20,
            seed=5,
            chunk_size=7,
        )
    )
    assert out["status"] == "ok", out
    assert out["backend"] == "cpp" and out["accepted"] == 20
    assert out["weight_variations"] == [
        "MUR=0.5__MUF=0.5__LHAPDF=-1",
        "MUR=2__MUF=2__LHAPDF=-1",
    ]
    events = event_io.load_events(str(base / "drv" / "events.root"))
    assert len(events.weights[0]) == 3
    assert float(events.weights[0][0]) == float(events.weight[0])
    manifest = json.loads((base / "drv" / "manifest.json").read_text())
    assert manifest["inputs"]["generator"]["backend"] == "cpp"
    assert manifest["outputs"]["weights"]["names"][0] == "Weight"
    # The Python interface on the same card warns that the variations are lost.
    plain = _json(
        SherpaFromRunCardTool(base_directory=str(base)).execute(
            data_dir="drv_py",
            card_path="var.yaml",
            n_events=3,
            seed=5,
        )
    )
    assert plain["status"] == "ok"
    assert any("variations" in w for w in plain.get("warnings", []))


DRELL_YAN_CARD = """\
BEAMS: 2212
BEAM_ENERGIES: 3500
PROCESSES:
- 93 93 -> 11 -11:
    Order: {QCD: 0, EW: 2}
SELECTORS:
- [Mass, 11, -11, 60, 120]
"""


@pytest.mark.skipif(
    not (HAVE_SHERPA and HAVE_AWKWARD and HAVE_PYHEPMC and HAVE_REAL_RIVET),
    reason="sherpa-mc, pyhepmc and RIVET_PATH needed",
)
def test_proton_collisions_give_rivet_prompt_leptons(sherpa_dir):
    """Drell-Yan at 7 TeV into the ATLAS phi* analysis and MC_ZINC. The
    beam proton breaks up into partons; were that written as a hadron
    decay, Rivet would read every lepton as non-prompt and fill nothing."""
    from tools.sherpa.generate import SherpaFromRunCardTool

    base = sherpa_dir
    (base / "dy.yaml").write_text(DRELL_YAN_CARD)
    out = _json(
        SherpaFromRunCardTool(
            base_directory=str(base), rivet_path=_REAL_RIVET
        ).execute(
            data_dir="dy",
            card_path="dy.yaml",
            n_events=60,
            seed=9,
            store="full",
            rivet_analyses=["ATLAS_2012_I1204784", "MC_ZINC:LMODE=EL"],
        )
    )
    assert out["status"] == "ok", out
    analyses = out["rivet"]["analyses"]
    assert analyses["MC_ZINC:LMODE=EL"]["filled"] == 7, analyses
    assert analyses["ATLAS_2012_I1204784"]["filled"] >= 1, analyses
    assert out["xsec"]["sigmaGen_pb"] == pytest.approx(900.0, rel=0.3)


@pytest.mark.skipif(
    not (HAVE_SHERPA and HAVE_AWKWARD and HAVE_PYHEPMC and HAVE_REAL_RIVET),
    reason="sherpa-mc, pyhepmc and RIVET_PATH needed",
)
def test_events_stream_into_rivet(sherpa_dir):
    from tools.sherpa.generate import SherpaFromRunCardTool

    base = sherpa_dir
    (base / "lep.yaml").write_text(LEP_CARD)
    out = _json(
        SherpaFromRunCardTool(
            base_directory=str(base), rivet_path=_REAL_RIVET
        ).execute(
            data_dir="rivet",
            card_path="lep.yaml",
            n_events=40,
            seed=6,
            store="full",
            rivet_analyses=["ALEPH_1996_I428072"],
        )
    )
    assert out["status"] == "ok", out
    assert out["rivet"]["n_events"] == 40.0
    assert out["rivet"]["analyses"]["ALEPH_1996_I428072"]["filled"] > 0
    assert not any(
        name.startswith("events.") for name in os.listdir(base / "rivet")
    )
    wrong = SherpaFromRunCardTool(
        base_directory=str(base), rivet_path=_REAL_RIVET
    ).execute(
        data_dir="rivet_bad",
        card_path="lep.yaml",
        n_events=5,
        store="full",
        rivet_analyses=["ATLAS_2012_I1204784"],
    )
    assert "Incompatible Analyses" in wrong
