"""
# test_event_io.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for the event dataset input/output layer.

The assertions that matter here are about layout stability: a caller must see
the same array shape whichever format a dataset came from, or every downstream
tool ends up branching on the container. The rest cover the store specs, which
are the largest lever on dataset size, and chunked writing, which is what makes
a sample larger than memory possible at all.
"""

import json
import sys
from pathlib import Path

import pytest

# Repository root on the path, so this file works both under pytest and when
# test_runner.py executes it directly. This file deliberately does not live in
# `tools/` itself: running a script from there puts `tools/` on sys.path[0],
# where the `tools/logging` package shadows the standard library's `logging`
# and breaks pytest's own imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import event_io as eio

ak = pytest.importorskip("awkward", reason="awkward not installed")
uproot = pytest.importorskip("uproot", reason="uproot not installed")
pythia8mc = pytest.importorskip("pythia8mc", reason="pythia8mc not installed")


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def batches():
    """Three batches of real Pythia events, with fragmentation variations on.

    Generated once for the module: initialising Pythia is far more expensive
    than any assertion here.
    """
    pythia = pythia8mc.Pythia("", printBanner=False)
    for command in (
        "Print:quiet = on",
        "Beams:eCM = 13000.",
        "HardQCD:all = on",
        "PhaseSpace:pTHatMin = 20.",
        "Random:setSeed = on",
        "Random:seed = 11",
        "VariationFrag:List = {var0 frag:aLund=0.6}",
    ):
        pythia.readString(command)
    assert pythia.init()
    return [pythia.nextBatch(25) for _ in range(3)]


# --------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path,expected",
    [
        ("a.root", "root"),
        ("dir/b.jsonl", "jsonl"),
        ("c.json", "jsonl"),
        ("d.hepmc", "hepmc3"),
        ("e.hepmc3", "hepmc3"),
    ],
)
def test_format_is_inferred_from_the_suffix(path, expected):
    assert eio.detect_format(path) == expected


def test_an_unknown_suffix_is_rejected():
    with pytest.raises(ValueError, match="cannot infer event format"):
        eio.detect_format("events.lhe")


# --------------------------------------------------------------------- #
# Store specs -- the largest lever on dataset size
# --------------------------------------------------------------------- #


def test_kinematics_spec_keeps_only_the_kinematic_fields(batches):
    reduced = eio.apply_store(batches[0], store="finals+kinematics")
    assert set(reduced.prt.fields) == {
        "id",
        "status",
        "px",
        "py",
        "pz",
        "e",
        "m",
    }


def test_mass_is_stored_rather_than_derived(batches):
    """At float32, m^2 = E^2 - |p|^2 is a catastrophic cancellation.

    For a 100 GeV pion E^2 is order 1e4, where the float32 spacing is order
    1e-3, while m^2 is 0.0195 -- several percent error on a mass that is known
    exactly at generation time. So `m` must survive the default store spec.
    """
    reduced = eio.apply_store(
        batches[0], store="finals+kinematics", precision="f32"
    )
    assert "m" in reduced.prt.fields
    masses = ak.to_numpy(ak.flatten(reduced.prt.m))
    assert (masses >= 0).all()


def test_finals_specs_drop_the_shower_history(batches):
    full = eio.apply_store(batches[0], store="full")
    finals = eio.apply_store(batches[0], store="finals")
    assert float(ak.mean(ak.num(finals.prt))) < float(
        ak.mean(ak.num(full.prt))
    )
    # Everything kept really is final state.
    assert bool(ak.all(finals.prt.status > 0))


def test_finals_spec_keeps_vertices_and_lifetimes(batches):
    """Displaced-vertex work needs vProd and tau, so `finals` must keep
    them."""
    finals = eio.apply_store(batches[0], store="finals")
    assert {"vx", "vy", "vz", "vt", "tau"} <= set(finals.prt.fields)


def test_precision_is_honoured(batches):
    import numpy as np

    f32 = eio.apply_store(
        batches[0], store="finals+kinematics", precision="f32"
    )
    f64 = eio.apply_store(
        batches[0], store="finals+kinematics", precision="f64"
    )
    assert ak.to_numpy(ak.flatten(f32.prt.px)).dtype == np.float32
    assert ak.to_numpy(ak.flatten(f64.prt.px)).dtype == np.float64


def test_the_nominal_weight_survives_every_store_spec(batches):
    """A weighted sample analysed as unweighted is wrong in shape.

    So the weight is never something a store spec can drop.
    """
    for store in eio.STORE_SPECS:
        reduced = eio.apply_store(batches[0], store=store)
        assert "weight" in reduced.fields, store


def test_hard_process_info_is_dropped_only_by_the_kinematics_spec(batches):
    kinematics = eio.apply_store(batches[0], store="finals+kinematics")
    finals = eio.apply_store(batches[0], store="finals")
    assert not [f for f in kinematics.fields if f.startswith("info_")]
    assert "info_x1" in finals.fields and "info_pT2Hat" in finals.fields


def test_an_unknown_store_spec_is_rejected(batches):
    with pytest.raises(ValueError, match="unknown store spec"):
        eio.apply_store(batches[0], store="nonsense")


def test_an_unknown_precision_is_rejected(batches):
    with pytest.raises(ValueError, match="unknown precision"):
        eio.apply_store(batches[0], store="full", precision="f16")


# --------------------------------------------------------------------- #
# Round-trip: the layout must not depend on the container
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", [".root", ".jsonl"])
def test_round_trip_preserves_the_layout(tmp_path, batches, suffix):
    path = str(tmp_path / f"events{suffix}")
    stats = eio.write_events(batches[0], path)
    back = eio.load_events(path)

    assert stats["n_events"] == len(batches[0]) == len(back)
    assert "prt" in back.fields and "weight" in back.fields
    assert set(back.prt.fields) == {"id", "status", "px", "py", "pz", "e", "m"}


def test_both_formats_agree_on_content(tmp_path, batches):
    """The container must not change the physics.

    Written through the same store spec, ROOT and JSONL must return the same
    multiplicities and the same particle ids.
    """
    root_path = str(tmp_path / "e.root")
    jsonl_path = str(tmp_path / "e.jsonl")
    eio.write_events(batches[0], root_path)
    eio.write_events(batches[0], jsonl_path)

    from_root = eio.load_events(root_path)
    from_jsonl = eio.load_events(jsonl_path)

    assert ak.to_list(ak.num(from_root.prt)) == ak.to_list(
        ak.num(from_jsonl.prt)
    )
    assert ak.to_list(from_root.prt.id[0]) == ak.to_list(from_jsonl.prt.id[0])


def test_root_counter_branches_are_not_surfaced(tmp_path, batches):
    """uproot writes an n<branch> counter per jagged branch.

    The length is implicit in the awkward layout, so the counters must not leak
    into the fields a caller sees.
    """
    path = str(tmp_path / "e.root")
    eio.write_events(batches[0], path, store="finals")
    back = eio.load_events(path)
    assert not [
        f for f in back.fields if f.startswith("n") and f[1:] in back.fields
    ]
    assert "nprt" not in back.fields


def test_root_output_is_a_ttree_not_an_rntuple(tmp_path, batches):
    """RNTuple is ~6% smaller but uproot cannot extend one.

    A format that cannot be appended to forces the whole sample into memory
    before writing, which is exactly what chunked generation avoids.
    """
    path = str(tmp_path / "e.root")
    eio.write_events(batches[0], path)
    assert list(uproot.open(path).classnames().values()) == ["TTree"]


def test_column_projection_reads_only_what_was_asked_for(tmp_path, batches):
    path = str(tmp_path / "e.root")
    eio.write_events(batches[0], path)
    projected = eio.load_events(path, columns=["prt_id"])
    assert projected.prt.fields == ["id"]


def test_jsonl_keeps_the_legacy_schema(tmp_path, batches):
    """Existing readers check for schema, event_id and data.particles."""
    path = str(tmp_path / "e.jsonl")
    eio.write_events(batches[0], path)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]

    assert rows[0]["schema"] == eio.JSONL_SCHEMA_VERSION
    assert [r["event_id"] for r in rows] == list(range(len(rows)))
    particle = rows[0]["data"]["particles"][0]
    # The legacy schema spells energy "E" and carries a per-particle index.
    assert "E" in particle and "i" in particle
    assert rows[0]["data"]["n"] == len(rows[0]["data"]["particles"])


@pytest.mark.parametrize("suffix", [".root", ".jsonl"])
def test_a_range_reads_exactly_those_events(tmp_path, batches, suffix):
    path = str(tmp_path / f"e{suffix}")
    with eio.EventWriter(path) as writer:
        for batch in batches:
            writer.write(batch)
    everything = eio.load_events(path)
    assert eio.count_events(path) == len(everything)

    middle = eio.read_events(path, 3, 7)
    assert ak.to_list(middle.prt.id) == ak.to_list(everything.prt.id[3:7])
    # A range running past the end is clipped, and one wholly past it is empty.
    assert (
        len(eio.read_events(path, len(everything) - 1, len(everything) + 5))
        == 1
    )
    assert (
        len(eio.read_events(path, len(everything) + 2, len(everything) + 3))
        == 0
    )


@pytest.mark.parametrize("suffix", [".root", ".jsonl", ".hepmc"])
def test_streaming_visits_every_event_in_order(tmp_path, batches, suffix):
    path = str(tmp_path / f"e{suffix}")
    if suffix == ".hepmc":
        _write_hepmc(tmp_path / "e.hepmc", batches)
        streamed = [
            ids
            for chunk in eio.iter_events(path, step_size=4)
            for ids in ak.to_list(chunk.prt.id)
        ]
        assert streamed == [
            ids
            for b in batches
            for ids in ak.to_list(eio.apply_store(b).prt.id)
        ]
        return
    with eio.EventWriter(path) as writer:
        for batch in batches:
            writer.write(batch)
    streamed = [
        ids
        for chunk in eio.iter_events(path, step_size=4)
        for ids in ak.to_list(chunk.prt.id)
    ]
    assert streamed == ak.to_list(eio.load_events(path).prt.id)


BEAMS = [
    {
        "id": 2212,
        "px": 0.0,
        "py": 0.0,
        "pz": 6799.9999,
        "e": 6800.0,
        "m": 0.93827,
    },
    {
        "id": 2212,
        "px": 0.0,
        "py": 0.0,
        "pz": -6799.9999,
        "e": 6800.0,
        "m": 0.93827,
    },
]


def _write_hepmc(path, batches, names=None, **options):
    with eio.EventWriter(str(path), beams=BEAMS, **options) as writer:
        width = (
            int(ak.max(ak.num(eio.apply_store(batches[0])["weights"])))
            if "weights" in eio.apply_store(batches[0]).fields
            else 1
        )
        writer.set_weight_names(
            names or ["Weight"] + [f"var{i}" for i in range(1, width)]
        )
        writer.set_cross_section(2.5e9, 1.0e7, 100, 120)
        for batch in batches:
            writer.write(batch)


def test_hepmc3_holds_the_final_state_between_two_beams(tmp_path, batches):
    import pyhepmc

    path = tmp_path / "e.hepmc"
    _write_hepmc(path, batches)
    written = eio.apply_store(batches[0])
    with pyhepmc.open(str(path)) as handle:
        event = next(iter(handle))
    statuses = event.numpy.particles.status
    # Delphes reads status 1; Pythia's own final-state codes (62, 83, 91, ...)
    # must not reach the file.
    assert set(statuses.tolist()) == {1, 4}
    assert [p.pid for p in event.beams] == [2212, 2212]
    assert (statuses == 1).sum() == len(written.prt[0])
    assert len(event.vertices) == 1


def test_hepmc3_carries_weights_their_names_and_the_cross_section(
    tmp_path, batches
):
    import pyhepmc

    path = tmp_path / "e.hepmc"
    _write_hepmc(path, batches)
    written = eio.apply_store(batches[0])
    with pyhepmc.open(str(path)) as handle:
        event = next(iter(handle))
    assert event.weights == ak.to_list(written["weights"][0])
    assert event.weight_names[0] == "Weight"
    assert event.cross_section.xsec(0) == pytest.approx(2.5e9)


def test_hepmc3_round_trip_matches_the_other_formats(tmp_path, batches):
    _write_hepmc(tmp_path / "e.hepmc", batches)
    eio.write_events(batches[0], str(tmp_path / "e.root"))
    from_hepmc = eio.read_events(str(tmp_path / "e.hepmc"), 0, len(batches[0]))
    from_root = eio.load_events(str(tmp_path / "e.root"))
    assert ak.to_list(from_hepmc.prt.id) == ak.to_list(from_root.prt.id)
    assert ak.all(abs(from_hepmc.prt.px - from_root.prt.px) < 1e-3)
    assert ak.to_list(from_hepmc.weight) == pytest.approx(
        ak.to_list(from_root.weight)
    )
    assert eio.count_events(str(tmp_path / "e.hepmc")) == sum(
        len(b) for b in batches
    )


def test_hepmc3_refuses_what_it_cannot_represent(tmp_path, batches):
    with pytest.raises(ValueError, match="two beam"):
        eio.EventWriter(str(tmp_path / "e.hepmc"))
    with pytest.raises(ValueError, match="HepMC3 record"):
        eio.EventWriter(
            str(tmp_path / "e.hepmc"), beams=BEAMS, hepmc_record="partial"
        )
    # A final-state record has no genealogy to write.
    with eio.EventWriter(
        str(tmp_path / "f.hepmc"), beams=BEAMS, hepmc_record="full"
    ) as writer:
        with pytest.raises(ValueError, match="complete record"):
            writer.write(batches[0])
    # Neither has a complete record filtered down to one species: its indices
    # point into the record it was cut from. (pi0 rather than pi+-: a species
    # that decays, so the filtered record still looks like a history.)
    filtered = eio.apply_store(batches[0], store="full", select_pids=[111])
    with eio.EventWriter(
        str(tmp_path / "g.hepmc"), store="full", beams=BEAMS
    ) as writer:
        writer.set_weight_names(
            ["Weight"]
            + [
                f"var{i}"
                for i in range(1, int(ak.max(ak.num(filtered["weights"]))))
            ]
        )
        with pytest.raises(ValueError, match="select_pids"):
            writer.write_reduced(filtered)


# --------------------------------------------------------------------- #
# HepMC3 genealogy
# --------------------------------------------------------------------- #

REFERENCE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "hepmc_reference"
)
REFERENCE_COLUMNS = (
    "id",
    "status",
    "mother1",
    "mother2",
    "daughter1",
    "daughter2",
    "px",
    "py",
    "pz",
    "e",
    "m",
    "vx",
    "vy",
    "vz",
    "vt",
    "status_hepmc",
)


def _reference_records(stem):
    """The raw Pythia records and the HepMC3 events Pythia's converter wrote
    for them (tools/tests/fixtures/hepmc_reference/README.md)."""
    import gzip
    import numpy as np
    import pyhepmc

    raw = [
        json.loads(line)
        for line in gzip.open(REFERENCE_DIR / f"{stem}.jsonl.gz", "rt")
    ]
    records = [
        {
            name: np.array([particle[i] for particle in event["prt"]])
            for i, name in enumerate(REFERENCE_COLUMNS)
        }
        for event in raw
    ]
    with pyhepmc.open(str(REFERENCE_DIR / f"{stem}.hepmc.gz")) as handle:
        reference = list(handle)
    return records, reference


def _particle_key(particle):
    momentum = particle.momentum
    return (
        particle.pid,
        particle.status,
        round(momentum.px, 5),
        round(momentum.py, 5),
        round(momentum.pz, 5),
        round(momentum.e, 5),
        round(particle.generated_mass, 5),
    )


def _vertex_structure(event):
    """Every vertex as (incoming, outgoing, position), independent of ids."""
    from collections import Counter

    structure = Counter()
    for vertex in event.vertices:
        position = vertex.position
        structure[
            (
                frozenset(
                    Counter(
                        _particle_key(p) for p in vertex.particles_in
                    ).items()
                ),
                frozenset(
                    Counter(
                        _particle_key(p) for p in vertex.particles_out
                    ).items()
                ),
                tuple(
                    round(x, 4)
                    for x in (position.x, position.y, position.z, position.t)
                ),
            )
        ] += 1
    return Counter(_particle_key(p) for p in event.particles), structure


@pytest.mark.parametrize("stem", ["softqcd", "zll", "ee", "bbbar"])
def test_hepmc3_genealogy_is_what_pythias_own_converter_writes(tmp_path, stem):
    """The vertices, statuses and positions match Pythia8ToHepMC3 exactly, on
    soft QCD, Drell-Yan with weight variations, e+e- and heavy flavour -- in
    memory, and after a round trip through the ASCII file."""
    import pyhepmc

    records, reference = _reference_records(stem)
    run_info = pyhepmc.GenRunInfo()
    run_info.weight_names = ["Weight"]
    with pyhepmc.open(str(tmp_path / "out.hepmc"), "w") as handle:
        for number, (record, expected) in enumerate(zip(records, reference)):
            statuses = eio.hepmc_status(
                record["id"], record["status"], record["daughter1"]
            )
            assert statuses.tolist() == record["status_hepmc"].tolist()
            event = eio.hepmc_event_from_record(record, number)
            assert _vertex_structure(event) == _vertex_structure(expected)
            event.run_info = run_info
            event.weights = [1.0]
            handle.write(event)
    with pyhepmc.open(str(tmp_path / "out.hepmc")) as handle:
        written = list(handle)
    assert len(written) == len(reference)
    for event, expected in zip(written, reference):
        assert _vertex_structure(event) == _vertex_structure(expected)


def test_hepmc3_full_record_carries_the_genealogy(tmp_path, batches):
    import pyhepmc

    path = tmp_path / "full.hepmc"
    _write_hepmc(path, batches, store="full")
    record = eio.apply_store(batches[0], store="full")
    with pyhepmc.open(str(path)) as handle:
        event = next(iter(handle))

    statuses = event.numpy.particles.status
    # Final state, decayed hadrons and beams, plus Pythia's own codes for the
    # rest of the history; nothing left at 0.
    assert {1, 2, 4} <= set(statuses.tolist()) and 0 not in statuses
    assert (statuses == 1).sum() == int(ak.sum(record.prt.status[0] > 0))
    assert [p.pid for p in event.beams] == [2212, 2212]
    assert all(
        not p.production_vertex or p.production_vertex.id == 0
        for p in event.beams
    )
    assert len(event.vertices) > 100
    # The system entry (id 90) is Pythia's, not the event's.
    assert 90 not in event.numpy.particles.pid

    # A decayed hadron's end vertex holds exactly its decay products.
    pythia = record.prt[0]
    index = next(
        i
        for i, (status, d1, d2) in enumerate(
            zip(
                ak.to_list(pythia.status),
                ak.to_list(pythia.daughter1),
                ak.to_list(pythia.daughter2),
            )
        )
        if status < 0
        and d1 > 0
        and d2 >= d1
        and abs(ak.to_list(pythia.status)[d1]) == 91
    )
    decayed = next(
        p
        for p in event.particles
        if p.status == 2
        and p.pid == pythia.id[index]
        and abs(p.momentum.e - pythia.e[index]) < 1e-3
    )
    products = sorted(p.pid for p in decayed.end_vertex.particles_out)
    assert products == sorted(
        ak.to_list(
            pythia.id[pythia.daughter1[index] : pythia.daughter2[index] + 1]
        )
    )


def test_hepmc3_full_record_keeps_the_pdf_information(tmp_path, batches):
    import pyhepmc

    path = tmp_path / "full.hepmc"
    _write_hepmc(path, batches, store="full")
    record = eio.apply_store(batches[0], store="full")
    with pyhepmc.open(str(path)) as handle:
        event = next(iter(handle))
    assert event.pdf_info.parton_id1 == int(record["info_id1"][0])
    assert event.pdf_info.x1 == pytest.approx(
        float(record["info_x1"][0]), rel=1e-6
    )
    assert event.pdf_info.scale == pytest.approx(
        float(record["info_Q2Fac"][0]) ** 0.5, rel=1e-6
    )


def test_hepmc3_final_state_is_the_same_from_either_record(tmp_path, batches):
    _write_hepmc(tmp_path / "full.hepmc", batches, store="full")
    _write_hepmc(tmp_path / "flat.hepmc", batches)
    from_full = eio.load_events(str(tmp_path / "full.hepmc"))
    from_flat = eio.load_events(str(tmp_path / "flat.hepmc"))

    # The genealogy numbers particles topologically, so the final state comes
    # back in a different order; the particles themselves are the same.
    def particles(events):
        return [
            sorted(zip(ak.to_list(e.prt.id), ak.to_list(e.prt.px)))
            for e in events
        ]

    assert particles(from_full) == particles(from_flat)
    assert ak.to_list(from_full.weights) == ak.to_list(from_flat.weights)


def test_hepmc3_record_choice_follows_the_batch_unless_told(tmp_path, batches):
    import pyhepmc

    def vertices(path):
        with pyhepmc.open(str(path)) as handle:
            return len(next(iter(handle)).vertices)

    _write_hepmc(tmp_path / "auto_full.hepmc", batches, store="full")
    _write_hepmc(tmp_path / "auto_finals.hepmc", batches, store="finals")
    _write_hepmc(
        tmp_path / "forced.hepmc",
        batches,
        store="full",
        hepmc_record="final-state",
    )
    assert vertices(tmp_path / "auto_full.hepmc") > 1
    assert vertices(tmp_path / "auto_finals.hepmc") == 1
    assert vertices(tmp_path / "forced.hepmc") == 1
    assert eio.has_genealogy(eio.apply_store(batches[0], store="full"))
    assert not eio.has_genealogy(eio.apply_store(batches[0], store="finals"))


def test_loading_a_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        eio.load_events(str(tmp_path / "absent.root"))


# --------------------------------------------------------------------- #
# Chunked writing -- what makes a sample larger than memory possible
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", [".root", ".jsonl"])
def test_chunked_writing_accumulates_every_batch(tmp_path, batches, suffix):
    path = str(tmp_path / f"events{suffix}")
    with eio.EventWriter(path) as writer:
        for batch in batches:
            writer.write(batch)

    expected = sum(len(b) for b in batches)
    assert writer.stats["n_events"] == expected
    assert len(eio.load_events(path)) == expected


def test_jsonl_event_ids_continue_across_chunks(tmp_path, batches):
    """event_id is a running index, not a per-chunk one."""
    path = str(tmp_path / "e.jsonl")
    with eio.EventWriter(path) as writer:
        for batch in batches:
            writer.write(batch)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert [r["event_id"] for r in rows] == list(range(len(rows)))


def test_writer_reports_size_and_settings(tmp_path, batches):
    path = str(tmp_path / "e.root")
    with eio.EventWriter(path, store="finals", precision="f64") as writer:
        writer.write(batches[0])
    stats = writer.stats
    assert stats["store"] == "finals" and stats["precision"] == "f64"
    assert stats["bytes"] > 0 and stats["format"] == "root"


def test_writer_rejects_bad_settings(tmp_path):
    with pytest.raises(ValueError, match="unknown store spec"):
        eio.EventWriter(str(tmp_path / "e.root"), store="nope")
    with pytest.raises(ValueError, match="unknown compression"):
        eio.EventWriter(str(tmp_path / "e.root"), compression="bzip9")


def test_an_empty_batch_writes_nothing(tmp_path, batches):
    path = str(tmp_path / "e.root")
    with eio.EventWriter(path) as writer:
        assert writer.write(batches[0][:0]) == 0
        writer.write(batches[0])
    assert writer.stats["n_events"] == len(batches[0])


# --------------------------------------------------------------------- #
# Summaries -- what an agent reads instead of the events
# --------------------------------------------------------------------- #


def test_summary_describes_the_sample(tmp_path, batches):
    path = str(tmp_path / "e.root")
    eio.write_events(batches[0], path)
    summary = eio.summarize(
        eio.load_events(path), particle_names={"22": "gamma"}
    )

    assert summary["n_events"] == len(batches[0])
    counts = summary["particles_per_event"]
    assert counts["min"] <= counts["mean"] <= counts["max"]
    assert summary["leading_pt_mean_gev"] > 0
    assert summary["top_species"], "expected at least one species"
    assert sum(s["count"] for s in summary["top_species"]) > 0


def test_summary_resolves_species_names_when_given_them(batches):
    summary = eio.summarize(
        eio.apply_store(batches[0]),
        particle_names={"22": "gamma", "211": "pi+"},
    )
    named = {s["id"]: s["name"] for s in summary["top_species"]}
    if 22 in named:
        assert named[22] == "gamma"


def test_summary_of_an_empty_array_is_empty(batches):
    assert eio.summarize(batches[0][:0]) == {}


def test_weight_stats_read_from_both_layouts(tmp_path, batches):
    """Weights live under info.weights straight from Pythia, and in a flat
    `weight` column once written; both must be understood."""
    from_batch = eio.weight_stats(batches[0])

    path = str(tmp_path / "e.root")
    eio.write_events(batches[0], path)
    from_disk = eio.weight_stats(eio.load_events(path))

    assert from_batch and from_disk
    assert from_batch["weighted"] == from_disk["weighted"]
    assert from_disk["weight_mean"] == pytest.approx(
        from_batch["weight_mean"], rel=1e-6
    )


def test_unweighted_runs_are_reported_as_such(batches):
    stats = eio.weight_stats(batches[0])
    assert stats["weighted"] is False
    assert stats["weight_min"] == stats["weight_max"] == 1.0


# --------------------------------------------------------------------- #
# Entry point for test_runner.py, which executes test files directly
# rather than through pytest.
# --------------------------------------------------------------------- #


def run_all_tests() -> bool:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    return pytest.main([__file__, "-v" if verbose else "-q"]) == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
