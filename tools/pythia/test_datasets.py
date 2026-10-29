"""
# test_datasets.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for the datasets PythiaFromRunCardTool writes: event weights, the run
summary and size guard, the ROOT/JSONL formats and store specs, and jet
clustering from those datasets.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("pythia8mc", reason="pythia8mc not installed")
ak = pytest.importorskip("awkward", reason="awkward not installed")
pytest.importorskip("uproot", reason="uproot not installed")

from tools import event_io as eio


# --------------------------------------------------------------------- #
# Event weights -- dropping these silently invalidates any histogram
# --------------------------------------------------------------------- #


def _generate(tmp_path, card_text, n_events=10, **kwargs):
    """Generate a sample and return (result, loaded events, manifest).

    The events come back through the IO layer rather than being parsed from a
    file, so these tests are independent of which format is the default.
    """
    from tools.pythia.generate import PythiaFromRunCardTool

    (tmp_path / "run.cmnd").write_text(card_text, encoding="utf-8")
    tool = PythiaFromRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(
        tool.execute(
            data_dir="out",
            cmnd_path="run.cmnd",
            n_events=n_events,
            seed=7,
            **kwargs,
        )
    )
    events = eio.load_events(str(tmp_path / payload["events"]))
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    return payload, events, manifest


UNWEIGHTED_CARD = (
    "Beams:eCM = 13000.\nHardQCD:all = on\nPhaseSpace:pTHatMin = 20.\n"
)
BIASED_CARD = UNWEIGHTED_CARD + (
    "PhaseSpace:bias2Selection = on\nPhaseSpace:bias2SelectionPow = 4.\n"
)
# A harder cut, so events reliably contain jets above the clustering threshold.
JET_CARD = "Beams:eCM = 13000.\nHardQCD:all = on\nPhaseSpace:pTHatMin = 50.\n"


def test_every_event_carries_a_weight(tmp_path):
    payload, events, manifest = _generate(tmp_path, UNWEIGHTED_CARD)
    assert payload["status"] == "ok"
    weights = ak.to_numpy(events.weight)
    assert len(weights) and (weights == 1.0).all()
    assert payload["weighted"] is False
    assert manifest["outputs"]["weights"]["weighted"] is False


def test_biased_generation_is_flagged_as_weighted(tmp_path):
    """Cross-section biasing produces weights spanning orders of magnitude.

    Before weights were recorded these events reached the analysis as
    weight-1, making every histogram built from them wrong.
    """
    payload, events, manifest = _generate(tmp_path, BIASED_CARD)
    weights = ak.to_numpy(events.weight)

    assert payload["weighted"] is True
    assert any("WEIGHTED" in w for w in payload["warnings"])
    assert len(set(weights.tolist())) > 1
    assert weights.max() / weights.min() > 10.0

    stats = manifest["outputs"]["weights"]
    assert stats["weighted"] is True
    assert stats["weight_min"] == pytest.approx(float(weights.min()), rel=1e-6)
    assert stats["weight_max"] == pytest.approx(float(weights.max()), rel=1e-6)


# --------------------------------------------------------------------- #
# Run summary and size guard -- events.jsonl is not readable at scale
# --------------------------------------------------------------------- #


def test_run_summary_describes_the_sample_without_reading_it(tmp_path):
    payload, events, manifest = _generate(tmp_path, UNWEIGHTED_CARD)
    summary = payload["summary"]

    assert summary["n_events"] == len(events)
    assert summary["particles_per_event"]["mean"] > 1
    assert (
        summary["particles_per_event"]["min"]
        <= summary["particles_per_event"]["mean"]
        <= summary["particles_per_event"]["max"]
    )
    assert summary["leading_pt_mean_gev"] > 0

    # Species are resolved to readable names, not bare PDG codes -- including
    # antiparticles, which is why the snapshot carries antiName.
    names = {entry["name"] for entry in summary["top_species"]}
    assert names & {"pi+", "pi-", "gamma"}, names

    # The manifest carries the same summary, plus the dataset size.
    assert manifest["outputs"]["summary"]["n_events"] == len(events)
    assert manifest["outputs"]["events_bytes"] > 0
    assert payload["bytes_per_event"] > 0


def test_species_counts_are_consistent_with_the_written_events(tmp_path):
    """The summary must describe what actually reached the file."""
    import numpy as np

    payload, events, _ = _generate(tmp_path, UNWEIGHTED_CARD)
    ids, counts = np.unique(
        ak.to_numpy(ak.flatten(events.prt.id)), return_counts=True
    )
    counted = dict(zip(ids.tolist(), counts.tolist()))
    for entry in payload["summary"]["top_species"]:
        assert counted[entry["id"]] == entry["count"]


def test_size_guard_stops_early_and_keeps_what_it_generated(tmp_path):
    """A hadron-level event costs tens of KiB, so a large n_events silently
    asks for tens of GB. The guard stops rather than filling the disk."""
    from tools.pythia.generate import PythiaFromRunCardTool

    (tmp_path / "run.cmnd").write_text(UNWEIGHTED_CARD, encoding="utf-8")
    tool = PythiaFromRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(
        tool.execute(
            data_dir="out",
            cmnd_path="run.cmnd",
            n_events=5000,
            seed=7,
            chunk_size=50,  # so the guard can act before the run ends
            max_output_gb=0.001,  # 1 MB
        )
    )

    assert payload["truncated"] is True
    assert 0 < payload["accepted"] < 5000
    assert any("reached the" in w for w in payload["warnings"])

    # Everything written is a complete, readable event -- nothing half-flushed.
    events = eio.load_events(str(tmp_path / payload["events"]))
    assert len(events) == payload["accepted"]
    assert int(ak.sum(ak.num(events.prt))) > 0


def test_size_guard_can_be_disabled(tmp_path):
    from tools.pythia.generate import PythiaFromRunCardTool

    (tmp_path / "run.cmnd").write_text(UNWEIGHTED_CARD, encoding="utf-8")
    tool = PythiaFromRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(
        tool.execute(
            data_dir="out",
            cmnd_path="run.cmnd",
            n_events=5,
            seed=7,
            max_output_gb=0,
        )
    )
    assert payload.get("truncated") is not True
    assert payload["accepted"] == 5


def test_a_normal_run_is_not_truncated(tmp_path):
    payload, events, manifest = _generate(tmp_path, UNWEIGHTED_CARD)
    assert payload.get("truncated") is not True
    assert manifest["outputs"]["truncated"] is False
    assert len(events) == 10


# --------------------------------------------------------------------- #
# Output format, store specs, and the legacy switches
# --------------------------------------------------------------------- #


def _run(tmp_path, card=UNWEIGHTED_CARD, n_events=20, seed=7, **kwargs):
    """Generate a small sample and return (result, data_dir)."""
    from tools.pythia.generate import PythiaFromRunCardTool

    (tmp_path / "run.cmnd").write_text(card, encoding="utf-8")
    tool = PythiaFromRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(
        tool.execute(
            data_dir="out",
            cmnd_path="run.cmnd",
            n_events=n_events,
            seed=seed,
            **kwargs,
        )
    )
    return payload, tmp_path / "out"


def test_root_is_the_default_format(tmp_path):
    payload, out = _run(tmp_path)
    assert payload["format"] == "root"
    assert payload["events"].endswith("events.root")
    assert (out / "events.root").exists()


def test_jsonl_remains_available(tmp_path):
    payload, out = _run(tmp_path, output_format="jsonl")
    assert payload["format"] == "jsonl"
    assert (out / "events.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (out / "events.jsonl").read_text().splitlines()
    ]
    assert rows[0]["schema"] == "evtjsonl-1.0"


def test_root_output_is_far_smaller_than_jsonl(tmp_path):
    """The whole reason ROOT is the default: ~7 KiB/event against ~60."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    root_payload, _ = _run(tmp_path / "a", output_format="root")
    jsonl_payload, _ = _run(tmp_path / "b", output_format="jsonl")
    assert (
        root_payload["bytes_per_event"] < jsonl_payload["bytes_per_event"] / 4
    )


def test_default_store_keeps_kinematics_only(tmp_path):
    payload, out = _run(tmp_path)
    assert payload["store"] == "finals+kinematics"
    events = eio.load_events(str(out / "events.root"))
    assert set(events.prt.fields) == {
        "id",
        "status",
        "px",
        "py",
        "pz",
        "e",
        "m",
    }


def test_full_store_keeps_vertices_and_lineage(tmp_path):
    payload, out = _run(tmp_path, store="full")
    events = eio.load_events(str(out / "events.root"))
    assert {"vx", "vy", "vz", "tau", "mother1", "col"} <= set(
        events.prt.fields
    )
    assert payload["store"] == "full"


def test_the_removed_legacy_switches_are_refused():
    """finals_only/full_history were superseded by `store` and are gone; a
    caller passing them must hear so rather than have them ignored."""
    from tools.pythia.generate import PythiaFromRunCardTool

    assert "finals_only" not in PythiaFromRunCardTool.model_fields
    assert "full_history" not in PythiaFromRunCardTool.model_fields


VARIATIONS_CARD = UNWEIGHTED_CARD + (
    "UncertaintyBands:doVariations = on\n"
    "UncertaintyBands:List = {fsr:murfac=0.5 fsr:murfac=2.0}\n"
)


@pytest.mark.parametrize("fmt", ["root", "jsonl"])
def test_variation_weights_are_stored_with_their_names(tmp_path, fmt):
    payload, events, manifest = _generate(
        tmp_path, VARIATIONS_CARD, n_events=5, output_format=fmt
    )
    names = manifest["outputs"]["weights"]["names"]
    assert names[0] == "Weight" and len(names) == 3
    assert sorted(payload["weight_variations"]) == sorted(names[1:])
    assert ak.all(ak.num(events["weights"]) == len(names))
    assert ak.all(events["weights"][:, 0] == events["weight"])


def test_weights_are_double_precision_whatever_the_kinematics_use(tmp_path):
    import numpy as np

    payload, events, _ = _generate(
        tmp_path, BIASED_CARD, n_events=5, precision="f32"
    )
    assert ak.to_numpy(events["weight"]).dtype == np.float64
    assert ak.to_numpy(events.prt.px[0]).dtype == np.float32


def test_manifest_names_its_own_schema_and_the_datasets(tmp_path):
    _, _, manifest = _generate(tmp_path, UNWEIGHTED_CARD, n_events=2)
    assert manifest["schema"] == "pythia-run-manifest-1.0"
    assert manifest["outputs"]["events_schema"] == "heptapod-events-root-1.0"


def test_precision_reaches_the_dataset(tmp_path):
    import numpy as np

    payload, out = _run(tmp_path, precision="f64")
    events = eio.load_events(str(out / "events.root"))
    assert ak.to_numpy(ak.flatten(events.prt.px)).dtype == np.float64


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"output_format": "hdf5"}, "unknown output_format"),
        ({"store": "everything"}, "unknown store spec"),
        ({"precision": "f16"}, "unknown precision"),
    ],
)
def test_invalid_dataset_options_are_rejected_before_generating(
    tmp_path, kwargs, expected
):
    """Rejected up front rather than after a generation run is spent."""
    from tools.pythia.generate import PythiaFromRunCardTool

    (tmp_path / "run.cmnd").write_text(UNWEIGHTED_CARD, encoding="utf-8")
    tool = PythiaFromRunCardTool(base_directory=str(tmp_path))
    out = tool.execute(
        data_dir="out", cmnd_path="run.cmnd", n_events=5, **kwargs
    )
    assert "Invalid Parameter" in out and expected in out


def test_chunking_does_not_change_the_result(tmp_path):
    """Batch size is a memory knob, not a physics one."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one_chunk, out_a = _run(tmp_path / "a", chunk_size=1000, seed=99)
    many_chunks, out_b = _run(tmp_path / "b", chunk_size=3, seed=99)

    assert one_chunk["accepted"] == many_chunks["accepted"] == 20
    left = eio.load_events(str(out_a / "events.root"))
    right = eio.load_events(str(out_b / "events.root"))
    assert ak.to_list(ak.num(left.prt)) == ak.to_list(ak.num(right.prt))


def test_manifest_records_the_dataset_settings(tmp_path):
    payload, out = _run(tmp_path, store="finals", precision="f64")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["inputs"]["output_format"] == "root"
    assert manifest["inputs"]["store"] == "finals"
    assert manifest["inputs"]["precision"] == "f64"
    assert manifest["outputs"]["events"] == "events.root"
    assert manifest["outputs"]["summary"]["n_events"] == payload["accepted"]


def test_weights_survive_the_columnar_path(tmp_path):
    """The weight must reach the dataset whatever the format."""
    payload, out = _run(tmp_path, card=BIASED_CARD)
    assert payload["weighted"] is True
    events = eio.load_events(str(out / "events.root"))
    weights = ak.to_numpy(events.weight)
    assert weights.min() != weights.max()
    assert any("WEIGHTED" in w for w in payload["warnings"])


# --------------------------------------------------------------------- #
# Jet clustering reads whatever generation wrote
# --------------------------------------------------------------------- #


def _cluster(tmp_path, events_rel, **kwargs):
    from tools.pythia.jets import JetClusterSlowJetTool

    tool = JetClusterSlowJetTool(base_directory=str(tmp_path))
    return json.loads(tool.execute(events_path=events_rel, **kwargs))


def test_clustering_reads_the_default_root_dataset(tmp_path):
    """Generation writes ROOT by default, so clustering must read it.

    Without this the default pipeline is broken end to end.
    """
    payload, _ = _run(tmp_path, card=JET_CARD)
    clustered = _cluster(
        tmp_path, payload["events"], event_index=0, R=0.4, ptmin=20.0
    )
    assert clustered["status"] == "ok"
    assert clustered["results"][0]["data"]["jets"]


def test_clustering_agrees_across_formats(tmp_path):
    """The container must not change the physics."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    root_run, _ = _run(tmp_path / "a", card=JET_CARD, output_format="root")
    jsonl_run, _ = _run(tmp_path / "b", card=JET_CARD, output_format="jsonl")

    from_root = _cluster(
        tmp_path / "a", root_run["events"], event_index=0, R=0.4, ptmin=20.0
    )
    from_jsonl = _cluster(
        tmp_path / "b", jsonl_run["events"], event_index=0, R=0.4, ptmin=20.0
    )

    pt_root = [j["pT"] for j in from_root["results"][0]["data"]["jets"]]
    pt_jsonl = [j["pT"] for j in from_jsonl["results"][0]["data"]["jets"]]
    assert pt_root and pt_root == pytest.approx(pt_jsonl)


def test_cluster_all_streams_every_event(tmp_path):
    payload, out = _run(tmp_path, card=JET_CARD)
    clustered = _cluster(
        tmp_path,
        payload["events"],
        cluster_all=True,
        output_path="out/jets.jsonl",
        R=0.4,
        ptmin=20.0,
    )
    assert clustered["n_events"] == payload["accepted"]
    rows = [
        json.loads(line)
        for line in (out / "jets.jsonl").read_text().splitlines()
    ]
    assert len(rows) == payload["accepted"]
    assert [r["event_index"] for r in rows] == list(range(len(rows)))


def test_one_event_clusters_as_it_does_within_the_whole_dataset(tmp_path):
    """A single event is read on its own; it must give what cluster_all
    gives."""
    from tools.pythia.jets import JetClusterSlowJetTool

    payload, _ = _run(tmp_path, card=JET_CARD, n_events=6)
    tool = JetClusterSlowJetTool(base_directory=str(tmp_path))
    tool.execute(
        events_path=payload["events"],
        cluster_all=True,
        output_path="all.jsonl",
    )
    streamed = [
        json.loads(line)
        for line in (tmp_path / "all.jsonl").read_text().splitlines()
    ]
    single = json.loads(
        tool.execute(events_path=payload["events"], event_index=4)
    )
    assert single["results"][0]["data"] == streamed[4]["data"]


def test_an_out_of_range_event_index_says_how_many_events_there_are(tmp_path):
    from tools.pythia.jets import JetClusterSlowJetTool

    payload, _ = _run(tmp_path, card=JET_CARD, n_events=3)
    tool = JetClusterSlowJetTool(base_directory=str(tmp_path))
    message = tool.execute(events_path=payload["events"], event_index=10)
    assert "out of range for 3 events" in message


def test_clustering_takes_exactly_one_input(tmp_path):
    from tools.pythia.jets import JetClusterSlowJetTool

    tool = JetClusterSlowJetTool(base_directory=str(tmp_path))
    assert "exactly one input" in tool.execute(
        events_path="a.root", npy_path="b.npy", event_index=0
    )


def test_clustering_rejects_a_missing_input(tmp_path):
    from tools.pythia.jets import JetClusterSlowJetTool

    tool = JetClusterSlowJetTool(base_directory=str(tmp_path))
    assert "exactly one input" in tool.execute(event_index=0)


# --------------------------------------------------------------------- #
# Entry point for test_runner.py, which executes test files directly
# rather than through pytest.
# --------------------------------------------------------------------- #


def run_all_tests() -> bool:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    return pytest.main([__file__, "-v" if verbose else "-q"]) == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
