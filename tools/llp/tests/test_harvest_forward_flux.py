"""
# test_harvest_forward_flux.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Unit tests for HarvestForwardFluxTool: turning a Pythia event JSONL
(evtjsonl-1.0) into weighted forward parent-flux files (parentflux-1.0) that
feed MesonDecayToLLPTool. Inputs are a small hand-built event file with a
known parent content so the forward selection, per-collision normalization,
prescaling and manifest-driven cross section are all checkable exactly.
"""
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent.parent                    # .../heptapod/tools/llp
REPO_ROOT = TOOL_DIR.parent.parent                      # .../heptapod
sys.path.insert(0, str(REPO_ROOT))

from tools.llp.harvest_forward_flux import HarvestForwardFluxTool

base_directory = str(TOOL_DIR / "test_files_harvest")

SIGMA_GEN = 60.0      # mb
SIGMA_INEL = 80.0     # mb
N_GEN = 50            # generated events
THETA_MAX = 0.01      # 10 mrad forward window

# PID -> name map we harvest (positive PID; sign folded by the tool)
PARENTS = {"321": "K", "411": "D", "211": "pi"}


def _particle(pid, p, theta, phi, m=0.4937):
    """One evtjsonl particle at momentum p, polar theta, azimuth phi."""
    px = p * math.sin(theta) * math.cos(phi)
    py = p * math.sin(theta) * math.sin(phi)
    pz = p * math.cos(theta)
    E = math.sqrt(p * p + m * m)
    return {"i": 0, "id": pid, "status": 1, "px": px, "py": py, "pz": pz,
            "E": E, "m": m}


def _write_events(path, n_gen=N_GEN):
    """Hand-built events: each event has one forward K (kept), one wide-angle
    K (cut), one forward D (kept), one forward pi (kept), one backward K
    (kept only when folding), and a non-parent (proton, ignored)."""
    with open(path, "w") as fh:
        for ev in range(n_gen):
            parts = [
                _particle(321, 100.0, 0.002, 0.3),      # forward K   -> keep
                _particle(321, 100.0, 0.5, 1.0),        # wide K      -> cut
                _particle(411, 80.0, 0.003, 2.0, m=1.869),   # forward D -> keep
                _particle(211, 60.0, 0.004, 0.7, m=0.1396),  # forward pi-> keep
                _particle(-321, 90.0, math.pi - 0.002, 1.5),  # backward K
                _particle(2212, 200.0, 0.001, 0.0, m=0.938),  # proton (ignored)
            ]
            rec = {"schema": "evtjsonl-1.0", "event_id": ev,
                   "data": {"n": len(parts), "particles": parts}}
            fh.write(json.dumps(rec) + "\n")


def _write_manifest(path, sigma_gen=SIGMA_GEN, n_written=N_GEN):
    doc = {"schema": "evtjsonl-1.0",
           "outputs": {"events_jsonl": "events.jsonl",
                       "n_events_written": n_written,
                       "xsec": {"sigmaGen_mb": sigma_gen}}}
    Path(path).write_text(json.dumps(doc, indent=2))


def _setup():
    base = Path(base_directory)
    base.mkdir(parents=True, exist_ok=True)
    _write_events(base / "events.jsonl")
    _write_manifest(base / "manifest.json")


def _tool(**kw):
    kw.setdefault("base_directory", base_directory)
    kw.setdefault("pythia_events_path", "events.jsonl")
    kw.setdefault("parents", PARENTS)
    kw.setdefault("theta_max_rad", THETA_MAX)
    kw.setdefault("sigma_inel_mb", SIGMA_INEL)
    kw.setdefault("output_dir", "flux")
    t = HarvestForwardFluxTool(**kw)
    t._setup()
    return t


def _species(res, name):
    return next(s for s in res["species"] if s["name"] == name)


def _records(rel):
    return [json.loads(ln) for ln in
            (Path(base_directory) / rel).read_text().splitlines() if ln.strip()]


def test_path_traversal_prevention():
    print(">> traversal prevention ...")
    _setup()
    t = _tool(pythia_events_path="../../../etc/passwd", sigma_gen_mb=SIGMA_GEN)
    assert "denied" in t._run().lower()
    t = _tool(output_dir="../../../tmp/evil", sigma_gen_mb=SIGMA_GEN)
    assert "denied" in t._run().lower()
    print("[OK] input and output traversal rejected")


def test_forward_selection_and_normalization():
    print(">> forward selection + per-collision weight (via manifest) ...")
    _setup()
    res = json.loads(_tool(manifest_path="manifest.json")._run())
    assert res["status"] == "ok", res
    assert res["n_gen"] == N_GEN
    assert abs(res["sigma_gen_mb"] - SIGMA_GEN) < 1e-9
    w_exp = SIGMA_GEN / (N_GEN * SIGMA_INEL)
    assert abs(res["w_per_collision_base"] - w_exp) / w_exp < 1e-12
    # forward K: one per event kept, plus the backward K folded in => 2 * N_GEN
    # RECORDS, but only 2x the STATISTICS -- the physical forward cone still
    # holds one K per collision, so each folded record carries w_exp / 2.
    kK = _species(res, "K")
    assert kK["n_parents"] == 2 * N_GEN, kK
    kD = _species(res, "D")
    kpi = _species(res, "pi")
    assert kD["n_parents"] == N_GEN
    assert kpi["n_parents"] == N_GEN
    # folding halves the per-record weight (cf. prescale / cap, which likewise
    # carry compensating factors)
    recs = _records(kK["path"])
    assert all(abs(r["weight_per_collision"] - 0.5 * w_exp) < 1e-18
               for r in recs)
    # folded backward K has pz > 0 in the output
    assert all(r["pz"] > 0.0 for r in recs), "folding must make pz forward"
    # sum of K weights = mean K per collision ENTERING ONE FORWARD CONE, which
    # the fixture fixes at exactly one per event
    assert abs(kK["sum_weights"] - N_GEN * w_exp) < 1e-15
    print(f"[OK] K={kK['n_parents']} D={kD['n_parents']} pi={kpi['n_parents']}, "
          f"w={w_exp:.3e}/coll")


def test_folding_preserves_normalisation():
    """Folding is variance reduction, NOT extra flux.

    This is the invariant the 2.4.0 tool violated: it harvested both cones and
    stored them all as forward WITHOUT halving the weight, so every flux -- and
    every yield built on one -- was 2x too large. A single-arm detector sees one
    cone; a genuinely backward parent can never reach it. Folding must double
    the sample size and leave the summed weight alone.
    """
    print(">> folding preserves the per-collision normalisation ...")
    _setup()
    on = json.loads(_tool(manifest_path="manifest.json",
                          fold_hemispheres=True)._run())
    _setup()
    off = json.loads(_tool(manifest_path="manifest.json",
                           fold_hemispheres=False)._run())
    assert on["status"] == "ok" and off["status"] == "ok"
    kon, koff = _species(on, "K"), _species(off, "K")
    # twice the records ...
    assert kon["n_parents"] == 2 * koff["n_parents"], (kon, koff)
    # ... and the SAME summed weight
    assert abs(kon["sum_weights"] - koff["sum_weights"]) < 1e-15, (
        f"folding changed the flux normalisation: "
        f"{kon['sum_weights']} vs {koff['sum_weights']}")
    print(f"[OK] records {koff['n_parents']} -> {kon['n_parents']}, "
          f"sum_weights unchanged at {kon['sum_weights']:.4e}")


def test_theta_cut_excludes_wide_and_folding_off():
    print(">> theta window + fold_hemispheres flag ...")
    _setup()
    # with folding off, the backward K (theta ~ pi) is excluded -> N_GEN Ks
    res = json.loads(_tool(sigma_gen_mb=SIGMA_GEN, n_gen=N_GEN,
                           fold_hemispheres=False)._run())
    assert _species(res, "K")["n_parents"] == N_GEN
    # a tighter window than the D angle (0.003) drops D but keeps the K (0.002)
    res2 = json.loads(_tool(sigma_gen_mb=SIGMA_GEN, n_gen=N_GEN,
                            theta_max_rad=0.0025)._run())
    assert _species(res2, "D")["n_parents"] == 0
    assert _species(res2, "K")["n_parents"] > 0
    # the empty species still gets a file
    assert (Path(base_directory) / _species(res2, "D")["path"]).exists()
    print("[OK] wide-angle cut applied; empty species still written")


def test_prescale_preserves_weight_sum():
    print(">> prescale keeps 1-in-N but preserves the summed weight ...")
    _setup()
    full = json.loads(_tool(sigma_gen_mb=SIGMA_GEN, n_gen=N_GEN)._run())
    pi_full = _species(full, "pi")
    res = json.loads(_tool(sigma_gen_mb=SIGMA_GEN, n_gen=N_GEN,
                           prescale={"pi": 10})._run())
    pi = _species(res, "pi")
    # keep ~1 in 10 of the pions ...
    assert pi["n_parents"] == N_GEN // 10, pi
    assert pi["prescale"] == 10
    # ... but each carries 10x the weight, so the total is preserved
    assert abs(pi["sum_weights"] - pi_full["sum_weights"]) < 1e-15, \
        (pi["sum_weights"], pi_full["sum_weights"])
    print("[OK] prescale=10 keeps 1/10 of pions with 10x weight; sum intact")


def test_cap_bounds_count_preserves_weight():
    print(">> cap subsamples abundant species, preserves the weight sum ...")
    _setup()
    # uncapped reference
    full = json.loads(_tool(sigma_gen_mb=SIGMA_GEN, n_gen=N_GEN, cap=0)._run())
    kf = _species(full, "K")
    n_total = kf["n_parents"]
    assert n_total > 50, n_total
    # cap to 50 -> fewer records, but summed weight identical (mean K/coll)
    res = json.loads(_tool(sigma_gen_mb=SIGMA_GEN, n_gen=N_GEN, cap=50)._run())
    k = _species(res, "K")
    assert k["n_parents"] == 50 and k["capped"] and k["n_total"] == n_total, k
    assert abs(k["sum_weights"] - kf["sum_weights"]) / kf["sum_weights"] < 1e-12
    # each kept parent carries the compensating n_total/cap factor
    assert abs(k["weight_per_collision"]
               - kf["weight_per_collision"] * n_total / 50) < 1e-18
    # a species already under the cap is untouched
    assert not _species(res, "D")["capped"]
    print(f"[OK] K {n_total}->50 capped, weight x{n_total/50:.2f}, sum intact")


def test_missing_cross_section_errors():
    print(">> missing sigma_gen -> error ...")
    _setup()
    out = _tool()._run()   # no sigma_gen_mb, no manifest
    assert "sigma_gen" in out.lower() and "error" in out.lower(), out
    print("[OK] normalization refuses to run without a cross section")


def test_root_dataset_harvests_identically():
    """The same events stored as ROOT harvest to exactly the JSONL result:
    generation writes ROOT by default, and the container must not change the
    flux."""
    try:
        import uproot  # noqa: F401
        from tools import event_io
    except ImportError:
        print("[SKIP] uproot not installed")
        return
    print(">> ROOT and JSONL datasets harvest identically ...")
    _setup()
    base = Path(base_directory)
    events = event_io.load_events(str(base / "events.jsonl"))
    with event_io.EventWriter(str(base / "events.root")) as writer:
        writer.write_reduced(events)

    from_jsonl = json.loads(_tool(manifest_path="manifest.json", output_dir="flux_jsonl")._run())
    from_root = json.loads(_tool(manifest_path="manifest.json", output_dir="flux_root",
                                 pythia_events_path="events.root")._run())
    assert from_root["status"] == "ok", from_root
    for name in set(PARENTS.values()):
        a, b = _species(from_jsonl, name), _species(from_root, name)
        assert a["n_parents"] == b["n_parents"], (name, a, b)
        assert math.isclose(a["sum_weights"], b["sum_weights"], rel_tol=1e-12)
    print("[OK] ROOT and JSONL give the same flux")


def run_all():
    try:
        test_path_traversal_prevention()
        test_forward_selection_and_normalization()
        test_theta_cut_excludes_wide_and_folding_off()
        test_prescale_preserves_weight_sum()
        test_cap_bounds_count_preserves_weight()
        test_missing_cross_section_errors()
        test_root_dataset_harvests_identically()
        print("\nAll HarvestForwardFlux tests passed.")
    finally:
        shutil.rmtree(base_directory, ignore_errors=True)


if __name__ == "__main__":
    run_all()
