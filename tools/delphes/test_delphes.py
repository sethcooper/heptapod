#!/usr/bin/env python3
"""
# test_delphes.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Delphes tools test suite.

Tests cover:
- setup/config errors (missing delphes_path, unknown card) with a fake
  install tree — no real Delphes needed
- card resolution: stock cards/ vs sandbox-local custom cards, path
  traversal prevention
- output-exists / overwrite semantics
- run-and-parse against a STUB DelphesHepMC3 (shell script emitting
  progress lines and touching the output) — pins the subprocess
  contract without ROOT
- the full real chain (evtjsonl -> HepMC -> Delphes CMS card -> ROOT ->
  recojsonl), gated on DELPHES_DIR + pyhepmc + uproot being available
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.delphes import DelphesFromHepMCTool, DelphesROOTToJSONLTool  # noqa: E402

try:
    import pyhepmc  # noqa: F401
    HAVE_PYHEPMC = True
except ImportError:
    HAVE_PYHEPMC = False
try:
    import pythia8mc  # noqa: F401
    HAVE_PYTHIA = True
except ImportError:
    HAVE_PYTHIA = False

try:
    import uproot  # noqa: F401
    HAVE_UPROOT = True
except ImportError:
    HAVE_UPROOT = False

_REAL_DELPHES = os.environ.get("DELPHES_DIR", "")
HAVE_REAL_DELPHES = bool(_REAL_DELPHES) and os.path.isfile(
    os.path.join(os.path.expanduser(_REAL_DELPHES), "DelphesHepMC3"))


def _fake_delphes_install(root: Path) -> Path:
    """A fake Delphes dir: stub DelphesHepMC3 + a cards/ dir."""
    root.mkdir(parents=True)
    (root / "cards").mkdir()
    (root / "cards" / "delphes_card_CMS.tcl").write_text("# fake card\n")
    stub = root / "DelphesHepMC3"
    stub.write_text(
        "#!/bin/sh\n"
        '[ -f "$3" ] || { echo "no input" >&2; exit 1; }\n'
        'echo "** 50 events processed"\n'
        'echo "** 123 events processed"\n'
        'touch "$2"\n'
    )
    stub.chmod(0o755)
    return root


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base = self.root / "sandbox"
        self.base.mkdir()
        (self.base / "in.hepmc").write_text("HepMC::Version 3.02.00\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _tool(self, install, **runtime):
        kwargs = dict(hepmc_path="in.hepmc", output_root="out/d.root")
        kwargs.update(runtime)
        tool = DelphesFromHepMCTool(**kwargs)
        tool.base_directory = str(self.base)
        tool.delphes_path = str(install)
        return tool


class TestSetupAndCards(_Tmp):
    def test_missing_delphes_path_clear_error(self):
        old = os.environ.pop("DELPHES_DIR", None)
        try:
            tool = self._tool("")
            out = tool._run()
            self.assertIn("Setup Error", out)
            self.assertIn("delphes_path", out)
        finally:
            if old is not None:
                os.environ["DELPHES_DIR"] = old

    def test_not_a_delphes_dir(self):
        out = self._tool(self.root)._run()      # exists, no binary
        self.assertIn("DelphesHepMC3", out)

    def test_unknown_card_lists_available(self):
        install = _fake_delphes_install(self.root / "delphes")
        out = self._tool(install, card="delphes_card_NOPE.tcl")._run()
        self.assertIn("delphes_card_CMS.tcl", out)   # the available list

    def test_custom_card_from_sandbox(self):
        install = _fake_delphes_install(self.root / "delphes")
        (self.base / "my_card.tcl").write_text("# custom\n")
        out = json.loads(self._tool(install, card="./my_card.tcl")._run())
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["card"], "my_card.tcl")

    def test_custom_card_escape_rejected(self):
        install = _fake_delphes_install(self.root / "delphes")
        out = self._tool(install, card="../escape.tcl")._run()
        self.assertIn("Setup Error", out)


class TestRunContract(_Tmp):
    def setUp(self):
        super().setUp()
        self.install = _fake_delphes_install(self.root / "delphes")

    def test_run_parses_progress_and_writes_output(self):
        out = json.loads(self._tool(self.install)._run())
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["n_events"], 123)       # last progress line
        self.assertTrue((self.base / "out" / "d.root").exists())

    def test_existing_output_refused_unless_overwrite(self):
        (self.base / "out").mkdir()
        (self.base / "out" / "d.root").write_text("old")
        out = self._tool(self.install)._run()
        self.assertIn("Output Exists", out)
        ok = json.loads(self._tool(self.install, overwrite=True)._run())
        self.assertEqual(ok["status"], "ok")

    def test_missing_input(self):
        out = self._tool(self.install, hepmc_path="nope.hepmc")._run()
        self.assertIn("File Not Found", out)

    def test_env_var_fallback(self):
        old = os.environ.get("DELPHES_DIR")
        os.environ["DELPHES_DIR"] = str(self.install)
        try:
            tool = self._tool("")
            tool.delphes_path = ""
            out = json.loads(tool._run())
            self.assertEqual(out["status"], "ok")
        finally:
            if old is None:
                del os.environ["DELPHES_DIR"]
            else:
                os.environ["DELPHES_DIR"] = old


@unittest.skipUnless(HAVE_REAL_DELPHES and HAVE_PYHEPMC and HAVE_UPROOT,
                     "needs DELPHES_DIR + pyhepmc + uproot")
class TestRealChain(unittest.TestCase):
    """The whole link this package exists for, against the real binary:
    events -> HepMC3 -> Delphes(CMS card) -> ROOT -> recojsonl."""

    def test_end_to_end(self):
        from tools.events import ConvertEventsTool

        with tempfile.TemporaryDirectory() as base:
            # Three e+e- + photon events, central and hard enough to
            # reconstruct as electrons/photons in the CMS card.
            with open(os.path.join(base, "events.jsonl"), "w") as f:
                for i in range(3):
                    particles = [
                        {"i": 0, "id": 11, "px": 60.0, "py": 1.0 + i,
                         "pz": 12.0, "E": 61.2, "m": 0.000511},
                        {"i": 1, "id": -11, "px": -58.0, "py": -2.0,
                         "pz": -9.0, "E": 58.7, "m": 0.000511},
                        {"i": 2, "id": 22, "px": 0.5, "py": 44.0,
                         "pz": 4.0, "E": 44.2, "m": 0.0},
                    ]
                    f.write(json.dumps({
                        "schema": "evtjsonl-1.0", "event_id": i,
                        "finals_only": True, "full_history": False,
                        "data": {"n": len(particles), "particles": particles},
                    }) + "\n")

            conv = ConvertEventsTool(input_path="events.jsonl",
                                     output_path="events.hepmc",
                                     beam_ids=[2212, 2212],
                                     beam_energies_gev=[6800.0, 6800.0])
            conv.base_directory = base
            self.assertEqual(json.loads(conv._run())["n_events"], 3)

            sim = DelphesFromHepMCTool(hepmc_path="events.hepmc",
                                       output_root="delphes/out.root",
                                       timeout_s=600)
            sim.base_directory = base
            sim.delphes_path = os.path.expanduser(_REAL_DELPHES)
            res = json.loads(sim._run())
            self.assertEqual(res["status"], "ok")
            self.assertTrue(os.path.exists(os.path.join(base, "delphes/out.root")))

            flat = DelphesROOTToJSONLTool(root_path="delphes/out.root",
                                          jsonl_path="reco.jsonl")
            flat.base_directory = base
            rres = json.loads(flat._run())
            self.assertEqual(rres["status"], "ok")
            self.assertEqual(rres["n_events"], 3)
            lines = [json.loads(line) for line in
                     Path(base, "reco.jsonl").read_text().splitlines()]
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0]["schema"], "recojsonl-1.0")
            self.assertIn("met", lines[0]["data"])
            # At least one event reconstructs at least one electron.
            self.assertTrue(any(ev["data"].get("electrons") for ev in lines))

    @unittest.skipUnless(HAVE_PYTHIA, "needs pythia8mc")
    def test_pythia_hepmc_reaches_delphes(self):
        """Pythia's final state must reach Delphes as status-1 particles.

        The JSONL bridge this replaced copied Pythia's status codes (62, 83,
        91, ...) into the HepMC file; Delphes reads only status 1, so every
        event reconstructed empty.
        """
        from tools.pythia.generate import PythiaFromRunCardTool

        with tempfile.TemporaryDirectory() as base:
            Path(base, "run.cmnd").write_text(
                "Beams:eCM = 13600.\nHardQCD:all = on\nPhaseSpace:pTHatMin = 100.\n")
            gen = json.loads(PythiaFromRunCardTool(base_directory=base).execute(
                data_dir="gen", cmnd_path="run.cmnd", n_events=5, seed=11,
                output_format="hepmc3"))
            self.assertEqual(gen["status"], "ok")

            sim = DelphesFromHepMCTool(hepmc_path=gen["events"],
                                       output_root="delphes/out.root", timeout_s=600)
            sim.base_directory = base
            sim.delphes_path = os.path.expanduser(_REAL_DELPHES)
            self.assertEqual(json.loads(sim._run())["status"], "ok")

            flat = DelphesROOTToJSONLTool(root_path="delphes/out.root",
                                          jsonl_path="reco.jsonl")
            flat.base_directory = base
            self.assertEqual(json.loads(flat._run())["n_events"], 5)
            lines = [json.loads(line) for line in
                     Path(base, "reco.jsonl").read_text().splitlines()]
            # Hard QCD above 100 GeV: every event has jets.
            self.assertTrue(all(ev["data"].get("jets") for ev in lines))


@unittest.skipUnless(HAVE_UPROOT, "needs uproot")
class TestRootToJsonlGuard(unittest.TestCase):
    """DelphesROOTToJSONL against a synthetic Delphes-layout ROOT (no real
    binary needed): the empty-output guard must fail loudly, and a populated
    ROOT must report per-collection object counts. Branches are written
    nested (`Jet/Jet.PT`, `MissingET/MissingET.MET`) via ak.zip to match the
    real Delphes leaf layout the tool indexes by leaf name.
    """

    @staticmethod
    def _write_root(path, *, jets, met):
        """jets: list-per-event of jet PTs (or None to omit the Jet branch);
        met: list-per-event MET magnitude. MissingET is always present."""
        import uproot
        import awkward as ak
        tree = {}
        if jets is not None:
            z = [[0.0] * len(ev) for ev in jets]
            iz = [[0] * len(ev) for ev in jets]
            tree["Jet"] = ak.zip({
                "PT": ak.Array(jets), "Eta": ak.Array(z), "Phi": ak.Array(z),
                "Mass": ak.Array(z), "BTag": ak.Array(iz)})
        tree["MissingET"] = ak.zip({
            "MET": ak.Array([[m] for m in met]),
            "Phi": ak.Array([[0.0] for _ in met])})
        with uproot.recreate(path) as f:
            f["Delphes"] = tree

    def test_empty_reco_fails_loudly(self):
        with tempfile.TemporaryDirectory() as base:
            rp = os.path.join(base, "empty.root")
            # No Jet branch and MET identically zero — the silent-failure
            # shape from the c009 T6gg audit (200 events, 0 objects, status ok).
            self._write_root(rp, jets=None, met=[0.0, 0.0, 0.0])
            tool = DelphesROOTToJSONLTool(root_path="empty.root",
                                          jsonl_path="reco.jsonl",
                                          collections="jets,met")
            tool.base_directory = base
            raw = tool._run()  # format_error returns text, not JSON
            self.assertIn("Empty Reconstruction", raw)
            self.assertIn("Jet.PT", raw)  # names the missing branch

    def test_populated_reco_reports_counts(self):
        with tempfile.TemporaryDirectory() as base:
            rp = os.path.join(base, "ok.root")
            self._write_root(rp, jets=[[50.0], [40.0], [30.0], [25.0]],
                             met=[120.0, 80.0, 5.0, 200.0])
            tool = DelphesROOTToJSONLTool(root_path="ok.root",
                                          jsonl_path="reco.jsonl",
                                          collections="jets,met")
            tool.base_directory = base
            res = json.loads(tool._run())
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["n_events"], 4)
            self.assertEqual(res["object_counts"]["jets"], 4)  # 1 jet/event
            self.assertEqual(res["object_counts"]["met"], 4)   # all MET > 0
            self.assertEqual(res["missing_branches"], [])


if __name__ == "__main__":
    unittest.main()
