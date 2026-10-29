"""
# delphes.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Delphes fast detector simulation, closing heptapod's chain:

    PythiaFromRunCardTool(output_format="hepmc3") -> .hepmc
    (or any event dataset -> ConvertEventsTool -> .hepmc)
        -> DelphesFromHepMCTool -> .root
        -> DelphesROOTToJSONLTool -> reco-level JSONL

Design note: a thin "3 lines of shell over the binary" wrapper adds
nothing for an agent that has a shell (ColliderBench deleted theirs for
exactly that reason). These tools earn their place by doing the work
the bare binary doesn't: detector-card resolution and validation,
refusing the silent skip-the-shower LHE path (HepMC input only),
progress/error surfacing, and converting Delphes's TClonesArray ROOT
output into a flat JSONL record downstream heptapod tools can consume
without the agent writing ROOT-reading code.

`DelphesFromHepMCTool` needs a built Delphes (the dir with
`DelphesHepMC3` and `cards/`), located via the `delphes_path` state /
config or the `DELPHES_DIR` environment variable.
"""

import json
import math
import os
import re
import subprocess
import time
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

# Reco-level output schema written by DelphesROOTToJSONLTool.
RECO_SCHEMA_VERSION = "recojsonl-1.0"

# Delphes branch layout per collection: (branch prefix, fields).
# `met` is handled separately (one record per event, not a list).
_COLLECTIONS = {
    "jets":      ("Jet",      ("PT", "Eta", "Phi", "Mass", "BTag")),
    "electrons": ("Electron", ("PT", "Eta", "Phi", "Charge")),
    "muons":     ("Muon",     ("PT", "Eta", "Phi", "Charge")),
    "photons":   ("Photon",   ("PT", "Eta", "Phi")),
}

# "** 1000 events processed" progress lines in DelphesHepMC3 stdout.
_PROGRESS_RE = re.compile(r"\*\*\s+(\d+)\s+events processed")


def _resolve_delphes_dir(delphes_path: str) -> str:
    """The Delphes install dir, from state/config else DELPHES_DIR env."""
    src = str(delphes_path or os.environ.get("DELPHES_DIR", "")).strip()
    if not src:
        raise ValueError(
            "delphes_path is not configured. Point it (or the DELPHES_DIR "
            "env var) at a built Delphes directory — the one containing the "
            "DelphesHepMC3 executable and cards/."
        )
    root = os.path.abspath(os.path.expanduser(src))
    if not os.path.isfile(os.path.join(root, "DelphesHepMC3")):
        raise ValueError(
            f"No DelphesHepMC3 executable in {root}. Build Delphes first "
            "(requires ROOT; `make` in the Delphes source dir)."
        )
    return root


class _SandboxedTool(BaseTool):
    """Shared base-directory plumbing (mirrors conversions.py)."""

    base_directory: str = StateField(default=".", description="Base sandbox root")

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(
                f"Base directory does not exist or is not a directory: {self.base_directory}"
            )

    def _safe_path(self, rel_or_abs: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        if not rel_or_abs:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel_or_abs))
        if full.startswith(self.base_directory + os.sep) or full == self.base_directory:
            return full
        return None


class DelphesFromHepMCTool(_SandboxedTool):
    """
    Run Delphes fast detector simulation on a HepMC3 event file.

    Input (runtime):
      - hepmc_path: input .hepmc file, from PythiaFromRunCardTool with
        output_format="hepmc3" or from ConvertEventsTool. HepMC only — feeding Delphes
        parton-level LHE would silently skip showering/hadronization,
        so that path is deliberately not offered.
      - output_root: output ROOT file path (default "delphes/delphes.root")
      - card: detector card. A bare name (e.g. "delphes_card_CMS.tcl",
        the default, or "delphes_card_ATLAS.tcl") resolves against the
        Delphes install's cards/; a path resolves inside the sandbox
        for custom cards.
      - overwrite: replace an existing output file (default False —
        Delphes itself refuses to overwrite).
      - timeout_s: subprocess wall-clock bound (default 3600).

    Output:
      JSON string: {"status": "ok", "output_root", "card",
                    "n_events": <int|null>, "runtime_s": <float>}
    """

    hepmc_path: str = RuntimeField(description="Path to the input HepMC3 ASCII file")
    output_root: str = RuntimeField(default="delphes/delphes.root",
                                    description="Output ROOT file path")
    card: str = RuntimeField(default="delphes_card_CMS.tcl",
                             description="Detector card: bare name from the Delphes "
                                         "cards/ dir, or a sandbox-relative custom card")
    overwrite: bool = RuntimeField(default=False,
                                   description="Replace an existing output file")
    timeout_s: int = RuntimeField(default=3600,
                                  description="Subprocess timeout in seconds")

    delphes_path: str = StateField(default="", description="Built Delphes install dir "
                                                           "(contains DelphesHepMC3 and cards/)")

    def _resolve_card(self, delphes_dir: str) -> str:
        """Bare card names come from the install's cards/; anything with
        a path separator is a sandbox-local custom card."""
        if os.sep in self.card or "/" in self.card:
            local = self._safe_path(self.card)
            if local is None:
                raise ValueError(f"card path escapes base_directory: {self.card}")
            if not os.path.isfile(local):
                raise FileNotFoundError(f"custom card not found: {self.card}")
            return local
        stock = os.path.join(delphes_dir, "cards", self.card)
        if not os.path.isfile(stock):
            available = sorted(
                f for f in os.listdir(os.path.join(delphes_dir, "cards"))
                if f.startswith("delphes_card") and f.endswith(".tcl")
            )
            raise FileNotFoundError(
                f"card {self.card!r} not in {delphes_dir}/cards. "
                f"Available: {available}"
            )
        return stock

    def _run(self) -> str:
        try:
            self._setup()
            delphes_dir = _resolve_delphes_dir(self.delphes_path)
            card = self._resolve_card(delphes_dir)
        except Exception as e:
            return self.format_error(error="Setup Error", reason=str(e),
                                     suggestion="Configure delphes_path / DELPHES_DIR "
                                                "and check the card name")

        src = self._safe_path(self.hepmc_path)
        dst = self._safe_path(self.output_root)
        if src is None:
            return self.format_error(error="Access Denied",
                                     reason="hepmc_path is outside allowed base_directory",
                                     context=self.hepmc_path)
        if dst is None:
            return self.format_error(error="Access Denied",
                                     reason="output_root escapes base_directory",
                                     context=self.output_root)
        if not os.path.exists(src):
            return self.format_error(error="File Not Found",
                                     reason="HepMC input not found", context=self.hepmc_path,
                                     suggestion="Produce one with PythiaFromRunCardTool "
                                                "(output_format='hepmc3') or ConvertEventsTool")
        if os.path.exists(dst):
            if not self.overwrite:
                return self.format_error(
                    error="Output Exists",
                    reason=f"{self.output_root} already exists and Delphes refuses to overwrite",
                    suggestion="Pass overwrite=True or choose another output_root")
            os.remove(dst)
        os.makedirs(os.path.dirname(dst) or self.base_directory, exist_ok=True)

        binary = os.path.join(delphes_dir, "DelphesHepMC3")
        t0 = time.time()
        try:
            proc = subprocess.run(
                [binary, card, dst, src],
                cwd=self.base_directory, capture_output=True, text=True,
                timeout=int(self.timeout_s),
            )
        except subprocess.TimeoutExpired:
            return self.format_error(error="Timeout",
                                     reason=f"Delphes exceeded {self.timeout_s}s",
                                     context=self.hepmc_path)
        except Exception as e:
            # Launch failures (binary missing / not executable / OOM at spawn)
            # would otherwise raise uncaught and crash the tool session.
            return self.format_error(
                error="Delphes Launch Error",
                reason=str(e),
                context=f"binary={binary}",
                suggestion="Verify DelphesHepMC3 is built and executable at delphes_path")
        if proc.returncode != 0 or not os.path.exists(dst):
            # Surface Delphes' own diagnostic: prefer error-looking lines over
            # progress chatter ("** N events processed"), across both streams.
            combined = "\n".join(s for s in (proc.stderr, proc.stdout) if s).strip()
            lines = combined.splitlines()
            err_lines = [l for l in lines if re.search(
                r"error|abort|exception|fatal|cannot|no such|segmentation|illegal|invalid|fail",
                l, re.I)]
            # Keep it lean: just the salient error lines, char-capped, so the
            # agent gets enough to fix the card without context-flooding chatter.
            diag = "\n".join((err_lines or lines)[-6:]).strip()[-500:]
            # Name the likely cause per failure mode so the agent fixes the
            # right input instead of blindly re-running the same command.
            if proc.returncode < 0:
                reason = f"DelphesHepMC3 killed by signal {-proc.returncode}"
                fix = ("A signal kill is usually an out-of-memory or a malformed "
                       "detector card (an invalid TCL module/parameter). Try a "
                       "stock card (e.g. delphes_card_CMS.tcl); if it was custom, "
                       "check its TCL syntax.")
            elif proc.returncode != 0:
                reason = f"DelphesHepMC3 exited with code {proc.returncode}"
                fix = ("A nonzero exit is usually the detector card (missing or "
                       "invalid TCL card) or the HepMC input (not a valid HepMC3 "
                       "ASCII file, or empty). Confirm the card resolves and the "
                       "HepMC was produced by ConvertEventsTool or "
                       "PythiaFromRunCardTool.")
            else:
                reason = "DelphesHepMC3 exited 0 but produced no ROOT output"
                fix = ("Delphes ran but wrote no ROOT file — almost always an "
                       "empty/zero-event HepMC input (nothing to simulate). Verify "
                       "the HepMC file is non-empty and contains events.")
            cmd = (f"DelphesHepMC3 {os.path.basename(card)} "
                   f"{os.path.relpath(dst, self.base_directory)} "
                   f"{os.path.relpath(src, self.base_directory)}")
            return self.format_error(
                error="Delphes Failed",
                reason=reason,
                context=(diag or "(no diagnostic output captured)"),
                suggestion=f"The lines above are Delphes' own diagnostics. {fix} "
                           f"Command: {cmd}.")

        matches = _PROGRESS_RE.findall(proc.stdout or "")
        n_events = int(matches[-1]) if matches else None
        return json.dumps({
            "status": "ok",
            "output_root": os.path.relpath(dst, self.base_directory),
            "card": os.path.basename(card),
            "n_events": n_events,
            "runtime_s": round(time.time() - t0, 2),
        })


class DelphesROOTToJSONLTool(_SandboxedTool):
    """
    Flatten Delphes ROOT output into reco-level JSONL (recojsonl-1.0).

    Reads the "Delphes" tree with uproot (no ROOT installation needed)
    and writes one JSON line per event with the requested reconstructed
    collections, so downstream heptapod analysis tools — and agents —
    never have to touch TClonesArrays.

    Input (runtime):
      - root_path: Delphes output ROOT file
      - jsonl_path: output JSONL path
      - collections: comma/space-separated subset of
        "jets,electrons,muons,photons,met" (default: all)

    Output:
      JSON string: {"status": "ok", "reco_jsonl", "n_events",
                    "collections": [...]}

    Schema per line:
      {"schema": "recojsonl-1.0", "event_id": <int>, "data": {
         "jets":      [{"pt","eta","phi","mass","btag","px","py","pz","E"}...],
         "electrons": [{"pt","eta","phi","charge","px","py","pz","E"}...],
         "muons":     [...], "photons": [...],
         "met":       {"met": <float>, "phi": <float>}}}

    Four-vectors are derived from (pt, eta, phi, mass) so kinematics
    tools that consume px/py/pz/E compose directly.
    """

    root_path: str = RuntimeField(description="Path to the Delphes output ROOT file")
    jsonl_path: str = RuntimeField(description="Relative path to write the reco JSONL")
    collections: str = RuntimeField(default="jets,electrons,muons,photons,met",
                                    description="Subset of jets,electrons,muons,photons,met")

    @staticmethod
    def _four_vector(pt: float, eta: float, phi: float, mass: float) -> dict:
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        e = math.sqrt(max(mass, 0.0) ** 2 + px * px + py * py + pz * pz)
        return {"px": px, "py": py, "pz": pz, "E": e}

    def _run(self) -> str:
        try:
            import uproot
        except ImportError as e:
            return self.format_error(error="Missing Dependency",
                                     reason=f"uproot is required: {e}",
                                     suggestion="pip install uproot")
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Path Error", reason=str(e))

        src = self._safe_path(self.root_path)
        dst = self._safe_path(self.jsonl_path)
        if src is None:
            return self.format_error(error="Access Denied",
                                     reason="root_path is outside allowed base_directory",
                                     context=self.root_path)
        if dst is None:
            return self.format_error(error="Access Denied",
                                     reason="jsonl_path escapes base_directory",
                                     context=self.jsonl_path)
        if not os.path.exists(src):
            return self.format_error(error="File Not Found",
                                     reason="ROOT file not found", context=self.root_path)

        wanted = [c.strip().lower() for c in re.split(r"[,\s]+", self.collections) if c.strip()]
        unknown = [c for c in wanted if c not in (*_COLLECTIONS, "met")]
        if unknown:
            return self.format_error(error="Unknown Collection",
                                     reason=f"{unknown} not in {sorted((*_COLLECTIONS, 'met'))}")

        try:
            with uproot.open(src) as f:
                tree = f["Delphes"]
                # Delphes trees nest leaves under their collection branch
                # (uproot key "Jet/Jet.PT" for leaf "Jet.PT") — index by
                # leaf name so lookups are layout-independent.
                by_leaf = {k.split("/")[-1]: k for k in tree.keys()}
                arrays = {}

                def _load(leaf: str) -> None:
                    key = by_leaf.get(leaf)
                    if key is not None:
                        arrays[leaf] = tree[key].array(library="np")

                for cname in wanted:
                    if cname == "met":
                        for fld in ("MET", "Phi"):
                            _load(f"MissingET.{fld}")
                        continue
                    prefix, fields = _COLLECTIONS[cname]
                    for fld in fields:
                        _load(f"{prefix}.{fld}")
                n_events = int(tree.num_entries)
        except Exception as e:
            return self.format_error(error="ROOT Read Error",
                                     reason=f"{type(e).__name__}: {e}",
                                     context=self.root_path,
                                     suggestion="Is this a Delphes output file?")

        def _objects(cname: str, i: int) -> list:
            prefix, fields = _COLLECTIONS[cname]
            pts = arrays.get(f"{prefix}.PT")
            if pts is None:
                return []
            out = []
            for j in range(len(pts[i])):
                rec = {}
                for fld in fields:
                    arr = arrays.get(f"{prefix}.{fld}")
                    if arr is not None:
                        rec[fld.lower()] = float(arr[i][j])
                rec.setdefault("mass", 0.0)
                rec.update(self._four_vector(rec.get("pt", 0.0), rec.get("eta", 0.0),
                                             rec.get("phi", 0.0), rec.get("mass", 0.0)))
                out.append(rec)
            return out

        # Which expected leaves are actually present in the ROOT? A missing
        # PT leaf for a requested collection means the branch is not in the
        # tree (wrong Delphes card, or an unrecognized key layout) — distinct
        # from a branch that is present but carries zero objects. Tracking it
        # lets the empty-output error below name the real cause.
        missing_branches = []
        for cname in wanted:
            if cname == "met":
                if arrays.get("MissingET.MET") is None:
                    missing_branches.append("MissingET.MET")
            else:
                prefix, _ = _COLLECTIONS[cname]
                if arrays.get(f"{prefix}.PT") is None:
                    missing_branches.append(f"{prefix}.PT")

        counts = {c: 0 for c in wanted}
        os.makedirs(os.path.dirname(dst) or self.base_directory, exist_ok=True)
        with open(dst, "w") as out:
            for i in range(n_events):
                data = {}
                for cname in wanted:
                    if cname == "met":
                        met_arr = arrays.get("MissingET.MET")
                        phi_arr = arrays.get("MissingET.Phi")
                        if met_arr is not None and len(met_arr[i]):
                            met_val = float(met_arr[i][0])
                            data["met"] = {"met": met_val,
                                           "phi": (float(phi_arr[i][0])
                                                   if phi_arr is not None else 0.0)}
                            if met_val > 0:
                                counts["met"] += 1
                    else:
                        objs = _objects(cname, i)
                        data[cname] = objs
                        counts[cname] += len(objs)
                out.write(json.dumps({
                    "schema": RECO_SCHEMA_VERSION,
                    "event_id": i,
                    "data": data,
                }) + "\n")

        # Loud-fail on a silently-empty conversion. A real reconstructed
        # sample of a hadronic signal carries objects (jets at least) in
        # nearly every event, so zero objects across ALL requested
        # collections and ALL events is a broken conversion, not physics —
        # most often the Delphes card did not define these branches, the
        # input HepMC was empty, or the ROOT leaf layout was unrecognized.
        # Returning success here is what previously let an empty reco flow
        # downstream undetected and pushed the agent to fabricate.
        total_objects = sum(counts.values())
        if n_events > 0 and total_objects == 0:
            return self.format_error(
                error="Empty Reconstruction",
                reason=(f"converted {n_events} events but extracted 0 objects "
                        f"across all requested collections {wanted} "
                        f"(per-collection counts {counts})."),
                context=(f"ROOT branches not found: {missing_branches}"
                         if missing_branches else
                         "expected branches were present but carried no objects"),
                suggestion=(
                    "This reco has nothing to analyze. Confirm the Delphes card "
                    "defines Jet/Photon/Electron/Muon/MissingET branches, that the "
                    "showered HepMC fed to Delphes was non-empty, and that this is "
                    "a Delphes output ROOT. Do NOT histogram this file and do NOT "
                    "substitute hand-generated events — fix the upstream step."))

        return json.dumps({
            "status": "ok",
            "reco_jsonl": os.path.relpath(dst, self.base_directory),
            "n_events": n_events,
            "collections": wanted,
            # Object-count manifest: a per-collection sanity signal so the
            # caller can confirm reconstruction actually populated objects
            # (e.g. notice 0 photons before building a photon signal region)
            # instead of discovering it only after an empty histogram.
            "object_counts": counts,
            "missing_branches": missing_branches,
        })
