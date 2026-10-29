"""
# harvest_forward_flux.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import math
import os
import random
from typing import Dict, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tools import event_io

SCHEMA_VERSION = "parentflux-1.0"
MANIFEST_SCHEMA_VERSION = "parentflux-manifest-1.0"
WEIGHT_CONVENTION = "per_inelastic_collision"


class HarvestForwardFluxTool(BaseTool):
    """
    Harvest a forward parent-hadron flux from a Pythia event sample (the
    events.root or events.jsonl PythiaFromRunCardTool writes): select the declared parent
    species in a forward angular window and write one weighted flux file per
    species. Step 1 of the reach chain
    `pythia -> HARVEST -> MesonDecayToLLP -> DecayInVolume`. Parents must be
    made stable in the run card (`<pid>:mayDecay = off`) so their production
    momenta are harvested directly; the parent -> LLP decay is applied
    downstream. Model- and experiment-agnostic: the species (`parents`),
    forward window (`theta_max_rad`), and normalization (generated + inelastic
    cross section) are all inputs, so a collider-forward and a beam-dump
    sample differ only in the run card and these values.

    Weight convention: each parent carries
        w = (sigma_gen_mb / (n_gen * sigma_inel_mb)) * prescale * cap_factor,
    so summed weights give the MEAN NUMBER of that parent per inelastic
    collision (`weight_per_collision`); downstream N_int multiplies these.

    Output: one file per species `output_dir/parents_<name>.jsonl` (schema
    "parentflux-1.0") of {parent, E, px, py, pz [GeV], weight_per_collision},
    plus `output_dir/harvest_manifest.json`. A species with no accepted
    parents still gets an empty file so the chain stays uniform.
    """

    pythia_events_path: str = RuntimeField(
        description="Pythia event dataset from PythiaFromRunCardTool "
                    "(events.root, or events.jsonl in the evtjsonl-1.0 schema)")
    parents: Dict[str, str] = RuntimeField(
        description="Map PID -> channel name (positive PID; sign folded), "
                    "e.g. {'321':'K','411':'D','431':'Ds'}")
    theta_max_rad: float = RuntimeField(
        description="Forward half-angle [rad]; keep parents with theta < this")
    sigma_inel_mb: float = RuntimeField(
        description="Inelastic cross section [mb] for per-collision weights")
    sigma_gen_mb: float = RuntimeField(
        default=0.0,
        description="Generated cross section [mb] of this sample "
                    "(0 => read from manifest_path)")
    n_gen: int = RuntimeField(
        default=0,
        description="Number of generated events (0 => manifest, else counted)")
    manifest_path: str = RuntimeField(
        default="",
        description="Optional PythiaFromRunCardTool manifest.json to read "
                    "sigmaGen_mb and n_events_written from")
    prescale: Dict[str, int] = RuntimeField(
        default={},
        description="Optional map name -> keep-1-in-N factor for abundant "
                    "species (weight compensated so the sum is preserved)")
    fold_hemispheres: bool = RuntimeField(
        default=True,
        description="Fold both beam directions into the forward window using "
                    "|pz| (default true). Both cones are harvested and stored "
                    "as forward-going, which DOUBLES the sample; each parent "
                    "therefore carries HALF the per-collision weight, so the "
                    "summed weight still equals the mean number of parents per "
                    "collision entering ONE forward cone -- which is what a "
                    "single-arm detector sees. Set false to keep only pz>0.")
    cap: int = RuntimeField(
        default=100000,
        description="Max harvested parents per species (default 100000; 0 = no "
                    "cap). An abundant species is subsampled to `cap` with a "
                    "compensating weight, so the summed weight is preserved but "
                    "the downstream record count stays bounded.")
    seed: int = RuntimeField(
        default=1,
        description="RNG seed for the cap subsampling (deterministic)")
    output_dir: str = RuntimeField(
        description="Relative directory for the per-species flux files")
    base_directory: str = StateField(
        default=".", description="Base directory for safe paths")

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _run(self) -> str:
        src = self._safe_path(self.pythia_events_path)
        out_dir = self._safe_path(self.output_dir)
        man = self._safe_path(self.manifest_path) if self.manifest_path else None
        if not src or not out_dir or (self.manifest_path and not man):
            return self.format_error(
                error="Access Denied",
                reason="pythia_events_path, output_dir or manifest_path "
                       "escapes base_directory",
                suggestion="Use relative paths inside base_directory")
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found", reason=f"events not found: {src}",
                suggestion="Provide the PythiaFromRunCardTool events.jsonl")

        # --------------------------- validate -------------------------- #
        theta_max = float(self.theta_max_rad)
        sigma_inel = float(self.sigma_inel_mb)
        if theta_max <= 0.0:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"theta_max_rad must be positive (got {theta_max})",
                suggestion="Provide a forward half-angle in radians")
        if sigma_inel <= 0.0:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"sigma_inel_mb must be positive (got {sigma_inel})",
                suggestion="Provide the inelastic cross section in mb")
        if not self.parents:
            return self.format_error(
                error="Invalid Parameter",
                reason="parents map is empty",
                suggestion="Declare PID -> name, e.g. {'321':'K'}")
        try:
            pid_to_name = {abs(int(k)): str(v) for k, v in self.parents.items()}
            # Reverse map so each flux record can carry its |PDG id|; downstream
            # (MesonDecayToLLP) uses it to auto-set the SM decay-in-flight length.
            name_to_pid = {name: pid for pid, name in pid_to_name.items()}
        except (TypeError, ValueError) as e:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"parents keys must be integer PIDs ({e})",
                suggestion="Use string integer keys, e.g. {'321':'K'}")
        prescale = {str(k): max(1, int(v))
                    for k, v in (self.prescale or {}).items()}

        # ------------- cross section + event count (manifest) ---------- #
        sigma_gen = float(self.sigma_gen_mb or 0.0)
        n_gen = int(self.n_gen or 0)
        if man is not None:
            try:
                with open(man) as fh:
                    mdoc = json.load(fh)
                outs = mdoc.get("outputs", {})
                if sigma_gen <= 0.0:
                    sigma_gen = float(outs.get("xsec", {}).get("sigmaGen_mb",
                                                               0.0))
                if n_gen <= 0:
                    n_gen = int(outs.get("n_events_written", 0))
            except Exception as e:
                return self.format_error(
                    error="Manifest Error", reason=str(e),
                    suggestion="manifest_path must be a PythiaFromRunCardTool "
                               "manifest.json")
        if sigma_gen <= 0.0:
            return self.format_error(
                error="Invalid Parameter",
                reason="sigma_gen_mb is unknown (not given and not in a "
                       "manifest)",
                suggestion="Provide sigma_gen_mb or a manifest_path with "
                           "outputs.xsec.sigmaGen_mb")
        # The per-collision weight normalizes to sigma_inel (the TOTAL pp
        # inelastic cross-section), so it must be the SAME across every sample
        # that will be combined downstream (soft QCD, charm, onia, ...). A
        # generated sub-process can never exceed the total inelastic, so
        # sigma_inel < sigma_gen means sigma_inel is wrong -- the classic slip is
        # passing THIS sample's process cross-section as sigma_inel, which makes
        # the per-collision weights incoherent when the parent fluxes are merged.
        if sigma_inel < sigma_gen:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"sigma_inel_mb ({sigma_inel}) < sigma_gen_mb "
                       f"({sigma_gen}): the total inelastic cross-section cannot "
                       f"be smaller than the generated sub-process.",
                suggestion="sigma_inel_mb is the TOTAL pp inelastic cross-section "
                           "(the same value for every sample), NOT this sample's "
                           "process cross-section. Use one consistent "
                           "sigma_inel_mb across all harvests you will combine.")

        names = sorted(set(pid_to_name.values()))
        buffers = {name: [] for name in names}
        seen = {name: 0 for name in names}          # for prescaling
        n_events = 0
        fold = bool(self.fold_hemispheres)

        # ----------------------- scan the events ----------------------- #
        # Read through the event IO layer, in chunks, so either format works
        # and a sample larger than memory is still a single pass. The columnar
        # layout spells energy "e"; the evtjsonl-1.0 schema spelled it "E".
        try:
            for chunk in event_io.iter_events(src):
                for parts in chunk["prt"].to_list():
                    n_events += 1
                    for p in parts:
                        pid = abs(int(p["id"]))
                        name = pid_to_name.get(pid)
                        if name is None:
                            continue
                        px, py, pz = float(p["px"]), float(p["py"]), float(p["pz"])
                        pz_used = abs(pz) if fold else pz
                        pmag = math.sqrt(px * px + py * py + pz * pz)
                        if pmag <= 0.0:
                            continue
                        theta = math.acos(max(-1.0, min(1.0, pz_used / pmag)))
                        if theta > theta_max:
                            continue
                        ps = prescale.get(name, 1)
                        if ps > 1:
                            seen[name] += 1
                            if seen[name] % ps != 0:
                                continue
                        buffers[name].append(
                            (float(p["e"]), px, py, pz_used))
        except KeyError as e:
            return self.format_error(
                error="Event Format Error",
                reason=f"event record missing field {e}",
                suggestion="Expected a PythiaFromRunCardTool dataset whose "
                           "particles carry id/px/py/pz/E")
        except Exception as e:
            return self.format_error(
                error="Read Error", reason=str(e),
                suggestion="Verify the events file is a .root or .jsonl "
                           "dataset from PythiaFromRunCardTool")

        if n_gen <= 0:
            n_gen = n_events
        if n_gen <= 0:
            return self.format_error(
                error="Invalid Parameter",
                reason="n_gen is zero (no events and none provided)",
                suggestion="Provide n_gen or a non-empty events file")

        # per-inelastic-collision base weight
        w_per = sigma_gen / (n_gen * sigma_inel)

        # Folding harvests BOTH beam cones and stores them all as forward, so
        # the sample is doubled and each parent must carry half the weight --
        # exactly as `prescale` and `cap` carry their own compensating factors.
        # Without this the flux, and every yield derived from it, is 2x too
        # large: a single-arm detector sees one cone, and a genuinely backward
        # parent can never reach it.
        fold_factor = 0.5 if fold else 1.0

        # --------------------------- write ----------------------------- #
        # Cap abundant species to bound downstream record volume: subsample
        # to `cap` and inflate the weight by n_total/cap so the summed weight
        # (mean parents per collision) is preserved exactly.
        cap = max(0, int(self.cap))
        rng = random.Random(int(self.seed))
        try:
            os.makedirs(out_dir, exist_ok=True)
            species = []
            for name in names:
                rows = buffers[name]
                n_total = len(rows)
                cap_factor = 1.0
                capped = bool(cap and n_total > cap)
                if capped:
                    rows = [rows[i] for i in rng.sample(range(n_total), cap)]
                    cap_factor = n_total / float(cap)
                w = w_per * prescale.get(name, 1) * cap_factor * fold_factor
                fpath = os.path.join(out_dir, f"parents_{name}.jsonl")
                with open(fpath, "w") as fh:
                    for (E, px, py, pz) in rows:
                        fh.write(json.dumps({
                            "schema": SCHEMA_VERSION, "parent": name,
                            "parent_pid": name_to_pid.get(name),
                            "E": E, "px": px, "py": py, "pz": pz,
                            "weight_per_collision": w},
                            separators=(",", ":")) + "\n")
                species.append({
                    "name": name,
                    "path": os.path.relpath(fpath, self.base_directory),
                    "n_parents": len(rows),
                    "n_total": n_total,
                    "capped": capped,
                    "weight_per_collision": w,
                    "prescale": prescale.get(name, 1),
                    "sum_weights": w * len(rows),
                })
        except Exception as e:
            return self.format_error(
                error="Write Error", reason=str(e),
                suggestion="Verify disk space and permissions")

        manifest = {
            "schema": MANIFEST_SCHEMA_VERSION,
            "weight_convention": WEIGHT_CONVENTION,
            "weight_definition": "weight_per_collision = (sigma_gen_mb / "
                                 "(n_gen * sigma_inel_mb)) * prescale * "
                                 "fold_factor * "
                                 "cap_factor (n_total/cap when subsampled); "
                                 "sum over parents = mean parents per "
                                 "inelastic collision",
            "pythia_events_path": os.path.relpath(src, self.base_directory),
            "sigma_gen_mb": sigma_gen,
            "sigma_inel_mb": sigma_inel,
            "n_gen": n_gen,
            "n_events_read": n_events,
            "theta_max_rad": theta_max,
            "fold_hemispheres": fold,
            "cap": cap,
            "w_per_collision_base": w_per,
            "species": species,
        }
        man_path = os.path.join(out_dir, "harvest_manifest.json")
        with open(man_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        result = {
            "status": "ok",
            "n_events_read": n_events,
            "n_gen": n_gen,
            "sigma_gen_mb": sigma_gen,
            "sigma_inel_mb": sigma_inel,
            "w_per_collision_base": w_per,
            "species": species,
            "output_dir": os.path.relpath(out_dir, self.base_directory),
            "manifest_path": os.path.relpath(man_path, self.base_directory),
            "weight_convention": WEIGHT_CONVENTION,
            "note": (f"per-collision weights are normalized to "
                     f"sigma_inel_mb={sigma_inel}; use this SAME value for every "
                     f"other sample you will merge (soft QCD, charm, onia) or the "
                     f"combined flux weights will be incoherent."),
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
