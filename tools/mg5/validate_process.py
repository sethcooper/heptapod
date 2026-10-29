"""
# validate_process.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import os
import re
import json
import shutil
import tempfile
import subprocess
from typing import Optional, List

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

# Reuse the heavy lifting already in the MadGraph tool: exec resolution, the
# generate/process error classifier, log distillation, and param_card parsing.
from .mg5 import (
    _resolve_mg5_exec, _classify_mg5_log, _distill_mg5_log, _ANSI_RE,
)

# "Total: N processes with M diagrams" (the symbolic-stage summary MG5 prints
# after `generate`/`output`). Also the per-subprocess "Process ...: K diagrams".
_DIAG_TOTAL_RE = re.compile(r"(\d+)\s+processes?\s+with\s+(\d+)\s+diagram", re.I)
_PROC_LINE_RE = re.compile(r"^\s*(\d+)\s+processes?", re.I)

# MG5 names of COLOUR-CHARGED particles (quarks, gluon, jet, gluino, squarks).
# Anchored token match so 'b1' but not 'b1' inside another token. Everything
# else (W/Z/photon/Higgs/leptons/neutralinos/charginos/gravitino) is a colour
# singlet. Used to flag a coloured particle decaying to an all-colourless final
# state — physically impossible (colour cannot vanish) and un-showerable.
_COLOURED_TOKEN_RE = re.compile(
    r"^(g|j|p|q|go|[udscbt]|[udscb][lr]|[tb][12])~?$", re.I)


def _colour_inconsistent_decays(process: str) -> list:
    """Find ``(parent > products)`` decays whose parent is colour-charged but
    whose products are all colour singlets — the gluino->W+chargino class that
    compiles (via an off-shell squark) yet leaves unbalanced colour, so Pythia
    cannot construct the colour flow and the shower aborts. Returns a list of the
    offending decay strings. Generic: keys on colour, not any process."""
    bad = []
    for m in re.finditer(r"\(([^()]*?)>([^()]*)\)", process):
        parent = m.group(1).strip().split()
        prods = m.group(2).strip().split()
        if not parent:
            continue
        parent_coloured = any(_COLOURED_TOKEN_RE.match(t) for t in parent)
        prod_coloured = any(_COLOURED_TOKEN_RE.match(t) for t in prods)
        if parent_coloured and prods and not prod_coloured:
            bad.append(m.group(0).strip())
    return bad


class ValidateProcessTool(BaseTool):
    """
    FAST (seconds) symbolic check of a MadGraph process + spectrum WITHOUT
    running event generation. Use this to converge on the right process/decay
    chain and confirm masses BEFORE committing to a multi-minute production run.

    It runs only MadGraph's `import model` + `generate` + `output` (the cheap
    symbolic stage), never `launch`. That stage is exactly where setup errors
    surface — an unknown particle name, a decay with no Feynman diagrams, or a
    typo'd model — so you get the same diagnostics you would otherwise wait
    ~40 minutes for, in seconds. It does NOT compute a cross-section or events
    (use MadGraphFromRunCard for that once this passes).

    It decides nothing about physics: it only reports whether YOUR requested
    process compiles in the model, how many diagrams each subprocess has (0 =
    your decay/process is empty), and the realized BSM spectrum after your
    masses are applied. The choice of process and spectrum remains yours.

    Inputs (runtime):
      - process: the generate line(s), e.g.
        "generate p p > go go, (go > q q~ n1)"  — or multiple lines separated by
        newlines (the first is `generate`, the rest become `add process`).
      - model: the model to import (default "MSSM_SLHA2"); a bare stock name is
        resolved by MG5, or a sandbox-relative UFO path.

    (Masses do not affect whether a process compiles or its diagram count, so
    they are not needed here — confirm the realized spectrum from the
    MadGraphFromRunCard `spectrum_bsm` echo on the actual generation run.)

    Output (JSON): {valid, n_subprocesses, total_diagrams,
    error/reason/suggestion (on failure), distilled_log}.
    """
    # --------------------------- Runtime fields --------------------------- #
    process: str = RuntimeField(
        description="MadGraph generate line(s), e.g. 'generate p p > go go, (go > q q~ n1)'. Multiple lines: first is generate, the rest become add process.")
    model: str = RuntimeField(
        default="MSSM_SLHA2",
        description="Model to import (bare stock name like 'MSSM_SLHA2'/'sm', or a sandbox-relative UFO path).")
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    mg5_path: str = StateField(description="Absolute path to top-level MG5_aMC install dir containing bin/mg5_aMC")
    base_directory: str = StateField(default=".", description="Base sandbox root")
    # ---------------------------------------------------------------------- #

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)

    def _model_arg(self) -> str:
        """The model as MG5 should see it.

        MG5 runs in a temporary directory, so a sandbox-relative UFO path
        ('inputs/model') must be made absolute against base_directory or MG5
        reports 'Path inputs/model is not a valid pathname'.  A restriction
        suffix ('inputs/model-no_b_mass') is kept.  Bare stock names and
        absolute paths pass through unchanged.
        """
        model = str(self.model).strip()
        if not model or os.path.isabs(model):
            return model
        base = os.path.abspath(self.base_directory)
        cand = os.path.join(base, model)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
        head, sep, tail = model.rpartition("-")
        if sep and head and os.path.isdir(os.path.join(base, head)):
            return os.path.abspath(os.path.join(base, head)) + "-" + tail
        return model

    def _build_card(self, outdir: str) -> str:
        lines = [f"import model {self._model_arg()}"]
        raw = [l.strip() for l in str(self.process).splitlines() if l.strip()]
        if not raw:
            raw = [str(self.process).strip()]
        for i, pl in enumerate(raw):
            body = re.sub(r"^(generate|add process)\s+", "", pl, flags=re.I)
            lines.append(("generate " if i == 0 else "add process ") + body)
        lines.append(f"output {outdir}")
        return "\n".join(lines) + "\n"

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Path Error", reason=str(e))

        try:
            mg5_exec = _resolve_mg5_exec(self.mg5_path)
        except Exception as e:
            return self.format_error(error="Dependency Missing", reason=str(e),
                                     suggestion="Pass a valid mg5_path pointing to MG5_aMC_v*/")

        work = tempfile.mkdtemp(prefix="validate_", dir=self.base_directory)
        try:
            proc_out = os.path.join(work, "PROC_validate")
            card_path = os.path.join(work, "validate_command.txt")
            log_path = os.path.join(work, "validate.log")
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(self._build_card(proc_out))

            with open(log_path, "w", encoding="utf-8") as logfp:
                completed = subprocess.run(
                    [mg5_exec, card_path], cwd=work,
                    stdout=logfp, stderr=subprocess.STDOUT, text=True, timeout=300)

            # A recognized generate/process failure → structured, actionable.
            classed = _classify_mg5_log(log_path)
            output_made = os.path.isdir(os.path.join(proc_out, "SubProcesses"))
            if classed is not None or completed.returncode != 0 or not output_made:
                diag = _distill_mg5_log(log_path)
                if classed is not None:
                    return json.dumps({
                        "valid": False,
                        "error": classed["error"],
                        "reason": classed["reason"].replace("the model", f"model '{self.model}'"),
                        "suggestion": classed["suggestion"],
                        "distilled_log": diag,
                    }, separators=(",", ":"), ensure_ascii=False)
                return json.dumps({
                    "valid": False,
                    "error": "Process Did Not Compile",
                    "reason": "MadGraph did not produce a process (no SubProcesses/ "
                              "output) — the generate/output stage failed.",
                    "suggestion": "Read the distilled log; fix the named cause "
                                  "(particle name, coupling order, or syntax) before generating.",
                    "distilled_log": diag,
                }, separators=(",", ":"), ensure_ascii=False)

            # Success: parse diagram counts + realized spectrum.
            try:
                logtxt = _ANSI_RE.sub("", open(log_path, errors="replace").read())
            except Exception:
                logtxt = ""
            n_sub = total_diag = 0
            m = _DIAG_TOTAL_RE.search(logtxt)
            if m:
                n_sub, total_diag = int(m.group(1)), int(m.group(2))

            result = {
                "valid": True,
                "n_subprocesses": n_sub,
                "total_diagrams": total_diag,
                "note": ("Process compiles. This is a SYMBOLIC check only — no "
                         "events or cross-section. Run MadGraphFromRunCard to "
                         "generate, then confirm its echoed spectrum_bsm matches "
                         "your benchmark point before the full production run."),
            }
            warnings = []
            if total_diag == 0:
                warnings.append(
                    "0 diagrams: the process/decay is EMPTY in this model even "
                    "though it parsed — a decay vertex likely does not exist "
                    "(e.g. a third-gen squark decaying to a light quark). Fix the "
                    "decay/process before generating; it will produce no events.")
            # Colour-flow check: compiles but a coloured parent -> all-colourless
            # final state cannot be showered (the gluino->W+chargino class).
            bad = _colour_inconsistent_decays(str(self.process))
            if bad:
                warnings.append(
                    "COLOUR-INCONSISTENT decay(s) " + "; ".join(bad) + ": a coloured "
                    "particle (gluino/squark/quark) is decaying to an all-colourless "
                    "final state. This may compile via an off-shell mediator but the "
                    "LHE will NOT shower (Pythia can't build the colour flow) — you are "
                    "almost certainly missing the quark jets, e.g. write `go > q q~ x1+` "
                    "(3-body) not `go > w+ x1-`.")
            if warnings:
                result["warning"] = warnings[0]
                result["warnings"] = warnings
            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except subprocess.TimeoutExpired:
            return self.format_error(error="Timeout",
                                     reason="symbolic validation exceeded 300s (unusual — a real generate/output is fast)")
        finally:
            shutil.rmtree(work, ignore_errors=True)
