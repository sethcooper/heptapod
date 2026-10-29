"""
# feynrules.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
FeynRules model tools.

FeynRulesToUFOTool compiles a ``.fr`` model to a UFO directory.
ValidateLagrangianTool runs the same FeynRules consistency checks without
writing a UFO, which is the faster loop while a model is still being edited --
``WriteUFO`` is the expensive half.

Both drive ``UFO_generator.wl`` through wolframscript and share
:func:`_invoke_generator`.  Neither assumes anything about the model: the
Lagrangian expression and whether to load the Standard Model alongside the
model file are both caller-supplied, because a wrong guess at either produces a
UFO that is silently wrong rather than one that fails.
"""

import os
import json
import re
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .lagrangian_checks import (
    parse_check_verdicts,
    parse_gauge,
    parse_mass_spectrum,
    parse_residual_fields,
    residual_fields_message,
    failed_checks,
    summarise,
)
from tools.ufo import is_ghost, is_goldstone, parse_particles, parse_vertices
from tools.logging.findings import append_finding

SCHEMA_VERSION = "tool-1.0"

# A UFO missing any of these is scaffolding, not a model. WriteUFO emits its
# boilerplate regardless of whether vertex extraction produced anything, so the
# generator asserts on these too; the check is repeated here because "the
# process exited 0" has never been sufficient evidence that it worked.
REQUIRED_UFO_FILES = (
    "particles.py",
    "vertices.py",
    "couplings.py",
    "parameters.py",
    "coupling_orders.py",
)

# Emitted by UFO_generator.wl when the Lagrangian is missing or misnamed, so the
# tool can hand the caller the symbols the model actually defines.
_CANDIDATES_MARKER = "[INFO] Lagrangian-like symbols defined by this model:"

_LAGRANGIAN_RE = re.compile(
    r"^HEPTAPOD-LAGRANGIAN\|expression=(?P<expr>[^|]*)\|"
    r"symbols=(?P<syms>[^|]*)\|terms=(?P<terms>\d+)",
    re.MULTILINE,
)


def _resolved_lagrangian(stdout: str) -> Optional[dict]:
    """What the Lagrangian expression actually became.

    The expression is caller text and is never guessed, which means a typo or
    an omitted term is silent: a total that excludes the Higgs potential
    compiles, passes every check, and produces a UFO missing a sector, with
    nothing in any output distinguishing it from the right one.
    """
    m = _LAGRANGIAN_RE.search(stdout or "")
    if not m:
        return None
    syms = [s for s in m.group("syms").split(",") if s]
    return {
        "expression": m.group("expr"),
        "symbols": syms,
        "n_terms": int(m.group("terms")),
    }

# Standard Model PDG codes. Classification goes by code rather than by name:
# a model may call its Higgses h1/h2 where SM.fr calls one H, and antiparticles
# appear as G__minus__ against a G__plus__ declaration, so name matching
# misclassifies routinely -- an early version of this call read 89 of the
# reference's 104 vertices as new physics because it did not recognise h1.
_SM_PDG = set(range(1, 7)) | set(range(11, 17)) | {21, 22, 23, 24, 25}


def _vertex_inventory(ufo_dir: str, mode: str) -> Optional[dict]:
    """What the model actually interacts through.

    A UFO's interactions are its content, and reading them meant parsing
    vertices.py against particles.py by hand -- an agent reviewing its own
    model wrote a Lorentz-structure decoder to do it, and called that the most
    valuable thing it built. The counts go in every result; the new-physics
    vertices go in full, because there are few of them beside the Standard
    Model ones and they are the part anyone is actually checking.

    Gauge artefacts are not new physics. Ghosts and Goldstones carry
    non-standard PDG codes and would otherwise be counted as new states, which
    would make every Feynman-gauge model look like it invented a sector.

    Args:
        ufo_dir: Directory to read.
        mode: "none", "new_physics" (default) or "full".

    Returns:
        Counts plus the vertex list, or None when disabled.
    """
    if mode == "none":
        return None
    verts = parse_vertices(ufo_dir)
    if not verts:
        return {"total": 0, "sm_only": 0, "new_physics": 0,
                "listed": mode, "vertices": []}

    artefact = {}
    for rec in parse_particles(os.path.join(ufo_dir, "particles.py")):
        var = str(rec.get("var") or rec.get("name") or "")
        artefact[var] = is_goldstone(rec) or is_ghost(rec)

    def _is_new(v: dict) -> bool:
        for name, pdg in zip(v["particles"], v["pdg"]):
            if artefact.get(name):
                continue
            if pdg is None or abs(int(pdg)) not in _SM_PDG:
                return True
        return False

    def _is_artefact(v: dict) -> bool:
        return any(artefact.get(n) for n in v["particles"])

    new = [v for v in verts if _is_new(v)]
    keep = verts if mode == "full" else new
    n_art = sum(1 for v in verts if _is_artefact(v))
    return {
        "total": len(verts),
        "sm_only": len(verts) - len(new),
        "new_physics": len(new),
        # Physics and bookkeeping, counted apart. A Feynman-gauge model's
        # Goldstone and ghost vertices arrive in the same list as its real
        # interactions and look identical: `ghWp ghWp~ Zp` reads exactly like
        # `u u~ Zp`. One author reviewed 97 undifferentiated entries, could not
        # see that most were artefacts, and learned the model's gauge only
        # afterwards. The counts make it one glance.
        "physical": len(verts) - n_art,
        "gauge_artifact": n_art,
        "listed": mode,
        "vertices": [
            {"particles": v["particles"], "pdg": v["pdg"], "lorentz": v["lorentz"],
             "kind": "gauge_artifact" if _is_artefact(v) else "physical"}
            for v in keep
        ],
    }



def _residual_check(stdout_text: Optional[str]) -> Optional[dict]:
    """The surviving-unphysical-field diagnosis, as a check record.

    Returned in the same shape as the FeynRules verdicts so it appears in
    `checks` and is counted by `summarise`. A model that loses whole Lagrangian
    terms to an unmatched replacement rule otherwise reports nothing but
    success, and this is the only signal that says so.
    """
    residual = parse_residual_fields(stdout_text)
    if residual is None:
        return None
    ok = bool(residual.get("ok"))
    return {
        "name": "no_residual_unphysical_fields",
        "label": "Unphysical fields removed by expansion",
        "verdict": "pass" if ok else "fail",
        "passed": ok,
        "pre_existing": False,
        "detail": ", ".join(str(f) for f in (residual.get("fields") or [])),
        "message": ("Unphysical fields removed by expansion: passed."
                    if ok else residual_fields_message(residual)),
    }



# Header stamped into the generated vertices.py. WriteUFO does not record the
# coupling dict's key order anywhere, and it is not guessable: a vertex with one
# colour structure and two Lorentz structures -- a fermion current split into
# chiral pieces -- returns the wrong coupling under a transposed lookup, or
# nothing, so the model reads as though its chirality were broken rather than
# as though the caller swapped an index. Two separate model builds made exactly
# that error while reading this file directly, one of them after being warned
# about it in a tool description it had already read. The warning belongs where
# the file is, not only where the tool is.
_VERTICES_HEADER = '''\
# Vertex coupling dictionaries are keyed (COLOUR index, LORENTZ index).
#
#     couplings = {(0,0): C.GC_1, (0,1): C.GC_2}
#                   |  |
#                   |  +-- index into this vertex's `lorentz` list
#                   +----- index into this vertex's `color` list
#
# NOT (lorentz, colour). A vertex with one colour structure and several Lorentz
# structures -- e.g. a fermion current split into left- and right-handed pieces
# -- silently yields the wrong coupling under the transposed reading.
#
# heptapod's InspectUFO tool reports both indices explicitly:
#     InspectUFO(ufo_dir=..., what="vertices", names=["Zp"])
# ---------------------------------------------------------------------------
'''



def _stamp_vertices_header(ufo_dir: str) -> bool:
    """Prepend the coupling-key-order note to a generated vertices.py.

    A comment block, so it changes nothing about how the file loads. Idempotent
    and best-effort: a UFO that cannot be rewritten is not a failed compile.

    Args:
        ufo_dir: The UFO directory just written.

    Returns:
        True when the header was added.
    """
    path = Path(ufo_dir) / "vertices.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "COLOUR index, LORENTZ index" in text:
        return False
    try:
        path.write_text(_VERTICES_HEADER + "\n" + text, encoding="utf-8")
    except OSError:
        return False
    return True



# Mathematica that leaks into a generated vertices.py when WriteUFO is handed a
# Lagrangian with unexpanded index structure. The affected Vertex blocks are
# unparseable, so every reader -- this toolkit, MadGraph, a grader -- silently
# sees a model with fewer interactions than it has.
_UFO_LEAK_MARKERS = ("PartNameMG", "CreateObjectParticleName")


def _vertex_leak(ufo_dir: str) -> Optional[str]:
    """Unexpanded Mathematica in the generated vertices.py, if any.

    Checked after every compile because the failure is otherwise invisible:
    the file exists, most of it parses, and the missing couplings look like a
    model that simply does not have them.

    Args:
        ufo_dir: The UFO directory just written.

    Returns:
        A diagnosis naming the marker and an affected line, or None when clean.
    """
    path = Path(ufo_dir) / "vertices.py"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for marker in _UFO_LEAK_MARKERS:
        if marker not in text:
            continue
        line = next((ln.strip() for ln in text.splitlines() if marker in ln), "")
        return (
            f"vertices.py contains unexpanded Mathematica ({marker}): "
            f"{line[:120]}. WriteUFO emits this for gauge self-couplings when "
            f"the Lagrangian reaches it with unexpanded indices; the affected "
            f"vertices are unreadable and the model is missing interactions. "
            f"Compile with expand_indices=true (the default), or pass the "
            f"Lagrangian already expanded."
        )
    return None


def _utc_now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()


def _bool_arg(name: str, value: Optional[bool]) -> list:
    """Render an optional boolean as a ``Name=true|false`` argument.

    Args:
        name: Parameter name understood by UFO_generator.wl.
        value: Tri-state -- None leaves the script's own default in place.

    Returns:
        A single-element list, or an empty one when value is None.
    """
    if value is None:
        return []
    return [f"{name}={'true' if value else 'false'}"]


def _candidate_symbols(stdout: str) -> str:
    """Pull the model's Lagrangian-like symbol names out of generator stdout.

    Args:
        stdout: Captured stdout from the wolframscript run.

    Returns:
        The comma-separated symbol list, or an empty string if absent.
    """
    for line in (stdout or "").splitlines():
        if _CANDIDATES_MARKER in line:
            return line.split(_CANDIDATES_MARKER, 1)[1].strip()
    return ""


def _descendants(pid: int) -> list:
    """PIDs descended from ``pid``, deepest last. Best effort, via ps."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,ppid"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                children.setdefault(int(parts[1]), []).append(int(parts[0]))
            except ValueError:
                continue
    found, stack = [], [pid]
    while stack:
        for kid in children.get(stack.pop(), []):
            found.append(kid)
            stack.append(kid)
    return found


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """SIGKILL the launcher and anything still running underneath it.

    ``wolframscript`` is a launcher; the process doing the work is a
    ``WolframKernel`` it spawns. Killing the launcher usually takes the kernel
    with it, but "usually" is not a property to rely on when the failure mode
    is a kernel holding a core at 100% until someone notices, so descendants
    are collected first and swept afterwards.

    Deliberately NOT ``killpg``. That would need the child in its own session,
    and a child in its own session no longer receives the signals sent to the
    host's process group -- which is the case that actually leaked here.
    """
    kids = _descendants(proc.pid)
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    for pid in kids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _invoke_generator(
    wolframscript_path: str,
    feynrules_path: Optional[str],
    model_path: str,
    output_dir: str,
    log_dir: str,
    timeout_sec: int,
    base_directory: str,
    extra_args: list,
    driver_name: str = "UFO_generator.wl",
) -> Tuple[Optional[subprocess.CompletedProcess], str, str, Optional[str]]:
    """Run UFO_generator.wl (or another driver here) and capture its logs.

    Args:
        wolframscript_path: Command or absolute path to wolframscript.
        feynrules_path: FeynRules installation root, or None to rely on the env.
        model_path: Absolute path to the .fr model file.
        output_dir: Absolute path for UFO output (unused when ChecksOnly).
        log_dir: Absolute path for the stdout/stderr logs.
        timeout_sec: Walltime limit for the Mathematica run.
        base_directory: Working directory for the subprocess.
        extra_args: Additional ``Key=value`` arguments for the script.
        driver_name: The driver script, in this directory.

    Returns:
        ``(proc, stdout_path, stderr_path, error)``. On failure to launch,
        ``proc`` is None and ``error`` describes why.
    """
    driver = Path(__file__).parent / driver_name
    if not driver.exists():
        return None, "", "", f"{driver} not found"

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stdout_path = str(Path(log_dir) / "wolframscript_stdout.log")
    stderr_path = str(Path(log_dir) / "wolframscript_stderr.log")

    env = os.environ.copy()
    if feynrules_path:
        env["FEYNRULES_PATH"] = feynrules_path

    cmd = [
        wolframscript_path, "-f", str(driver),
        f"ModelPath={model_path}",
        f"FeynRulesPath={feynrules_path or ''}",
        f"OutputDir={output_dir}",
    ] + extra_args

    # Left in the host's own session ON PURPOSE.
    #
    # A wolframscript started by this tool was found alive after six days at
    # 3% CPU, reparented to init, still holding the path of a sandbox that no
    # longer existed. Its parent had died while it ran, so no timeout ever
    # fired: the thing counting the seconds was the thing that went away.
    # Keeping the child in the host's session means a signal sent to that
    # group reaches it too, which is the only in-process protection against
    # that case. `start_new_session=True` would buy a tidy killpg and lose
    # exactly this, so it is not used.
    try:
        proc_h = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=base_directory,
            env=env,
            text=True,
        )
    except FileNotFoundError:
        return None, stdout_path, stderr_path, (
            f"wolframscript_path not found: {wolframscript_path}"
        )

    timed_out = False
    try:
        try:
            out, err = proc_h.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc_h)
            out, err = proc_h.communicate()
        except BaseException:
            # KeyboardInterrupt, SystemExit, a harness tearing the trial down:
            # the kernel must not outlive the call that started it.
            _kill_process_tree(proc_h)
            raise
    finally:
        if proc_h.poll() is None:
            _kill_process_tree(proc_h)

    # Partial output is evidence, and a run that timed out is exactly when it
    # is wanted, so the logs are written on both paths.
    Path(stdout_path).write_text(out or "")
    Path(stderr_path).write_text(err or "")
    if timed_out:
        return None, stdout_path, stderr_path, (
            f"wolframscript exceeded {timeout_sec}s (killed, with its kernel)"
        )

    proc = subprocess.CompletedProcess(
        args=cmd, returncode=proc_h.returncode, stdout=out, stderr=err)
    return proc, stdout_path, stderr_path, None


class _FeynRulesToolBase(BaseTool):
    """Shared configuration for the tools that drive UFO_generator.wl.

    Subclasses supply the arguments specific to what they are asking the script
    to do, and interpret the result; everything here is common to both.
    """

    # --------------------------- Runtime fields --------------------------- #
    model_path: Optional[str] = RuntimeField(
        default=None, description="Path to FeynRules .fr model file")
    lagrangian: Optional[str] = RuntimeField(
        default=None,
        description=(
            "Total Lagrangian expression handed to FeynRules, e.g. "
            "'LSM + LBSM', 'LNP', or 'LSM + LkinS1 + L1YukRR'. Omit to have the "
            "tool report which Lagrangian-like symbols the model defines "
            "instead of guessing one"))
    load_sm: Optional[bool] = RuntimeField(
        default=None,
        description=(
            "Load FeynRules' SM.fr alongside the model file (default true). Set "
            "false for a standalone model that does not extend the SM. "
            "ADD-ON SEMANTICS, the central question for any model built on "
            "SM.fr: the files are read in order and declarations ACCUMULATE. A "
            "class index or parameter name re-declared by the add-on does not "
            "replace SM.fr's -- SM.fr was read first and its declaration wins. "
            "To re-express an SM gauge eigenstate in a BSM basis, go through "
            "MR$Definitions rather than re-declaring the field; the rules are "
            "applied lazily by ExpandIndices. REPLACE OR PREPEND that rule -- "
            "appending does nothing, because ReplaceAll takes the first "
            "matching rule and SM.fr's own rules are already in the list. Match "
            "BOTH stored forms: FeynRules keeps each definition twice, bare "
            "(Phi[2], Wi[mu_,1]) and index-wrapped (Phi[Index[SU2D,2]], "
            "Wi[Index[Lorentz,mu_],Index[SU2W,3]]). A lone Lorentz slot absorbs "
            "the wrapper, so B[mu_] matches both, but Phi[2] and Wi[mu_,1] "
            "match only the bare form -- which is how the same idiom silently "
            "works for one field and fails for another in the same rule list. "
            "A rule that never matches fails SILENTLY: the unphysical field "
            "survives expansion, vertex extraction drops every term containing "
            "it, and the model still compiles and passes hermiticity. The "
            "no_residual_unphysical_fields check exists to catch exactly that. "
            "Internal parameters are topologically sorted before parameters.py "
            "is written, so an add-on parameter may depend on an SM.fr one "
            "regardless of declaration order"))
    feynman_gauge: Optional[bool] = RuntimeField(
        default=None,
        description=(
            "Gauge to build in. Omit to inherit whatever the model files set, "
            "which for an SM.fr add-on is FEYNMAN gauge -- SM.fr assigns "
            "FeynmanGauge = True itself, silently, and the add-on acquires a "
            "Goldstone and ghost sector with it. That is tens of extra "
            "vertices MadGraph ignores but every vertex count includes; one "
            "model carried 138 gauge-artefact vertices against 122 physical "
            "ones. Set false for unitary gauge, which is conventional for a "
            "tree-level UFO. The gauge actually used is reported in the result "
            "either way"))
    expand_indices: Optional[bool] = RuntimeField(
        default=None,
        description=(
            "Expand the Lagrangian's indices before writing the UFO (default "
            "true). WriteUFO given unexpanded SU(2) index structure emits "
            "Mathematica source where a particle name belongs, and those "
            "vertex blocks are then unreadable -- the model compiles and is "
            "silently missing its gauge self-couplings. Set false only if the "
            "expansion is too expensive or your `lagrangian` expression "
            "already does it"))
    run_checks: Optional[bool] = RuntimeField(
        default=None,
        description="Run the FeynRules consistency checks (default true)")
    checks: Optional[str] = RuntimeField(
        default=None,
        description=(
            "KEEP mass_spectrum ON. It is the only check here that can catch "
            "a wrong mass, and a model with a sign error in its mass matrix is "
            "hermitian, compiles, and has exactly the right interactions. It "
            "agrees within 0.1% (FeynRules' own tolerance) and reports "
            "declared, derived and rel_diff per state at full precision. "
            "Which checks to run: omit for the default set (hermiticity, "
            "kinetic_diagonal, mass_diagonal, mass_spectrum), 'all' to add "
            "quadratic_diagonal and kinetic_norm, or a comma-separated list. "
            "The two extras fail for the Standard Model itself, so turn on "
            "baseline_compare if you ask for them. Separately, the caveat "
            "about mixing angles applies to kinetic_diagonal and mass_diagonal "
            "ONLY: those two report a symbolic cross term that vanishes at the "
            "physical value of the angle. mass_spectrum is NOT affected -- it "
            "compares numbers, so a mixing angle cannot confuse it"))
    baseline_compare: Optional[bool] = RuntimeField(
        default=None,
        description=(
            "Also run the checks against the SM alone and mark results it "
            "already exhibits as inherited rather than as model failures "
            "(default false). Doubles check time; worth it only when running "
            "checks the SM itself fails"))
    timeout_sec: Optional[int] = RuntimeField(
        default=1200,
        description=(
            "Walltime limit for the Mathematica run. On expiry the kernel is "
            "killed along with the launcher that spawned it. Was 3600, which "
            "is long enough for a hung kernel to burn an hour of CPU before "
            "anyone notices; raise it per call for a model that needs longer"))
    log_dir: Optional[str] = RuntimeField(
        default=None,
        description=(
            "Directory for the wolframscript stdout/stderr logs. Defaults to a "
            "sibling of output_dir named '<output_dir>_logs', so the UFO "
            "directory stays exactly the file list MadGraph expects"))
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    feynrules_path: str = StateField(
        description="Path to FeynRules installation root")
    wolframscript_path: str = StateField(
        description="Command/path to wolframscript")
    base_directory: str = StateField(
        description="Base working directory for relative paths")
    # ---------------------------------------------------------------------- #

    def _abs_path(self, maybe_rel: Optional[str]) -> Optional[str]:
        """Convert a possibly relative path to an absolute one under base_directory."""
        if maybe_rel is None:
            return None
        p = Path(maybe_rel)
        if not p.is_absolute():
            p = Path(self.base_directory) / p
        return str(p.resolve())

    def _ensure_dir(self, path: str) -> None:
        """Create a directory, including parents, if it does not exist."""
        Path(path).mkdir(parents=True, exist_ok=True)

    def _common_args(self) -> list:
        """Build the generator arguments shared by both tools."""
        args = []
        if self.lagrangian:
            args.append(f"Lagrangian={self.lagrangian}")
        args += _bool_arg("LoadSM", self.load_sm)
        args += _bool_arg("FeynmanGauge", self.feynman_gauge)
        args += _bool_arg("ExpandIndices", self.expand_indices)
        # run_checks is the on/off switch; checks names which ones. Passing
        # neither leaves the generator's own default set in place.
        if self.run_checks is False:
            args.append("Checks=false")
        elif self.checks:
            args.append(f"Checks={self.checks}")
        args += _bool_arg("BaselineCompare", self.baseline_compare)
        return args

    def _lagrangian_error(self, proc: subprocess.CompletedProcess) -> Optional[str]:
        """Return a helpful message if the run failed on the Lagrangian.

        Exit code 1 covers several setup failures, but the one worth expanding
        is an absent or misnamed Lagrangian: the generator has already listed
        the symbols the model defines, and repeating them here saves the caller
        a second FeynRules load to find out.

        Args:
            proc: The completed wolframscript run.

        Returns:
            A reason string, or None if this was not a Lagrangian failure.
        """
        stdout = proc.stdout or ""
        for line in stdout.splitlines():
            if not line.startswith("Error:"):
                continue
            if "Lagrangian" not in line and "undefined symbol" not in line:
                continue
            candidates = _candidate_symbols(stdout)
            if candidates:
                return f"{line.strip()} Symbols this model defines: {candidates}"
            return line.strip()
        return None


class FeynRulesToUFOTool(_FeynRulesToolBase):
    """
    Generate a UFO model directory from a FeynRules .fr model file by calling
    Mathematica's `wolframscript`.

    The Lagrangian expression is required rather than inferred. Models name
    their total Lagrangian idiosyncratically (LBSM, LNP, LLeptoQuark, ...), and
    some define only the pieces, so pass whatever sums to the full Lagrangian --
    including the SM part if you loaded it.

    The UFO is emitted as Python 2. FeynRules 2.3.49 writes `raise UFOError,
    "..."` and similar, so `object_library.py` will not parse under Python 3 and
    a script that reads the model by importing it will fail. Read it statically
    instead -- tools/ufo parses particles and vertices with `ast` for exactly
    this reason -- or let MadGraph convert it, which it does on first import.

    Success means a complete UFO exists, not that Mathematica exited cleanly:
    WriteUFO writes its boilerplate even when vertex extraction produced
    nothing, so the output is checked for the files that carry the physics.

    Returns:
      JSON summary including paths, logs, consistency-check results, and the
      file listing of the generated UFO.
    """

    output_dir: str = RuntimeField(
        default="UFO_Output", description="Directory for UFO output")
    add_decays: Optional[bool] = RuntimeField(
        default=None,
        description=(
            "Have FeynRules compute 1->2 partial widths into decays.py. "
            "Defaults on for Wolfram < 14 and off above, where the routine is "
            "unreliable and MadGraph should compute widths from the UFO"))
    vertex_inventory: Optional[str] = RuntimeField(
        default="new_physics",
        description=(
            "Return the model's interactions in the result: 'new_physics' "
            "(default) lists every vertex involving a non-SM field plus counts, "
            "'full' lists all of them, 'none' omits it. This is the model's "
            "actual content -- what it couples to and how -- which otherwise "
            "requires parsing vertices.py against particles.py by hand"))
    strict_checks: Optional[bool] = RuntimeField(
        default=None,
        description=(
            "Abort before writing the UFO if a consistency check fails that the "
            "SM baseline passes (default false)"))

    toy_spec: Optional[Union[Dict[str, Any], str]] = RuntimeField(
        default=None,
        description=("Instead of model_path: a WriteToyModel spec (name, gauge, fields, parameters, "
                     "interactions); the .fr is written and compiled with its lagrangian and load_sm"))

    def _run(self) -> str:
        """Compile the model to a UFO and report what was produced."""
        if self.toy_spec is not None:
            from .toy_model import apply_toy_spec
            err = apply_toy_spec(self, self.toy_spec, self._abs_path("."), "ToyModel")
            if err:
                return self.format_error(error="Invalid Model Spec", reason=err)
        if not self.model_path:
            return self.format_error(
                error="Missing Parameter", reason="model_path is required")
        if not self.output_dir:
            return self.format_error(
                error="Missing Parameter", reason="output_dir is required")

        abs_model = self._abs_path(self.model_path)
        if not os.path.isfile(abs_model):
            return self.format_error(
                error="Model Not Found",
                reason=f"no file at {abs_model} (model_path is resolved against the "
                       f"sandbox, {self.base_directory})")
        if os.path.isfile(abs_model):
            # What FeynRules would misread without an error (it falls back
            # to SM.fr's fields and writes a UFO without the model's own).
            from .feynarts_model import fr_syntax_problems, TOY_TEMPLATE, template_text
            # Lines that drop part of a sum are reported with the result, not
            # refused here: existing models (the feynrules skill's own template
            # had two) compile, only without those terms.
            problems = fr_syntax_problems(Path(abs_model).read_text(errors="replace"),
                                          self.load_sm is not False, dangling=False)
            if problems:
                return self.format_error(
                    error="Invalid FeynRules Model",
                    reason="; ".join(problems) + f". A worked standalone model: {TOY_TEMPLATE}."
                           + template_text(self.load_sm is not False))
        abs_out = self._abs_path(self.output_dir)
        abs_fr = self._abs_path(self.feynrules_path) if self.feynrules_path else None
        self._ensure_dir(abs_out)

        # Beside the UFO, never inside it. A UFO directory is a deliverable
        # with a fixed expected file list: MadGraph reads it, graders read it,
        # and a _logs/ subdirectory in the middle of it is something the caller
        # then has to move out by hand. One did.
        log_dir = self._abs_path(self.log_dir) if self.log_dir else str(
            Path(abs_out).parent / f"{Path(abs_out).name}_logs")

        args = self._common_args()
        args += _bool_arg("AddDecays", self.add_decays)
        args += _bool_arg("StrictChecks", self.strict_checks)

        proc, stdout_path, stderr_path, launch_error = _invoke_generator(
            wolframscript_path=self.wolframscript_path,
            feynrules_path=abs_fr,
            model_path=abs_model,
            output_dir=abs_out,
            log_dir=log_dir,
            timeout_sec=self.timeout_sec,
            base_directory=self.base_directory,
            extra_args=args,
        )
        if proc is None:
            return self.format_error(
                error="UFO Generation Failed", reason=launch_error)

        checks = parse_check_verdicts(proc.stdout)
        _rc = _residual_check(proc.stdout)
        if _rc is not None:
            checks.append(_rc)
        missing = [
            f for f in REQUIRED_UFO_FILES
            if not (Path(abs_out) / f).exists()
        ]
        leak = _vertex_leak(abs_out) if proc.returncode == 0 else None
        ok = proc.returncode == 0 and not missing and leak is None
        if ok:
            # Best-effort, after the UFO is known complete: record the coupling
            # key order in the file people actually read it from.
            _stamp_vertices_header(abs_out)

        files_created = []
        if Path(abs_out).exists():
            files_created = sorted(p.name for p in Path(abs_out).iterdir())

        summary = {
            "schema_version": SCHEMA_VERSION,
            "ok": ok,
            "created_at_utc": _utc_now_iso(),
            "model_path": abs_model,
            "output_dir": abs_out,
            "lagrangian": self.lagrangian,
            "lagrangian_resolved": _resolved_lagrangian(proc.stdout),
            "wolframscript_path": self.wolframscript_path,
            "feynrules_path": abs_fr,
            "logs": {"stdout": stdout_path, "stderr": stderr_path},
            "files_created": files_created,
            "checks": checks,
            "checks_summary": summarise(checks),
            "mass_spectrum": parse_mass_spectrum(proc.stdout),
            # Which gauge the model was actually built in. It is inherited from
            # SM.fr unless the caller says otherwise, so reporting it is the
            # only way a caller finds out without counting ghosts.
            "gauge": parse_gauge(proc.stdout),
            "vertex_inventory": _vertex_inventory(
                abs_out, self.vertex_inventory or "new_physics"),
        }

        if not ok:
            reason = self._lagrangian_error(proc)
            # Named first: a leak means the compile "succeeded" and produced a
            # model quietly missing interactions, which is the failure most
            # likely to be mistaken for a correct result.
            if reason is None and leak:
                reason = leak
            if reason is None and missing:
                reason = (
                    f"wolframscript exited {proc.returncode} but the UFO is "
                    f"incomplete: missing {', '.join(missing)}. The Lagrangian "
                    f"may contain no interaction terms, or vertex extraction "
                    f"failed -- see {stdout_path}"
                )
                msgs = [m for m in dict.fromkeys(
                    ln.strip() for ln in (proc.stdout or "").splitlines()
                    if re.match(r"^[\w`$]+::\w+:", ln.strip()))][:6]
                if msgs:
                    reason += f". FeynRules' messages: {msgs}"
            if reason is None:
                reason = self._stdout_hint(proc) or (
                    f"wolframscript returned exit status {proc.returncode}")
            summary["error"] = reason
            return self.format_error(error="UFO Generation Failed", reason=reason)

        from .feynarts_model import dangling_lines
        lost = dangling_lines(Path(abs_model).read_text(errors="replace")) if os.path.isfile(abs_model) else []
        if lost:
            summary.setdefault("warnings", []).append(
                f".fr lines {lost[:5]} start with + or - outside any bracket: Mathematica ended the "
                "statement on the line before, so those terms are NOT in the Lagrangian. Move the "
                "operator to the end of the previous line")
        self._record_finding(abs_model, abs_out, checks)
        return json.dumps(summary, indent=2)

    def _stdout_hint(self, proc: subprocess.CompletedProcess) -> Optional[str]:
        """Find the most informative error line in the run's output."""
        for stream in (proc.stdout, proc.stderr):
            for line in (stream or "").splitlines():
                if line.startswith("Error:"):
                    return line.strip()
        for line in (proc.stderr or "").splitlines()[-20:]:
            if "Error" in line or "LoadModel" in line or "WriteUFO" in line:
                return line.strip()
        return None

    def _record_finding(self, model: str, out: str, checks: list) -> None:
        """Append a short entry to the shared findings ledger.

        Best-effort: a ledger failure must not turn a successful compile into a
        reported failure.
        """
        try:
            entries = [
                f"Model: {Path(model).name}",
                f"Lagrangian: {self.lagrangian}",
                f"UFO: {out}",
            ]
            if checks:
                entries.append(f"Checks: {summarise(checks)}")
                for failure in failed_checks(checks):
                    entries.append(f"FAILED — {failure['message']}")
            append_finding(
                self.base_directory, f"FeynRules → UFO: {Path(model).stem}", entries)
        except Exception:
            pass


class ValidateLagrangianTool(_FeynRulesToolBase):
    """
    Run FeynRules' consistency checks on a .fr model without writing a UFO.

By default it checks hermiticity, diagonal kinetic terms, diagonal mass
    terms and the mass spectrum. Each verdict comes from the FeynRules
    function's return value rather than its printed prose, and is three-valued:
    a check that could not be decided is reported as inconclusive, never as a
    pass.

    FeynRules' other two model checks are off by default because they fail for
    the Standard Model itself and so say nothing about the model under test --
    CheckDiagonalQuadraticTerms reports the R_xi gauge-Goldstone mixing term,
    and CheckKineticTermNormalisation trips on the massless neutrinos. Ask for
    them with checks="all", and set baseline_compare to have their results
    reported against an SM-only run rather than in isolation.

    This is the fast loop while a model is being edited; UFO generation is the
    expensive half and is skipped entirely.

    Returns:
      JSON summary with a verdict, message and remedy per check.
    """

    def _run(self) -> str:
        """Load the model, run the consistency checks, and report verdicts."""
        if not self.model_path:
            return self.format_error(
                error="Missing Parameter", reason="model_path is required")

        abs_model = self._abs_path(self.model_path)
        if not os.path.isfile(abs_model):
            return self.format_error(
                error="Model Not Found",
                reason=f"no file at {abs_model} (model_path is resolved against the "
                       f"sandbox, {self.base_directory})")
        abs_fr = self._abs_path(self.feynrules_path) if self.feynrules_path else None
        log_dir = self._abs_path(self.log_dir) if self.log_dir else str(
            Path(self.base_directory) / "_logs" / "feynrules_validate")

        args = self._common_args() + ["ChecksOnly=true"]

        proc, stdout_path, stderr_path, launch_error = _invoke_generator(
            wolframscript_path=self.wolframscript_path,
            feynrules_path=abs_fr,
            model_path=abs_model,
            output_dir=log_dir,
            log_dir=log_dir,
            timeout_sec=self.timeout_sec,
            base_directory=self.base_directory,
            extra_args=args,
        )
        if proc is None:
            return self.format_error(
                error="Model Validation Failed", reason=launch_error)

        if proc.returncode != 0:
            reason = self._lagrangian_error(proc) or (
                f"wolframscript returned exit status {proc.returncode}; "
                f"see {stdout_path}")
            return self.format_error(error="Model Validation Failed", reason=reason)

        checks = parse_check_verdicts(proc.stdout)
        _rc = _residual_check(proc.stdout)
        if _rc is not None:
            checks.append(_rc)
        failures = failed_checks(checks)

        summary = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "created_at_utc": _utc_now_iso(),
            "model_path": abs_model,
            "lagrangian": self.lagrangian,
            "feynrules_path": abs_fr,
            "logs": {"stdout": stdout_path, "stderr": stderr_path},
            "checks": checks,
            "checks_summary": summarise(checks),
            # Declared vs derived mass per state. This is the fast loop, so it
            # is the place the numbers matter most: a wrong mass is invisible
            # to every other check here.
            "mass_spectrum": parse_mass_spectrum(proc.stdout),
            "gauge": parse_gauge(proc.stdout),
            # Failures the SM baseline does not share: the ones this model
            # introduced, and the only ones worth acting on.
            "new_failures": [f["name"] for f in failures],
        }

        try:
            entries = [f"Model: {Path(abs_model).name}", f"Checks: {summarise(checks)}"]
            entries += [f"FAILED — {f['message']}" for f in failures]
            append_finding(
                self.base_directory,
                f"FeynRules validation: {Path(abs_model).stem}",
                entries,
            )
        except Exception:
            pass

        return json.dumps(summary, indent=2)
