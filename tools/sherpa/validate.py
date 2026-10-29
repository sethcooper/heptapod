"""
# validate.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Validating a Sherpa 3 run card: ValidateSherpaRunCardTool.

Sherpa reads a YAML card and ignores, without a word, every key it does not
know. A misspelt ``BEAM_ENERGIES`` leaves the beams at zero energy and a
misspelt ``SHOWER`` option leaves the shower at its default; the run
proceeds and the result is wrong. A value Sherpa cannot parse, a process it
cannot build or a selector it has never heard of, on the other hand, end
initialisation with a fatal error after the integration has started.

The validator works in two layers. The static layer reads the card as YAML
and checks its keys against a snapshot of every setting Sherpa 3.0.3 read
in a set of reference runs (``data/settings_snapshot.json``), plus the
documented keys of the ``PROCESSES`` block, and checks the shape of the
settings every card has: beams, energies, processes, selectors. The dry run
then initialises Sherpa on the card without generating (``-e0`` with
``CHECK_SETTINGS: true``): it reports Sherpa's own fatal error if there is
one, the settings Sherpa never read, and the effective configuration from
the Settings_Report Sherpa writes.
"""

import difflib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestral.tools.base.field_utils import RuntimeField, StateField
from orchestral.tools.base.tool import BaseTool

from tools.cpp.build import BuildError, resolve_install
from tools.sherpa.card import (
    CONTAINERS,
    CardError,
    has_variations,
    key_lines,
    parse_card,
    parse_process,
)
from tools.sherpa.runtime import (
    clean,
    fatal_messages,
    sherpa_executable,
    sherpa_version,
    unused_settings,
    wheel_prefix,
)

SNAPSHOT_PATH = Path(__file__).with_name("data") / "settings_snapshot.json"

#: The settings a card cannot do without: Sherpa's defaults for them (beams
#: of id 0 at zero energy, no processes) end initialisation with an error.
REQUIRED = ("BEAMS", "BEAM_ENERGIES", "PROCESSES")

#: EVENT_GENERATION_MODE as Sherpa spells it, with the short forms it also
#: accepts.
GENERATION_MODES = {
    "Weighted",
    "Unweighted",
    "PartiallyUnweighted",
    "W",
    "U",
    "PU",
    "P",
}

_snapshot_cache: Dict[str, Any] = {}


def load_snapshot() -> Dict[str, Any]:
    """The settings snapshot, read once."""
    if not _snapshot_cache:
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as handle:
            _snapshot_cache.update(json.load(handle))
    return _snapshot_cache


def known_settings() -> Dict[str, Dict[str, str]]:
    return load_snapshot()["settings"]


def known_list(name: str) -> List[str]:
    """One of the name lists Sherpa prints on request: "selectors",
    "me_generators", "shower_generators", "scale_setters", "kfactors",
    "nlomc_generators", "ps_generators"."""
    return list(load_snapshot().get("lists", {}).get(name, []))


def top_level_keys() -> List[str]:
    return sorted({name.split(":")[0] for name in known_settings()})


def children_of(group: str) -> List[str]:
    """The keys the snapshot knows under a block, as ``A`` for ``GROUP:A``;
    ``*`` stands for any entry of a list or a per-particle mapping."""
    prefix = group + ":"
    return sorted(
        {
            name[len(prefix) :].split(":")[0]
            for name in known_settings()
            if name.startswith(prefix)
        }
    )


def suggest(
    name: str, candidates: List[str], cutoff: float = 0.75
) -> List[str]:
    """Close matches to a misspelt key, case-insensitively."""
    lowered = {c.lower(): c for c in candidates}
    hits = difflib.get_close_matches(
        name.lower(), list(lowered), n=3, cutoff=cutoff
    )
    return [lowered[h] for h in hits]


def _finding(
    severity: str,
    message: str,
    *,
    key: str = "",
    line: Optional[int] = None,
    suggestion: str = "",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"severity": severity, "message": message}
    if key:
        out["key"] = key
    if line is not None:
        out["line"] = line
    if suggestion:
        out["suggestion"] = suggestion
    return out


def lint_card(text: str) -> List[Dict[str, Any]]:
    """
    The static findings for a card: errors that will or did break the run,
    warnings for what Sherpa would accept silently and probably should not.
    """
    findings: List[Dict[str, Any]] = []
    try:
        card = parse_card(text)
    except CardError as e:
        return [_finding("error", str(e), line=e.line)]
    lines = key_lines(text)
    known = known_settings()
    tops = top_level_keys()

    for key in REQUIRED:
        if key not in card:
            findings.append(
                _finding(
                    "error",
                    f"{key} is not set; Sherpa's default (none) ends "
                    "initialisation with an error",
                    key=key,
                )
            )

    for key, value in card.items():
        line = lines.get(str(key))
        name = str(key)
        if ":" in name and name.split(":")[0] not in tops:
            findings.append(
                _finding(
                    "error",
                    f"'{name}' looks like a Pythia setting; Sherpa cards "
                    "use upper-case YAML keys such as BEAMS and PROCESSES",
                    key=name,
                    line=line,
                )
            )
            continue
        if name not in tops and not any(
            n.startswith(name + ":") for n in known
        ):
            close = suggest(name, tops)
            if close:
                findings.append(
                    _finding(
                        "error",
                        f"'{name}' is not a Sherpa setting; it would be "
                        "ignored without a word",
                        key=name,
                        line=line,
                        suggestion=f"did you mean {', '.join(close)}?",
                    )
                )
            else:
                findings.append(
                    _finding(
                        "warning",
                        f"'{name}' is not among the settings Sherpa 3.0.3 "
                        "read in the reference runs; Sherpa ignores a key it "
                        "does not know without a word, so check the "
                        "spelling, or run with dry_run_init to see whether "
                        "Sherpa reads it",
                        key=name,
                        line=line,
                    )
                )
            continue
        if isinstance(value, dict):
            findings.extend(_check_block(name, value, line))

    findings.extend(_check_shapes(card, lines))
    return findings


def _check_block(
    group: str, block: Dict[str, Any], line: Optional[int]
) -> List[Dict[str, Any]]:
    """Unknown keys inside a block the snapshot knows the children of."""
    children = children_of(group)
    if not children or "*" in children:
        return []
    out: List[Dict[str, Any]] = []
    for key in block:
        name = str(key)
        if name in children:
            continue
        close = suggest(name, children)
        if close:
            out.append(
                _finding(
                    "error",
                    f"{group}:{name} is not a setting of {group}; Sherpa "
                    "would ignore it",
                    key=f"{group}:{name}",
                    line=line,
                    suggestion=f"did you mean {', '.join(close)}?",
                )
            )
        else:
            out.append(
                _finding(
                    "warning",
                    f"{group}:{name} is not among the {group} settings the "
                    "reference runs read; check the spelling",
                    key=f"{group}:{name}",
                    line=line,
                )
            )
    return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_shapes(
    card: Dict[str, Any], lines: Dict[str, int]
) -> List[Dict[str, Any]]:
    """The settings every card has, checked for the shape Sherpa reads."""
    out: List[Dict[str, Any]] = []

    beams = card.get("BEAMS")
    if beams is not None:
        values = beams if isinstance(beams, list) else [beams]
        if not 1 <= len(values) <= 2 or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in values
        ):
            out.append(
                _finding(
                    "error",
                    "BEAMS is one PDG id for both beams or a pair, e.g. "
                    "2212 or [11, -11]",
                    key="BEAMS",
                    line=lines.get("BEAMS"),
                )
            )
    energies = card.get("BEAM_ENERGIES")
    if energies is not None:
        values = energies if isinstance(energies, list) else [energies]
        if not 1 <= len(values) <= 2 or not all(
            _is_number(v) and v > 0 for v in values
        ):
            out.append(
                _finding(
                    "error",
                    "BEAM_ENERGIES is one positive beam energy in GeV for "
                    "both beams or a pair, e.g. 6500 or [45.6, 45.6]",
                    key="BEAM_ENERGIES",
                    line=lines.get("BEAM_ENERGIES"),
                )
            )
        elif all(_is_number(v) for v in values) and max(values) > 1.0e5:
            out.append(
                _finding(
                    "warning",
                    "BEAM_ENERGIES is the energy per beam in GeV, not the "
                    "centre-of-mass energy; 6500 gives a 13 TeV collider",
                    key="BEAM_ENERGIES",
                    line=lines.get("BEAM_ENERGIES"),
                )
            )

    for key in ("EVENTS", "RANDOM_SEED"):
        value = card.get(key)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        if not all(
            isinstance(v, int) and not isinstance(v, bool) for v in values
        ):
            out.append(
                _finding(
                    "error",
                    f"{key} takes an integer",
                    key=key,
                    line=lines.get(key),
                )
            )

    mode = card.get("EVENT_GENERATION_MODE")
    if mode is not None and str(mode) not in GENERATION_MODES:
        out.append(
            _finding(
                "error",
                "EVENT_GENERATION_MODE is Weighted, Unweighted or "
                "PartiallyUnweighted",
                key="EVENT_GENERATION_MODE",
                line=lines.get("EVENT_GENERATION_MODE"),
            )
        )

    processes = card.get("PROCESSES")
    if processes is not None:
        out.extend(_check_processes(processes, lines.get("PROCESSES")))

    selectors = card.get("SELECTORS")
    if selectors is not None:
        out.extend(_check_selectors(selectors, lines.get("SELECTORS")))

    for key, names in (
        ("ME_GENERATORS", "me_generators"),
        ("SHOWER_GENERATOR", "shower_generators"),
    ):
        value = card.get(key)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        allowed = known_list(names)
        for entry in values:
            if str(entry) not in allowed:
                close = suggest(str(entry), allowed, cutoff=0.6)
                out.append(
                    _finding(
                        "error",
                        f"{key}: '{entry}' is not one of {', '.join(allowed)}"
                        + (
                            " in this Sherpa; a loop or external generator "
                            "needs an install built with it"
                            if key == "ME_GENERATORS"
                            else ""
                        ),
                        key=key,
                        line=lines.get(key),
                        suggestion=(
                            f"did you mean {', '.join(close)}?"
                            if close
                            else ""
                        ),
                    )
                )
    return out


def _check_selectors(
    selectors: Any, line: Optional[int]
) -> List[Dict[str, Any]]:
    """Every selector by a name Sherpa has. An unknown selector is ignored
    by Sherpa without a word, so this is the only check there is."""
    if not isinstance(selectors, list):
        return [
            _finding(
                "error",
                "SELECTORS is a list of selectors, e.g. [[Mass, 11, -11, "
                "66, 116]]",
                key="SELECTORS",
                line=line,
            )
        ]
    names = known_list("selectors")
    out: List[Dict[str, Any]] = []
    for entry in selectors:
        if isinstance(entry, list) and entry:
            name = str(entry[0])
        elif isinstance(entry, dict) and len(entry) == 1:
            name = str(next(iter(entry)))
        else:
            out.append(
                _finding(
                    "error",
                    f"a selector is [Name, args...] or {{Name: {{...}}}}, not "
                    f"{entry!r}",
                    key="SELECTORS",
                    line=line,
                )
            )
            continue
        if name not in names:
            close = suggest(name, names, cutoff=0.6)
            out.append(
                _finding(
                    "error",
                    f"'{name}' is not a Sherpa selector; Sherpa would ignore "
                    "it and apply no cut",
                    key="SELECTORS",
                    line=line,
                    suggestion=(
                        f"did you mean {', '.join(close)}?"
                        if close
                        else "SHOW_SELECTOR_SYNTAX: 1 makes Sherpa list its "
                        "selectors"
                    ),
                )
            )
    return out


def _check_processes(
    processes: Any, line: Optional[int]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(processes, list) or not processes:
        return [
            _finding(
                "error",
                "PROCESSES is a non-empty list; each entry maps a process "
                "string such as '93 93 -> 11 -11' to its options",
                key="PROCESSES",
                line=line,
            )
        ]
    options = children_of("PROCESSES:*")
    for entry in processes:
        if isinstance(entry, str):
            specs: Dict[str, Any] = {entry: {}}
        elif isinstance(entry, dict):
            specs = entry
        else:
            out.append(
                _finding(
                    "error",
                    f"a PROCESSES entry is a mapping, not {entry!r}",
                    key="PROCESSES",
                    line=line,
                )
            )
            continue
        for spec, opts in specs.items():
            try:
                incoming, outgoing = parse_process(str(spec))
            except ValueError as e:
                out.append(
                    _finding("error", str(e), key="PROCESSES", line=line)
                )
                continue
            if len(incoming) != 2:
                out.append(
                    _finding(
                        "error",
                        f"'{spec}': a process has two incoming particles",
                        key="PROCESSES",
                        line=line,
                    )
                )
            if len(outgoing) < 1:
                out.append(
                    _finding(
                        "error",
                        f"'{spec}': a process has at least one outgoing "
                        "particle",
                        key="PROCESSES",
                        line=line,
                    )
                )
            for pid, _ in outgoing:
                if 90 <= abs(pid) <= 99 and abs(pid) not in CONTAINERS:
                    out.append(
                        _finding(
                            "warning",
                            f"'{spec}': {pid} is not one of Sherpa's "
                            "particle containers "
                            f"({', '.join(str(c) for c in CONTAINERS)})",
                            key="PROCESSES",
                            line=line,
                        )
                    )
            if opts is None:
                continue
            if not isinstance(opts, dict):
                out.append(
                    _finding(
                        "error",
                        f"'{spec}': the options are a mapping ({{}} for none)",
                        key="PROCESSES",
                        line=line,
                    )
                )
                continue
            for key, value in opts.items():
                name = str(key)
                # Multiplicity-specific options ("2->2-4") and the
                # documented options pass; anything else is checked by
                # spelling.
                if re.match(r"^\d+->\d+(-\d+)?$", name) or name in options:
                    if name == "Scales" and isinstance(value, str):
                        setter = value.split("{", 1)[0].strip()
                        setters = known_list("scale_setters")
                        if setters and setter not in setters:
                            out.append(
                                _finding(
                                    "error",
                                    f"'{spec}': Scales '{setter}' is not a "
                                    f"scale setter ({', '.join(setters)})",
                                    key="PROCESSES",
                                    line=line,
                                )
                            )
                    continue
                close = suggest(name, options, cutoff=0.7)
                out.append(
                    _finding(
                        "error" if close else "warning",
                        f"'{spec}': {name} is not a process option "
                        + ("" if close else "the reference runs read"),
                        key="PROCESSES",
                        line=line,
                        suggestion=(
                            f"did you mean {', '.join(close)}?"
                            if close
                            else ""
                        ),
                    )
                )
            order = opts.get("Order")
            if order is not None and not isinstance(order, dict):
                out.append(
                    _finding(
                        "error",
                        f"'{spec}': Order is a mapping of coupling orders, "
                        "e.g. {QCD: 0, EW: 2}",
                        key="PROCESSES",
                        line=line,
                    )
                )
    return out


# ====================================================================== #
# ============================== Dry run =============================== #
# ====================================================================== #


def sherpa_prefix(sherpa_path: str) -> Optional[str]:
    """The install prefix to run: the configured local Sherpa, else the
    wheel's own."""
    configured = (sherpa_path or os.environ.get("SHERPA_PATH", "")).strip()
    if configured:
        try:
            return resolve_install("sherpa", configured).prefix
        except BuildError:
            return None
    return wheel_prefix()


def effective_settings(report_dir: str) -> List[Dict[str, str]]:
    """
    The customised settings of a run from its Settings_Report: each with
    its default, the value the card gave and the value Sherpa used.
    """
    path = os.path.join(report_dir, "Settings_Report.md")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return []
    section = text.split("Customised settings", 1)[-1].split(
        "Settings kept at their default value", 1
    )[0]
    out: List[Dict[str, str]] = []
    for line in section.splitlines():
        cells = [c.strip().replace("\\", "") for c in line.split("|")]
        if len(cells) < 6 or cells[0] in ("parameter", "-", ""):
            continue
        out.append(
            {
                "setting": cells[0],
                "default": cells[1],
                "card": cells[3],
                "command_line": cells[4],
                "final": cells[5],
            }
        )
    return out


def dry_run(
    card_path: str, prefix: str, work_dir: str, timeout_s: int
) -> Dict[str, Any]:
    """
    Initialise Sherpa on the card without generating an event.

    Runs ``Sherpa -e0 -O0 'CHECK_SETTINGS: true' card`` in ``work_dir``, so
    Sherpa's Results directory lands where a later generation reuses it.
    """
    exe = sherpa_executable(prefix)
    if exe is None:
        return {"ok": False, "error": f"no bin/Sherpa under {prefix}"}
    os.makedirs(work_dir, exist_ok=True)
    argv = [
        exe,
        "-e0",
        "-O0",
        "CHECK_SETTINGS: true",
        os.path.abspath(card_path),
    ]
    env = dict(os.environ)
    lib = os.path.join(prefix, "lib", "SHERPA-MC")
    var = (
        "DYLD_LIBRARY_PATH"
        if os.uname().sysname == "Darwin"
        else "LD_LIBRARY_PATH"
    )
    env[var] = lib + (":" + env[var] if env.get(var) else "")
    log_path = os.path.join(work_dir, "sherpa_init.log")
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                argv,
                cwd=work_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
            )
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        returncode = -1
        timed_out = True
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        text = clean(handle.read())
    fatal = fatal_messages(text)
    result: Dict[str, Any] = {
        "ok": returncode == 0 and not fatal and not timed_out,
        "returncode": returncode,
        "log": log_path,
        "fatal": fatal,
        "unused_settings": unused_settings(text),
        "effective_settings": effective_settings(
            os.path.join(work_dir, "Settings_Report")
        ),
    }
    if timed_out:
        result["error"] = f"initialisation did not finish in {timeout_s}s"
    return result


class ValidateSherpaRunCardTool(BaseTool):
    """
    Validate a Sherpa 3 YAML run card before generating events with it.

    Sherpa ignores a key it does not know without a word -- a misspelt
    BEAM_ENERGIES leaves the beams at zero energy -- and reports a value it
    cannot use only by ending initialisation after the integration has
    started. This tool checks the card first: its YAML, its keys against the
    settings Sherpa 3.0.3 reads, the shape of the beams, energies,
    processes and selectors, and Pythia settings that strayed in. With
    dry_run_init it also initialises Sherpa on the card without generating,
    which catches what no static check can -- an unknown selector, a process
    Sherpa cannot build -- and reports the settings Sherpa never read and
    the effective configuration.

    Inputs (runtime):
      - card_path: relative path to the .yaml card (or card_text instead)
      - card_text: the card's contents inline, for a card not yet on disk
      - dry_run_init: if True, initialise Sherpa on the card (no events).
        Integrates the process, so it costs what a run's set-up costs; done
        in work_dir, a later generation there reuses the integration
      - work_dir: relative directory for the dry run's files; use the
        data_dir of the planned generation
      - timeout_s: wall-clock limit for the dry run (default 1800)
      - base_directory: sandbox root for all file operations
      - sherpa_path: a local Sherpa install prefix; the sherpa-mc wheel's
        own Sherpa is used when unset

    Output (JSON):
      {
        "status": "ok",
        "valid": <bool>,
        "counts": {"error": 0, "warning": 1},
        "findings": [{"severity": "error", "line": 4, "key": "BEAM_ENERGIEZ",
                      "message": "...", "suggestion": "did you mean ..."}],
        "variations": <bool>,
        "dry_run": {"ok": true, "fatal": [], "unused_settings": {...},
                    "effective_settings": [{"setting": "BEAMS", "default":
                    "0, 0", "card": "2212", "final": "2212, 2212"}, ...],
                    "log": "<path>"},
        "sherpa_version": "3.0.3"
      }

    "valid" is false when any finding is an error or the dry run failed. A
    warning marks what Sherpa accepts but probably should not; read it.
    """

    # --------------------------- Runtime fields --------------------------- #
    card_path: str = RuntimeField(
        default="", description="Relative path to the Sherpa .yaml run card"
    )
    card_text: str = RuntimeField(
        default="",
        description="The card's contents inline, instead of card_path",
    )
    dry_run_init: bool = RuntimeField(
        default=False,
        description="Also initialise Sherpa on the card without generating: "
        "catches unknown selectors and processes Sherpa cannot build, and "
        "reports unread settings and the effective configuration. Costs a "
        "run's set-up (the process integration)",
    )
    work_dir: str = RuntimeField(
        default="",
        description="Relative directory for the dry run's Results and "
        "Settings_Report; the data_dir of the planned generation lets it "
        "reuse the integration. Default: a temporary directory",
    )
    timeout_s: int = RuntimeField(
        default=1800, description="Wall-clock limit for the dry run"
    )

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(
        default=".", description="Base directory for safe path resolution"
    )
    sherpa_path: str = StateField(
        default="",
        description="Local Sherpa install prefix (holds bin/Sherpa-config); "
        "the sherpa-mc wheel's Sherpa is used when unset",
    )

    def _setup(self) -> None:
        self.base_directory = os.path.abspath(self.base_directory)

    def _safe_path(self, rel: str) -> Optional[str]:
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        if not (full + os.sep).startswith(self.base_directory + os.sep):
            return None
        return full

    def _run(self) -> str:
        if not self.card_text and not self.card_path:
            return self.format_error(
                error="Missing Parameter",
                reason="card_path or card_text is required",
            )
        text = self.card_text
        card_file: Optional[str] = None
        if not text:
            card_file = self._safe_path(self.card_path)
            if not card_file:
                return self.format_error(
                    error="Access Denied",
                    reason="card_path escapes base_directory",
                    context=f"card_path={self.card_path}",
                )
            try:
                with open(card_file, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except OSError as e:
                return self.format_error(
                    error="File Not Found",
                    reason=str(e),
                    context=f"card_path={self.card_path}",
                )

        findings = lint_card(text)
        counts = {"error": 0, "warning": 0}
        for finding in findings:
            counts[finding["severity"]] = (
                counts.get(finding["severity"], 0) + 1
            )
        try:
            card = parse_card(text)
        except CardError:
            card = {}
        result: Dict[str, Any] = {
            "status": "ok",
            "valid": counts["error"] == 0,
            "counts": counts,
            "findings": findings,
            "variations": has_variations(card),
        }

        prefix = sherpa_prefix(self.sherpa_path)
        result["sherpa_version"] = sherpa_version(prefix)
        if self.dry_run_init:
            if prefix is None:
                return self.format_error(
                    error="Sherpa Not Available",
                    reason="neither sherpa_path nor the sherpa-mc wheel "
                    "provides a Sherpa to initialise",
                    suggestion="Set sherpa_path, or install sherpa-mc; the "
                    "static checks above ran without it",
                )
            if counts["error"]:
                result["dry_run"] = {
                    "ok": False,
                    "skipped": "the static checks found errors; fix them "
                    "first",
                }
                return json.dumps(result, ensure_ascii=False)
            if self.work_dir:
                work = self._safe_path(self.work_dir)
                if not work:
                    return self.format_error(
                        error="Access Denied",
                        reason="work_dir escapes base_directory",
                    )
            else:
                work = tempfile.mkdtemp(prefix="sherpa_dry_run_")
            if card_file is None:
                os.makedirs(work, exist_ok=True)
                card_file = os.path.join(work, "card.yaml")
                with open(card_file, "w", encoding="utf-8") as handle:
                    handle.write(text)
            outcome = dry_run(card_file, prefix, work, int(self.timeout_s))
            outcome["log"] = (
                os.path.relpath(outcome["log"], self.base_directory)
                if outcome.get("log")
                else ""
            )
            result["dry_run"] = outcome
            result["valid"] = result["valid"] and bool(outcome.get("ok"))
            for source, keys in outcome.get("unused_settings", {}).items():
                for key in keys:
                    result["findings"].append(
                        _finding(
                            "warning",
                            f"Sherpa never read {key} (from {source}); a "
                            "misspelt key, or a block none of whose entries "
                            "it knows",
                            key=key,
                        )
                    )
                    counts["warning"] += 1
        return json.dumps(result, ensure_ascii=False)
