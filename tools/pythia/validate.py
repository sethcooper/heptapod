"""
# validate.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Validating a Pythia 8 ``.cmnd`` run card: ValidatePythiaRunCardTool.

The validator exists because of how Pythia reports bad input. An unknown key
or an out-of-range value does not raise: Pythia prints the complaint and
makes ``init()`` return ``false``. Validating the card first turns that into
a precise message naming the line -- and catches the mistakes Pythia accepts
without a word, such as a boolean typo that it silently reads as ``off``.

Checks run in two layers. The static layer reimplements the reading rules of
``Settings::readString`` and ``ParticleData::readString``, and works without
Pythia installed. When pythia8mc is available, every line is also put to the
installed Pythia (``pythia_check.line_verdicts``), whose verdict on whether a
line is accepted is final; the static layer then contributes only what Pythia
does not say. How a card is read is
matched to Pythia in ``card``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .card import (
    FALSE_WORDS,
    FLOAT_RE,
    INT_RE,
    SUBRUN_DEFAULT,
    TRUE_WORDS,
    Statement,
    make_finding,
    read_statements,
)
from .settings_db import load_settings, search_settings

# Properties ParticleData::readString understands on an `<id>:<property>` line.
_PARTICLE_PROPERTIES = {
    "addchannel",
    "all",
    "antiname",
    "bratio",
    "chargetype",
    "coltype",
    "doexternaldecay",
    "doforcewidth",
    "isresonance",
    "isvisible",
    "m0",
    "maydecay",
    "memode",
    "mmax",
    "mmin",
    "mwidth",
    "name",
    "names",
    "new",
    "offifall",
    "offifany",
    "offifmatch",
    "onechannel",
    "onifall",
    "onifany",
    "onifmatch",
    "onmode",
    "onnegifall",
    "onnegifany",
    "onnegifmatch",
    "onposifall",
    "onposifany",
    "onposifmatch",
    "products",
    "rescalebr",
    "spintype",
    "tau0",
    "taucalc",
    "varwidth",
}
# ...and on an `<id>:<channel>:<property>` decay-channel line.
_CHANNEL_PROPERTIES = {"onmode", "bratio", "memode", "products", "all"}

# Value types of the particle properties that take a single value.
_PARTICLE_BOOL = {
    "maydecay",
    "isresonance",
    "isvisible",
    "doexternaldecay",
    "varwidth",
    "doforcewidth",
    "taucalc",
}
_PARTICLE_FLOAT = {"m0", "mwidth", "mmin", "mmax", "tau0", "rescalebr"}
_PARTICLE_INT = {"spintype", "chargetype", "coltype"}

# Properties that create a particle, after which later lines may address it.
_PARTICLE_CREATORS = {"new", "all"}

# ====================================================================== #
# ============================ Value checks ============================ #
# ====================================================================== #


def _leading(regex: re.Pattern[str], value: str) -> Optional[str]:
    match = regex.match(value)
    return match.group(0) if match else None


def _check_range(record: Dict[str, Any], value: float) -> Optional[str]:
    """Return a message if ``value`` violates the setting's declared range."""
    low, high = record.get("min"), record.get("max")
    if low is not None and value < float(low):
        return f"below the declared minimum {low}"
    if high is not None and value > float(high):
        return f"above the declared maximum {high}"
    return None


def _out_of_range(
    record: Dict[str, Any], value: str, message: str, version: Optional[str]
) -> Tuple[str, str, str, str]:
    return (
        "error",
        f"{value} is {message}. Pythia {version or '8.3'} refuses the value "
        f"and init() "
        "fails (some older 8.3 releases silently clamp it to the limit "
        "instead)",
        _range_hint(record) + "; `force` before the value bypasses the limit",
        "out_of_range",
    )


def _check_number(
    record: Dict[str, Any], value: str, kind: str, version: Optional[str]
) -> List[Tuple[str, str, str, str]]:
    """Check an int or float value the way istringstream reads it."""
    lead = _leading(INT_RE if kind == "int" else FLOAT_RE, value)
    if lead is None:
        noun = "an integer" if kind == "int" else "a number"
        return [
            (
                "error",
                f"'{value}' is not {noun}; Pythia rejects it and init() fails",
                "use a numeric value",
                "bad_value",
            )
        ]
    out: List[Tuple[str, str, str, str]] = []
    if lead != value:
        out.append(
            (
                "warning",
                f"Pythia reads only the leading '{lead}' of '{value}' and "
                f"ignores the rest",
                f"write `{lead}`" if kind == "int" else "write a plain number",
                "truncated_number",
            )
        )
    number = float(lead)
    message = _check_range(record, number)
    if message:
        out.append(_out_of_range(record, value, message, version))
    elif kind == "int" and record.get("options"):
        allowed = {option["value"] for option in record["options"]}
        if str(int(number)) not in allowed:
            out.append(
                (
                    "warning",
                    f"{lead} is not one of the documented options "
                    f"({', '.join(sorted(allowed, key=_sort_key))})",
                    "check the option list for this setting",
                    "undocumented_option",
                )
            )
    return out


def _check_vector(
    record: Dict[str, Any], value: str, version: Optional[str]
) -> List[Tuple[str, str, str, str]]:
    """Check each element of a vector value against the element type and
    range."""
    element = record["type"][len("vector<") : -1]
    body = (
        value[1:-1] if value.startswith("{") and value.endswith("}") else value
    )
    items = [item.strip() for item in body.split(",") if item.strip()]
    out: List[Tuple[str, str, str, str]] = []
    for item in items:
        if element == "bool":
            if item.lower() not in TRUE_WORDS | FALSE_WORDS:
                out.append(
                    (
                        "error",
                        f"element '{item}' is not a boolean Pythia "
                        f"recognises; it is "
                        "silently read as off",
                        "use on/off",
                        "bool_unrecognised",
                    )
                )
        elif element in ("int", "float"):
            for severity, message, suggestion, code in _check_number(
                record, item, element, version
            ):
                out.append((severity, f"element {message}", suggestion, code))
    return out


def _check_value(
    record: Dict[str, Any],
    value: str,
    statement: Optional[Statement] = None,
    version: Optional[str] = None,
) -> List[Tuple[str, str, str, str]]:
    """Type/range-check one value. Returns (severity, message, suggestion,
    code)."""
    kind = record["type"]

    # Pythia matches both keywords case-sensitively.
    if value in ("DEFAULT", "?"):
        return []
    if value.lower() == "default":
        return [
            (
                "error",
                f"'{value}' is not the reset keyword: Pythia recognises only "
                f"upper-case "
                f"DEFAULT, and reads '{value}' as an ordinary value"
                + (", i.e. off" if kind == "bool" else ""),
                "write `DEFAULT`",
                "bad_value" if kind != "bool" else "bool_unrecognised",
            )
        ]
    if "force" in value and kind in (
        "int",
        "float",
        "vector<int>",
        "vector<float>",
    ):
        # The value is the next word; its range is deliberately not checked.
        tokens = statement.tokens if statement else []
        forced = tokens[2] if len(tokens) > 2 else ""
        if not forced:
            return [
                (
                    "error",
                    "`force` with no value after it",
                    "write `force <value>`",
                    "no_value",
                )
            ]
        return (
            []
            if kind.startswith("vector")
            else [
                f for f in _check_number({"type": kind}, forced, kind, version)
            ]
        )

    if not value:
        if kind == "string":
            return []
        return [
            (
                "error",
                "no value given; Pythia rejects a recognised name with no "
                "value",
                f"write `{record['name']} = <{kind}>`",
                "no_value",
            )
        ]

    if kind == "bool":
        if value.lower() not in TRUE_WORDS | FALSE_WORDS:
            return [
                (
                    "error",
                    f"'{value}' is not a boolean Pythia recognises, and "
                    f"unrecognised "
                    "values are silently treated as off rather than reported",
                    "use on/off (true/false, yes/no, 1/0 also work)",
                    "bool_unrecognised",
                )
            ]
        return []
    if kind in ("int", "float"):
        return _check_number(record, value, kind, version)
    if kind.startswith("vector"):
        out = _check_vector(record, value, version)
        if (
            not value.startswith("{")
            and statement is not None
            and len(statement.tokens) > 2
        ):
            out.append(
                (
                    "warning",
                    f"{record['name']} is a {kind}; unbraced, Pythia reads "
                    f"only up to the "
                    "first space, so the remaining values are dropped",
                    "brace the list, e.g. `= {1, 2, 3}`",
                    "vector_truncated",
                )
            )
        return out
    return []


def _sort_key(value: str) -> Tuple[int, str]:
    lead = _leading(FLOAT_RE, value)
    return (0, f"{float(lead):012.3f}") if lead == value else (1, value)


def _range_hint(record: Dict[str, Any]) -> str:
    low, high = record.get("min"), record.get("max")
    if low is not None and high is not None:
        return f"allowed range is {low} to {high}"
    if low is not None:
        return f"minimum is {low}"
    if high is not None:
        return f"maximum is {high}"
    return ""


def _suggest_names(
    key: str, snapshot: Dict[str, Any], limit: int = 3
) -> List[str]:
    """Nearest known setting names for an unrecognised key.

    A misspelling is the common case here, and relevance search is the wrong
    instrument for it: 'PhaseSpace:pTHatMinn' shares no token with anything, so
    the search falls back to the group name and suggests unrelated PhaseSpace
    settings. Edit distance finds the intended name; relevance search is kept
    as the fallback for a key that is wrong rather than merely mistyped.
    """
    import difflib

    names = [record["name"] for record in snapshot["settings"]]
    lowered = {name.lower(): name for name in names}
    close = difflib.get_close_matches(
        key.lower(), list(lowered), n=limit, cutoff=0.85
    )
    if close:
        return [lowered[name] for name in close]

    hits = search_settings(
        key.replace(":", " "), snapshot=snapshot, limit=limit
    )
    return [record["name"] for record in hits]


# ====================================================================== #
# ================================ Linting ============================= #
# ====================================================================== #

# Codes of findings that assert Pythia will reject the line. When the
# installed Pythia is consulted and accepts the line, these were wrong and are
# dropped; every other finding describes something Pythia accepts silently
# (a boolean typo read as off, a truncated number) and stands.
_REJECTION_CODES = {
    "unknown_setting",
    "bad_value",
    "out_of_range",
    "no_value",
    "append_to_scalar",
    "unknown_particle",
    "unknown_property",
}


def lint_card(
    text: str,
    *,
    snapshot: Optional[Dict[str, Any]] = None,
    lhe_mode: Optional[bool] = None,
    card_dir: Optional[str] = None,
    ask_pythia: bool = True,
) -> Dict[str, Any]:
    """Validate a ``.cmnd`` card's syntax and semantics.

    Parameters:
        text: Card contents
        snapshot: Settings database to check against; by default the
            installed Pythia's, or the bundled snapshot without pythia8mc
        lhe_mode: Whether the card will be run by PythiaFromRunCardTool with
            shower_lhe=True; when None it is inferred from Beams:frameType
        card_dir: Directory of the card on disk, for resolving includes
        ask_pythia: Also put every line to the installed Pythia, whose
            accept/reject verdict then overrides the static one
    """
    data = snapshot if snapshot is not None else load_settings()
    version = data.get("pythia_version")
    by_name = {record["name"].lower(): record for record in data["settings"]}
    particles = set(data.get("particles", {}))

    statements, findings = read_statements(text, card_dir=card_dir)
    per_statement: List[List[Dict[str, Any]]] = []
    seen: Dict[Tuple[int, str], int] = {}
    resolved: Dict[str, str] = {}

    for statement in statements:
        here: List[Dict[str, Any]] = []
        per_statement.append(here)
        if statement.kind == "particle":
            here.extend(_lint_particle_line(statement, particles))
            continue

        record = by_name.get(statement.key)
        if record is None:
            written = (
                statement.tokens[0] if statement.tokens else statement.key
            )
            suggestions = _suggest_names(statement.key, data)
            here.append(
                make_finding(
                    "error",
                    statement.line_no,
                    f"unknown setting '{written}'; Pythia reports "
                    "'input string not found in settings databases' and "
                    "init() returns false",
                    line=statement.text,
                    code="unknown_setting",
                    source=statement.source,
                    suggestion=(
                        "did you mean " + ", ".join(suggestions) + "?"
                        if suggestions
                        else "check the spelling against the settings database"
                    ),
                )
            )
            continue

        canonical = record["name"]
        if statement.subrun == SUBRUN_DEFAULT:
            resolved[canonical] = statement.value
        scope = (statement.subrun, statement.key)
        if scope in seen and statement.key != "main:subrun":
            here.append(
                make_finding(
                    "warning",
                    statement.line_no,
                    f"{canonical} is set again here (first set on line "
                    f"{seen[scope]}); "
                    "the last assignment wins",
                    line=statement.text,
                    source=statement.source,
                )
            )
        seen[scope] = statement.line_no

        if statement.append and not record["type"].startswith("vector"):
            here.append(
                make_finding(
                    "error",
                    statement.line_no,
                    f"{canonical}: `+=` only works on vector settings; Pythia "
                    f"rejects it here",
                    line=statement.text,
                    suggestion="use `=`",
                    code="append_to_scalar",
                    source=statement.source,
                )
            )
        if record.get("read_only"):
            here.append(
                make_finding(
                    "warning",
                    statement.line_no,
                    f"{canonical} is a read-only setting and is not meant to "
                    f"be assigned",
                    line=statement.text,
                    source=statement.source,
                )
            )
        for severity, message, suggestion, code in _check_value(
            record, statement.value, statement, version
        ):
            here.append(
                make_finding(
                    severity,
                    statement.line_no,
                    f"{canonical}: {message}",
                    line=statement.text,
                    suggestion=suggestion,
                    code=code,
                    source=statement.source,
                )
            )
        if (
            record["type"] == "string"
            and len(statement.text.split("!")[0].split()) > 3
        ):
            here.append(
                make_finding(
                    "warning",
                    statement.line_no,
                    f"{canonical} is a string setting and Pythia keeps only "
                    f"the "
                    "first whitespace-delimited token; a value containing "
                    "spaces is truncated",
                    line=statement.text,
                    source=statement.source,
                )
            )

    checked_by = "static"
    has_process: Optional[bool] = None
    if ask_pythia:
        answer = _ask_pythia(statements)
        if answer is not None:
            verdicts, has_process = answer
            checked_by = "installed Pythia"
            _apply_verdicts(statements, per_statement, verdicts)

    for here in per_statement:
        findings.extend(here)
    findings.extend(
        _lint_card_semantics(resolved, statements, lhe_mode, data, has_process)
    )

    counts = {"error": 0, "warning": 0, "note": 0}
    for finding in findings:
        counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
        finding.pop("code", None)
    findings.sort(
        key=lambda f: (f.get("file", ""), f.get("line", 0), f["severity"])
    )

    return {
        "valid": counts["error"] == 0,
        "counts": counts,
        "findings": findings,
        "settings_seen": len(
            {s.key for s in statements if s.kind == "setting"}
        ),
        "pythia_version": version,
        "checked_by": checked_by,
        "database": data.get("database", {"source": "given"}),
    }


def _ask_pythia(
    statements: List[Statement],
) -> Optional[Tuple[List[Optional[Dict[str, Any]]], bool]]:
    """The installed Pythia's verdict on each statement, and whether the card
    switches on a hard process by Pythia's own reckoning; None without it.

    Only statements outside any `Main:subrun` block are asked about: Pythia
    reads a subrun block only when that subrun is selected, so it would
    "accept" every line inside one without looking at it.
    """
    from .pythia_check import line_verdicts

    asked = [
        s
        for s in statements
        if s.subrun == SUBRUN_DEFAULT and s.key != "main:subrun"
    ]
    answer = line_verdicts([s.text for s in asked])
    if answer is None:
        return None
    by_id = {id(s): v for s, v in zip(asked, answer["lines"])}
    return [by_id.get(id(s)) for s in statements], answer["has_hard_process"]


def _apply_verdicts(
    statements: List[Statement],
    per_statement: List[List[Dict[str, Any]]],
    verdicts: List[Optional[Dict[str, Any]]],
) -> None:
    """Reconcile the static findings with what Pythia said about each line."""
    for statement, here, verdict in zip(statements, per_statement, verdicts):
        if verdict is None:
            continue
        if verdict["accepted"]:
            here[:] = [
                f for f in here if f.get("code") not in _REJECTION_CODES
            ]
            continue
        errors = [f for f in here if f["severity"] == "error"]
        said = "; ".join(verdict["messages"]) or "the line was rejected"
        if errors:
            errors[0]["pythia_says"] = said
        else:
            here.append(
                make_finding(
                    "error",
                    statement.line_no,
                    f"Pythia rejects this line: {said}",
                    line=statement.text,
                    source=statement.source,
                )
            )


def _lint_particle_line(
    statement: Statement, particles: set
) -> List[Dict[str, Any]]:
    """Validate an `<id>:<property>` or `<id>:<channel>:<property>` line."""
    findings: List[Dict[str, Any]] = []
    parts = statement.key.split(":")
    pid = parts[0]
    channel = parts[1] if len(parts) > 2 and parts[1].isdigit() else None
    prop = (
        (parts[2] if channel is not None else parts[1])
        if len(parts) > 1
        else ""
    )
    value = statement.value
    line, source = statement.text, statement.source

    if prop in _PARTICLE_CREATORS:
        # Defines the particle; later lines may address it.
        particles.add(pid)
        particles.add(f"-{pid}")
        return findings

    if particles and pid not in particles and f"-{pid}" not in particles:
        findings.append(
            make_finding(
                "error",
                statement.line_no,
                f"particle id {pid} is not in Pythia's particle data table; "
                "Pythia reports 'input particle not found' and init() returns "
                "false",
                line=line,
                code="unknown_particle",
                source=source,
                suggestion=(
                    f"define it first with `{pid}:new = <name> <antiName> "
                    f"...` or "
                    f"`{pid}:all = ...`; an SLHA or LHEF header is read at "
                    f"init(), "
                    "after the card, so it cannot introduce a particle the "
                    "card "
                    "then modifies"
                ),
            )
        )

    allowed = (
        _CHANNEL_PROPERTIES if channel is not None else _PARTICLE_PROPERTIES
    )
    if prop and prop not in allowed:
        findings.append(
            make_finding(
                "error",
                statement.line_no,
                f"'{prop}' is not a "
                f"{'decay-channel' if channel is not None else 'particle'} "
                "property Pythia understands",
                line=line,
                code="unknown_property",
                source=source,
                suggestion=(
                    "channel properties are onMode, bRatio, meMode, products"
                    if channel is not None
                    else "valid ones include m0, mWidth, tau0, mayDecay, "
                    "onMode, oneChannel"
                ),
            )
        )
        return findings

    if (
        channel is None
        and prop in _PARTICLE_BOOL
        and value.lower() not in TRUE_WORDS | FALSE_WORDS
    ):
        findings.append(
            make_finding(
                "error",
                statement.line_no,
                f"{prop} value '{value}' is not a boolean; unrecognised "
                f"values read as off",
                line=line,
                suggestion="use on/off",
                code="bool_unrecognised",
                source=source,
            )
        )
    elif channel is None and prop in _PARTICLE_FLOAT | _PARTICLE_INT:
        regex = INT_RE if prop in _PARTICLE_INT else FLOAT_RE
        if _leading(regex, value) is None:
            findings.append(
                make_finding(
                    "error",
                    statement.line_no,
                    f"{prop} value '{value}' is not a number; Pythia accepts "
                    f"the line without "
                    f"a word and sets {prop} to 0",
                    line=line,
                    suggestion="use a numeric value",
                    code="particle_not_a_number",
                    source=source,
                )
            )
    return findings


# Prefixes whose flags switch on an internal hard process. With Pythia
# installed, Settings::hasHardProc() is consulted as well, but it cannot stand
# alone: it misses processes switched on through a vector of flags (the
# Charmonium/Bottomonium switches), with which init() succeeds.
_PROCESS_PREFIXES = (
    "SoftQCD:",
    "HardQCD:",
    "Top:",
    "Weak",
    "PromptPhoton:",
    "Charmonium:",
    "Bottomonium:",
    "Onia:",
    "HiggsSM:",
    "HiggsBSM:",
    "SUSY:",
    "HiddenValley:",
    "DM:",
    "ExtraDimensions",
    "LeftRightSymmmetry:",
    "LeptoQuark:",
    "NewGaugeBoson:",
    "ExcitedFermion:",
    "ContactInteractions:",
    "FourthGeneration:",
    "PhotonCollision:",
    "PhotonParton:",
    "Photon:",
    "LowEnergyQCD:",
    "TopThreshold:",
)


def _switched_on(value: str) -> bool:
    """True if a flag, or any element of a flag vector, is on."""
    items = value.strip("{}").split(",") if value.startswith("{") else [value]
    return any(item.strip().lower() in TRUE_WORDS for item in items)


def _has_process(resolved: Dict[str, str]) -> bool:
    return any(
        name.startswith(_PROCESS_PREFIXES) and _switched_on(value)
        for name, value in resolved.items()
    )


def _lint_card_semantics(
    resolved: Dict[str, str],
    statements: List[Statement],
    lhe_mode: Optional[bool],
    data: Dict[str, Any],
    has_process: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Cross-line checks, including how the card meets
    PythiaFromRunCardTool."""
    findings: List[Dict[str, Any]] = []
    first_line = {
        s.key: s.line_no for s in reversed(statements) if s.kind == "setting"
    }
    frame = (
        _leading(INT_RE, resolved.get("Beams:frameType", "1") or "1") or "1"
    )
    is_lhe = bool(lhe_mode) or frame == "4"

    if is_lhe:
        missing = [
            name
            for name in ("Beams:frameType", "Beams:LHEF")
            if name not in resolved
            or (name == "Beams:frameType" and frame != "4")
        ]
        if missing:
            findings.append(
                make_finding(
                    "note",
                    0,
                    "PythiaFromRunCardTool sets Beams:frameType = 4 and "
                    "Beams:LHEF from its "
                    "lhe_path, overriding the card; run anywhere else, this "
                    "card does not "
                    f"read an LHE file ({', '.join(missing)} not set for it)",
                    suggestion="pass lhe_path to the tool, or add "
                    "`Beams:frameType = 4` and "
                    "`Beams:LHEF = <file>` for use elsewhere",
                )
            )
    elif frame == "5":
        pass  # an external generator supplies the beams and the process
    else:
        energy = {
            "1": ("Beams:eCM",),
            "2": ("Beams:eA", "Beams:eB"),
            "3": ("Beams:pzA", "Beams:pzB"),
        }.get(frame, ("Beams:eCM",))
        if not set(energy) & set(resolved):
            default = next(
                (
                    r.get("default")
                    for r in data["settings"]
                    if r["name"] == energy[0]
                ),
                None,
            )
            findings.append(
                make_finding(
                    "warning",
                    0,
                    f"no beam energy set for Beams:frameType = {frame}; "
                    f"Pythia falls back to "
                    f"its default {energy[0]}"
                    + (f" = {default}" if default else ""),
                    suggestion=f"set {' and '.join(energy)}",
                )
            )
        if not (has_process or _has_process(resolved)):
            findings.append(
                make_finding(
                    "error",
                    0,
                    "no hard process switched on; Pythia aborts init with "
                    "'no process switched on'",
                    suggestion=(
                        "switch one on, e.g. `HardQCD:all = on`, "
                        "`SoftQCD:nonDiffractive = on`, "
                        "or `Top:all = on` -- or set `Beams:frameType = 4` to "
                        "read an LHE file"
                    ),
                )
            )

    if "Random:seed" in resolved:
        findings.append(
            make_finding(
                "note",
                first_line.get("random:seed", 0),
                "PythiaFromRunCardTool's `seed`, when given, overrides this "
                "Random:seed; the card's seed applies only when the tool is "
                "run without one",
            )
        )
        if "Random:setSeed" not in resolved or not _switched_on(
            resolved["Random:setSeed"]
        ):
            findings.append(
                make_finding(
                    "warning",
                    first_line.get("random:seed", 0),
                    "Random:seed has no effect unless Random:setSeed is on",
                    suggestion="add `Random:setSeed = on`",
                )
            )
    if "Main:numberOfEvents" in resolved:
        findings.append(
            make_finding(
                "note",
                first_line.get("main:numberofevents", 0),
                "Main:numberOfEvents is ignored by PythiaFromRunCardTool, "
                "which loops "
                "over its own `n_events` runtime field",
                suggestion="pass n_events to the tool; keep this line only "
                "for standalone main programs",
            )
        )
    return findings


class ValidatePythiaRunCardTool(BaseTool):
    """
    Validate a Pythia 8 `.cmnd` run card before generating events with it.

    Pythia does not raise on a bad card: it prints a complaint and makes
    `init()` return false, and it accepts some mistakes without a word -- a
    boolean typo is read as `off`. This tool checks every line and reports
    the offending ones by number.

    When pythia8mc is installed, the card is checked against that Pythia's own
    settings database and every line is also put to its own parser, so the
    verdict holds for the release that will run the card ("checked_by" in the
    result says so). Without it, the bundled database is used.

    Checks performed:
      - unknown setting names, with nearest-match suggestions
      - values of the wrong type, out of the declared min/max range, or outside
        a setting's documented option list
      - booleans Pythia does not recognise -- silently read as `off`
      - numbers Pythia truncates (`5.5` for an integer setting reads as 5)
      - particle-data and decay-channel lines (`6:m0 = 172.5`,
        `23:3:onMode = off`): unknown ids, properties and values
      - `include` files that Pythia will not find
      - a card with no hard process switched on, or no beam energy
      - settings assigned twice, and how the card meets PythiaFromRunCardTool

    Inputs (runtime):
      - cmnd_path: relative path to the card to validate
      - card_text: card contents inline (use instead of cmnd_path when the card
        has not been written to disk yet)
      - lhe_mode: True when the card will be run with shower_lhe=True; a card
        with Beams:frameType = 4 is treated as one either way
      - dry_run_init: if True, additionally run a real Pythia init() on the
        card. Generates no events. Returns:
          * Pythia's own messages when initialisation fails
          * "changed_settings": the EFFECTIVE configuration -- every setting
            whose value after init differs from its default
          * "overridden_settings": settings whose value after init is not what
            the card asked for. A `Tune:` line resets the parameters it owns
            when it is read, so a value set above it is silently discarded;
            this is the only way to detect that.
      - base_directory: sandbox root for all file operations

    Output (JSON):
      {
        "status": "ok",
        "valid": <bool>,
        "counts": {"error": 0, "warning": 1, "note": 0},
        "findings": [
          {"severity": "error", "line": 4, "message": "...",
           "text": "PhaseSpace:pTHatMinn = 25.",
           "suggestion": "did you mean ..."}
        ],
        "pythia_version": "8.317",
        "checked_by": "installed Pythia" | "static"
      }

    Notes:
      - `valid: true` means the card will be read and initialised. It does not
        mean the physics is what you intended.
    """

    # Optional fields take non-None defaults: RuntimeField(default=None) is
    # read by the schema generator as "no default", which marks the field
    # required.
    cmnd_path: str = RuntimeField(
        default="",
        description="Relative path to the .cmnd run card to validate",
    )
    card_text: str = RuntimeField(
        default="",
        description="Card contents inline, as an alternative to cmnd_path",
    )
    lhe_mode: bool = RuntimeField(
        default=False,
        description="True if this card will be used with shower_lhe=True; "
        "implied by Beams:frameType = 4",
    )
    dry_run_init: bool = RuntimeField(
        default=False,
        description="Also run a real Pythia init() to surface Pythia's own "
        "error text (generates no events)",
    )

    base_directory: str = StateField(
        description="Base directory for safe path resolution"
    )

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(
                f"Base directory does not exist: {self.base_directory}"
            )

    def _safe_path(self, rel: str) -> Optional[str]:
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        if full != self.base_directory and not full.startswith(
            self.base_directory + os.sep
        ):
            return None
        return full

    def _run(self) -> str:
        if not self.cmnd_path and not self.card_text:
            return self.format_error(
                error="Missing Parameter",
                reason="one of cmnd_path or card_text is required",
                suggestion="Pass the path to a .cmnd file, or the card "
                "contents inline",
            )

        card_path: Optional[str] = None
        if self.card_text:
            text = self.card_text
        else:
            card_path = self._safe_path(self.cmnd_path)
            if not card_path:
                return self.format_error(
                    error="Access Denied",
                    reason="Path escapes base_directory",
                    context=f"cmnd_path={self.cmnd_path}",
                    suggestion="Use a path inside the allowed base directory",
                )
            if not os.path.exists(card_path):
                return self.format_error(
                    error="File Not Found",
                    reason="Run card does not exist",
                    context=f"path={self.cmnd_path}",
                    suggestion="Provide a valid .cmnd file path",
                )
            try:
                text = Path(card_path).read_text(encoding="utf-8")
            except Exception as exc:
                return self.format_error(
                    error="Read Error",
                    reason=str(exc),
                    context=f"path={self.cmnd_path}",
                    suggestion="Verify the file exists and is readable",
                )

        card_dir = os.path.dirname(card_path) if card_path else None
        try:
            report = lint_card(
                text, lhe_mode=bool(self.lhe_mode) or None, card_dir=card_dir
            )
        except Exception as exc:
            return self.format_error(
                error="Database Unavailable",
                reason=str(exc),
                suggestion="Regenerate with `python -m "
                "tools.pythia.settings_db`",
            )

        result: Dict[str, Any] = {"status": "ok", **report}
        if self.cmnd_path and not self.card_text:
            result["cmnd_path"] = self.cmnd_path

        if self.dry_run_init:
            from .pythia_check import dry_run

            outcome = dry_run(text)
            effective = outcome.pop("effective", None)
            statements, _ = read_statements(text, card_dir=card_dir)
            if outcome.get("ran"):
                outcome["changed_settings"] = [
                    {
                        "name": v["name"],
                        "value": v["value"],
                        "default": v["default"],
                    }
                    for v in (effective or {}).values()
                    if v["value"] != v["default"]
                ]
                outcome["particle_data_changed"] = any(
                    s.kind == "particle" for s in statements
                )
            result["dry_run"] = outcome
            if outcome.get("init_ok") is False:
                result["valid"] = False

            # A value the card set that Pythia does not hold after init was
            # silently replaced. This cannot be found statically -- it needs
            # the initialised generator to report what it actually holds.
            if outcome.get("init_ok") and effective is not None:
                overridden = find_overridden_settings(statements, effective)
                for entry in overridden:
                    result["findings"].append(
                        make_finding(
                            "warning",
                            entry["line"],
                            f"{entry['name']} = {entry['requested']} did not "
                            f"survive "
                            f"initialisation; the effective value is "
                            f"{entry['effective']}",
                            suggestion=entry["suggestion"],
                        )
                    )
                if overridden:
                    result["counts"]["warning"] = result["counts"].get(
                        "warning", 0
                    ) + len(overridden)
                    result["overridden_settings"] = overridden

        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


# Settings that only control printout. Pythia rewrites these itself (setting
# Print:quiet resets every Init:show* and Next:* flag), so a difference between
# the card and the effective value there is not an override worth reporting.
_PRINTOUT_PREFIXES = ("print:", "init:show", "next:")


def find_overridden_settings(
    statements: List[Statement], effective: Dict[str, Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Settings whose value after init is not the one the card assigned.

    Each setting's last assignment outside any subrun block is compared with
    the value Pythia holds after init(), with the setting's own type
    semantics. Comparing values -- not merely checking whether a setting
    appears changed -- is what catches a tune overwriting a value with a
    different non-default one.

    Returns:
        [{"name", "line", "requested", "effective", "suggestion"}]
    """
    last: Dict[str, Statement] = {}
    for statement in statements:
        if (
            statement.kind == "setting"
            and statement.subrun == SUBRUN_DEFAULT
            and not statement.key.startswith(_PRINTOUT_PREFIXES)
            and statement.value not in ("DEFAULT", "?", "")
            and "force" not in statement.value
            and not statement.append
        ):
            last[statement.key] = statement

    overridden: List[Dict[str, Any]] = []
    for key, statement in last.items():
        held = effective.get(key)
        if held is None or _same_value(
            held["type"], statement.value, held["value"]
        ):
            continue
        tune_after = next(
            (
                s
                for s in statements
                if s.kind == "setting"
                and s.key.startswith("tune:")
                and s.line_no > statement.line_no
                and s.source == statement.source
            ),
            None,
        )
        overridden.append(
            {
                "name": held["name"],
                "line": statement.line_no,
                "requested": statement.value,
                "effective": held["value"],
                "suggestion": (
                    f"move `{tune_after.text.strip()}` (line "
                    f"{tune_after.line_no}) above this "
                    "setting -- a tune resets the parameters it owns when it "
                    "is read"
                    if tune_after
                    else "something read later reset it; check later lines "
                    "and any tune or model settings that own this parameter"
                ),
            }
        )
    return overridden


def _same_value(kind: str, requested: str, held: str) -> bool:
    """Compare a card value with an effective one, with Pythia's own
    reading."""
    if kind == "bool":
        return (requested.lower() in TRUE_WORDS) == (
            held.lower() in TRUE_WORDS
        )
    if kind in ("int", "float"):
        a = _leading(INT_RE if kind == "int" else FLOAT_RE, requested)
        b = _leading(FLOAT_RE, held)
        if a is None or b is None:
            return True  # an unreadable value is reported by the static checks
        x, y = float(a), float(b)
        return abs(x - y) <= 1e-6 * max(1.0, abs(x), abs(y))
    if kind.startswith("vector"):
        element = kind[len("vector<") : -1]
        left = [
            v.strip() for v in requested.strip("{}").split(",") if v.strip()
        ]
        right = [v.strip() for v in held.strip("{}").split(",") if v.strip()]
        return len(left) == len(right) and all(
            _same_value(element, a, b) for a, b in zip(left, right)
        )
    return requested == held
