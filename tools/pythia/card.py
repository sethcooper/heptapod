"""
# card.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Reading a Pythia 8 run card the way Pythia reads it.

Line semantics are matched to Settings::readString and
ParticleData::readString in the Pythia source:

  * A line is a comment unless its first non-blank character is a letter or a
    digit -- ``!``, ``#`` and anything else are ignored.
  * A leading digit routes the line to the particle data table
    (``6:m0 = 172.5``), including decay channels (``23:3:onMode = off``).
  * Names are case-insensitive and ``::`` is silently folded to ``:``.
  * A braced value may continue onto following lines until its ``}``.
  * ``include <file>`` reads another card, looked up in the working
    directory, $PYTHIA8CMND and Pythia's share directory -- not the card's
    own directory.
  * ``Main:subrun = N`` opens a block read only for that subrun.

Used by the validator, and by the generation tool to inline includes.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .settings_db import default_xmldoc_dir

# Values Pythia's boolString() accepts as true. Anything else reads as false,
# with no warning -- so a card is validated against both spellings and anything
# outside the union is reported.
TRUE_WORDS = {"true", "1", "on", "yes", "ok"}
FALSE_WORDS = {"false", "0", "off", "no"}

# Pythia's default subrun: lines outside any `Main:subrun` block.
SUBRUN_DEFAULT = -999

# Leading numbers, read the way istringstream reads them.
INT_RE = re.compile(r"[+-]?\d+")
FLOAT_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def make_finding(
    severity: str,
    line_no: int,
    message: str,
    *,
    line: str = "",
    suggestion: str = "",
    code: str = "",
    source: str = "",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "severity": severity,
        "line": line_no,
        "message": message,
    }
    if source:
        out["file"] = source
    if line:
        out["text"] = line.strip()
    if suggestion:
        out["suggestion"] = suggestion
    if code:
        out["code"] = code
    return out


def parse_card_line(raw: str) -> Optional[Tuple[str, str, str]]:
    """Split one card line the way Pythia does.

    Returns ``(kind, key, value)`` where kind is ``"setting"`` or
    ``"particle"``, or ``None`` for blank and comment lines. ``key`` is
    lowercased, matching Pythia's case-insensitive lookup.
    """
    stripped = raw.strip()
    if not stripped:
        return None
    first = stripped[0]
    if first.isdigit():
        kind = "particle"
    elif first.isalpha():
        kind = "setting"
    else:
        # Pythia treats every other leading character as a comment, which is
        # why both `!` and `#` work.
        return None

    # Pythia replaces '=' with a space before the first '{', then reads two
    # whitespace-separated tokens.
    brace = stripped.find("{")
    head = stripped if brace == -1 else stripped[:brace]
    tail = "" if brace == -1 else stripped[brace:]
    head = head.replace("+=", " ").replace("=", " ")
    tokens = head.split()
    if not tokens:
        return None

    key = tokens[0].lower()
    while "::" in key:
        key = key.replace("::", ":")

    if tail:
        close = tail.find("}")
        value = tail[: close + 1] if close != -1 else tail
    else:
        value = tokens[1] if len(tokens) > 1 else ""
    return kind, key, value


@dataclass
class Statement:
    """One statement of a card, as Pythia reads it.

    A braced value that continues onto later lines is one statement; `text`
    is what Pythia sees after joining them, `line_no` where it starts.
    """

    line_no: int
    text: str
    kind: str
    key: str
    value: str
    append: bool
    subrun: int
    source: str = ""

    @property
    def tokens(self) -> List[str]:
        brace = self.text.find("{")
        head = self.text if brace == -1 else self.text[:brace]
        return head.replace("+=", " ").replace("=", " ").split()


def include_search_path(card_dir: Optional[str]) -> List[Path]:
    """
    Directories an include is looked up in, in order.

    The card's own directory comes first: that is what a card's author means
    by a relative include, and it is where PythiaFromRunCardTool looks, since
    it inlines includes when it copies the card into its run directory. Then
    each $PYTHIA8CMND directory and Pythia's share directory, as Pythia does.
    Pythia itself would try its working directory first instead of the
    card's; the validator notes when that difference matters.
    """
    directories: List[Path] = [Path(card_dir)] if card_dir else []
    directories += [
        Path(d) for d in os.environ.get("PYTHIA8CMND", "").split(":") if d
    ]
    xmldoc = default_xmldoc_dir()
    if xmldoc is not None:
        directories.append(xmldoc.parent)
    return directories


def resolve_include(name: str, card_dir: Optional[str]) -> Optional[Path]:
    """The file an `include <name>` line reads, or None if there is none."""
    if Path(name).is_absolute():
        return Path(name) if Path(name).is_file() else None
    for directory in include_search_path(card_dir):
        if (directory / name).is_file():
            return directory / name
    return None


class IncludeError(ValueError):
    """An include that cannot be inlined: missing, outside the sandbox, or
    cyclic."""


def inline_includes(
    text: str,
    *,
    card_dir: Optional[str],
    allowed_roots: Sequence[Path],
    _depth: int = 0,
) -> str:
    """
    Replace every `include` line of a card with the file it includes.

    The result reads exactly as the original would, and does not depend on
    the working directory or on files that may later change or disappear, so
    it is a complete record of the run.

    Parameters:
        text: Card contents
        card_dir: Directory the card came from, for relative includes
        allowed_roots: Directories an included file must lie within, so a
            card cannot copy an arbitrary readable file into a run directory

    Raises:
        IncludeError: if an include is missing, outside allowed_roots, or
            nested more than 10 deep (an include cycle)
    """
    if _depth >= 10:
        raise IncludeError(
            "includes nested more than 10 deep; is a file including itself?"
        )
    roots = [Path(root).resolve() for root in allowed_roots]
    out: List[str] = []
    for line in text.splitlines():
        parsed = parse_card_line(line)
        if parsed is None or parsed[:2] != ("setting", "include"):
            out.append(line)
            continue
        tokens = line.split("!")[0].replace("=", " ").split()
        name = tokens[1] if len(tokens) > 1 else ""
        path = resolve_include(name, card_dir) if name else None
        if path is None:
            raise IncludeError(
                f"include file '{name}' not found in "
                + ", ".join(str(d) for d in include_search_path(card_dir))
            )
        resolved = path.resolve()
        if not any(
            resolved == root or root in resolved.parents for root in roots
        ):
            raise IncludeError(
                f"include file '{name}' is outside the allowed directories"
            )
        body = inline_includes(
            resolved.read_text(encoding="utf-8", errors="replace"),
            card_dir=str(resolved.parent),
            allowed_roots=roots,
            _depth=_depth + 1,
        )
        out += [
            f"! ---- begin {line.strip()}  ({resolved}) ----",
            body.rstrip("\n"),
            f"! ---- end include {name} ----",
        ]
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def read_statements(
    text: str,
    *,
    card_dir: Optional[str] = None,
    source: str = "",
    _depth: int = 0,
    _subrun: int = SUBRUN_DEFAULT,
) -> Tuple[List[Statement], List[Dict[str, Any]]]:
    """
    Split a card into the statements Pythia reads, expanding includes.

    Parameters:
        text: Card contents
        card_dir: Directory of the card on disk, if any, for include lookup

    Returns:
        The statements in reading order, and findings about the reading
        itself (an unterminated brace, an include that cannot be found)
    """
    statements: List[Statement] = []
    findings: List[Dict[str, Any]] = []
    subrun = _subrun
    pending: Optional[Tuple[int, str]] = None

    for line_no, raw in enumerate(text.splitlines(), start=1):
        if pending is not None:
            # Pythia appends the next line verbatim to an unfinished one.
            start, joined = pending
            joined += raw
            if "}" not in joined[joined.find("{") :]:
                pending = (start, joined)
                continue
            pending = None
            line_no, raw = start, joined
        parsed = parse_card_line(raw)
        if parsed is None:
            continue
        kind, key, value = parsed

        if kind == "setting" and value.startswith("{") and "}" not in value:
            pending = (line_no, raw)
            continue

        head = raw.split("{", 1)[0]
        statement = Statement(
            line_no,
            raw,
            kind,
            key,
            value,
            "+=" in head.replace(" ", ""),
            subrun,
            source,
        )

        if kind == "setting" and key == "main:subrun":
            number = INT_RE.match(value)
            if number:
                subrun = int(number.group(0))
        if kind == "setting" and key == "include":
            findings.extend(
                _expand_include(
                    statement, card_dir, _depth, statements, subrun
                )
            )
            continue
        statements.append(statement)

    if pending is not None:
        start, joined = pending
        findings.append(
            make_finding(
                "error",
                start,
                "a braced value is never closed; Pythia keeps appending the "
                "following "
                "lines to it, so this setting and every line after it are "
                "misread",
                line=joined[:120],
                suggestion="close the value with `}`",
                code="unterminated_brace",
                source=source,
            )
        )
    return statements, findings


def _expand_include(
    statement: Statement,
    card_dir: Optional[str],
    depth: int,
    statements: List[Statement],
    subrun: int,
) -> List[Dict[str, Any]]:
    """Read an included card in place, or explain why it cannot be found."""
    name = statement.tokens[1] if len(statement.tokens) > 1 else ""
    if not name:
        return [
            make_finding(
                "error",
                statement.line_no,
                "include with no file name",
                line=statement.text,
                code="no_value",
                source=statement.source,
            )
        ]
    if depth >= 10:
        return [
            make_finding(
                "error",
                statement.line_no,
                f"include nesting deeper than 10 at '{name}'; is it including "
                f"itself?",
                line=statement.text,
                source=statement.source,
            )
        ]

    path = resolve_include(name, card_dir)
    if path is None:
        return [
            make_finding(
                "error",
                statement.line_no,
                f"include file '{name}' not found; it is looked up next to "
                f"the card, in "
                "$PYTHIA8CMND and in Pythia's share directory, and a missing "
                "include fails "
                "the whole card",
                line=statement.text,
                suggestion="check the path, or give an absolute one",
                code="include_missing",
                source=statement.source,
            )
        ]

    findings: List[Dict[str, Any]] = []
    if (
        card_dir
        and not Path(name).is_absolute()
        and path.parent == Path(card_dir)
    ):
        findings.append(
            make_finding(
                "note",
                statement.line_no,
                f"'{name}' is found next to the card. PythiaFromRunCardTool "
                f"inlines it into "
                "run.cmnd; a standalone Pythia program looks in its working "
                "directory "
                "instead, so it must run from the card's directory",
                line=statement.text,
                source=statement.source,
            )
        )
    included, inner = read_statements(
        path.read_text(encoding="utf-8", errors="replace"),
        card_dir=str(path.parent),
        source=str(path),
        _depth=depth + 1,
        _subrun=subrun,
    )
    statements.extend(included)
    return findings + inner
