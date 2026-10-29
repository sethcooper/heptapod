"""
# quote_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Locate a statement in a fetched paper and return a citable locator for it.

Evidence that says "the paper states X" is only checkable if it says where:
which file of which version, which lines, which section, which equation. This
tool finds a pattern in a source the literature tools fetched and returns the
verbatim text with that locator and the sha256 of the file it was found in.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .arxiv_interface import split_version, validate_arxiv_id
from ..provenance import attach, file_entry, record_call, sha256_file

SCHEMA_VERSION = "quote-1.0"
MAX_LOCATIONS = 20
_TEXT_EXT = (".tex", ".ltx", ".bbl", ".txt")

_HEADING_RE = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection|paragraph)\*?\s*(?:\[[^\]]*\]\s*)?\{"
)
_LEVEL = {"part": 0, "chapter": 1, "section": 2, "subsection": 3, "subsubsection": 4, "paragraph": 5}
_ENVS = (
    "equation|align|alignat|eqnarray|gather|multline|flalign|subequations|"
    "table|figure"
)
_ENV_RE = re.compile(r"\\(begin|end)\{((?:" + _ENVS + r")\*?)\}")
_LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
_ROW_RE = re.compile(r"\\\\")
_PAGE_RE = re.compile(r"^%% page (\d+)\s*$", re.MULTILINE)


# ------------------------------------------------------------- TeX structure


def _commented(text: str, pos: int) -> bool:
    """True if ``pos`` lies after an unescaped ``%`` on its line."""
    i = text.rfind("\n", 0, pos) + 1
    while i < pos:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "%":
            return True
        i += 1
    return False


def _braced(text: str, open_pos: int) -> Optional[str]:
    """Contents of the brace group whose ``{`` is at ``open_pos``."""
    depth = 0
    for i in range(open_pos, len(text)):
        c = text[i]
        if c == "\\":
            continue
        if c == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif c == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[open_pos + 1:i]
    return None


def _headings(text: str) -> List[Tuple[int, int, str]]:
    out = []
    for m in _HEADING_RE.finditer(text):
        if _commented(text, m.start()):
            continue
        title = _braced(text, m.end() - 1)
        if title is not None:
            out.append((m.start(), _LEVEL[m.group(1)], " ".join(title.split())))
    return out


def _section_at(heads: List[Tuple[int, int, str]], pos: int) -> Optional[str]:
    """Title of the innermost heading in force at ``pos``."""
    current: Dict[int, str] = {}
    for start, level, title in heads:
        if start > pos:
            break
        current = {k: v for k, v in current.items() if k < level}
        current[level] = title
    return current[max(current)] if current else None


def _env_spans(text: str) -> List[Tuple[int, int, str]]:
    stack: List[Tuple[str, int]] = []
    spans = []
    for m in _ENV_RE.finditer(text):
        if _commented(text, m.start()):
            continue
        kind, name = m.group(1), m.group(2)
        if kind == "begin":
            stack.append((name, m.start()))
        else:
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][0] == name:
                    spans.append((stack[j][1], m.end(), name))
                    del stack[j:]
                    break
    return spans


def _label_at(text: str, spans: List[Tuple[int, int, str]], start: int, end: int) -> Optional[str]:
    """The \\label of the innermost equation/table environment around a match.

    In a multi-row environment (align, eqnarray) with several labels, the
    label of the row containing the match; None if that row has none.
    """
    around = sorted((s for s in spans if s[0] <= start and end <= s[1]), key=lambda s: s[1] - s[0])
    for s0, s1, _name in around:
        labels = [(m.start(), m.group(1)) for m in _LABEL_RE.finditer(text, s0, s1)
                  if not _commented(text, m.start())]
        if not labels:
            continue
        if len(labels) == 1:
            return labels[0][1]
        rows = [s0] + [m.end() for m in _ROW_RE.finditer(text, s0, s1)] + [s1]
        anchor = max(start, end - 1)
        for a, b in zip(rows, rows[1:]):
            if a <= anchor < b:
                in_row = [lab for p, lab in labels if a <= p < b]
                return in_row[0] if in_row else None
        return None
    return None


def _page_at(text: str, pos: int) -> Optional[int]:
    page = None
    for m in _PAGE_RE.finditer(text):
        if m.start() > pos:
            break
        page = int(m.group(1))
    return page


# ------------------------------------------------------------------ the tool


class QuoteSourceTool(BaseTool):
    """
    Find a statement in a fetched paper and return it verbatim with a locator.

    Input:
        source: a versioned arXiv id previously fetched with ArxivSourceTool /
                ArxivPDFTool + PDFToTeXTool (e.g. "2103.02708v2"), or a path
                under base_directory to a fetched source directory, a single
                .tex file, or PDFToTeX output.
        pattern: the text to find, literally (whitespace runs may differ from
                 the file's line wrapping); a regular expression if regex=true.
        regex: treat pattern as a Python regular expression (default false).
        context_chars: characters of surrounding text to return as "context"
                       (default 0: none).
        output_dir: corpus root the arXiv tools wrote to (default "papers").

    For an arXiv id the ORIGINAL source files (source/) are searched first,
    since their line numbers are the citable ones; then the normalized
    source.tex; then PDFToTeX output in the paper directory.

    Returns JSON:
        {"status": "ok", "source_id": "arXiv:2103.02708v2",
         "source_sha256": "<sha256 of the file the quote is in>",
         "file": "papers/2103.02708v2/source/ms.tex",
         "line_start": 120, "line_end": 121,      (1-based, inclusive)
         "char_start": 5012, "char_end": 5090,    (0-based, end exclusive)
         "section": "The model", "label": "eq:lag", "page": null,
         "quote": "<verbatim match>", "match_mode": "exact",
         "provenance": {...}}

        An absent pattern returns status "error" with the files searched; an
        ambiguous one returns status "error" with every match's location
        (capped at 20), so the pattern can be lengthened or the file named.
    """

    # ======================== Runtime fields ======================== #
    source: str = RuntimeField(
        description=(
            "Versioned arXiv id already fetched (e.g. '2103.02708v2'), or a path "
            "under base_directory to a fetched source dir/file or PDFToTeX output"
        )
    )
    pattern: str = RuntimeField(
        description="Text to locate (literal; whitespace-insensitive fallback), or a regex if regex=true"
    )
    regex: Optional[bool] = RuntimeField(
        default=False, description="Treat pattern as a Python regular expression"
    )
    context_chars: Optional[int] = RuntimeField(
        default=0, description="Characters of surrounding context to return (default 0)"
    )
    output_dir: Optional[str] = RuntimeField(
        default="papers", description="Corpus root the arXiv tools wrote to (default 'papers')"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    # --------------------------------------------------------------- run

    def _run(self) -> str:
        base = os.path.realpath(self.base_directory)
        request = {"source": self.source, "pattern": self.pattern, "regex": bool(self.regex),
                   "context_chars": self.context_chars or 0}
        result, files_used = self._quote(base)
        source_ids = [result.get("source_id")] if result.get("status") == "ok" else []
        files = [file_entry(base, f) for f in files_used]
        prov = record_call(base, "QuoteSource", request, result,
                           api={"name": "local", "version": SCHEMA_VERSION},
                           source_ids=source_ids, files=files or None)
        return attach(result, prov)

    @staticmethod
    def _error(error: str, reason: str, **extra: Any) -> Dict[str, Any]:
        return dict({"status": "error", "schema": SCHEMA_VERSION, "error": error, "reason": reason}, **extra)

    def _quote(self, base: str) -> Tuple[Dict[str, Any], List[str]]:
        if not self.pattern:
            return self._error("Missing Parameter", "pattern is empty"), []
        tiers, err = self._candidates(base)
        if err:
            return err, []
        if not any(tiers):
            return self._error("No Text Files", f"nothing to search in {self.source!r}"), []

        try:
            matchers = self._matchers()
        except re.error as e:
            return self._error("Invalid Regex", str(e)), []

        texts: Dict[str, str] = {}
        for mode, rx in matchers:
            for tier in tiers:
                hits = []
                for path in tier:
                    if path not in texts:
                        try:
                            with open(path, encoding="utf-8", errors="replace", newline="") as fh:
                                texts[path] = fh.read()
                        except OSError:
                            texts[path] = ""
                    for m in rx.finditer(texts[path]):
                        if m.end() > m.start():
                            hits.append((path, m.start(), m.end()))
                if hits:
                    return self._report(base, hits, texts, mode)

        searched = [os.path.relpath(p, base) for tier in tiers for p in tier]
        return self._error(
            "Pattern Not Found",
            f"{self.pattern!r} does not occur in {self.source!r}",
            searched=searched[:50],
            suggestion=("Quote a shorter verbatim fragment; equations in the source are LaTeX "
                        "(e.g. '\\bar{\\psi}'), not rendered symbols"),
        ), []

    def _matchers(self) -> List[Tuple[str, "re.Pattern[str]"]]:
        if self.regex:
            return [("regex", re.compile(self.pattern, re.MULTILINE))]
        out = [("exact", re.compile(re.escape(self.pattern)))]
        tokens = self.pattern.split()
        if tokens and r"\s+".join(map(re.escape, tokens)) != re.escape(self.pattern):
            out.append(("whitespace-insensitive", re.compile(r"\s+".join(map(re.escape, tokens)))))
        return out

    def _locate(self, base: str, path: str, text: str, start: int, end: int) -> Dict[str, Any]:
        return {
            "file": os.path.relpath(path, base),
            "line_start": text.count("\n", 0, start) + 1,
            "line_end": text.count("\n", 0, max(start, end - 1)) + 1,
            "char_start": start,
            "char_end": end,
            "section": _section_at(_headings(text), start),
            "label": _label_at(text, _env_spans(text), start, end),
            "page": _page_at(text, start),
        }

    def _report(self, base, hits, texts, mode) -> Tuple[Dict[str, Any], List[str]]:
        if len(hits) > 1:
            locs = [self._locate(base, p, texts[p], s, e) for p, s, e in hits[:MAX_LOCATIONS]]
            return self._error(
                "Ambiguous Pattern",
                f"{self.pattern!r} occurs {len(hits)} times",
                count=len(hits),
                matches=locs,
                truncated=len(hits) > MAX_LOCATIONS,
                match_mode=mode,
                suggestion="Lengthen the pattern, or pass the specific file as source",
            ), []
        path, s, e = hits[0]
        text = texts[path]
        result = {"status": "ok", "schema": SCHEMA_VERSION,
                  "source_id": self._source_id(base, path),
                  "source_sha256": sha256_file(path)}
        result.update(self._locate(base, path, text, s, e))
        result["quote"] = text[s:e]
        result["match_mode"] = mode
        c = max(0, int(self.context_chars or 0))
        if c:
            result["context"] = text[max(0, s - c):e + c]
        return result, [path]

    # ---------------------------------------------------- what to search

    def _candidates(self, base: str) -> Tuple[List[List[str]], Optional[Dict[str, Any]]]:
        src = (self.source or "").strip()
        valid = validate_arxiv_id(src)
        if valid and not os.path.exists(os.path.join(base, src)):
            root = os.path.join(base, self.output_dir or "papers")
            if not os.path.realpath(root).startswith(base):
                return [], self._error("Access Denied", "output_dir escapes base_directory")
            id_base, version = split_version(valid)
            flat = id_base.replace("/", "_")
            fetched = sorted(d for d in (os.listdir(root) if os.path.isdir(root) else [])
                             if re.fullmatch(re.escape(flat) + r"v\d+", d))
            if not version:
                return [], self._error(
                    "Unversioned Id",
                    "quote from a specific version, e.g. " + (fetched[-1] if fetched else f"{valid}v1"),
                    fetched_versions=fetched,
                )
            paper = os.path.join(root, valid.replace("/", "_"))
            if not os.path.isdir(paper):
                return [], self._error(
                    "Not Fetched", f"{valid} has not been fetched into {self.output_dir or 'papers'}/",
                    fetched_versions=fetched,
                    suggestion="Run ArxivSourceTool (or ArxivPDFTool + PDFToTeXTool) first",
                )
            return self._paper_tiers(paper), None

        full = os.path.realpath(os.path.join(base, src))
        if full != base and not full.startswith(base + os.sep):
            return [], self._error("Access Denied", "source escapes base_directory")
        if os.path.isfile(full):
            return [[full]], None
        if os.path.isdir(full):
            if os.path.isdir(os.path.join(full, "source")) or os.path.isfile(os.path.join(full, "retrieval.json")):
                return self._paper_tiers(full), None
            return [self._text_files(full)], None
        return [], self._error("Source Not Found", f"no such arXiv fetch or path: {src!r}")

    @staticmethod
    def _text_files(d: str) -> List[str]:
        out = []
        for root, dirs, files in os.walk(d):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(root, f)
                if f.lower().endswith(_TEXT_EXT) and not os.path.islink(p):
                    out.append(p)
        return out

    def _paper_tiers(self, paper: str) -> List[List[str]]:
        """Originals first (citable lines), then source.tex, then PDF extracts."""
        originals = self._text_files(os.path.join(paper, "source"))
        normalized = [p for p in [os.path.join(paper, "source.tex")] if os.path.isfile(p)]
        extracts = sorted(
            os.path.join(paper, f) for f in os.listdir(paper)
            if f.lower().endswith(".tex") and f != "source.tex"
            and os.path.isfile(os.path.join(paper, f))
        )
        return [t for t in (originals, normalized, extracts) if t]

    @staticmethod
    def _source_id(base: str, path: str) -> Optional[str]:
        """Versioned id of the fetch a file belongs to, from the sidecars the
        literature tools write (``<file>.provenance.json``, ``retrieval.json``)."""
        try:
            with open(path + ".provenance.json", encoding="utf-8") as fh:
                sid = json.load(fh).get("source_id")
            if sid:
                return sid
        except (OSError, ValueError, AttributeError):
            pass
        d = os.path.dirname(path)
        while d.startswith(base):
            meta = os.path.join(d, "retrieval.json")
            if os.path.isfile(meta):
                try:
                    with open(meta, encoding="utf-8") as fh:
                        return json.load(fh).get("source_id")
                except (OSError, ValueError, AttributeError):
                    return None
            if d == base:
                break
            d = os.path.dirname(d)
        return None
