"""
# literature_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

arXiv literature tools for agent use.

Provides BaseTool implementations for searching arXiv, downloading paper PDFs,
and extracting their full text. These complement the INSPIRE tools, which cover
metadata/citation search but neither arXiv nor full-text retrieval — the text an
agent needs to identify and extract a BSM Lagrangian.
"""

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .arxiv_interface import (
    ArxivInterface,
    arxiv_id_from_url,
    is_allowed_pdf_url,
    split_version,
    validate_arxiv_id,
    version_from_response,
)
from ..digest import clip, digest
from ..provenance import (
    arxiv_source_id,
    attach,
    file_entry,
    now_iso,
    record_call,
)
from .source_archive import (
    detect_payload,
    gunzip_capped,
    inline_inputs,
    resolve_main_tex,
    safe_extract_tar,
    strip_comments,
)

SCHEMA_VERSION = "literature-1.3"
ARXIV_API = {"name": "arXiv", "version": None}

#: Per-paper record of what was fetched, from where, when, with which hash.
RETRIEVAL_FILE = "retrieval.json"

_SEARCH_SORT_BY = {"relevance", "submittedDate", "lastUpdatedDate"}
_SEARCH_SORT_ORDER = {"ascending", "descending"}


def _safe_join(base_directory: str, rel_or_abs: str) -> Optional[str]:
    """Resolve ``rel_or_abs`` under ``base_directory``; return None if it escapes.

    Uses ``realpath`` on both sides so a symlinked component cannot redirect a
    write/read outside the sandbox (see CONTRIBUTING.md "Path Safety and
    Sandboxing").
    """
    if not rel_or_abs:
        return None
    base = os.path.realpath(base_directory)
    full = os.path.realpath(os.path.join(base, rel_or_abs))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full


def _pdf_stem(arxiv_id: Optional[str], pdf_url: Optional[str]) -> Optional[str]:
    """Deterministic, collision-free filename stem for a PDF.

    Derived from a validated arXiv id (version preserved, path separators
    flattened) or, when only a URL is given, from a hash of the URL — so two
    different sources never map to the same cached file. Returns None if an
    ``arxiv_id`` is supplied but malformed.
    """
    if arxiv_id:
        valid = validate_arxiv_id(arxiv_id)
        if not valid:
            return None
        return valid.replace("/", "_").replace("\\", "_")
    if pdf_url:
        return "url_" + hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:16]
    return None


def _paper_dir(base_directory: str, output_dir: Optional[str], stem: str) -> Optional[str]:
    """Absolute path of the per-paper directory, or None if it escapes.

    Everything retrieved for one paper lives together under
    ``<output_dir>/<stem>/`` — the PDF, the extracted source tree, and the
    normalized LaTeX — so a corpus is one directory per paper rather than the
    same id scattered across three parallel trees.
    """
    root = _safe_join(base_directory, output_dir or "papers")
    if root is None:
        return None
    return os.path.join(root, stem)


def _sha256_file(path: str) -> str:
    """Chunked SHA-256 of a file (avoids reading large PDFs fully into memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _flat(arxiv_id: str) -> str:
    """Filesystem-safe form of an arXiv id (old-style ids contain a slash)."""
    return arxiv_id.replace("/", "_").replace("\\", "_")


def _read_meta(paper_dir: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(paper_dir, RETRIEVAL_FILE), encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_meta(paper_dir: str, source_id: Optional[str], kind: str, entry: Dict[str, Any]) -> None:
    meta = _read_meta(paper_dir)
    if source_id:
        meta["source_id"] = source_id
    meta[kind] = entry
    try:
        with open(os.path.join(paper_dir, RETRIEVAL_FILE), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except OSError:
        pass


def _mtime_iso(path: str) -> str:
    return now_iso(datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc))


def _resolve(interface: ArxivInterface, arxiv_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(versioned_id, metadata_request_url, error)`` for an arXiv id.

    "Latest" is pinned to a concrete vN BEFORE any cache lookup, so a cached
    copy of v1 is never served for a paper that has since gained a v2.
    """
    try:
        res = interface.resolve_version(arxiv_id)
        return res["versioned_id"], res["request_url"], None
    except Exception as e:  # noqa: BLE001 - network, parse, or no version
        return None, None, str(e)


def _with_provenance(
    base_directory: str,
    tool: str,
    request: Dict[str, Any],
    result: Dict[str, Any],
    http: List[str],
    **kw: Any,
) -> str:
    if len(http) > 1:
        request = dict(request, http_requests=list(http))
    prov = record_call(base_directory, tool, request, result, api=ARXIV_API, **kw)
    return attach(result, prov)


# ============================ Search ============================ #


class ArxivSearchTool(BaseTool):
    """
    Search arXiv for physics papers via the Atom export API.

    Use this to discover candidate papers for a BSM scenario when you need the
    arXiv preprint (and, via the other literature tools, its full text). For
    citation-ranked metadata search, prefer InspireSearchTool; use this when you
    need arXiv ids / PDF links or an abstract-level keyword search.

    Input:
        query: A raw arXiv search_query (e.g. 'cat:hep-ph AND ti:"leptoquark"').
               If omitted, one is built from model_name/keywords/category.
        model_name: BSM scenario or model name (used to build a query when
                    `query` is not given), e.g. "scalar leptoquark".
        keywords: Extra abstract keywords to AND into a built query.
        category: arXiv category filter for a built query (default "hep-ph").
        max_results: Max papers to return (default 25, max 100).
        sort_by: "relevance", "submittedDate", or "lastUpdatedDate".
        sort_order: "descending" or "ascending".

    Returns a JSON digest:
        {"status": "ok", "schema": "literature-1.3", "query": "...",
         "request_url": "...", "count": N,
         "results_path": "research/results/<call_id>.json",
         "papers": [{"id": "2103.02708v2", "title", "authors", "published",
                     "category"}, ...],
         "omitted": N, "note", "provenance": {...}}

        ``id`` is versioned: the version the metadata describes (the latest
        at query time). Titles are cut to 150 characters; ``authors`` is the
        first author, or the first two, with "et al." when there are more;
        ``category`` is the primary category. The file at results_path holds
        every paper in full: {"arxiv_id", "version", "versioned_id",
        "source_id" ("arXiv:<id>v<N>"), "title", "authors", "abstract",
        "categories", "doi", "published", "updated", "pdf_url", "abs_url"}.
        ``omitted`` (present only when non-zero) counts papers left out of
        the digest to keep it short; they are in the file.
    """

    # ======================== Runtime fields ======================== #
    query: Optional[str] = RuntimeField(
        default=None,
        description="Raw arXiv search_query; if omitted it is built from model_name/keywords/category",
    )
    model_name: Optional[str] = RuntimeField(
        default=None,
        description="BSM model/scenario name used to build a query when `query` is not provided",
    )
    keywords: Optional[List[str]] = RuntimeField(
        default=None,
        description="Extra abstract keywords to AND into a built query",
    )
    category: Optional[str] = RuntimeField(
        default="hep-ph",
        description="arXiv category filter for a built query (default hep-ph)",
    )
    max_results: Optional[int] = RuntimeField(
        default=25, description="Max papers to return (default 25, max 100)"
    )
    sort_by: Optional[str] = RuntimeField(
        default="relevance",
        description="Sort field: 'relevance', 'submittedDate', or 'lastUpdatedDate'",
    )
    sort_order: Optional[str] = RuntimeField(
        default="descending", description="'descending' or 'ascending'"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations",
        default=".",
    )
    # ================================================================ #

    def _run(self) -> str:
        query = self.query
        if not query:
            if not self.model_name and not self.keywords:
                return self.format_error(
                    error="Missing Parameter",
                    reason="Provide `query`, or `model_name`, or `keywords`",
                    suggestion="Pass a raw arXiv query, or a model_name and/or keywords to build one",
                )
            query = ArxivInterface.build_query(
                self.model_name, self.keywords, self.category
            )

        sort_by = self.sort_by or "relevance"
        sort_order = self.sort_order or "descending"
        if sort_by not in _SEARCH_SORT_BY:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"sort_by must be one of {sorted(_SEARCH_SORT_BY)}",
                context=f"sort_by={self.sort_by}",
            )
        if sort_order not in _SEARCH_SORT_ORDER:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"sort_order must be one of {sorted(_SEARCH_SORT_ORDER)}",
                context=f"sort_order={self.sort_order}",
            )
        max_results = max(1, min(self.max_results or 25, 100))

        try:
            interface = ArxivInterface()
            papers, url = interface.search(
                query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        except Exception as e:  # noqa: BLE001 - surface network/parse errors to the agent
            return self.format_error(
                error="arXiv Search Failed",
                reason=str(e),
                context=f"query={query}",
                suggestion="Check the query syntax or retry (arXiv may be rate-limiting)",
            )

        result = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "query": query,
            "request_url": url,
            "count": len(papers),
            "papers": papers,
        }
        prov = record_call(
            self.base_directory,
            "ArxivSearch",
            {"query": query, "max_results": max_results, "sort_by": sort_by,
             "sort_order": sort_order},
            result,
            api=ARXIV_API,
            request_url=url,
            source_ids=[p.get("source_id") for p in papers],
            raw=interface.last_raw,
            raw_ext="xml",
        )
        compact = [{
            "id": p.get("versioned_id") or p.get("arxiv_id"),
            "title": clip(p.get("title"), 150),
            "authors": clip(_short_authors(p.get("authors") or []), 80),
            "published": p.get("published"),
            "category": (p.get("categories") or [None])[0],
        } for p in papers]
        head = {k: result[k] for k in ("status", "schema", "query", "request_url", "count")}
        return digest(self.base_directory, result, prov, head, "papers", compact, what="papers")


def _short_authors(authors: List[str]) -> Optional[str]:
    """First author, first two, or the first two with "et al." for more."""
    if not authors:
        return None
    if len(authors) <= 2:
        return " and ".join(authors)
    return f"{authors[0]}, {authors[1]} et al."


# ========================= PDF retrieval ======================== #


class ArxivPDFTool(BaseTool):
    """
    Download an arXiv paper PDF into the sandbox for later text extraction.

    Input:
        arxiv_id: arXiv identifier, e.g. "2103.02708" or "2103.02708v2". An
                  unversioned id is pinned to the current version (from the
                  arXiv metadata) before anything is fetched or reused.
        pdf_url: Explicit HTTPS arXiv PDF URL, used only when arxiv_id is not
                 given (must be on an arXiv host). If both are given, arxiv_id
                 wins.
        output_dir: Corpus root (relative to base_directory), one directory
                    per paper version beneath it (default "papers").

    Behavior:
        Downloads to {output_dir}/{id}v{N}/{id}v{N}.pdf under base_directory,
        alongside any source ArxivSourceTool has fetched for the same version,
        reusing a cached valid PDF of that exact version if present. Validates
        the %PDF- header. The fetch (URL, time, sha256) is recorded in
        {output_dir}/{id}v{N}/retrieval.json.

    Returns JSON:
        {"status": "ok", "schema": "literature-1.3", "arxiv_id": "2103.02708v2",
         "requested_id": "2103.02708", "source_id": "arXiv:2103.02708v2",
         "pdf_path": "papers/2103.02708v2/2103.02708v2.pdf", "bytes": N,
         "sha256": "...", "url": "https://arxiv.org/pdf/2103.02708v2.pdf",
         "retrieved_at": "...Z", "cached": false, "provenance": {...}}
    """

    # ======================== Runtime fields ======================== #
    arxiv_id: Optional[str] = RuntimeField(
        default=None,
        description="arXiv identifier, e.g. '2103.02708' (latest) or '2103.02708v2'",
    )
    pdf_url: Optional[str] = RuntimeField(
        default=None,
        description="Explicit HTTPS arXiv PDF URL; used only when arxiv_id is not given",
    )
    output_dir: Optional[str] = RuntimeField(
        default="papers",
        description=(
            "Corpus root relative to base_directory; the PDF lands in "
            "<output_dir>/<arxiv_id>v<N>/ next to any source fetched for it"
        ),
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _run(self) -> str:
        if not self.arxiv_id and not self.pdf_url:
            return self.format_error(
                error="Missing Parameter",
                reason="Provide either `arxiv_id` or `pdf_url`",
            )
        if self.arxiv_id and validate_arxiv_id(self.arxiv_id) is None:
            return self.format_error(
                error="Invalid Parameter",
                reason="arxiv_id is not a well-formed arXiv identifier",
                context=f"arxiv_id={self.arxiv_id}",
                suggestion="Use e.g. '2103.02708' or 'hep-ph/9905221' (version optional)",
            )
        if self.pdf_url and not is_allowed_pdf_url(self.pdf_url):
            return self.format_error(
                error="Access Denied",
                reason="pdf_url must be an HTTPS URL on an arXiv host",
                context=f"pdf_url={self.pdf_url}",
                suggestion="Pass an arxiv_id, or an https://arxiv.org/... PDF URL",
            )
        root = _safe_join(self.base_directory, self.output_dir or "papers")
        if root is None:
            return self.format_error(
                error="Access Denied",
                reason="output_dir escapes base_directory",
                context=self.output_dir,
                suggestion="Use a relative path inside base_directory",
            )

        base = os.path.realpath(self.base_directory)
        request = {"arxiv_id": self.arxiv_id, "pdf_url": self.pdf_url,
                   "output_dir": self.output_dir}
        requested = validate_arxiv_id(self.arxiv_id) if self.arxiv_id else arxiv_id_from_url(self.pdf_url)
        interface = ArxivInterface()
        http: List[str] = []

        versioned, resolve_err = None, None
        if requested:
            versioned, meta_url, resolve_err = _resolve(interface, requested)
            if meta_url:
                http.append(meta_url)

        if versioned:
            stem = _flat(versioned)
        elif requested:
            # Metadata unreachable: fetch the current PDF and read its version
            # from what arXiv names the file; place it once that is known.
            stem = None
        else:
            stem = _pdf_stem(None, self.pdf_url)

        try:
            os.makedirs(root, exist_ok=True)
            if stem is not None:
                out_dir = os.path.join(root, stem)
                os.makedirs(out_dir, exist_ok=True)
                dest = os.path.join(out_dir, f"{stem}.pdf")
                cached = self._cached(dest, out_dir, versioned)
                if cached is not None:
                    return _with_provenance(
                        base, "ArxivPDF", request, cached, http,
                        request_url=cached["url"], from_cache=True,
                        cached_at=cached["retrieved_at"],
                        source_ids=[cached["source_id"]],
                        files=[file_entry(base, dest, cached["sha256"])],
                    )
            else:
                dest = os.path.join(root, f".download-{secrets.token_hex(4)}.pdf")
        except OSError as e:
            return self.format_error(
                error="Filesystem Error",
                reason=str(e),
                context=f"output_dir={self.output_dir}",
            )

        fetch_id = versioned or requested
        try:
            if fetch_id:
                meta = interface.download_pdf(dest, arxiv_id=fetch_id)
            else:
                meta = interface.download_pdf(dest, pdf_url=self.pdf_url)
        except Exception as e:  # noqa: BLE001
            return self.format_error(
                error="PDF Download Failed",
                reason=str(e),
                context=f"arxiv_id={self.arxiv_id} pdf_url={self.pdf_url}",
                suggestion="Verify the arXiv id/URL; arXiv may be rate-limiting",
            )
        http.append(meta["url"])
        retrieved_at = now_iso()

        if requested:
            id_base, want = split_version(fetch_id)
            served = version_from_response(id_base, meta.get("content_disposition"), meta.get("final_url"))
            if want and served and served != want:
                os.remove(dest)
                return self.format_error(
                    error="Version Mismatch",
                    reason=f"asked for {fetch_id}, arXiv served {id_base}{served}",
                    suggestion="Retry, or pass the version you want explicitly",
                )
            if not versioned:
                if not served:
                    os.remove(dest)
                    return self.format_error(
                        error="Version Resolution Failed",
                        reason=f"could not determine which version of {requested} arXiv serves",
                        context=resolve_err or "",
                        suggestion="Pass an explicit version, e.g. '2103.02708v2'",
                    )
                versioned = f"{id_base}{served}"
                stem = _flat(versioned)
                out_dir = os.path.join(root, stem)
                os.makedirs(out_dir, exist_ok=True)
                final = os.path.join(out_dir, f"{stem}.pdf")
                os.replace(dest, final)
                dest = final
        else:
            out_dir = os.path.dirname(dest)

        source_id = arxiv_source_id(versioned)
        rel_dest = os.path.relpath(dest, base)
        _write_meta(out_dir, source_id, "pdf", {
            "url": meta["url"], "final_url": meta.get("final_url"),
            "retrieved_at": retrieved_at, "sha256": meta["sha256"],
            "bytes": meta["bytes"], "path": rel_dest,
        })
        result = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "arxiv_id": versioned,
            "requested_id": self.arxiv_id or self.pdf_url,
            "source_id": source_id,
            "pdf_path": rel_dest,
            "bytes": meta["bytes"],
            "sha256": meta["sha256"],
            "url": meta["url"],
            "retrieved_at": retrieved_at,
            "cached": False,
        }
        return _with_provenance(
            base, "ArxivPDF", request, result, http,
            request_url=meta["url"], source_ids=[source_id],
            files=[file_entry(base, dest, meta["sha256"])],
        )

    def _cached(self, dest: str, out_dir: str, versioned: Optional[str]) -> Optional[Dict[str, Any]]:
        """Result for a valid cached PDF of this exact version, else None.

        Never follows a symlink at the leaf (a download would os.replace it
        safely, but a cache read must not escape the sandbox).
        """
        if not (os.path.exists(dest) and not os.path.islink(dest) and os.path.getsize(dest) > 0):
            return None
        with open(dest, "rb") as fh:
            if fh.read(5) != b"%PDF-":
                return None
        meta = _read_meta(out_dir).get("pdf", {})
        url = meta.get("url") or (f"https://arxiv.org/pdf/{versioned}.pdf" if versioned else self.pdf_url)
        return {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "arxiv_id": versioned,
            "requested_id": self.arxiv_id or self.pdf_url,
            "source_id": arxiv_source_id(versioned),
            "pdf_path": os.path.relpath(dest, os.path.realpath(self.base_directory)),
            "bytes": os.path.getsize(dest),
            "sha256": _sha256_file(dest),
            "url": url,
            "retrieved_at": meta.get("retrieved_at") or _mtime_iso(dest),
            "cached": True,
        }


# ======================= LaTeX source (e-print) ================= #

_EPRINT_EXT = {"tar": ".tar", "tex": ".tex", "gzip-tar": ".tar.gz", "gzip-tex": ".tex.gz"}


class ArxivSourceTool(BaseTool):
    """
    Fetch a paper's LaTeX SOURCE from arXiv (e-print) — the preferred input for
    Lagrangian extraction, since PDF text mangles equations while the .tex
    source preserves them exactly.

    Input:
        arxiv_id: arXiv identifier, e.g. "2103.02708" or "2103.02708v2". An
                  unversioned id is pinned to the current version before any
                  cache lookup or download.
        output_dir: Corpus root (relative to base_directory), default "papers".
        inline_one_level: Inline \\input{...}/\\include{...} one level into the
                          main file (default true).
        preview_chars: Length of the inline LaTeX preview returned (default 2000).

    Behavior:
        Downloads https://export.arxiv.org/e-print/<id>v<N> (size-capped),
        keeps the payload as <output_dir>/<id>v<N>/<id>v<N>.eprint.<ext>,
        extracts archives with strict safety checks into source/ (the original
        files, verbatim, so line numbers can be cited), resolves the main .tex,
        strips % comments, optionally inlines one level of \\input, and writes
        that normalized LaTeX to source.tex. retrieval.json records the URL,
        time and sha256 of the payload.

    Returns JSON:
        {"status": "ok", "schema": "literature-1.3", "arxiv_id": "2103.02708v2",
         "requested_id": "2103.02708", "source_id": "arXiv:2103.02708v2",
         "source_type": "tar"|"single_tex"|"pdf_only",
         "source_dir": ".../source", "main_tex": ".../source/ms.tex",
         "tex_path": ".../source.tex", "n_files": N, "chars": M,
         "sha256": "<e-print payload>", "eprint_path": "...",
         "tex_sha256": "<source.tex>", "url": "...", "retrieved_at": "...Z",
         "preview": "...", "cached": bool, "provenance": {...}}

        source_type "pdf_only" means arXiv has no LaTeX source for this paper.
        That is a normal outcome, not an error: the PDF arXiv served instead is
        saved to "pdf_path" in the same directory, ready for PDFToTeXTool.
    """

    # ======================== Runtime fields ======================== #
    arxiv_id: str = RuntimeField(
        description="arXiv identifier, e.g. '2103.02708' (latest) or '2103.02708v2'"
    )
    output_dir: Optional[str] = RuntimeField(
        default="papers",
        description=(
            "Corpus root relative to base_directory; this paper's files land in "
            "<output_dir>/<arxiv_id>v<N>/ next to any PDF fetched for it"
        ),
    )
    inline_one_level: Optional[bool] = RuntimeField(
        default=True,
        description="Inline \\input/\\include one level into the main file (default true)",
    )
    preview_chars: Optional[int] = RuntimeField(
        default=2000, description="Length of the inline LaTeX preview (default 2000)"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _run(self) -> str:
        valid = validate_arxiv_id(self.arxiv_id)
        if not valid:
            return self.format_error(
                error="Invalid Parameter",
                reason="arxiv_id is not a well-formed arXiv identifier",
                context=f"arxiv_id={self.arxiv_id}",
                suggestion="Use e.g. '2103.02708' or 'hep-ph/9905221' (version optional)",
            )
        base = os.path.realpath(self.base_directory)
        if _paper_dir(self.base_directory, self.output_dir, "x") is None:
            return self.format_error(
                error="Access Denied",
                reason="output_dir escapes base_directory",
                context=self.output_dir,
                suggestion="Use a relative path inside base_directory",
            )
        request = {"arxiv_id": self.arxiv_id, "output_dir": self.output_dir,
                   "inline_one_level": self.inline_one_level}
        n_preview = self.preview_chars if self.preview_chars is not None else 2000
        interface = ArxivInterface()
        http: List[str] = []

        versioned, meta_url, resolve_err = _resolve(interface, valid)
        if meta_url:
            http.append(meta_url)

        # Cache: this exact version was fetched before and its payload kept.
        if versioned:
            paper_dir = _paper_dir(self.base_directory, self.output_dir, _flat(versioned))
            cached = self._cached(paper_dir, versioned, base, n_preview)
            if cached is not None:
                files = [file_entry(base, cached["eprint_path"], cached["sha256"])]
                if cached.get("tex_path"):
                    files.append(file_entry(base, cached["tex_path"], cached.get("tex_sha256")))
                return _with_provenance(
                    base, "ArxivSource", request, cached, http,
                    request_url=cached["url"], from_cache=True,
                    cached_at=cached["retrieved_at"],
                    source_ids=[cached["source_id"]], files=files,
                )

        # 1. Download the e-print payload.
        id_base, want = split_version(versioned or valid)
        try:
            payload = interface.download_eprint(versioned or valid)
        except Exception as e:  # noqa: BLE001
            return self.format_error(
                error="E-print Download Failed",
                reason=str(e),
                context=f"arxiv_id={self.arxiv_id}",
                suggestion="Verify the arXiv id; arXiv may be rate-limiting",
            )
        http.append(payload["url"])
        retrieved_at = now_iso()
        served = version_from_response(id_base, payload.get("content_disposition"), payload.get("final_url"))
        if want and served and served != want:
            return self.format_error(
                error="Version Mismatch",
                reason=f"asked for {id_base}{want}, arXiv served {id_base}{served}",
                suggestion="Retry, or pass the version you want explicitly",
            )
        if not versioned:
            if not served:
                return self.format_error(
                    error="Version Resolution Failed",
                    reason=f"could not determine which version of {valid} arXiv serves",
                    context=resolve_err or "",
                    suggestion="Pass an explicit version, e.g. '2103.02708v2'",
                )
            versioned = f"{id_base}{served}"
        stem = _flat(versioned)
        source_id = arxiv_source_id(versioned)
        paper_dir = _paper_dir(self.base_directory, self.output_dir, stem)
        src_dir = os.path.join(paper_dir, "source")
        tex_abs = os.path.join(paper_dir, "source.tex")
        data = payload["data"]
        common = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "arxiv_id": versioned,
            "requested_id": self.arxiv_id,
            "source_id": source_id,
        }
        fetched = {"sha256": payload["sha256"], "url": payload["url"],
                   "retrieved_at": retrieved_at, "cached": False}

        # 2. Classify + unwrap.
        try:
            os.makedirs(paper_dir, exist_ok=True)
            kind = detect_payload(data)
            gz = kind == "gzip"
            if gz:
                data = gunzip_capped(data)
                kind = detect_payload(data)
                if kind == "gzip":  # double-wrapped is not a thing; treat as tex
                    kind = "tex"
            if kind == "pdf":
                # arXiv served the PDF because no source was submitted. We
                # already hold those bytes, so save them under the same
                # per-paper directory rather than telling the caller to fetch
                # the identical file again through the 3s rate limiter.
                pdf_abs = os.path.join(paper_dir, f"{stem}.pdf")
                with open(pdf_abs, "wb") as fh:
                    fh.write(data)
                result = dict(common, source_type="pdf_only", **fetched)
                result["pdf_path"] = result["eprint_path"] = os.path.relpath(pdf_abs, base)
                result["bytes"] = len(data)
                result["suggestion"] = (
                    "No LaTeX source on arXiv for this paper. The PDF was "
                    "saved at pdf_path; run PDFToTeXTool on it to recover "
                    "TeX-faithful text."
                )
                _write_meta(paper_dir, source_id, "eprint", {
                    "url": payload["url"], "final_url": payload.get("final_url"),
                    "retrieved_at": retrieved_at, "sha256": payload["sha256"],
                    "bytes": payload["bytes"], "path": result["eprint_path"],
                    "source_type": "pdf_only",
                })
                return _with_provenance(
                    base, "ArxivSource", request, result, http,
                    request_url=payload["url"], source_ids=[source_id],
                    files=[file_entry(base, pdf_abs, payload["sha256"])],
                )

            kind_key = ("gzip-" if gz else "") + ("tar" if kind == "tar" else "tex")
            eprint_abs = os.path.join(paper_dir, f"{stem}.eprint{_EPRINT_EXT[kind_key]}")
            with open(eprint_abs, "wb") as fh:
                fh.write(payload["data"])

            os.makedirs(src_dir, exist_ok=True)
            if kind == "tar":
                files = safe_extract_tar(data, src_dir)
                source_type = "tar"
                main_rel = resolve_main_tex(src_dir)
                if main_rel is None:
                    return self.format_error(
                        error="No TeX Found",
                        reason="archive extracted but contains no .tex files",
                        context=f"n_files={len(files)}",
                    )
            else:  # single tex (or unknown-but-texish)
                main_rel = "main.tex"
                files = [main_rel]
                with open(os.path.join(src_dir, main_rel), "wb") as fh:
                    fh.write(data)
                source_type = "single_tex"

            main_abs = os.path.join(src_dir, main_rel)
            with open(main_abs, "r", encoding="utf-8", errors="replace") as fh:
                tex = fh.read()
            if self.inline_one_level is not False:
                tex = inline_inputs(tex, src_dir, main_rel)
            tex = strip_comments(tex)

            with open(tex_abs, "w", encoding="utf-8") as fh:
                fh.write(tex)
        except ValueError as e:  # archive-safety violations, caps, bombs
            return self.format_error(
                error="Unsafe or Invalid Archive",
                reason=str(e),
                context=f"arxiv_id={self.arxiv_id}",
            )
        except OSError as e:
            return self.format_error(error="Filesystem Error", reason=str(e))

        tex_sha = _sha256_file(tex_abs)
        result = dict(
            common,
            source_type=source_type,
            source_dir=os.path.relpath(src_dir, base),
            main_tex=os.path.join(os.path.relpath(src_dir, base), main_rel),
            tex_path=os.path.relpath(tex_abs, base),
            n_files=len(files),
            chars=len(tex),
            eprint_path=os.path.relpath(eprint_abs, base),
            tex_sha256=tex_sha,
            note=(
                "source.tex is comment-stripped with \\input inlined; the "
                "original files under source_dir are kept verbatim, so cite "
                "line numbers from them (QuoteSource does)."
            ),
            preview=tex[:n_preview],
            **fetched,
        )
        _write_meta(paper_dir, source_id, "eprint", {
            "url": payload["url"], "final_url": payload.get("final_url"),
            "retrieved_at": retrieved_at, "sha256": payload["sha256"],
            "bytes": payload["bytes"], "path": result["eprint_path"],
            "source_type": source_type, "source_dir": result["source_dir"],
            "main_tex": result["main_tex"], "tex_path": result["tex_path"],
            "n_files": len(files),
        })
        return _with_provenance(
            base, "ArxivSource", request, result, http,
            request_url=payload["url"], source_ids=[source_id],
            files=[file_entry(base, eprint_abs, payload["sha256"]),
                   file_entry(base, tex_abs, tex_sha)],
        )

    def _cached(self, paper_dir: str, versioned: str, base: str, n_preview: int) -> Optional[Dict[str, Any]]:
        """Result for a previously fetched copy of this exact version, else None.

        Requires the kept e-print payload, so the returned sha256 is always
        recomputed from the bytes on disk rather than trusted from a record.
        """
        meta = _read_meta(paper_dir).get("eprint")
        if not meta or not meta.get("path"):
            return None
        eprint_abs = _safe_join(base, meta["path"])
        if not eprint_abs or not os.path.isfile(eprint_abs) or os.path.islink(eprint_abs):
            return None
        result: Dict[str, Any] = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "arxiv_id": versioned,
            "requested_id": self.arxiv_id,
            "source_id": arxiv_source_id(versioned),
            "source_type": meta.get("source_type"),
            "sha256": _sha256_file(eprint_abs),
            "eprint_path": meta["path"],
            "url": meta.get("url"),
            "retrieved_at": meta.get("retrieved_at") or _mtime_iso(eprint_abs),
            "cached": True,
        }
        if meta.get("source_type") == "pdf_only":
            result["pdf_path"] = meta["path"]
            result["bytes"] = os.path.getsize(eprint_abs)
            return result
        tex_abs = _safe_join(base, meta.get("tex_path") or "")
        if not tex_abs or not os.path.isfile(tex_abs):
            return None
        try:
            with open(tex_abs, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            return None
        result.update(
            source_dir=meta.get("source_dir"),
            main_tex=meta.get("main_tex"),
            tex_path=meta.get("tex_path"),
            n_files=meta.get("n_files"),
            chars=len(text),
            tex_sha256=_sha256_file(tex_abs),
            preview=text[:n_preview],
        )
        return result
