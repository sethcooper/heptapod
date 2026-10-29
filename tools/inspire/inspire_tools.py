"""
# inspire_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

INSPIRE HEP API tools for agent use.

Provides BaseTool implementations for searching papers, authors,
institutions, conferences, journals, experiments, and managing
a personal research library.
"""

import json
import os
from typing import Optional, List

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .inspire_interface import InspireInterface
from .query_builder import QueryBuilder
from .library import InspireLibrary
from ..digest import clip, digest
from ..provenance import attach, inspire_source_id, record_call

SCHEMA_VERSION = "inspire-hep-1.0"
INSPIRE_API = {"name": "INSPIRE-HEP REST API", "version": "v1"}
_MAX_SOURCE_IDS = 200


def _inspire_records(data, out):
    """Collect (recid, revision, updated, collection) from INSPIRE JSON.

    A record is any dict with a ``metadata`` object carrying a
    ``control_number``; search responses nest them under ``hits.hits``.
    """
    if isinstance(data, dict):
        md = data.get("metadata")
        if isinstance(md, dict) and md.get("control_number") is not None:
            schema = str(md.get("$schema", ""))
            coll = schema.rsplit("/", 1)[-1].replace(".json", "") if schema else "hep"
            coll = "literature" if coll in ("hep", "") else coll
            out.append((md["control_number"], data.get("revision_id"),
                        data.get("updated"), coll))
        for v in data.values():
            if isinstance(v, (dict, list)):
                _inspire_records(v, out)
    elif isinstance(data, list):
        for v in data:
            _inspire_records(v, out)
    return out


class _InspireProvenance:
    """Wraps each INSPIRE tool's ``_execute`` so its JSON result carries a
    ``provenance`` block and the call is logged under research/.

    The exact request URLs (with parameters), cache hits and the time a
    cached response was originally fetched come from the interface's
    ``call_log``; the raw API responses are saved verbatim. A tool that
    defines ``_digest(result, prov)`` returns that instead of the full result.
    """

    def _run(self) -> str:
        iface = getattr(self, "_interface", None)
        if iface is not None:
            iface.call_log = []
        out = self._execute()
        log = list(getattr(iface, "call_log", []) or [])
        records = []
        for c in log:
            _inspire_records(c.get("data"), records)
        source_ids = [inspire_source_id(r, rev, upd, coll) for r, rev, upd, coll in records]
        request = {k: getattr(self, k) for k in type(self).model_fields
                   if k not in BaseTool.model_fields
                   and k not in ("base_directory", "cache_enabled")}
        if len(log) > 1:
            request["http_requests"] = [
                {"url": c["url"], "from_cache": c["from_cache"]} for c in log
            ]
        cached = [c["cached_at"] for c in log if c.get("from_cache")]
        raw = [c.get("data") for c in log] if len(log) != 1 else log[0].get("data")
        prov = record_call(
            self.base_directory,
            type(self).__name__.removesuffix("Tool"),
            request,
            out,
            request_url=log[0]["url"] if log else None,
            api=INSPIRE_API if log else {"name": "local library", "version": None},
            from_cache=bool(log) and all(c.get("from_cache") for c in log),
            cached_at=min(cached) if cached else None,
            source_ids=source_ids[:_MAX_SOURCE_IDS],
            raw=raw if log else None,
            raw_ext="bib" if log and isinstance(raw, str) else "json",
        )
        if hasattr(self, "_digest"):
            return self._digest(out, prov)
        return attach(out, prov)

def _inspire_interface(base_directory: str, cache_enabled: bool) -> "InspireInterface":
    """Construct an InspireInterface with cache_file resolved against
    ``base_directory`` and caching toggled by ``cache_enabled``.

    Centralised because most _setup methods used to call
    ``InspireInterface()`` with no args, which silently defaulted to
    ``enable_cache=True`` and loaded ``~/.heptapod/inspire_cache.json``
    (close to 1 GB in some installs) into a fresh dict for EVERY tool
    instance. The cache_enabled state field from toolkit.yaml's
    ``config:`` block now flows through here instead.
    """
    cache_file = os.path.join(os.path.abspath(base_directory), ".inspire_cache.json")
    return InspireInterface(cache_file=cache_file, enable_cache=cache_enabled)


# ==================== Literature Tools ==================== #


class InspireSearchTool(_InspireProvenance, BaseTool):
    """
    Search INSPIRE HEP for physics papers.

    This tool provides natural language search for physics literature
    with support for author, title, topic, and citation-based queries.

    Input:
        query: Natural language or SPIRES query
               Examples:
               - "papers by Witten on string theory"
               - "highly cited papers on supersymmetry"
               - "recent papers about dark matter"
               - "a witten and t strings" (direct SPIRES)
        sort: How to sort results ("mostrecent", "mostcited")
        size: Number of results (default 10, max 100)

    Returns a JSON digest:
        {
            "status": "ok",
            "schema": "inspire-hep-1.0",
            "total": 1234,
            "returned": 10,
            "query_info": {"original": "papers by Witten", "spires": "a witten",
                           "sort": "mostcited", "size": 10},
            "results_path": "research/results/<call_id>.json",
            "papers": [
                {
                    "recid": "451647",
                    "arxiv_id": "hep-th/9503124",
                    "title": "String theory dynamics...",
                    "first_author": "Witten, Edward",
                    "year": 1995,
                    "citation_count": 3500
                }
            ],
            "omitted": 0,
            "note": "...",
            "provenance": {...}
        }

    Titles are cut to 150 characters; "first_author" is the collaboration
    when the paper has one, with "et al." when there are more authors;
    "year" is the publication year, else the year of the earliest date. The
    file at results_path holds every paper in full (authors, abstract, DOI,
    publication info, keywords, ...). "omitted" (present only when non-zero)
    counts papers left out of the digest to keep it short; they are in the
    file. InspirePaperTool fetches one paper's details.
    """

    # ======================== Runtime fields ======================== #
    query: str = RuntimeField(
        description="Search query (natural language or SPIRES format)"
    )
    sort: Optional[str] = RuntimeField(
        description="Sort order: 'mostrecent' or 'mostcited' (default: mostcited)",
        default="mostcited"
    )
    size: Optional[int] = RuntimeField(
        description="Number of results to return (default 10, max 100)",
        default=10
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    cache_enabled: bool = StateField(
        description="Whether to use local cache",
        default=True
    )
    # ================================================================ #

    def _setup(self):
        """Initialize interface."""
        self.base_directory = os.path.abspath(self.base_directory)
        cache_file = os.path.join(self.base_directory, ".inspire_cache.json")
        self._interface = InspireInterface(
            cache_file=cache_file,
            enable_cache=self.cache_enabled
        )
        self._query_builder = QueryBuilder()

    def _execute(self) -> str:
        """Execute search and return JSON result."""
        try:
            # Convert natural language to SPIRES query
            spires_query = self._query_builder.build(self.query)

            # Perform search
            results = self._interface.search_papers(
                query=spires_query,
                sort=self.sort or "mostcited",
                size=min(self.size or 10, 100)
            )

            return json.dumps({
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "total": results.total,
                "returned": len(results.papers),
                "papers": [p.to_dict() for p in results.papers],
                "query_info": {
                    "original": self.query,
                    "spires": spires_query,
                    "sort": self.sort,
                    "size": self.size
                }
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "query": self.query
            }, indent=2)

    def _digest(self, out: str, prov: dict) -> str:
        """Compact paper list; the full result goes to research/results/."""
        result = json.loads(out)
        if result.get("status") != "ok":
            return attach(result, prov)
        papers = []
        for p in result["papers"]:
            authors = p.get("authors") or []
            collab = p.get("collaborations") or []
            if collab:
                first = collab[0] + (" Collaboration" if "ollaboration" not in collab[0] else "")
            elif authors:
                first = authors[0] + (" et al." if len(authors) > 1 else "")
            else:
                first = None
            year = p.get("year")
            if not year and str(p.get("date") or "")[:4].isdigit():
                year = int(str(p["date"])[:4])
            papers.append({
                "recid": p.get("recid"),
                "arxiv_id": p.get("arxiv_id"),
                "title": clip(p.get("title"), 150),
                "first_author": clip(first, 80),
                "year": year,
                "citation_count": p.get("citation_count"),
            })
        head = {k: result[k] for k in ("status", "schema", "total", "returned", "query_info")}
        return digest(self.base_directory, result, prov, head, "papers", papers, what="papers")


class InspirePaperTool(_InspireProvenance, BaseTool):
    """
    Get detailed information about a specific INSPIRE paper.

    Input:
        recid: INSPIRE record ID (e.g., "451647")
        include_citations: Whether to include top citing papers (slower)
        include_references: Whether to include references

    Returns JSON with full paper details including abstract,
    authors, publication info, and optionally citations.
    """

    # ======================== Runtime fields ======================== #
    recid: str = RuntimeField(
        description="INSPIRE record ID"
    )
    include_citations: Optional[bool] = RuntimeField(
        description="Include top citing papers (default False)",
        default=False
    )
    include_references: Optional[bool] = RuntimeField(
        description="Include references (default False)",
        default=False
    )
    citation_limit: Optional[int] = RuntimeField(
        description="Max citations/references to include (default 10)",
        default=10
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    cache_enabled: bool = StateField(
        description="Use the on-disk INSPIRE cache (default: True). Toolkit config sets the path; see toolkit.yaml config block.",
        default=True,
    )
    # ================================================================ #

    def _setup(self):
        """Initialize INSPIRE interface (and any helpers)."""
        self._interface = _inspire_interface(self.base_directory, self.cache_enabled)

    def _execute(self) -> str:
        """Get paper details."""
        try:
            paper = self._interface.get_paper(self.recid)

            result = {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "paper": paper.to_dict()
            }

            # Optionally include citations
            if self.include_citations:
                citations = self._interface.get_paper_citations(
                    self.recid,
                    size=self.citation_limit or 10
                )
                result["top_citing_papers"] = [c.to_dict() for c in citations]
                result["citing_papers_returned"] = len(citations)

            # Optionally include references
            if self.include_references:
                refs = self._interface.get_paper_references(
                    self.recid,
                    size=self.citation_limit or 10
                )
                result["references"] = [r.to_dict() for r in refs]
                result["references_returned"] = len(refs)

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "recid": self.recid
            }, indent=2)


class InspireCitationTool(_InspireProvenance, BaseTool):
    """
    Analyze citations for a paper.

    Use this tool to:
    - Find who cites a paper
    - Analyze citation trends over time
    - Find co-citation networks (papers frequently cited together)

    Input:
        recid: INSPIRE paper record ID
        analysis_type: "citing_papers", "citation_trend", "co_citations"
        limit: Max results (default 50)

    Returns citation analysis including citing papers,
    trends over time, and optionally co-citation data.
    """

    # ======================== Runtime fields ======================== #
    recid: str = RuntimeField(
        description="INSPIRE record ID of paper to analyze"
    )
    analysis_type: Optional[str] = RuntimeField(
        description="Analysis type: 'citing_papers', 'citation_trend', 'co_citations' (default: citing_papers)",
        default="citing_papers"
    )
    limit: Optional[int] = RuntimeField(
        description="Max results (default 50)",
        default=50
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory"
    )
    cache_enabled: bool = StateField(
        description="Use the on-disk INSPIRE cache (default: True). Toolkit config sets the path; see toolkit.yaml config block.",
        default=True,
    )
    # ================================================================ #

    def _setup(self):
        """Initialize INSPIRE interface (and any helpers)."""
        self._interface = _inspire_interface(self.base_directory, self.cache_enabled)

    def _execute(self) -> str:
        try:
            # Get paper info first
            paper = self._interface.get_paper(self.recid)

            result = {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "paper": {
                    "recid": self.recid,
                    "title": paper.title,
                    "total_citations": paper.citation_count
                }
            }

            analysis_type = self.analysis_type or "citing_papers"

            if analysis_type == "citing_papers":
                citations = self._interface.get_paper_citations(
                    self.recid,
                    size=self.limit or 50
                )
                result["citing_papers"] = [c.to_dict() for c in citations]
                result["citing_papers_returned"] = len(citations)

            elif analysis_type == "citation_trend":
                # Get citations and aggregate by year
                citations = self._interface.get_paper_citations(
                    self.recid,
                    size=min(self.limit or 50, 500)
                )

                year_counts = {}
                for c in citations:
                    if c.date:
                        year = c.date[:4]
                        year_counts[year] = year_counts.get(year, 0) + 1

                result["citation_trend"] = dict(sorted(year_counts.items()))
                result["trend_years"] = len(year_counts)
                result["citations_analyzed"] = len(citations)

            elif analysis_type == "co_citations":
                # Find papers frequently cited together with this one
                citations = self._interface.get_paper_citations(
                    self.recid,
                    size=min(self.limit or 50, 100)
                )

                # Count co-cited papers
                co_cited = {}
                for citing_paper in citations[:20]:  # Limit for performance
                    try:
                        refs = self._interface.get_paper_references(
                            citing_paper.recid,
                            size=50
                        )
                        for ref in refs:
                            if ref.recid != self.recid:
                                if ref.recid not in co_cited:
                                    co_cited[ref.recid] = {
                                        "title": ref.title,
                                        "count": 0
                                    }
                                co_cited[ref.recid]["count"] += 1
                    except Exception:
                        continue

                # Sort by co-citation count
                sorted_co_cited = sorted(
                    co_cited.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True
                )[:20]

                result["co_cited_papers"] = [
                    {"recid": recid, **data}
                    for recid, data in sorted_co_cited
                ]
                result["citing_papers_analyzed"] = min(len(citations), 20)

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "recid": self.recid
            }, indent=2)


class InspireBibTeXTool(_InspireProvenance, BaseTool):
    """
    Get BibTeX entry for a paper.

    Input:
        recid: INSPIRE record ID

    Returns BibTeX string suitable for LaTeX documents.
    """

    # ======================== Runtime fields ======================== #
    recid: str = RuntimeField(
        description="INSPIRE record ID"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory"
    )
    cache_enabled: bool = StateField(
        description="Use the on-disk INSPIRE cache (default: True). Toolkit config sets the path; see toolkit.yaml config block.",
        default=True,
    )
    # ================================================================ #

    def _setup(self):
        """Initialize INSPIRE interface (and any helpers)."""
        self._interface = _inspire_interface(self.base_directory, self.cache_enabled)

    def _execute(self) -> str:
        try:
            bibtex = self._interface.get_bibtex(self.recid)

            return json.dumps({
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "recid": self.recid,
                "bibtex": bibtex
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "recid": self.recid
            }, indent=2)


# ==================== People & Institutions Tools ==================== #


class InspireAuthorTool(_InspireProvenance, BaseTool):
    """
    Get information about a physicist and their publication history.

    Input:
        author: Author name (e.g., 'Edward Witten') or INSPIRE ID
        include_papers: Whether to include top papers
        paper_limit: Max papers to include (default 20)
        paper_sort: Sort papers by 'mostcited' or 'mostrecent'

    Returns JSON with author info, affiliations, and optionally top papers.
    """

    # ======================== Runtime fields ======================== #
    author: str = RuntimeField(
        description="Author name (e.g., 'Edward Witten') or INSPIRE ID"
    )
    include_papers: Optional[bool] = RuntimeField(
        description="Include author's top papers (default True)",
        default=True
    )
    paper_limit: Optional[int] = RuntimeField(
        description="Max papers to include (default 20)",
        default=20
    )
    paper_sort: Optional[str] = RuntimeField(
        description="Sort papers by: 'mostcited' or 'mostrecent' (default: mostcited)",
        default="mostcited"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory"
    )
    cache_enabled: bool = StateField(
        description="Use the on-disk INSPIRE cache (default: True). Toolkit config sets the path; see toolkit.yaml config block.",
        default=True,
    )
    # ================================================================ #

    def _setup(self):
        """Initialize INSPIRE interface (and any helpers)."""
        self._interface = _inspire_interface(self.base_directory, self.cache_enabled)

    def _execute(self) -> str:
        try:
            # Search for author
            authors = self._interface.search_authors(self.author, size=1)

            if not authors:
                return json.dumps({
                    "status": "error",
                    "schema": SCHEMA_VERSION,
                    "reason": f"Author not found: {self.author}"
                }, indent=2)

            author_info = authors[0]

            result = {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "author": author_info.to_dict()
            }

            if self.include_papers:
                papers = self._interface.get_author_papers(
                    author_info.inspire_id,
                    size=self.paper_limit or 20,
                    sort=self.paper_sort or "mostcited"
                )
                result["papers"] = [p.to_dict() for p in papers]
                result["papers_returned"] = len(papers)

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "author": self.author
            }, indent=2)


class InspireInstitutionTool(_InspireProvenance, BaseTool):
    """
    Search for physics institutions.

    Input:
        query: Institution name or search query
        size: Maximum results (default 10)

    Returns JSON with institution info including location and affiliations.
    """

    # ======================== Runtime fields ======================== #
    query: str = RuntimeField(
        description="Institution name or search query"
    )
    size: Optional[int] = RuntimeField(
        description="Maximum results (default 10)",
        default=10
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory"
    )
    cache_enabled: bool = StateField(
        description="Use the on-disk INSPIRE cache (default: True). Toolkit config sets the path; see toolkit.yaml config block.",
        default=True,
    )
    # ================================================================ #

    def _setup(self):
        """Initialize INSPIRE interface (and any helpers)."""
        self._interface = _inspire_interface(self.base_directory, self.cache_enabled)

    def _execute(self) -> str:
        try:
            institutions = self._interface.search_institutions(
                self.query,
                size=self.size or 10
            )

            return json.dumps({
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "query": self.query,
                "count": len(institutions),
                "institutions": [i.to_dict() for i in institutions]
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "query": self.query
            }, indent=2)


# ==================== Events & Publications Tools ==================== #


class InspireConferenceTool(_InspireProvenance, BaseTool):
    """
    Search for physics conferences and seminars.

    Input:
        query: Conference name, series, or search query
        size: Maximum results (default 10)
        sort: Sort order ('dateasc' or 'datedesc')

    Returns JSON with conference info including dates and locations.
    """

    # ======================== Runtime fields ======================== #
    query: str = RuntimeField(
        description="Conference name, series, or search query"
    )
    size: Optional[int] = RuntimeField(
        description="Maximum results (default 10)",
        default=10
    )
    sort: Optional[str] = RuntimeField(
        description="Sort order: 'dateasc' or 'datedesc' (default: dateasc)",
        default="dateasc"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory"
    )
    cache_enabled: bool = StateField(
        description="Use the on-disk INSPIRE cache (default: True). Toolkit config sets the path; see toolkit.yaml config block.",
        default=True,
    )
    # ================================================================ #

    def _setup(self):
        """Initialize INSPIRE interface (and any helpers)."""
        self._interface = _inspire_interface(self.base_directory, self.cache_enabled)

    def _execute(self) -> str:
        try:
            conferences = self._interface.search_conferences(
                self.query,
                size=self.size or 10,
                sort=self.sort or "dateasc"
            )

            return json.dumps({
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "query": self.query,
                "count": len(conferences),
                "conferences": [c.to_dict() for c in conferences]
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "query": self.query
            }, indent=2)


class InspireJournalTool(_InspireProvenance, BaseTool):
    """
    Search for physics journals.

    Input:
        query: Journal name or search query
        size: Maximum results (default 10)

    Returns JSON with journal info including short names and publishers.
    """

    # ======================== Runtime fields ======================== #
    query: str = RuntimeField(
        description="Journal name or search query"
    )
    size: Optional[int] = RuntimeField(
        description="Maximum results (default 10)",
        default=10
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory"
    )
    cache_enabled: bool = StateField(
        description="Use the on-disk INSPIRE cache (default: True). Toolkit config sets the path; see toolkit.yaml config block.",
        default=True,
    )
    # ================================================================ #

    def _setup(self):
        """Initialize INSPIRE interface (and any helpers)."""
        self._interface = _inspire_interface(self.base_directory, self.cache_enabled)

    def _execute(self) -> str:
        try:
            journals = self._interface.search_journals(
                self.query,
                size=self.size or 10
            )

            return json.dumps({
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "query": self.query,
                "count": len(journals),
                "journals": [j.to_dict() for j in journals]
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "query": self.query
            }, indent=2)


class InspireExperimentTool(_InspireProvenance, BaseTool):
    """
    Search for physics experiments.

    Input:
        query: Experiment name or search query
        size: Maximum results (default 10)

    Returns JSON with experiment info including institutions and status.
    """

    # ======================== Runtime fields ======================== #
    query: str = RuntimeField(
        description="Experiment name or search query"
    )
    size: Optional[int] = RuntimeField(
        description="Maximum results (default 10)",
        default=10
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory"
    )
    cache_enabled: bool = StateField(
        description="Use the on-disk INSPIRE cache (default: True). Toolkit config sets the path; see toolkit.yaml config block.",
        default=True,
    )
    # ================================================================ #

    def _setup(self):
        """Initialize INSPIRE interface (and any helpers)."""
        self._interface = _inspire_interface(self.base_directory, self.cache_enabled)

    def _execute(self) -> str:
        try:
            experiments = self._interface.search_experiments(
                self.query,
                size=self.size or 10
            )

            return json.dumps({
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "query": self.query,
                "count": len(experiments),
                "experiments": [e.to_dict() for e in experiments]
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "query": self.query
            }, indent=2)


# ==================== Library Tools ==================== #


class InspireReadingListTool(_InspireProvenance, BaseTool):
    """
    Manage your personal reading list of INSPIRE papers.

    Actions:
    - "add": Add a paper to your reading list
    - "remove": Remove a paper from your reading list
    - "list": List papers in your reading list
    - "check": Check if a paper is in your reading list

    Input:
        action: The action to perform
        recid: INSPIRE record ID (required for add/remove/check)
        tags: Tags to associate with the paper (for add action)
        tag_filter: Filter by tag when listing

    Returns JSON with the result of the action.
    """

    # ======================== Runtime fields ======================== #
    action: str = RuntimeField(
        description="Action: 'add', 'remove', 'list', or 'check'"
    )
    recid: Optional[str] = RuntimeField(
        description="INSPIRE record ID (required for add/remove/check)",
        default=None
    )
    tags: Optional[List[str]] = RuntimeField(
        description="Tags for the paper (for add action)",
        default=None
    )
    tag_filter: Optional[str] = RuntimeField(
        description="Filter by tag (for list action)",
        default=None
    )
    limit: Optional[int] = RuntimeField(
        description="Max papers to list (default 50)",
        default=50
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory"
    )
    cache_enabled: bool = StateField(
        description="Use the on-disk INSPIRE cache (default: True). Toolkit config sets the path; see toolkit.yaml config block.",
        default=True,
    )
    # ================================================================ #

    def _setup(self):
        """Initialize INSPIRE interface (and any helpers)."""
        self._interface = _inspire_interface(self.base_directory, self.cache_enabled)
        library_file = os.path.join(self.base_directory, ".inspire_library.json")
        self._library = InspireLibrary(library_file)

    def _execute(self) -> str:
        try:
            action = self.action.lower()

            if action == "add":
                if not self.recid:
                    return json.dumps({
                        "status": "error",
                        "schema": SCHEMA_VERSION,
                        "reason": "recid is required for add action"
                    }, indent=2)

                # Get paper info for display
                title = None
                authors = []
                try:
                    paper = self._interface.get_paper(self.recid)
                    title = paper.title
                    authors = paper.authors[:5]
                except Exception:
                    pass

                entry = self._library.add_to_reading_list(
                    self.recid,
                    title=title,
                    authors=authors,
                    tags=self.tags
                )

                return json.dumps({
                    "status": "ok",
                    "schema": SCHEMA_VERSION,
                    "action": "added",
                    "entry": entry.to_dict()
                }, indent=2)

            elif action == "remove":
                if not self.recid:
                    return json.dumps({
                        "status": "error",
                        "schema": SCHEMA_VERSION,
                        "reason": "recid is required for remove action"
                    }, indent=2)

                removed = self._library.remove_from_reading_list(self.recid)

                return json.dumps({
                    "status": "ok",
                    "schema": SCHEMA_VERSION,
                    "action": "removed" if removed else "not_found",
                    "recid": self.recid
                }, indent=2)

            elif action == "list":
                entries = self._library.get_reading_list(
                    tag_filter=self.tag_filter,
                    limit=self.limit or 50
                )

                return json.dumps({
                    "status": "ok",
                    "schema": SCHEMA_VERSION,
                    "count": len(entries),
                    "tag_filter": self.tag_filter,
                    "entries": [e.to_dict() for e in entries],
                    "all_tags": self._library.get_all_tags()
                }, indent=2)

            elif action == "check":
                if not self.recid:
                    return json.dumps({
                        "status": "error",
                        "schema": SCHEMA_VERSION,
                        "reason": "recid is required for check action"
                    }, indent=2)

                is_saved = self._library.is_in_reading_list(self.recid)

                return json.dumps({
                    "status": "ok",
                    "schema": SCHEMA_VERSION,
                    "recid": self.recid,
                    "in_reading_list": is_saved
                }, indent=2)

            else:
                return json.dumps({
                    "status": "error",
                    "schema": SCHEMA_VERSION,
                    "reason": f"Unknown action: {action}. Valid actions: add, remove, list, check"
                }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "action": self.action
            }, indent=2)


class InspireNotesTool(_InspireProvenance, BaseTool):
    """
    Manage notes on INSPIRE papers.

    Actions:
    - "add": Add a note to a paper
    - "get": Get all notes for a paper
    - "search": Search notes for matching text

    Input:
        action: The action to perform
        recid: INSPIRE record ID (required for add/get)
        note: The note text (required for add)
        search_query: Search text (required for search)

    Returns JSON with the result of the action.
    """

    # ======================== Runtime fields ======================== #
    action: str = RuntimeField(
        description="Action: 'add', 'get', or 'search'"
    )
    recid: Optional[str] = RuntimeField(
        description="INSPIRE record ID (required for add/get)",
        default=None
    )
    note: Optional[str] = RuntimeField(
        description="Note text (required for add)",
        default=None
    )
    search_query: Optional[str] = RuntimeField(
        description="Search text (required for search)",
        default=None
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory"
    )
    # ================================================================ #

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        library_file = os.path.join(self.base_directory, ".inspire_library.json")
        self._library = InspireLibrary(library_file)

    def _execute(self) -> str:
        try:
            action = self.action.lower()

            if action == "add":
                if not self.recid:
                    return json.dumps({
                        "status": "error",
                        "schema": SCHEMA_VERSION,
                        "reason": "recid is required for add action"
                    }, indent=2)
                if not self.note:
                    return json.dumps({
                        "status": "error",
                        "schema": SCHEMA_VERSION,
                        "reason": "note is required for add action"
                    }, indent=2)

                paper_note = self._library.add_note(self.recid, self.note)

                return json.dumps({
                    "status": "ok",
                    "schema": SCHEMA_VERSION,
                    "action": "added",
                    "recid": self.recid,
                    "note": paper_note.to_dict()
                }, indent=2)

            elif action == "get":
                if not self.recid:
                    return json.dumps({
                        "status": "error",
                        "schema": SCHEMA_VERSION,
                        "reason": "recid is required for get action"
                    }, indent=2)

                notes = self._library.get_notes(self.recid)

                return json.dumps({
                    "status": "ok",
                    "schema": SCHEMA_VERSION,
                    "recid": self.recid,
                    "count": len(notes),
                    "notes": [n.to_dict() for n in notes]
                }, indent=2)

            elif action == "search":
                if not self.search_query:
                    return json.dumps({
                        "status": "error",
                        "schema": SCHEMA_VERSION,
                        "reason": "search_query is required for search action"
                    }, indent=2)

                results = self._library.search_notes(self.search_query)

                return json.dumps({
                    "status": "ok",
                    "schema": SCHEMA_VERSION,
                    "search_query": self.search_query,
                    "count": len(results),
                    "results": results
                }, indent=2)

            else:
                return json.dumps({
                    "status": "error",
                    "schema": SCHEMA_VERSION,
                    "reason": f"Unknown action: {action}. Valid actions: add, get, search"
                }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "action": self.action
            }, indent=2)
