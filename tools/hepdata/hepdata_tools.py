"""
# hepdata_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

HEPData (https://www.hepdata.net) tools.

    HEPDataSearchTool   search records; each hit with its INSPIRE id, version
                        and table count
    HEPDataRecordTool   one record at a version: metadata and its tables
    HEPDataTableTool    one table at a version: saved as YAML in the sandbox,
                        its columns with their uncertainties in the results file

Records are addressed by INSPIRE id and versioned; every result names what it
read as ``HEPData:ins<inspire_id>/v<version>`` (a record) or
``HEPData:ins<inspire_id>/v<version>/<table>`` (a table), and carries the
provenance block of ``tools/provenance.py``.

Each tool returns a digest (``tools/digest.py``): identifiers, counts and a
compact list, with the full structured result in
``research/results/<call_id>.json`` (``results_path``).
"""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
import yaml

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from ..digest import clip, digest
from ..provenance import attach, file_entry, hepdata_source_id, record_call

SCHEMA_VERSION = "hepdata-1.0"
HEPDATA_URL = "https://www.hepdata.net"
API = {"name": "hepdata.net", "version": None}
TIMEOUT = 60

_TAG = re.compile(r"<[^>]+>")
_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


# ------------------------------------------------------------------ helpers


class HEPDataError(Exception):
    def __init__(self, error: str, reason: str, url: Optional[str] = None):
        super().__init__(reason)
        self.error, self.reason, self.url = error, reason, url


def _session() -> requests.Session:
    """A plain requests session. HEPData refuses unrecognised user agents
    (403), so the library's own is kept."""
    return requests.Session()


def _get(session: requests.Session, url: str, params: Optional[Dict[str, Any]] = None,
         expect: str = "json") -> Tuple[requests.Response, str]:
    """GET; the response and the exact URL sent. HEPData answers a missing
    record or table with an HTML page, sometimes with status 200, so the
    content type is checked as well as the status."""
    sent = requests.Request("GET", url, params=params).prepare().url
    try:
        resp = session.get(url, params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise HEPDataError("Request Failed", str(e), sent)
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if resp.status_code == 404 or (resp.status_code == 200 and "html" in ctype):
        raise HEPDataError("Not Found", f"HEPData has nothing at {sent}", sent)
    if resp.status_code != 200:
        raise HEPDataError("HTTP Error", f"HEPData returned {resp.status_code} for {sent}", sent)
    if expect == "yaml" and "yaml" not in ctype and "text/plain" not in ctype:
        raise HEPDataError("Unexpected Response", f"expected YAML, got {ctype or 'no content type'}", sent)
    return resp, sent


def _inspire_id(value: Any) -> str:
    s = str(value or "").strip()
    s = re.sub(r"(?i)^(inspire:|ins)", "", s)
    if not s.isdigit():
        raise HEPDataError("Invalid Parameter",
                           f"inspire_id must be an INSPIRE record number (e.g. 2964565), got {value!r}")
    return s


def _text(s: Any, limit: Optional[int] = None) -> Optional[str]:
    """HEPData descriptions are HTML fragments; return plain text."""
    if s is None:
        return None
    t = html.unescape(_TAG.sub("", str(s)))
    t = re.sub(r"\s+", " ", t).strip()
    if limit and len(t) > limit:
        t = t[:limit].rstrip() + " ..."
    return t


def _resources(items: Any) -> List[Dict[str, Any]]:
    out = []
    for r in items or []:
        if isinstance(r, dict):
            out.append({"description": _text(r.get("description")), "type": r.get("type"),
                        "url": r.get("url")})
    return out


def _error(error: str, reason: str, **extra: Any) -> Dict[str, Any]:
    return dict({"status": "error", "schema": SCHEMA_VERSION, "error": error, "reason": reason}, **extra)


def _inside(base: str, path: str) -> str:
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(base, path))
    if full != base and not full.startswith(base + os.sep):
        raise HEPDataError("Access Denied", f"{path!r} is outside base_directory")
    return full


def _columns(table: Dict[str, Any], max_points: Optional[int]) -> Dict[str, Any]:
    """HEPData table YAML -> independent and dependent columns.

    An independent value is a number or ``[low, high]`` for a bin. A dependent
    value is ``{"value", "errors": [{"label", "symerror"} | {"label", "plus",
    "minus"}]}``; signs are kept as the record gives them.
    """
    def cut(values: List[Any]) -> List[Any]:
        return values if not max_points or max_points <= 0 else values[:max_points]

    indep = []
    for v in table.get("independent_variables") or []:
        head = v.get("header") or {}
        vals = []
        for p in v.get("values") or []:
            if "value" in p:
                vals.append(p["value"])
            else:
                vals.append([p.get("low"), p.get("high")])
        indep.append({"name": head.get("name"), "units": head.get("units"), "values": cut(vals)})

    dep = []
    for v in table.get("dependent_variables") or []:
        head = v.get("header") or {}
        quals = {}
        for q in v.get("qualifiers") or []:
            name = q.get("name")
            quals[name] = q.get("value") if name not in quals else f"{quals[name]}; {q.get('value')}"
        vals = []
        for p in v.get("values") or []:
            point: Dict[str, Any] = {"value": p.get("value")}
            errs = []
            for e in p.get("errors") or []:
                entry: Dict[str, Any] = {"label": e.get("label")}
                if "symerror" in e:
                    entry["symerror"] = e["symerror"]
                elif "asymerror" in e:
                    entry["plus"] = (e["asymerror"] or {}).get("plus")
                    entry["minus"] = (e["asymerror"] or {}).get("minus")
                errs.append(entry)
            if errs:
                point["errors"] = errs
            vals.append(point)
        dep.append({"name": head.get("name"), "units": head.get("units"),
                    "qualifiers": quals, "values": cut(vals)})

    n = max([len(v.get("values") or []) for v in (table.get("independent_variables") or [])
             + (table.get("dependent_variables") or [])] or [0])
    return {"points": n, "truncated": bool(max_points and 0 < max_points < n),
            "independent_variables": indep, "dependent_variables": dep}


# --------------------------------------------------------------- the tools


class HEPDataSearchTool(BaseTool):
    """
    Search HEPData for records.

    Input:
        query: HEPData search text, e.g. "heavy neutral lepton", or a field
               query such as 'collaborations:CMS AND title:"heavy neutrino"'.
        max_results: hits to return (default 10, at most 50).

    Returns a JSON digest:
        {"status": "ok", "schema", "query", "max_results", "total", "count",
         "results_path": "research/results/<call_id>.json",
         "results": [{"inspire_id", "version", "source_id": "HEPData:ins<id>/v<N>",
                      "title", "collaborations", "arxiv_id", "year",
                      "total_tables"}],
         "omitted": N, "note", "provenance": {...}}

    Titles are cut to 150 characters. The file at results_path holds every
    hit in full: also "journal_info", "hepdata_doi", "last_updated" and
    "tables" (the table titles). "omitted" (present only when non-zero)
    counts hits left out of the digest to keep it short; they are in the
    file. Fetch a hit's tables and descriptions with HEPDataRecordTool.
    """

    # ======================== Runtime fields ======================== #
    query: str = RuntimeField(description="HEPData search text or field query")
    max_results: Optional[int] = RuntimeField(default=10, description="Number of hits (default 10, max 50)")
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory for file operations")
    # ================================================================ #

    def _run(self) -> str:
        base = os.path.realpath(self.base_directory)
        size = max(1, min(int(self.max_results or 10), 50))
        request = {"query": self.query, "max_results": size}
        sent, raw = None, None
        try:
            if not (self.query or "").strip():
                raise HEPDataError("Missing Parameter", "query is empty")
            resp, sent = _get(_session(), f"{HEPDATA_URL}/search/",
                              {"q": self.query, "format": "json", "size": size})
            raw = resp.text
            data = resp.json()
            hits = []
            for r in (data.get("results") or [])[:size]:
                ins, ver = str(r.get("inspire_id") or ""), r.get("version")
                hits.append({
                    "inspire_id": ins or None,
                    "version": ver,
                    "source_id": hepdata_source_id(ins, ver),
                    "title": _text(r.get("title")),
                    "collaborations": r.get("collaborations") or [],
                    "arxiv_id": r.get("arxiv_id"),
                    "journal_info": r.get("journal_info"),
                    "year": r.get("year"),
                    "hepdata_doi": r.get("hepdata_doi"),
                    "last_updated": r.get("last_updated"),
                    "total_tables": r.get("total_tables"),
                    "tables": [t.get("title") for t in (r.get("data") or []) if isinstance(t, dict)],
                })
            result = {"status": "ok", "schema": SCHEMA_VERSION, "query": self.query,
                      "total": data.get("total"), "count": len(hits), "results": hits}
        except (HEPDataError, ValueError) as e:
            result = (_error(e.error, e.reason) if isinstance(e, HEPDataError)
                      else _error("Invalid Response", f"HEPData returned malformed JSON: {e}"))
        prov = record_call(base, "HEPDataSearch", request, result, request_url=sent, api=API,
                           source_ids=[h["source_id"] for h in result.get("results", [])],
                           raw=raw)
        if result["status"] != "ok":
            return attach(result, prov)
        hits = [{k: (clip(h[k], 150) if k == "title" else h[k])
                 for k in ("inspire_id", "version", "source_id", "title", "collaborations",
                           "arxiv_id", "year", "total_tables")}
                for h in result["results"]]
        head = {"status": "ok", "schema": SCHEMA_VERSION, "query": self.query,
                "max_results": size, "total": result["total"], "count": result["count"]}
        return digest(base, result, prov, head, "results", hits, what="hits")


class HEPDataRecordTool(BaseTool):
    """
    Fetch one HEPData record: metadata, resources and its tables.

    Input:
        inspire_id: the publication's INSPIRE record number ("2964565",
                    "ins2964565" also accepted).
        version: record version (default: the latest). Pin it once chosen:
                 a new version can change every table.

    Returns a JSON digest:
        {"status": "ok", "schema", "inspire_id", "version", "version_count",
         "source_id": "HEPData:ins<id>/v<N>", "title", "collaborations",
         "arxiv_id", "doi", "resource_count", "table_count",
         "results_path": "research/results/<call_id>.json",
         "tables": [{"name", "location", "description"}],
         "omitted": N, "note", "provenance": {...}}

    Descriptions are cut to 120 characters and the title to 150. The file at
    results_path holds the whole record: also "hepdata_doi",
    "journal_info", "last_updated", "abstract", "resources" ([{"description",
    "type", "url"}]) and every table with its "doi", longer "description" and
    "source_id" ("HEPData:ins<id>/v<N>/<name>"). "omitted" (present only when
    non-zero) counts tables left out of the digest; they are in the file.
    Fetch a table's values with HEPDataTableTool, using the name exactly as
    listed.

    An INSPIRE id with no HEPData record is an error with "error": "Not
    Found"; HEPData holds data for only some papers, and HEPDataSearchTool
    finds the ones it has.
    """

    # ======================== Runtime fields ======================== #
    inspire_id: str = RuntimeField(description="INSPIRE record number of the publication, e.g. '2964565'")
    version: Optional[int] = RuntimeField(default=None, description="Record version (default: latest)")
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory for file operations")
    # ================================================================ #

    def _run(self) -> str:
        base = os.path.realpath(self.base_directory)
        request = {"inspire_id": self.inspire_id, "version": self.version}
        sent, raw, ids = None, None, []
        try:
            ins = _inspire_id(self.inspire_id)
            params: Dict[str, Any] = {"format": "json"}
            if self.version:
                params["version"] = int(self.version)
            try:
                resp, sent = _get(_session(), f"{HEPDATA_URL}/record/ins{ins}", params)
            except HEPDataError as e:
                if e.error == "Not Found":
                    which = (f"no HEPData record at version {params['version']}, or none at all"
                             if "version" in params else "no HEPData record")
                    e.reason = (f"INSPIRE record {ins} has {which}: HEPData holds data for "
                                f"only some papers. Use HEPDataSearchTool to find the records "
                                f"it has. (Requested {e.url})")
                raise
            raw = resp.text
            data = resp.json()
            rec = data.get("record") or {}
            ver = data.get("version")
            tables = [{
                "name": t.get("name"),
                "location": _text(t.get("location")),
                "doi": t.get("doi"),
                "description": _text(t.get("description"), 600),
                "source_id": hepdata_source_id(ins, ver, t.get("name")),
            } for t in data.get("data_tables") or []]
            ids = [hepdata_source_id(ins, ver)]
            result = {
                "status": "ok", "schema": SCHEMA_VERSION,
                "inspire_id": ins, "version": ver, "version_count": data.get("version_count"),
                "source_id": ids[0],
                "title": _text(rec.get("title")),
                "collaborations": rec.get("collaborations") or [],
                "arxiv_id": rec.get("arxiv_id"),
                "doi": rec.get("doi"),
                "hepdata_doi": rec.get("hepdata_doi"),
                "journal_info": rec.get("journal_info"),
                "last_updated": rec.get("last_updated"),
                "abstract": _text(rec.get("abstract"), 1500),
                "resources": _resources(rec.get("resources")),
                "tables": tables,
            }
        except (HEPDataError, ValueError) as e:
            result = (_error(e.error, e.reason) if isinstance(e, HEPDataError)
                      else _error("Invalid Response", f"HEPData returned malformed JSON: {e}"))
        prov = record_call(base, "HEPDataRecord", request, result, request_url=sent, api=API,
                           source_ids=ids, raw=raw)
        if result["status"] != "ok":
            return attach(result, prov)
        head = {k: result[k] for k in ("status", "schema", "inspire_id", "version",
                                       "version_count", "source_id", "title", "collaborations",
                                       "arxiv_id", "doi")}
        head["title"] = clip(head["title"], 150)
        head["resource_count"] = len(result["resources"])
        head["table_count"] = len(result["tables"])
        tables = [{"name": t["name"], "location": clip(t["location"], 80),
                   "description": clip(t["description"], 120)} for t in result["tables"]]
        return digest(base, result, prov, head, "tables", tables, what="tables")


class HEPDataTableTool(BaseTool):
    """
    Fetch one HEPData table at a pinned record version.

    Input:
        inspire_id: the publication's INSPIRE record number.
        table: the table name exactly as HEPDataRecordTool lists it
               (e.g. "Figure 4a", "Table 3").
        version: record version (default: the latest, which the result then
                 names).
        output_dir: where to save the table under base_directory (default
                    "hepdata"); the file is <output_dir>/ins<id>/v<N>/<table>.yaml.
        max_points: points per column in the results file (default 200; 0 for
                    all). The saved YAML always holds the whole table.

    Returns a JSON digest:
        {"status": "ok", "schema", "inspire_id", "version", "table",
         "source_id": "HEPData:ins<id>/v<N>/<table>", "file", "sha256",
         "points", "truncated",
         "independent_variables": [{"name", "units"}],
         "dependent_variables": [{"name", "units", "qualifiers": {...}}],
         "results_path": "research/results/<call_id>.json",
         "rows": [[x..., y...]], "omitted": N, "note", "provenance": {...}}

    "rows" are the first few points, one value per listed column
    (independent variables first; a bin is [low, high]), without
    uncertainties. The file at results_path holds the columns in full:
        "independent_variables": [{"name", "units", "values": [x | [low, high]]}],
        "dependent_variables": [{"name", "units", "qualifiers": {...},
                                 "values": [{"value", "errors": [...]}]}]
    with at most max_points values per column ("truncated" says whether
    that cut applied). At most 10 dependent variables are listed in the
    digest; "dependent_count" gives the number when there are more. Values
    are as the record gives them, including the signs of asymmetric
    uncertainties. Cite the table by its source_id and the saved YAML's
    sha256.
    """

    # ======================== Runtime fields ======================== #
    inspire_id: str = RuntimeField(description="INSPIRE record number of the publication")
    table: str = RuntimeField(description="Table name exactly as HEPDataRecordTool lists it")
    version: Optional[int] = RuntimeField(default=None, description="Record version (default: latest)")
    output_dir: Optional[str] = RuntimeField(default="hepdata",
                                             description="Directory under base_directory (default 'hepdata')")
    max_points: Optional[int] = RuntimeField(default=200,
                                             description="Points per column in the results file (default 200; 0 for all)")
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory for file operations")
    # ================================================================ #

    def _run(self) -> str:
        base = os.path.realpath(self.base_directory)
        request = {"inspire_id": self.inspire_id, "table": self.table, "version": self.version,
                   "output_dir": self.output_dir or "hepdata"}
        sent, raw, ids, files = None, None, [], None
        http_requests: List[str] = []
        try:
            ins = _inspire_id(self.inspire_id)
            if not (self.table or "").strip():
                raise HEPDataError("Missing Parameter", "table is empty")
            root = _inside(base, self.output_dir or "hepdata")
            session = _session()
            ver = int(self.version) if self.version else None
            if ver is None:
                resp, url = _get(session, f"{HEPDATA_URL}/record/ins{ins}",
                                 {"format": "json", "light": "true"})
                http_requests.append(url)
                ver = resp.json().get("version")
                if not ver:
                    raise HEPDataError("Invalid Response", f"no version in the record at {url}", url)
            resp, sent = _get(session, f"{HEPDATA_URL}/download/table/ins{ins}/"
                              f"{quote(self.table, safe='')}/{ver}/yaml", expect="yaml")
            http_requests.append(sent)
            raw = resp.content
            try:
                table = yaml.safe_load(raw.decode("utf-8"))
            except (UnicodeDecodeError, yaml.YAMLError) as e:
                raise HEPDataError("Invalid Response", f"table is not valid YAML: {e}", sent)
            if not isinstance(table, dict):
                raise HEPDataError("Invalid Response", "table YAML is not a mapping", sent)

            folder = os.path.join(root, f"ins{ins}", f"v{ver}")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, (_SLUG.sub("_", self.table).strip("_") or "table") + ".yaml")
            with open(path, "wb") as fh:
                fh.write(raw)
            entry = file_entry(base, path)
            files = [entry]
            ids = [hepdata_source_id(ins, ver, self.table)]
            result = dict({
                "status": "ok", "schema": SCHEMA_VERSION,
                "inspire_id": ins, "version": ver, "table": self.table,
                "source_id": ids[0], "file": entry["path"], "sha256": entry["sha256"],
            }, **_columns(table, 200 if self.max_points is None else int(self.max_points)))
        except HEPDataError as e:
            result = _error(e.error, e.reason)
        except ValueError as e:
            result = _error("Invalid Response", f"HEPData returned malformed JSON: {e}")
        except OSError as e:
            result = _error("Write Failed", str(e))
        request["http_requests"] = http_requests
        prov = record_call(base, "HEPDataTable", request, result,
                           request_url=sent or (http_requests[0] if http_requests else None),
                           api=API, source_ids=ids, raw=raw, raw_ext="yaml" if raw else "json",
                           files=files)
        if result["status"] != "ok":
            return attach(result, prov)
        return _table_digest(base, result, prov)


_DIGEST_ROWS = 5
_DIGEST_DEPENDENT = 10


def _table_digest(base: str, result: Dict[str, Any], prov: Dict[str, Any]) -> str:
    """Identifiers, column headers and the first rows of a HEPDataTableTool result."""
    indep = result["independent_variables"]
    dep = result["dependent_variables"]
    shown = dep[:_DIGEST_DEPENDENT]
    head = {k: result[k] for k in ("status", "schema", "inspire_id", "version", "table",
                                   "source_id", "file", "sha256", "points", "truncated")}
    head["independent_variables"] = [{"name": clip(v["name"], 80), "units": v["units"]}
                                     for v in indep]
    head["dependent_variables"] = [{
        "name": clip(v["name"], 80), "units": v["units"],
        "qualifiers": {clip(str(k), 40): clip(str(q), 80) for k, q in v["qualifiers"].items()},
    } for v in shown]
    if len(dep) > len(shown):
        head["dependent_count"] = len(dep)
    columns = ([v["values"] for v in indep]
               + [[p.get("value") for p in v["values"]] for v in shown])
    n_rows = max([len(c) for c in columns] or [0])
    rows = [[clip(c[i], 40) if i < len(c) else None for c in columns]
            for i in range(min(n_rows, _DIGEST_ROWS))]
    return digest(base, result, prov, head, "rows", rows, available=n_rows, what="rows")
