"""
# provenance.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Provenance for research tools (PDG, INSPIRE, arXiv, HEPData, PDF extraction,
quotes).

Every call a research tool makes is recorded in the sandbox:

  <base_directory>/research/calls/<call_id>.<ext>   the raw response
  <base_directory>/research/calls.jsonl             one line per call

and the tool's JSON result carries the same record as a top-level
``provenance`` object::

    {"call_id", "tool", "retrieved_at", "request", "request_url",
     "api": {"name", "version"}, "from_cache", "cached_at", "source_ids",
     "response_sha256", "saved_path"}

plus ``files`` ([{"path", "sha256", "bytes"}]) when the tool downloaded or
wrote files. ``source_ids`` are versioned: ``arXiv:<id>v<N>``,
``INSPIRE:<recid>/rev<N>`` (or ``/<updated>``), ``PDG:<pdgid>/<edition>``,
``HEPData:ins<inspire_id>/v<N>`` (a record) and ``.../v<N>/<table>`` (a table).

Recording never breaks the tool: if the sandbox cannot be written,
``saved_path`` is None and the result is still returned.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

CALLS_DIR = os.path.join("research", "calls")
CALLS_LOG = os.path.join("research", "calls.jsonl")

_VERSIONED_ARXIV = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[A-Za-z]{2})?/\d{7})v\d+$"
)


# ------------------------------------------------------------------ helpers


def now_iso(dt: Optional[datetime] = None) -> str:
    """UTC timestamp, ISO 8601 with a trailing Z, second precision."""
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_iso(value: Union[None, int, float, str, datetime]) -> Optional[str]:
    """Epoch seconds, datetime or string -> ISO UTC string (strings pass through)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return now_iso(value)
    if isinstance(value, (int, float)):
        return now_iso(datetime.fromtimestamp(float(value), tz=timezone.utc))
    return str(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(base_directory: str, path: str, sha256: Optional[str] = None) -> Dict[str, Any]:
    """``{"path", "sha256", "bytes"}`` for a file, path relative to base_directory."""
    base = os.path.realpath(base_directory)
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(base, path))
    return {
        "path": os.path.relpath(full, base),
        "sha256": sha256 or sha256_file(full),
        "bytes": os.path.getsize(full),
    }


def arxiv_source_id(arxiv_id: Optional[str]) -> Optional[str]:
    """``arXiv:<id>vN`` for a versioned id; None for an unversioned one.

    An unversioned id names whatever arXiv serves today, so it is never
    reported as the identity of evidence.
    """
    if not arxiv_id:
        return None
    s = re.sub(r"(?i)^arxiv:", "", arxiv_id.strip())
    return f"arXiv:{s}" if _VERSIONED_ARXIV.match(s) else None


def pdg_source_id(pdgid: Optional[str], edition: Optional[str]) -> Optional[str]:
    """``PDG:<pdgid>/<edition>``, e.g. ``PDG:S044W/2026``."""
    if not pdgid:
        return None
    pid = str(pdgid).split("/")[0].strip()
    if not pid:
        return None
    return f"PDG:{pid}/{edition}" if edition else f"PDG:{pid}"


def inspire_source_id(
    recid: Any,
    revision: Any = None,
    updated: Optional[str] = None,
    collection: Optional[str] = None,
) -> Optional[str]:
    """``INSPIRE:<recid>/rev<N>``, else ``INSPIRE:<recid>/<updated>``.

    Non-literature records are prefixed by collection:
    ``INSPIRE:authors/<id>/rev<N>``.
    """
    if recid in (None, ""):
        return None
    head = f"INSPIRE:{recid}" if collection in (None, "", "literature") else f"INSPIRE:{collection}/{recid}"
    if revision not in (None, ""):
        return f"{head}/rev{revision}"
    if updated:
        return f"{head}/{updated}"
    return head


def hepdata_source_id(inspire_id: Any, version: Any, table: Optional[str] = None) -> Optional[str]:
    """``HEPData:ins<inspire_id>/v<N>``, or ``.../v<N>/<table>`` for a table.

    Without a version the id would name whatever HEPData serves today, so it
    is None.
    """
    ins = str(inspire_id or "").strip()
    if not ins or version in (None, ""):
        return None
    head = f"HEPData:ins{ins}/v{version}"
    return f"{head}/{table}" if table else head


def _new_call_id(tool: str, when: datetime) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool or "tool")
    return f"{safe}-{when.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}"


def _raw_bytes(raw: Any) -> bytes:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return json.dumps(raw, indent=2, sort_keys=True, default=str).encode("utf-8")


def _dedup(ids: Iterable[Optional[str]]) -> List[str]:
    out: List[str] = []
    for i in ids:
        if i and i not in out:
            out.append(i)
    return out


# ---------------------------------------------------------------- recording


def record_call(
    base_directory: str,
    tool: str,
    request: Any,
    result: Any,
    *,
    request_url: Optional[str] = None,
    api: Optional[Dict[str, Any]] = None,
    from_cache: bool = False,
    cached_at: Any = None,
    source_ids: Sequence[Optional[str]] = (),
    raw: Any = None,
    raw_ext: str = "json",
    files: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Save the raw response, append to calls.jsonl, return the provenance dict.

    ``raw`` (bytes, str, or anything JSON-serialisable) defaults to ``result``.
    ``files`` lists files the call downloaded or wrote, as built by
    :func:`file_entry`.
    """
    when = datetime.now(timezone.utc)
    call_id = _new_call_id(tool, when)
    ts = now_iso(when)
    payload = _raw_bytes(result if raw is None else raw)
    response_sha256 = sha256_bytes(payload)
    ext = (raw_ext or "json").lstrip(".")
    api_block = {"name": (api or {}).get("name"), "version": (api or {}).get("version")}
    ids = _dedup(source_ids)
    cached_iso = to_iso(cached_at)

    saved_path: Optional[str] = None
    try:
        base = os.path.realpath(base_directory)
        calls_dir = os.path.join(base, CALLS_DIR)
        os.makedirs(calls_dir, exist_ok=True)
        rel = os.path.join(CALLS_DIR, f"{call_id}.{ext}")
        with open(os.path.join(base, rel), "wb") as fh:
            fh.write(payload)
        saved_path = rel
        line = {
            "call_id": call_id,
            "ts": ts,
            "tool": tool,
            "request": request,
            "request_url": request_url,
            "api": api_block,
            "from_cache": bool(from_cache),
            "cached_at": cached_iso,
            "source_ids": ids,
            "response_sha256": response_sha256,
            "saved_path": saved_path,
        }
        if files:
            line["files"] = list(files)
        with open(os.path.join(base, CALLS_LOG), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass

    prov = {
        "call_id": call_id,
        "tool": tool,
        "retrieved_at": ts,
        "request": request,
        "request_url": request_url,
        "api": api_block,
        "from_cache": bool(from_cache),
        "cached_at": cached_iso,
        "source_ids": ids,
        "response_sha256": response_sha256,
        "saved_path": saved_path,
    }
    if files:
        prov["files"] = list(files)
    return prov


def attach(result: Union[str, Dict[str, Any]], provenance: Dict[str, Any]) -> str:
    """Return the tool's JSON result with ``provenance`` added at top level.

    A result that is not a JSON object is wrapped as ``{"result": ...}``.
    """
    if isinstance(result, str):
        try:
            obj = json.loads(result)
        except ValueError:
            obj = {"status": "ok", "result": result}
    else:
        obj = dict(result)
    if not isinstance(obj, dict):
        obj = {"result": obj}
    obj["provenance"] = provenance
    return json.dumps(obj, indent=2, default=str)


def read_calls(base_directory: str) -> List[Dict[str, Any]]:
    """All lines of calls.jsonl (empty list if none)."""
    path = os.path.join(os.path.realpath(base_directory), CALLS_LOG)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
