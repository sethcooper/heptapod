"""
# test_provenance.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

record_call: raw response on disk, one calls.jsonl line, matching hashes.
"""

import hashlib
import json
import re

from tools.provenance import (
    arxiv_source_id,
    attach,
    inspire_source_id,
    pdg_source_id,
    read_calls,
    record_call,
)


def test_record_call_writes_file_and_line(tmp_path):
    p = record_call(
        str(tmp_path), "ArxivSearch", {"query": "q"}, {"status": "ok"},
        request_url="https://export.arxiv.org/api/query?search_query=q",
        api={"name": "arXiv API", "version": "1"},
        source_ids=["arXiv:2103.02708v2", None, "arXiv:2103.02708v2"],
        raw=b"<feed/>", raw_ext="xml", cached_at=0,
    )
    assert re.fullmatch(r"ArxivSearch-\d{8}T\d{6}Z-[0-9a-f]{8}", p["call_id"])
    saved = tmp_path / p["saved_path"]
    assert saved.read_bytes() == b"<feed/>" and saved.suffix == ".xml"
    assert p["response_sha256"] == hashlib.sha256(b"<feed/>").hexdigest()
    assert p["source_ids"] == ["arXiv:2103.02708v2"]
    assert p["cached_at"] == "1970-01-01T00:00:00Z"
    (line,) = read_calls(str(tmp_path))
    assert line["call_id"] == p["call_id"] and line["ts"] == p["retrieved_at"]
    assert line["response_sha256"] == p["response_sha256"]
    assert set(line) == {"call_id", "ts", "tool", "request", "request_url", "api", "from_cache",
                         "cached_at", "source_ids", "response_sha256", "saved_path"}
    assert set(p) == {"call_id", "tool", "retrieved_at", "request", "request_url", "api",
                      "from_cache", "cached_at", "source_ids", "response_sha256", "saved_path"}


def test_default_raw_is_result_and_attach(tmp_path):
    p = record_call(str(tmp_path), "T", {}, {"a": 1})
    assert json.loads((tmp_path / p["saved_path"]).read_text()) == {"a": 1}
    out = json.loads(attach('{"status": "ok"}', p))
    assert out["provenance"]["call_id"] == p["call_id"]


def test_source_id_forms():
    assert arxiv_source_id("2103.02708v2") == "arXiv:2103.02708v2"
    assert arxiv_source_id("hep-ph/9905221v1") == "arXiv:hep-ph/9905221v1"
    assert arxiv_source_id("2103.02708") is None
    assert pdg_source_id("S044W/2026", "2026") == "PDG:S044W/2026"
    assert inspire_source_id(451647, 12) == "INSPIRE:451647/rev12"
    assert inspire_source_id(451647, None, "2024-01-02T03:04:05") == "INSPIRE:451647/2024-01-02T03:04:05"
