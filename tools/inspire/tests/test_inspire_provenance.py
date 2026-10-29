"""
# test_inspire_provenance.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

INSPIRE results carry provenance: exact request URL, versioned record ids,
cache hits with the original fetch time. The HTTP layer is mocked.
"""

import json
from unittest import mock

import pytest

from tools.inspire import inspire_interface
from tools.inspire.inspire_tools import InspireBibTeXTool, InspirePaperTool, InspireSearchTool
from tools.provenance import read_calls



@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    monkeypatch.setattr(inspire_interface.RateLimiter, "wait_if_needed", lambda self: 0.0)

RECORD = {
    "id": "451647", "revision_id": 37, "updated": "2025-06-01T10:00:00+00:00",
    "metadata": {
        "$schema": "https://inspirehep.net/schemas/records/hep.json",
        "control_number": 451647,
        "titles": [{"title": "String theory dynamics in various dimensions"}],
        "authors": [{"full_name": "Witten, Edward"}],
        "arxiv_eprints": [{"value": "hep-th/9503124", "categories": ["hep-th"]}],
    },
}
SEARCH = {"hits": {"total": 1, "hits": [{
    "id": "451647", "updated": "2025-06-01T10:00:00+00:00",
    "metadata": RECORD["metadata"]}]}}


class _Resp:
    def __init__(self, url, params, body):
        self.status_code = 200
        self.url = inspire_interface.InspireInterface.full_url(url, params)
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _fake_get(self, url, params=None, timeout=None):
    if params and params.get("format") == "bibtex":
        return _Resp(url, params, "@article{Witten:1995ex, title={X}}")
    if url.endswith("literature/451647"):
        return _Resp(url, params, RECORD)
    return _Resp(url, params, SEARCH)


def test_paper_provenance_and_cache(tmp_path):
    with mock.patch.object(inspire_interface.requests.Session, "get", _fake_get) as _:
        r1 = json.loads(InspirePaperTool(recid="451647", base_directory=str(tmp_path))._run())
    p = r1["provenance"]
    assert r1["status"] == "ok"
    assert p["request_url"] == "https://inspirehep.net/api/literature/451647"
    assert p["source_ids"] == ["INSPIRE:451647/rev37"]
    assert p["from_cache"] is False and p["cached_at"] is None
    assert json.loads((tmp_path / p["saved_path"]).read_text())["revision_id"] == 37

    # Second call: served from the 24 h cache, no network.
    with mock.patch.object(inspire_interface.requests.Session, "get",
                           side_effect=AssertionError("network used")):
        r2 = json.loads(InspirePaperTool(recid="451647", base_directory=str(tmp_path))._run())
    p2 = r2["provenance"]
    assert p2["from_cache"] is True and p2["cached_at"].endswith("Z")
    assert p2["source_ids"] == ["INSPIRE:451647/rev37"]
    assert [c["call_id"] for c in read_calls(str(tmp_path))] == [p["call_id"], p2["call_id"]]


def test_search_records_exact_url(tmp_path):
    with mock.patch.object(inspire_interface.requests.Session, "get", _fake_get):
        r = json.loads(InspireSearchTool(query="a witten", size=5, cache_enabled=False,
                                         base_directory=str(tmp_path))._run())
    p = r["provenance"]
    assert p["request_url"].startswith("https://inspirehep.net/api/literature?q=a+witten")
    assert "size=5" in p["request_url"] and "sort=mostcited" in p["request_url"]
    assert p["source_ids"] == ["INSPIRE:451647/2025-06-01T10:00:00+00:00"]
    assert p["request"]["query"] == "a witten" and p["api"]["name"].startswith("INSPIRE")


def test_bibtex_saved_raw(tmp_path):
    with mock.patch.object(inspire_interface.requests.Session, "get", _fake_get):
        r = json.loads(InspireBibTeXTool(recid="451647", base_directory=str(tmp_path))._run())
    p = r["provenance"]
    assert p["request_url"].endswith("literature/451647?format=bibtex")
    assert p["saved_path"].endswith(".bib")
    assert (tmp_path / p["saved_path"]).read_text().startswith("@article")


def test_search_digest_and_full_result(tmp_path):
    with mock.patch.object(inspire_interface.requests.Session, "get", _fake_get):
        r = json.loads(InspireSearchTool(query="a witten", size=5, cache_enabled=False,
                                         base_directory=str(tmp_path))._run())
    assert r["status"] == "ok" and r["total"] == 1 and r["returned"] == 1
    assert r["papers"] == [{"recid": "451647", "arxiv_id": "hep-th/9503124",
                            "title": "String theory dynamics in various dimensions",
                            "first_author": "Witten, Edward", "year": None,
                            "citation_count": 0}]
    assert r["query_info"]["spires"] == "a witten" and "omitted" not in r
    p = r["provenance"]
    assert r["results_path"] == f"research/results/{p['call_id']}.json"
    full = json.loads((tmp_path / r["results_path"]).read_text())
    assert full["papers"][0]["authors"] == ["Witten, Edward"]
    assert full["papers"][0]["arxiv_category"] == "hep-th"


def test_search_digest_budget(tmp_path):
    from tools.digest import DIGEST_BUDGET
    hits = [{"id": str(i), "updated": "2025-06-01T10:00:00+00:00", "metadata": {
        "control_number": 1000 + i,
        "titles": [{"title": f"{i}: Searches for heavy neutral leptons " * 8}],
        "abstracts": [{"value": "A long abstract. " * 200}],
        "authors": [{"full_name": f"Author {k}"} for k in range(50)],
        "collaborations": [{"value": "ATLAS"}] if i % 2 else [],
        "earliest_date": "2024-02-03", "citation_count": i,
        "arxiv_eprints": [{"value": f"2402.{i:05d}", "categories": ["hep-ex"]}]}}
        for i in range(30)]
    big = {"hits": {"total": 900, "hits": hits}}
    with mock.patch.object(inspire_interface.requests.Session, "get",
                           lambda self, url, params=None, timeout=None: _Resp(url, params, big)):
        r = json.loads(InspireSearchTool(query="t heavy neutral lepton", size=30,
                                         cache_enabled=False, base_directory=str(tmp_path))._run())
    body = {k: v for k, v in r.items() if k != "provenance"}
    assert len(json.dumps(body, indent=2)) <= DIGEST_BUDGET
    shown = r["papers"]
    assert r["omitted"] == 30 - len(shown) and 0 < len(shown) < 30
    assert [p["recid"] for p in shown] == [str(1000 + i) for i in range(len(shown))]
    assert shown[0]["first_author"] == "Author 0 et al." and shown[0]["year"] == 2024
    assert shown[1]["first_author"] == "ATLAS Collaboration"
    assert all("abstract" not in p and len(p["title"]) <= 150 for p in shown)
    full = json.loads((tmp_path / r["results_path"]).read_text())
    assert len(full["papers"]) == 30 and full["papers"][29]["abstract"].startswith("A long")


def test_search_error_is_unchanged(tmp_path):
    def boom(self, url, params=None, timeout=None):
        raise inspire_interface.requests.ConnectionError("down")
    with mock.patch.object(inspire_interface.requests.Session, "get", boom):
        r = json.loads(InspireSearchTool(query="a witten", cache_enabled=False,
                                         base_directory=str(tmp_path))._run())
    assert r["status"] == "error" and "results_path" not in r and "papers" not in r
