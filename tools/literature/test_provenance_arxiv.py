"""
# test_provenance_arxiv.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

arXiv results carry versioned source ids and provenance (network mocked).
"""

import hashlib
import json
from unittest import mock

from tools.literature import arxiv_interface
from tools.literature.literature_tools import ArxivPDFTool, ArxivSearchTool, ArxivSourceTool
from tools.literature.test_arxiv import _ATOM_FIXTURE, _FakeStreamResponse, _gz, _router
from tools.provenance import read_calls

arxiv_interface._LIMITER.interval = 0.0
PDF = b"%PDF-1.4\n" + b"z" * 64


def _get(router):
    return mock.patch.object(arxiv_interface.requests.Session, "get", router)


def test_search_source_ids_and_raw_feed(tmp_path):
    with _get(lambda self, url, **kw: _FakeStreamResponse(_ATOM_FIXTURE.encode())):
        r = json.loads(ArxivSearchTool(query="ti:leptoquark", base_directory=str(tmp_path))._run())
    p = r["provenance"]
    assert r["papers"][0]["id"] == "2103.02708v2"
    assert p["source_ids"] == ["arXiv:2103.02708v2"]
    assert p["request_url"] == r["request_url"] and "search_query=ti%3Aleptoquark" in p["request_url"]
    saved = (tmp_path / p["saved_path"]).read_bytes()
    assert saved == _ATOM_FIXTURE.encode() and p["response_sha256"] == hashlib.sha256(saved).hexdigest()


def test_latest_is_not_served_stale(tmp_path):
    """v1 cached; arXiv now reports v2: 'latest' must fetch v2, not reuse v1."""
    with _get(_router(PDF, "v1")):
        r1 = json.loads(ArxivPDFTool(arxiv_id="2103.02708", base_directory=str(tmp_path))._run())
    with _get(_router(PDF + b"new", "v2")):
        r2 = json.loads(ArxivPDFTool(arxiv_id="2103.02708", base_directory=str(tmp_path))._run())
    assert r1["source_id"] == "arXiv:2103.02708v1" and r2["source_id"] == "arXiv:2103.02708v2"
    assert r2["cached"] is False and r2["sha256"] != r1["sha256"]
    assert r2["url"] == "https://arxiv.org/pdf/2103.02708v2.pdf"
    p = r2["provenance"]
    assert p["request_url"] == r2["url"]
    assert p["request"]["http_requests"][0].startswith("https://export.arxiv.org/api/query?id_list=")
    assert p["files"] == [{"path": r2["pdf_path"], "sha256": r2["sha256"], "bytes": r2["bytes"]}]
    assert len(read_calls(str(tmp_path))) == 2


def test_version_from_download_when_metadata_unreachable(tmp_path):
    def get(self, url, **kw):
        if "api/query" in url:
            raise arxiv_interface.requests.ConnectionError("down")
        return _FakeStreamResponse(_gz(b"\\documentclass{a}\nx"), url=url,
                                   headers={"Content-Disposition": 'attachment; filename="arXiv-2103.02708v4.tar.gz"'})
    with _get(get):
        r = json.loads(ArxivSourceTool(arxiv_id="2103.02708", base_directory=str(tmp_path))._run())
    assert r["arxiv_id"] == "2103.02708v4" and r["source_id"] == "arXiv:2103.02708v4", r


def test_no_version_no_file(tmp_path):
    def get(self, url, **kw):
        if "api/query" in url:
            raise arxiv_interface.requests.ConnectionError("down")
        return _FakeStreamResponse(PDF, url=url)
    with _get(get):
        out = ArxivPDFTool(arxiv_id="2103.02708", base_directory=str(tmp_path))._run()
    assert "Version Resolution Failed" in out
    assert not list(tmp_path.rglob("*.pdf"))


# ----------------------------------------------------------- search digest


def _feed(n):
    entry = _ATOM_FIXTURE.split("<entry>")[1].split("</entry>")[0]
    title = "A scalar leptoquark model with a very long and descriptive title " * 3
    entries = []
    for i in range(n):
        e = (entry.replace("2103.02708v2", f"2401.{i:05d}v3")
             .replace("A scalar leptoquark model", f"{i}: {title}")
             .replace("We study a scalar leptoquark.", "Abstract text. " * 150))
        entries.append(f"<entry>{e}</entry>")
    head, _, _ = _ATOM_FIXTURE.partition("<entry>")
    return head + "".join(entries) + "</feed>"


def test_search_digest_and_results_file(tmp_path):
    with _get(lambda self, url, **kw: _FakeStreamResponse(_ATOM_FIXTURE.encode())):
        r = json.loads(ArxivSearchTool(query="ti:leptoquark", base_directory=str(tmp_path))._run())
    assert r["papers"] == [{"id": "2103.02708v2", "title": "A scalar leptoquark model",
                            "authors": "Jane Physicist and John Theorist",
                            "published": "2021-03-04", "category": "hep-ph"}]
    assert r["count"] == 1 and "omitted" not in r
    p = r["provenance"]
    assert r["results_path"] == f"research/results/{p['call_id']}.json"
    full = json.loads((tmp_path / r["results_path"]).read_text())
    assert full["papers"][0]["abstract"] == "We study a scalar leptoquark."
    assert full["papers"][0]["source_id"] == "arXiv:2103.02708v2"


def test_search_digest_budget(tmp_path):
    from tools.digest import DIGEST_BUDGET
    feed = _feed(50).encode()
    with _get(lambda self, url, **kw: _FakeStreamResponse(feed)):
        r = json.loads(ArxivSearchTool(query="ti:leptoquark", max_results=50,
                                       base_directory=str(tmp_path))._run())
    assert r["count"] == 50
    assert len(json.dumps({k: v for k, v in r.items() if k != "provenance"}, indent=2)) <= DIGEST_BUDGET
    shown = r["papers"]
    assert r["omitted"] == 50 - len(shown) and 0 < len(shown) < 50
    assert [x["id"] for x in shown] == [f"2401.{i:05d}v3" for i in range(len(shown))]
    assert all(len(x["title"]) <= 150 and "abstract" not in x for x in shown)
    full = json.loads((tmp_path / r["results_path"]).read_text())
    assert len(full["papers"]) == 50 and full["papers"][49]["abstract"].startswith("Abstract")


# ------------------------------------------------------ e-print retries


class _Busy:
    """A 429 response with a Retry-After header."""

    status_code = 429

    def __init__(self, retry_after=None):
        self.headers = {"Retry-After": retry_after} if retry_after else {}
        self.url = None

    def raise_for_status(self):
        raise arxiv_interface.requests.HTTPError("429 Client Error: Too Many Requests",
                                                 response=self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _CutShort(_FakeStreamResponse):
    def iter_content(self, chunk_size=65536):
        yield b"\x1f\x8b"
        raise arxiv_interface.requests.exceptions.ChunkedEncodingError(
            "Connection broken: IncompleteRead(2 bytes read, 100 more expected)")


def _eprint_sequence(*responses):
    queue = list(responses)

    def get(self, url, **kw):
        return queue.pop(0)
    return get


def _source(tmp_path):
    return json.loads(ArxivSourceTool(arxiv_id="2103.02708v2", base_directory=str(tmp_path))._run())


def test_eprint_retried_after_429(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(arxiv_interface, "_sleep", waits.append)
    ok = _FakeStreamResponse(_gz(b"\\documentclass{a}\nx"))
    with _get(_eprint_sequence(_Busy("7"), _Busy(), ok)):
        r = _source(tmp_path)
    assert r["status"] == "ok" and r["source_type"] == "single_tex", r
    assert waits == [7.0, 15.0]                 # Retry-After honoured, then the backoff


def test_eprint_retried_after_incomplete_read(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(arxiv_interface, "_sleep", waits.append)
    ok = _FakeStreamResponse(_gz(b"\\documentclass{a}\nx"))
    with _get(_eprint_sequence(_CutShort(b""), ok)):
        r = _source(tmp_path)
    assert r["status"] == "ok", r
    assert waits == [5.0]


def test_eprint_retries_exhausted(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(arxiv_interface, "_sleep", waits.append)
    with _get(_eprint_sequence(*[_Busy() for _ in range(4)])):
        out = ArxivSourceTool(arxiv_id="2103.02708v2", base_directory=str(tmp_path))._run()
    assert "E-print Download Failed" in out and "429" in out
    assert waits == [5.0, 15.0, 45.0]


def test_eprint_retry_wait_is_capped(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(arxiv_interface, "_sleep", waits.append)
    with _get(_eprint_sequence(_Busy("100"), _Busy("100"))):
        out = ArxivSourceTool(arxiv_id="2103.02708v2", base_directory=str(tmp_path))._run()
    assert "E-print Download Failed" in out
    assert waits == [100.0]                     # a second 100 s pause would pass the cap


def test_eprint_other_errors_not_retried(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(arxiv_interface, "_sleep", waits.append)
    with _get(_eprint_sequence(_FakeStreamResponse(b"", status=404))):
        out = ArxivSourceTool(arxiv_id="2103.02708v2", base_directory=str(tmp_path))._run()
    assert "E-print Download Failed" in out and waits == []
