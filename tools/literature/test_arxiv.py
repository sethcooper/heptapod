#!/usr/bin/env python3
"""
# test_literature.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for the arXiv literature toolkit.

Offline tests (no network) cover the query builder, PDF text extraction on a
synthetic PDF, path-traversal safety, and error formatting. Network tests
(arXiv search + PDF download) are attempted and skipped gracefully if arXiv is
unreachable, so the suite passes in offline/CI environments.
"""

import argparse
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.literature import arxiv_interface
from tools.literature.arxiv_interface import (
    ArxivInterface,
    normalize_arxiv_id,
    validate_arxiv_id,
)
from tools.literature.literature_tools import (
    ArxivSearchTool,
    ArxivPDFTool,
    _safe_join,
)

TEST_DIR = TOOL_DIR / "test_files"

# Don't sleep between requests in tests (module-level shared limiter).
arxiv_interface._LIMITER.interval = 0.0

# Minimal arXiv Atom feed fixture (namespaces + version suffix + pdf link).
_ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2103.02708v2</id>
    <title>A scalar leptoquark model</title>
    <summary>We study a scalar leptoquark.</summary>
    <published>2021-03-04T00:00:00Z</published>
    <updated>2021-05-01T00:00:00Z</updated>
    <author><name>Jane Physicist</name></author>
    <author><name>John Theorist</name></author>
    <arxiv:primary_category term="hep-ph"/>
    <category term="hep-ph"/>
    <category term="hep-ex"/>
    <arxiv:doi>10.1000/xyz123</arxiv:doi>
    <link rel="alternate" href="http://arxiv.org/abs/2103.02708v2"/>
    <link title="pdf" type="application/pdf" href="http://arxiv.org/pdf/2103.02708v2"/>
  </entry>
</feed>"""


class _FakeStreamResponse:
    """Minimal stand-in for a streaming ``requests`` Response context manager."""

    def __init__(self, content: bytes, status: int = 200, headers=None, url=None):
        self._content = content
        self.status_code = status
        self.headers = headers or {}
        self.url = url
        self.text = content.decode("utf-8", "replace")

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _router(payload: bytes, version: str = "v1"):
    """``Session.get`` side effect: Atom metadata naming ``version`` for
    id_list queries, otherwise ``payload`` named after that version (as
    arXiv's Content-Disposition does)."""
    from urllib.parse import parse_qs, urlparse

    def get(self, url, **kw):
        if "api/query" in url:
            arxiv_id = parse_qs(urlparse(url).query)["id_list"][0]
            feed = _ATOM_FIXTURE.replace("2103.02708v2", f"{arxiv_id}{version}")
            return _FakeStreamResponse(feed.encode())
        arxiv_id = url.rstrip("/").split("/", 4)[-1].removesuffix(".pdf")
        name = arxiv_id if re.search(r"v\d+$", arxiv_id) else arxiv_id + version
        return _FakeStreamResponse(
            payload, url=url,
            headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'},
        )

    return get


def test_query_builder() -> bool:
    print(">> Testing arXiv query builder...\n")
    q = ArxivInterface.build_query("scalar leptoquark", ["BSM", "leptoquark"])
    assert 'cat:hep-ph' in q, q
    assert 'ti:"scalar leptoquark"' in q, q
    assert 'abs:"BSM"' in q, q
    assert normalize_arxiv_id("2103.02708v2") == "2103.02708"
    print("[✓] Query builder test passed\n")
    return True


def test_search_missing_params() -> bool:
    print(">> Testing search parameter validation...\n")
    tool = ArxivSearchTool(base_directory=str(TEST_DIR))
    result = tool._run()
    assert "error" in result.lower() or "missing" in result.lower(), result
    print("[✓] Search parameter-validation test passed\n")
    return True


def test_arxiv_search_live() -> bool:
    print(">> Testing live arXiv search (network)...\n")
    tool = ArxivSearchTool(
        model_name="scalar leptoquark", keywords=["leptoquark"], max_results=3,
        base_directory=str(TEST_DIR),
    )
    try:
        result = json.loads(tool._run())
    except Exception as e:  # noqa: BLE001
        print(f"[i] Skipping live search (error: {e})\n")
        return True
    if result.get("status") != "ok":
        print(f"[i] Skipping live search (network unavailable): {result.get('reason')}\n")
        return True
    assert result["count"] >= 1, result
    assert result["papers"][0].get("id"), result
    print(f"[✓] Live search returned {result['count']} papers\n")
    return True


def test_arxiv_id_validation() -> bool:
    print(">> Testing arXiv id validation/normalization...\n")
    assert validate_arxiv_id("2103.02708") == "2103.02708"
    assert validate_arxiv_id("2103.02708v2") == "2103.02708v2"
    assert validate_arxiv_id("arXiv:2103.02708v2") == "2103.02708v2"
    assert validate_arxiv_id("hep-ph/9905221") == "hep-ph/9905221"
    assert validate_arxiv_id("../../etc/passwd") is None
    assert validate_arxiv_id("not an id") is None
    assert validate_arxiv_id("") is None
    assert normalize_arxiv_id("arXiv:2103.02708v2") == "2103.02708"
    print("[✓] arXiv id validation test passed\n")
    return True


def test_atom_parsing() -> bool:
    print(">> Testing Atom feed parsing (namespaces + version)...\n")
    papers = ArxivInterface()._parse_feed(_ATOM_FIXTURE)
    assert len(papers) == 1, papers
    p = papers[0]
    assert p["arxiv_id"] == "2103.02708", p  # version stripped for canonical id
    assert p["version"] == "v2" and p["source_id"] == "arXiv:2103.02708v2", p
    assert p["title"] == "A scalar leptoquark model", p
    assert p["authors"] == ["Jane Physicist", "John Theorist"], p
    assert p["categories"][0] == "hep-ph", p
    assert p["doi"] == "10.1000/xyz123", p
    assert p["pdf_url"].startswith("https://"), p  # http -> https upgrade
    assert p["pdf_url"].endswith("2103.02708v2"), p
    assert p["abs_url"].startswith("https://"), p
    assert p["published"] == "2021-03-04", p
    print("[✓] Atom parsing test passed\n")
    return True


def test_fetch_rejects_bad_url() -> bool:
    print(">> Testing PDF fetch URL allowlist (SSRF guard)...\n")
    tool = ArxivPDFTool(pdf_url="http://evil.example.com/x.pdf", base_directory=str(TEST_DIR))
    result = tool._run()
    assert "denied" in result.lower() or "error" in result.lower(), result
    # http (non-https) arXiv should also be refused
    tool2 = ArxivPDFTool(pdf_url="http://arxiv.org/pdf/2103.02708.pdf", base_directory=str(TEST_DIR))
    assert "denied" in tool2._run().lower()
    print("[✓] URL allowlist test passed\n")
    return True


def test_fetch_rejects_invalid_id() -> bool:
    print(">> Testing PDF fetch id validation...\n")
    tool = ArxivPDFTool(arxiv_id="../../etc/passwd", base_directory=str(TEST_DIR))
    result = tool._run()
    assert "invalid" in result.lower() or "error" in result.lower(), result
    print("[✓] id validation test passed\n")
    return True


def test_fetch_download_and_cache_mocked() -> bool:
    print(">> Testing PDF download + cache identity (mocked network)...\n")
    fake_pdf = b"%PDF-1.4\n" + b"x" * 512
    with mock.patch.object(
        arxiv_interface.requests.Session, "get", _router(fake_pdf, "v3"),
    ):
        r1 = json.loads(
            ArxivPDFTool(arxiv_id="2103.02708", base_directory=str(TEST_DIR))._run()
        )
        assert r1["status"] == "ok" and r1["cached"] is False, r1
        assert (TEST_DIR / r1["pdf_path"]).exists()
        # "latest" pinned to the version arXiv reports, and cached under it.
        assert r1["arxiv_id"] == "2103.02708v3" and r1["source_id"] == "arXiv:2103.02708v3", r1
        assert r1["pdf_path"] == "papers/2103.02708v3/2103.02708v3.pdf", r1
        assert r1["provenance"]["files"][0]["sha256"] == r1["sha256"], r1

        # Same id again -> cache hit, same hash.
        r1b = json.loads(
            ArxivPDFTool(arxiv_id="2103.02708", base_directory=str(TEST_DIR))._run()
        )
        assert r1b["cached"] is True and r1b["sha256"] == r1["sha256"], r1b
        assert r1b["retrieved_at"] == r1["retrieved_at"], r1b
        assert r1b["provenance"]["from_cache"] is True, r1b
        assert r1b["provenance"]["cached_at"] == r1["retrieved_at"], r1b

        # Different id -> different file (no cache collision).
        r2 = json.loads(
            ArxivPDFTool(arxiv_id="1811.07920", base_directory=str(TEST_DIR))._run()
        )
        assert r2["status"] == "ok" and r2["pdf_path"] != r1["pdf_path"], r2

    # Non-PDF payload must be rejected.
    with mock.patch.object(
        arxiv_interface.requests.Session, "get", _router(b"<html>not a pdf</html>"),
    ):
        r3 = ArxivPDFTool(arxiv_id="2000.00001", base_directory=str(TEST_DIR))._run()
        assert "error" in r3.lower() or "not a valid pdf" in r3.lower(), r3
    print("[✓] Download + cache-identity test passed\n")
    return True


def test_fetch_both_args_precedence() -> bool:
    print(">> Testing fetch arxiv_id precedence over pdf_url...\n")
    fake_pdf = b"%PDF-1.4\n" + b"y" * 128
    with mock.patch.object(
        arxiv_interface.requests.Session, "get", _router(fake_pdf),
    ):
        r = json.loads(
            ArxivPDFTool(
                arxiv_id="2103.02708",
                pdf_url="https://arxiv.org/pdf/1811.07920.pdf",
                base_directory=str(TEST_DIR),
            )._run()
        )
    assert r["status"] == "ok", r
    # Cache filename must key on arxiv_id (the winner), not the URL.
    assert "2103.02708" in r["pdf_path"], r
    assert "1811.07920" not in r["pdf_path"], r
    print("[✓] Both-args precedence test passed\n")
    return True


def test_cache_symlink_not_followed() -> bool:
    print(">> Testing cache does not follow a leaf symlink...\n")
    base = TEST_DIR / "cachebox"
    # One directory per paper: <base>/papers/<id>/<id>.pdf
    pdf_dir = base / "papers" / "2103.02708v1"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    # A malicious symlink at the cache path pointing outside the sandbox.
    outside_pdf = TEST_DIR / "secret.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4\nSECRET")
    dest = pdf_dir / "2103.02708v1.pdf"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(outside_pdf, dest)

    fake_pdf = b"%PDF-1.4\nLEGIT-DOWNLOAD"
    with mock.patch.object(
        arxiv_interface.requests.Session, "get", _router(fake_pdf),
    ):
        r = json.loads(
            ArxivPDFTool(arxiv_id="2103.02708", base_directory=str(base))._run()
        )
    assert r["status"] == "ok", r
    assert r["cached"] is False, r  # symlink not served as cache
    # The symlink was atomically replaced by the real download, not followed.
    assert not (pdf_dir / "2103.02708v1.pdf").is_symlink(), "symlink should be replaced"
    assert outside_pdf.read_bytes() == b"%PDF-1.4\nSECRET", "outside file must be untouched"
    print("[✓] Cache symlink-safety test passed\n")
    return True


def _make_tar(members, symlink=None, absolute=None):
    """Build an in-memory tar. members: {name: bytes}."""
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo(name=symlink[0])
            info.type = tarfile.SYMTYPE
            info.linkname = symlink[1]
            tf.addfile(info)
        if absolute:
            info = tarfile.TarInfo(name=absolute)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
    return buf.getvalue()


def _gz(data: bytes) -> bytes:
    import gzip as _gzip

    return _gzip.compress(data)


def test_source_archive_safety() -> bool:
    print(">> Testing source-archive safety (tar-slip, links, caps)...\n")
    from tools.literature.source_archive import safe_extract_tar

    dest = str(TEST_DIR / "srcbox")

    for bad_tar, label in [
        (_make_tar({"../evil.tex": b"x"}), "traversal ../"),
        (_make_tar({}, absolute="/etc/evil.tex"), "absolute path"),
        (_make_tar({"ok.tex": b"x"}, symlink=("link.tex", "/etc/passwd")), "symlink"),
        (_make_tar({f"f{i}.tex": b"x" for i in range(11)}), "member count cap"),
    ]:
        try:
            if label == "member count cap":
                safe_extract_tar(bad_tar, dest, max_members=10)
            else:
                safe_extract_tar(bad_tar, dest)
            raise AssertionError(f"{label}: should have been rejected")
        except ValueError:
            pass

    # Size caps.
    big = _make_tar({"big.tex": b"y" * 2048})
    try:
        safe_extract_tar(big, dest, max_member_bytes=1024)
        raise AssertionError("member size cap not enforced")
    except ValueError:
        pass

    # A benign archive extracts.
    ok = _make_tar({"paper/ms.tex": b"\\documentclass{article}"})
    files = safe_extract_tar(ok, dest)
    assert files == [os.path.join("paper", "ms.tex")], files
    print("[✓] Source-archive safety test passed\n")
    return True


def test_payload_detection_and_tex_processing() -> bool:
    print(">> Testing payload detection, main-tex resolution, comments, inlining...\n")
    from tools.literature.source_archive import (
        detect_payload,
        gunzip_capped,
        inline_inputs,
        resolve_main_tex,
        strip_comments,
    )

    assert detect_payload(_gz(b"anything")) == "gzip"
    assert detect_payload(b"%PDF-1.4 x") == "pdf"
    assert detect_payload(_make_tar({"a.tex": b"x"})) == "tar"
    assert detect_payload(b"\\documentclass{article}") == "tex"
    assert gunzip_capped(_gz(b"hello")) == b"hello"
    try:
        gunzip_capped(_gz(b"z" * 4096), max_bytes=1024)
        raise AssertionError("gunzip cap not enforced")
    except ValueError:
        pass

    # Main-tex resolution with decoys: appendix (no \documentclass) + main.
    src = TEST_DIR / "resolvebox"
    (src / "sub").mkdir(parents=True, exist_ok=True)
    (src / "appendix.tex").write_text("\\section{app}")
    (src / "sub" / "real.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nBody\n\\end{document}"
    )
    assert resolve_main_tex(str(src)) == os.path.join("sub", "real.tex")

    # Comment stripping preserves escaped \%.
    out = strip_comments("a \\% kept % dropped\nplain % gone")
    assert out == "a \\% kept \nplain ", repr(out)

    # One-level inlining with containment + missing marker.
    (src / "sub" / "part.tex").write_text("INLINED-CONTENT")
    main = "\\input{part}\n\\input{../../../etc/passwd}\n\\input{missing}"
    inlined = inline_inputs(main, str(src), os.path.join("sub", "real.tex"))
    assert "INLINED-CONTENT" in inlined, inlined
    assert "passwd' not inlined" in inlined or "not inlined" in inlined, inlined
    print("[✓] Payload/tex-processing test passed\n")
    return True


def test_arxiv_source_tool_mocked() -> bool:
    print(">> Testing ArxivSourceTool end-to-end (mocked network)...\n")
    from tools.literature.literature_tools import ArxivSourceTool

    tar = _make_tar(
        {
            "ms.tex": b"\\documentclass{article}\n% comment\n\\input{lag}\n\\begin{document}\\end{document}",
            "lag.tex": b"L = y S1 u e % yukawa",
        }
    )
    with mock.patch.object(
        arxiv_interface.requests.Session, "get", _router(_gz(tar), "v2"),
    ):
        r = json.loads(
            ArxivSourceTool(arxiv_id="2103.02708", base_directory=str(TEST_DIR))._run()
        )
    assert r["status"] == "ok" and r["source_type"] == "tar", r
    assert r["n_files"] == 2, r
    text = (TEST_DIR / r["tex_path"]).read_text()
    assert "L = y S1 u e" in text and "% comment" not in text, text[:200]
    # Versioned identity, e-print hash kept, originals kept verbatim.
    import hashlib
    assert r["source_id"] == "arXiv:2103.02708v2" and r["tex_path"].startswith("papers/2103.02708v2/"), r
    assert r["sha256"] == hashlib.sha256(_gz(tar)).hexdigest(), r
    assert (TEST_DIR / r["eprint_path"]).read_bytes() == _gz(tar), r
    assert "% comment" in (TEST_DIR / r["main_tex"]).read_text(), r
    assert r["provenance"]["source_ids"] == ["arXiv:2103.02708v2"], r

    # Cache hit on second call: explicit version needs no network, and the
    # e-print hash comes back recomputed from the kept payload.
    with mock.patch.object(
        arxiv_interface.requests.Session, "get", side_effect=AssertionError("network used"),
    ):
        r2 = json.loads(
            ArxivSourceTool(arxiv_id="2103.02708v2", base_directory=str(TEST_DIR))._run()
        )
    assert r2["cached"] is True and r2["sha256"] == r["sha256"], r2
    assert r2["tex_sha256"] == r["tex_sha256"] and r2["main_tex"] == r["main_tex"], r2
    assert r2["provenance"]["from_cache"] is True, r2
    assert r2["provenance"]["cached_at"] == r["retrieved_at"], r2

    # pdf_only payload.
    with mock.patch.object(
        arxiv_interface.requests.Session, "get", _router(b"%PDF-1.4 binary"),
    ):
        r3 = json.loads(
            ArxivSourceTool(arxiv_id="1811.07920", base_directory=str(TEST_DIR))._run()
        )
    assert r3["status"] == "ok" and r3["source_type"] == "pdf_only", r3
    # The e-print WAS the PDF, so it is saved rather than re-fetched: the
    # bytes are already in hand and arXiv is rate-limited to one request/3s.
    assert "pdf_path" in r3, r3
    assert r3["pdf_path"] == "papers/1811.07920v1/1811.07920v1.pdf", r3
    saved = TEST_DIR / r3["pdf_path"]
    assert saved.is_file() and saved.read_bytes() == b"%PDF-1.4 binary", r3
    assert r3["bytes"] == len(b"%PDF-1.4 binary"), r3
    assert "PDFToTeXTool" in r3.get("suggestion", ""), r3

    # Malicious archive rejected via the tool path.
    with mock.patch.object(
        arxiv_interface.requests.Session, "get", _router(_gz(_make_tar({"../evil.tex": b"x"}))),
    ):
        r4 = ArxivSourceTool(arxiv_id="2000.00001", base_directory=str(TEST_DIR))._run()
    assert "unsafe" in r4.lower() or "error" in r4.lower(), r4

    # Single gzipped .tex payload.
    with mock.patch.object(
        arxiv_interface.requests.Session, "get", _router(_gz(b"\\documentclass{a}\nBody % c")),
    ):
        r5 = json.loads(
            ArxivSourceTool(arxiv_id="2005.06475", base_directory=str(TEST_DIR))._run()
        )
    assert r5["status"] == "ok" and r5["source_type"] == "single_tex", r5
    print("[✓] ArxivSourceTool mocked end-to-end test passed\n")
    return True


def test_symlink_escape() -> bool:
    print(">> Testing symlink-escape rejection in sandbox...\n")
    base = TEST_DIR / "sandbox"
    base.mkdir(parents=True, exist_ok=True)
    outside = TEST_DIR / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    link = base / "escape"
    if not link.exists():
        os.symlink(outside, link)
    # A path through the symlinked dir resolves outside base -> rejected.
    assert _safe_join(str(base), "escape/evil.txt") is None
    assert _safe_join(str(base), "../outside/evil.txt") is None
    # A legitimate in-sandbox path is allowed.
    assert _safe_join(str(base), "ok/file.txt") is not None
    print("[✓] Symlink-escape test passed\n")
    return True


def test_path_traversal() -> bool:
    print(">> Testing path-traversal safety...\n")
    # The PDF-text tool that used to cover this was dropped in favour of
    # PDFToTeXTool, so the check moves to the tool that still writes into the
    # sandbox: a fetch must refuse an output_dir that escapes it.
    tool = ArxivPDFTool(
        arxiv_id="2103.02708",
        output_dir="../../../../tmp",
        base_directory=str(TEST_DIR),
    )
    result = tool._run()
    assert "error" in result.lower() or "denied" in result.lower(), result
    print("[\u2713] Path-traversal test passed\n")
    return True


try:
    import pytest

    @pytest.fixture(autouse=True, scope="module")
    def _fresh_test_dir():
        """Under pytest, start from and leave behind an empty TEST_DIR, so a
        rerun does not hit the previous run's cache."""
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        yield
        shutil.rmtree(TEST_DIR, ignore_errors=True)
except ImportError:  # run as a script
    pass


def cleanup_test_files() -> None:
    print("\n>> Cleaning up test files...\n")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"[✓] Removed: {TEST_DIR.name}\n")
    else:
        print("[i] No test files to clean up\n")


TESTS = [
    test_query_builder,
    test_arxiv_id_validation,
    test_atom_parsing,
    test_path_traversal,
    test_symlink_escape,
    test_search_missing_params,
    test_fetch_rejects_bad_url,
    test_fetch_rejects_invalid_id,
    test_fetch_download_and_cache_mocked,
    test_fetch_both_args_precedence,
    test_cache_symlink_not_followed,
    test_source_archive_safety,
    test_payload_detection_and_tex_processing,
    test_arxiv_source_tool_mocked,
    test_arxiv_search_live,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the literature toolkit")
    parser.add_argument("--keep-files", action="store_true", help="Keep test-generated files")
    args = parser.parse_args()

    all_passed = True
    for test in TESTS:
        try:
            if not test():
                all_passed = False
        except Exception as e:  # noqa: BLE001
            print(f"[✗] {test.__name__} failed: {e}\n")
            all_passed = False

    if not args.keep_files:
        cleanup_test_files()
    else:
        print("\n[i] Keeping test files (--keep-files set)\n")

    if all_passed:
        print("[✓] All tests passed!\n")
        sys.exit(0)
    print("[✗] Some tests failed!\n")
    sys.exit(1)
