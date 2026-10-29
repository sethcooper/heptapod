"""
# test_pdf_to_tex_provenance.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

PDFToTeX returns JSON, marks pages, and carries the PDF hash and arXiv id.
The PDF backend is stubbed, so neither pypdfium2 nor pdflatex is needed.
"""

import hashlib
import json
import sys
import types

from tools.literature.pdf_to_tex_tool import PDFToTeXTool, join_pages


def _stub_backend(monkeypatch, pages):
    fake_pdfium = types.ModuleType("pypdfium2")
    fake_pdfium.PdfDocument = lambda path: list(pages)
    fake_extract = types.ModuleType("tools.literature.tex_extract")
    fake_extract.page_to_tex = lambda page: page
    monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
    monkeypatch.setitem(sys.modules, "tools.literature.tex_extract", fake_extract)


def _run(base, **fields):
    tool = PDFToTeXTool(base_directory=str(base), **fields)
    tool._setup()
    return json.loads(tool._run())


def test_join_pages():
    assert join_pages([(1, "a"), (3, "b")]) == "%% page 1\na\n%% page 3\nb"


def test_page_markers_hash_and_source_id(tmp_path, monkeypatch):
    _stub_backend(monkeypatch, ["Intro $x$", "Lagrangian \\mathcal{L}", "Refs"])
    paper = tmp_path / "papers" / "2103.02708v2"
    paper.mkdir(parents=True)
    pdf = paper / "2103.02708v2.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    (paper / "retrieval.json").write_text(json.dumps(
        {"source_id": "arXiv:2103.02708v2", "pdf": {"sha256": sha}}))

    r = _run(tmp_path, pdf_path="papers/2103.02708v2/2103.02708v2.pdf", pages="2-3")
    text = (tmp_path / r["output_path"]).read_text()
    assert text == "%% page 2\nLagrangian \\mathcal{L}\n%% page 3\nRefs"
    assert r["status"] == "ok" and r["pages_extracted"] == [2, 3] and r["n_pages"] == 3
    assert r["pdf_sha256"] == sha and r["source_id"] == "arXiv:2103.02708v2"
    p = r["provenance"]
    assert p["source_ids"] == ["arXiv:2103.02708v2"]
    assert [f["sha256"] for f in p["files"]] == [sha, r["output_sha256"]]
    side = json.loads((tmp_path / (r["output_path"] + ".provenance.json")).read_text())
    assert side["source_id"] == "arXiv:2103.02708v2" and side["pdf_sha256"] == sha


def test_replaced_pdf_is_not_attributed(tmp_path, monkeypatch):
    _stub_backend(monkeypatch, ["text"])
    (tmp_path / "retrieval.json").write_text(json.dumps(
        {"source_id": "arXiv:2103.02708v2", "pdf": {"sha256": "0" * 64}}))
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4 other")
    r = _run(tmp_path, pdf_path="x.pdf")
    assert r["source_id"] is None and r["provenance"]["source_ids"] == []
