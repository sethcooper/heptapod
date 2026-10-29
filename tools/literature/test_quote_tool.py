"""
# test_quote_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

QuoteSource: verbatim quote, line/char span, section, label, page, hash.
"""

import hashlib
import json

from tools.literature.quote_tool import QuoteSourceTool

MAIN = r"""\documentclass{article}
\begin{document}
\section{Introduction}
We extend the SM by a scalar.
% \section{Old title}
\section{The model}
\subsection{Yukawa sector}
The Lagrangian is
\begin{equation}
  \mathcal{L} \supset y_{ij}\, \bar{Q}_i S_1 L_j + \text{h.c.}
  \label{eq:yukawa}
\end{equation}
\begin{align}
  m_S &= 1.5\,\mathrm{TeV} \label{eq:mass} \\
  \Gamma_S &= 0.1\, m_S \label{eq:width}
\end{align}
The coupling is real.
The coupling is real.
\end{document}
"""


def _paper(tmp_path):
    paper = tmp_path / "papers" / "2103.02708v2"
    (paper / "source").mkdir(parents=True)
    (paper / "source" / "ms.tex").write_text(MAIN)
    (paper / "source.tex").write_text(MAIN.replace("% \\section{Old title}\n", ""))
    (paper / "retrieval.json").write_text(json.dumps({"source_id": "arXiv:2103.02708v2"}))
    return paper


def _q(tmp_path, **kw):
    kw.setdefault("source", "2103.02708v2")
    return json.loads(QuoteSourceTool(base_directory=str(tmp_path), **kw)._run())


def test_quote_in_equation(tmp_path):
    paper = _paper(tmp_path)
    r = _q(tmp_path, pattern=r"y_{ij}\, \bar{Q}_i S_1 L_j")
    assert r["status"] == "ok", r
    assert r["source_id"] == "arXiv:2103.02708v2"
    assert r["file"] == "papers/2103.02708v2/source/ms.tex"  # the original, not source.tex
    assert r["source_sha256"] == hashlib.sha256((paper / "source" / "ms.tex").read_bytes()).hexdigest()
    assert r["line_start"] == r["line_end"] == 10
    assert MAIN[r["char_start"]:r["char_end"]] == r["quote"] == r"y_{ij}\, \bar{Q}_i S_1 L_j"
    # A commented-out heading does not count as the enclosing section.
    (tmp_path / "c.tex").write_text("\\section{A}\n% \\section{B}\ntext here\n")
    r2 = _q(tmp_path, source="c.tex", pattern="text here")
    assert r2["section"] == "A" and r2["line_start"] == 3 and r2["source_id"] is None
    assert r["section"] == "Yukawa sector" and r["label"] == "eq:yukawa" and r["page"] is None
    assert r["provenance"]["source_ids"] == ["arXiv:2103.02708v2"]
    assert r["provenance"]["files"][0]["sha256"] == r["source_sha256"]


def test_align_row_label_and_multiline_whitespace(tmp_path):
    _paper(tmp_path)
    r = _q(tmp_path, pattern=r"\Gamma_S &= 0.1")
    assert r["label"] == "eq:width", r
    r = _q(tmp_path, pattern="We extend the SM by a scalar. \\section{The",
           context_chars=5)
    assert r["status"] == "ok" and r["match_mode"] == "whitespace-insensitive", r
    # Only the normalized source.tex (comment line stripped) has this text.
    assert r["file"] == "papers/2103.02708v2/source.tex", r
    assert (r["line_start"], r["line_end"]) == (4, 5) and r["label"] is None
    assert r["section"] == "Introduction"
    assert r["context"].startswith("ion}\n")


def test_absent_and_ambiguous(tmp_path):
    _paper(tmp_path)
    r = _q(tmp_path, pattern="leptoquark")
    assert r["status"] == "error" and r["error"] == "Pattern Not Found"
    assert "papers/2103.02708v2/source/ms.tex" in r["searched"]
    r = _q(tmp_path, pattern="The coupling is real.")
    assert r["status"] == "error" and r["error"] == "Ambiguous Pattern" and r["count"] == 2
    assert [m["line_start"] for m in r["matches"]] == [17, 18]
    assert r["provenance"]["source_ids"] == []


def test_regex_unversioned_and_page_markers(tmp_path):
    _paper(tmp_path)
    r = _q(tmp_path, pattern=r"m_S &= [0-9.]+", regex=True)
    assert r["quote"] == "m_S &= 1.5" and r["label"] == "eq:mass"
    r = _q(tmp_path, source="2103.02708", pattern="x")
    assert r["error"] == "Unversioned Id" and r["fetched_versions"] == ["2103.02708v2"]

    (tmp_path / "x.tex").write_text("%% page 1\nIntro text\n%% page 2\nBound: m > 1 TeV\n")
    (tmp_path / "x.tex.provenance.json").write_text(json.dumps({"source_id": "arXiv:1811.07920v1"}))
    r = _q(tmp_path, source="x.tex", pattern="m > 1 TeV")
    assert r["page"] == 2 and r["line_start"] == 4 and r["source_id"] == "arXiv:1811.07920v1"
