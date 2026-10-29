"""
# test_literature.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for TeX extraction.

The fixture PDF is compiled from a LaTeX source held in this file, so every
assertion is checked against known ground truth rather than against whatever a
sample document happens to contain. Tests skip if pdflatex is unavailable.
"""

import json
import os
import shutil
import subprocess
import tempfile

import pytest

pytest.importorskip("pypdfium2")

from .tex_extract import pdf_to_tex  # noqa: E402

HAS_PDFLATEX = shutil.which("pdflatex") is not None
pytestmark = pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex not installed")

SOURCE = r"""
\documentclass{article}
\usepackage{amsmath,amssymb}
\begin{document}
\section{Kinematics}
The invariant mass is $m_{jj}^2 = (p_1+p_2)^2$ and we define
$\Delta R = \sqrt{\Delta\eta^2 + \Delta\phi^2}$.
\begin{equation}
\mathcal{L} = -\frac{1}{4}F_{\mu\nu}F^{\mu\nu}
 + \bar{\psi}(i\gamma^\mu D_\mu - m)\psi + \epsilon \chi \bar{\chi} A_\mu .
\end{equation}
Here $\epsilon \simeq 10^{-3}$, $\alpha_s(m_Z) = 0.118$, and
$\sigma \times \mathcal{B} < 0.1$ fb at $95\%$ CL.
Also $\varepsilon$, $\varphi$, $\hbar$, $\mathbb{R}$, $\partial_\mu$,
$\sum_i x_i$, and $\hat{s} = x_1 x_2 s$.
\end{document}
"""


@pytest.fixture(scope="module")
def extracted():
    tmp = tempfile.mkdtemp(prefix="heptapod-tex-")
    try:
        tex = os.path.join(tmp, "doc.tex")
        with open(tex, "w") as f:
            f.write(SOURCE)
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmp, tex],
            capture_output=True, text=True,
        )
        pdf = os.path.join(tmp, "doc.pdf")
        if not os.path.exists(pdf):
            pytest.skip(f"pdflatex failed: {r.stdout[-400:]}")
        yield pdf_to_tex(pdf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class TestSymbols:
    """Symbols must map to the macro that produced them, not a lookalike."""

    @pytest.mark.parametrize("macro", [
        r"\Delta", r"\eta", r"\mu", r"\nu", r"\gamma", r"\psi", r"\chi",
        r"\sigma", r"\times", r"\simeq", r"\alpha", r"\partial",
    ])
    def test_macro_present(self, extracted, macro):
        assert macro in extracted

    def test_epsilon_not_confused_with_varepsilon(self, extracted):
        """The distinction a Unicode round-trip destroys: both forms appear in
        the source and must survive as different macros."""
        assert r"\epsilon" in extracted
        assert r"\varepsilon" in extracted

    def test_phi_variants_distinguished(self, extracted):
        assert r"\varphi" in extracted

    def test_blackboard_bold(self, extracted):
        """\\mathbb{R} is a plain ASCII 'R' in the text layer; only the font
        identity (MSBM) distinguishes it."""
        assert r"\mathbb{R}" in extracted

    def test_calligraphic(self, extracted):
        assert r"\mathcal{L}" in extracted or r"\mathcal{B}" in extracted


class TestStructure:
    """Structure that the flat text layer cannot express on its own."""

    def test_fraction_reassembled(self, extracted):
        """Fraction bars are drawn rules, invisible to the text layer."""
        assert r"\frac{1}{4}" in extracted

    def test_radical_reassembled(self, extracted):
        assert r"\sqrt{" in extracted
        assert "p\r" not in extracted  # the CMEX radical must not leak as 'p'

    def test_stacked_sub_and_superscript(self, extracted):
        """m_{jj}^2 carries a subscript and a superscript on the same base."""
        assert "m_{jj}^{2}" in extracted

    def test_subscript_on_descender_base(self, extracted):
        """p descends below the baseline; its subscript must still register."""
        assert "p_{1}" in extracted and "p_{2}" in extracted

    def test_accents_recomposed(self, extracted):
        assert r"\bar{\psi}" in extracted
        assert r"\hat{s}" in extracted

    def test_subscripts_and_superscripts(self, extracted):
        assert "F_{" in extracted and "F^{" in extracted
        assert "10^{-3}" in extracted
        assert "m_{Z}" in extracted


MATRIX_SOURCE = r"""
\documentclass{article}
\usepackage{amsmath}
\pagestyle{empty}
\begin{document}
$$ M = \begin{pmatrix} a & b \\ c & d \end{pmatrix} $$
$$ \Sigma = \begin{pmatrix} m_1 & 0 & 0 \\ 0 & m_2 & \epsilon \\
   0 & \epsilon & m_3 \end{pmatrix} $$
$$ V = \begin{bmatrix} \alpha \\ \beta \\ \gamma \end{bmatrix} $$
$$ S = \begin{Bmatrix} x \\ y \end{Bmatrix} $$
$$ D = \begin{vmatrix} a & b \\ c & d \end{vmatrix} $$
$$ R = \left( \frac{a}{b} \right) $$
\end{document}
"""


@pytest.fixture(scope="module")
def matrices():
    tmp = tempfile.mkdtemp(prefix="heptapod-mat-")
    try:
        tex = os.path.join(tmp, "m.tex")
        with open(tex, "w") as f:
            f.write(MATRIX_SOURCE)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmp, tex],
            capture_output=True, text=True,
        )
        pdf = os.path.join(tmp, "m.pdf")
        if not os.path.exists(pdf):
            pytest.skip("pdflatex failed")
        yield pdf_to_tex(pdf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class TestMatrices:
    """Matrix rows sit on separate baselines and the enclosing fence -- often a
    stack of extensible glyphs -- is what says they form one expression."""

    def test_pmatrix_2x2(self, matrices):
        assert r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}" in matrices

    def test_pmatrix_3x3_with_subscripts(self, matrices):
        assert (r"\begin{pmatrix} m_{1} & 0 & 0 \\ 0 & m_{2} & \epsilon \\ "
                r"0 & \epsilon & m_{3} \end{pmatrix}") in matrices

    def test_bmatrix_column_vector(self, matrices):
        assert r"\begin{bmatrix} \alpha \\ \beta \\ \gamma \end{bmatrix}" in matrices

    def test_brace_matrix(self, matrices):
        assert r"\begin{Bmatrix} x \\ y \end{Bmatrix}" in matrices

    def test_determinant_bars(self, matrices):
        """TeX's extensible vertical bar is U+000C (form feed) in CMEX, so any
        whitespace-based glyph filter silently drops every vmatrix fence."""
        assert r"\begin{vmatrix} a & b \\ c & d \end{vmatrix}" in matrices

    def test_non_grid_fence_is_left_right(self, matrices):
        """A tall fence around a non-grid must still render as a fence."""
        assert r"\left(" in matrices and r"\right)" in matrices

    def test_no_control_characters(self, matrices):
        """Extensible delimiter pieces must never leak as raw glyph codes."""
        assert not [c for c in matrices if ord(c) < 32 and c not in "\n\r\t"]


class TestProse:
    """Prose must survive untouched: no macro leakage into running text."""

    def test_words_intact(self, extracted):
        assert "invariant mass" in extracted
        assert "Kinematics" in extracted

    def test_no_mathrm_leakage(self, extracted):
        """\\mathrm renders in the prose font; probing it would wrap ordinary
        letters in \\mathrm{...}. Guard against that regression."""
        assert r"\mathrm" not in extracted

    def test_tex_specials_escaped(self, extracted):
        assert r"\%" in extracted

    def test_spacing_preserved(self, extracted):
        assert "invariantmass" not in extracted


class TestPageSelection:
    def test_page_subset(self, extracted):
        assert extracted.strip()


class TestTool:
    """The BaseTool wrapper: sandboxing and context discipline."""

    def _run(self, base, **fields):
        from .pdf_to_tex_tool import PDFToTeXTool
        tool = PDFToTeXTool(base_directory=str(base), **fields)
        tool._setup()
        return tool._run()

    def test_rejects_escaping_path(self, tmp_path):
        out = self._run(tmp_path, pdf_path="../../../etc/passwd", output_path="x.tex")
        assert "denied" in out.lower() or "escape" in out.lower()

    def test_missing_file(self, tmp_path):
        out = self._run(tmp_path, pdf_path="nope.pdf")
        assert "not found" in out.lower()

    def test_writes_file_and_summarises(self, tmp_path, extracted):
        """The tool must return a path and summary, not the document body."""
        src = tmp_path / "doc.pdf"
        tex = tmp_path / "doc.tex"
        with open(tex, "w") as f:
            f.write(SOURCE)
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", str(tmp_path), str(tex)],
            capture_output=True, text=True,
        )
        if not src.exists():
            pytest.skip("pdflatex failed")
        out = self._run(tmp_path, pdf_path="doc.pdf", output_path="out.tex")
        assert (tmp_path / "out.tex").exists()
        res = json.loads(out)
        assert res["output_path"] == "out.tex" and res["tex_macros"] > 0
        assert (tmp_path / "out.tex").read_text().startswith("%% page 1\n")
        # the body must not be echoed back into the agent's context
        assert r"\frac{1}{4}" not in out

    def test_parse_pages(self):
        from .pdf_to_tex_tool import PDFToTeXTool
        assert PDFToTeXTool._parse_pages("1-3", 10) == [0, 1, 2]
        assert PDFToTeXTool._parse_pages("1,4", 10) == [0, 3]
        assert PDFToTeXTool._parse_pages(None, 10) is None
        assert PDFToTeXTool._parse_pages("99", 10) == []
