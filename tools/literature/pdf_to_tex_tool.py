"""
# pdf_to_tex_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Convert a TeX-produced PDF into TeX-flavoured text.
"""

import json
import os
import re
from typing import Iterable, Optional, Tuple

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from ..provenance import attach, file_entry, record_call, sha256_file

SCHEMA_VERSION = "pdftex-1.1"

#: Line written before each page's text; QuoteSource reads it back.
PAGE_MARKER = "%% page {n}"
PAGE_MARKER_RE = re.compile(r"^%% page (\d+)\s*$", re.MULTILINE)


def join_pages(pages: Iterable[Tuple[int, str]]) -> str:
    """Join ``(1-based page number, text)`` pairs, each preceded by a
    ``%% page N`` line so every extracted line maps back to its page."""
    return "\n".join(f"{PAGE_MARKER.format(n=n)}\n{text}" for n, text in pages)


def _arxiv_source_for(pdf_abs: str, pdf_sha: str) -> Optional[str]:
    """The versioned arXiv id of a PDF the arXiv tools fetched, if this is it.

    Read from the paper directory's retrieval.json and trusted only when the
    recorded hash matches the file, so a replaced PDF is not misattributed.
    """
    try:
        with open(os.path.join(os.path.dirname(pdf_abs), "retrieval.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    for kind in ("pdf", "eprint"):
        if isinstance(meta.get(kind), dict) and meta[kind].get("sha256") == pdf_sha:
            return meta.get("source_id")
    return None

# tex_extract imports pypdfium2 and loads the glyph table at module scope;
# both are deferred to _run() so toolkit startup stays cheap.


class PDFToTeXTool(BaseTool):
    """
    Extract TeX-flavoured text from a PDF produced by LaTeX.

    Recovers math semantics from the embedded font identity rather than from
    Unicode, so ``\\epsilon`` and ``\\varepsilon`` stay distinct, variables stay
    distinguishable from prose, and ``\\frac``/``\\sqrt`` are reassembled from
    the drawn rules that carry them.

    The extracted text is written to a file under ``base_directory``, each
    page preceded by a ``%% page N`` line; the tool returns JSON with the
    output path, a structural summary (not the document body, so a long
    paper does not flood the agent's context), the input PDF's sha256 and,
    for a PDF fetched by the arXiv tools, its versioned ``source_id``. A
    ``<output>.provenance.json`` sidecar records the same for QuoteSource.
    """

    # --------------------------- Runtime fields --------------------------- #
    pdf_path: str = RuntimeField(
        description="Relative path to the source PDF, inside base_directory."
    )
    output_path: Optional[str] = RuntimeField(
        default=None,
        description=(
            "Relative path for the extracted text (e.g. 'papers/2401.12345.tex'). "
            "Defaults to the PDF path with a .tex suffix."
        ),
    )
    pages: Optional[str] = RuntimeField(
        default=None,
        description=(
            "Optional 1-based page selection, e.g. '1-4' or '1,3,7'. "
            "Omit to extract the whole document."
        ),
    )
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(default=".", description="Base directory for safe paths")
    # ---------------------------------------------------------------------- #

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PDFToTeX"
        self.description = (
            "Extract TeX-flavoured text from a LaTeX-produced PDF. Recovers Greek "
            "letters, operators, sub/superscripts, accents, fractions, radicals and "
            "matrices as TeX macros by reading the embedded math fonts. Writes the "
            "result to a file (each page preceded by a '%% page N' line) and returns "
            "JSON with the path, a summary, the PDF's sha256 and provenance. Best for papers "
            "with no arXiv source available; prefer the LaTeX source when there is "
            "one, since it is authoritative. Output is TeX-flavoured for reading, "
            "not guaranteed to compile: upright math styles (\\mathrm) are "
            "unrecoverable, aligned environments flatten, and a fraction inside a "
            "matrix cell may read as an extra row."
        )

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    @staticmethod
    def _parse_pages(spec: str, n_pages: int):
        """'1-4,7' -> [0,1,2,3,6]; returns None for the whole document."""
        if not spec:
            return None
        out = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, _, b = part.partition("-")
                out.extend(range(int(a) - 1, int(b)))
            else:
                out.append(int(part) - 1)
        return [p for p in out if 0 <= p < n_pages]

    def _run(self) -> str:
        try:
            import pypdfium2 as pdfium
            from .tex_extract import page_to_tex
        except ImportError as e:
            return self.format_error(
                error="Missing Dependency",
                reason=str(e),
                suggestion="Install the literature bundle: tb install heptapod --bundle literature",
            )

        src = self._safe_path(self.pdf_path)
        rel_out = self.output_path or (os.path.splitext(self.pdf_path)[0] + ".tex")
        dst = self._safe_path(rel_out)

        if not src or not dst:
            return self.format_error(
                error="Access Denied",
                reason="pdf_path or output_path escapes base_directory",
                suggestion="Use relative paths inside base_directory",
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason="PDF file not found",
                context=f"path={self.pdf_path}",
                suggestion="Provide a valid PDF path relative to base_directory",
            )

        try:
            doc = pdfium.PdfDocument(src)
            n_pages = len(doc)
            selected = self._parse_pages(self.pages, n_pages)
            indices = range(n_pages) if selected is None else selected
            if selected is not None and not selected:
                return self.format_error(
                    error="Invalid Page Selection",
                    reason=f"no pages matched {self.pages!r}",
                    context=f"document has {n_pages} pages",
                    suggestion="Use 1-based indices within the document length",
                )
            indices = list(indices)
            text = join_pages((i + 1, page_to_tex(doc[i])) for i in indices)
        except Exception as e:
            return self.format_error(
                error="Extraction Error",
                reason=str(e),
                context=f"path={self.pdf_path}",
                suggestion="Verify the file is a valid PDF; scanned PDFs have no text layer",
            )

        body = PAGE_MARKER_RE.sub("", text)
        macros = re.findall(r"\\([a-zA-Z]+)", body)
        n_math = body.count("$") // 2
        if not body.strip():
            return self.format_error(
                error="No Text Layer",
                reason="the PDF contains no extractable text",
                context=f"path={self.pdf_path}",
                suggestion="This is likely a scanned document; it needs OCR, which this tool does not do",
            )

        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={rel_out}",
                suggestion="Check that the output directory is writable",
            )

        base = os.path.realpath(self.base_directory)
        pdf_sha = sha256_file(src)
        out_sha = sha256_file(dst)
        source_id = _arxiv_source_for(src, pdf_sha)
        rel_pdf = os.path.relpath(os.path.realpath(src), base)
        rel_dst = os.path.relpath(os.path.realpath(dst), base)
        try:
            with open(dst + ".provenance.json", "w", encoding="utf-8") as f:
                json.dump({"source_id": source_id, "pdf_path": rel_pdf, "pdf_sha256": pdf_sha,
                           "output_sha256": out_sha, "pages": [i + 1 for i in indices],
                           "schema": SCHEMA_VERSION}, f, indent=2)
        except OSError:
            pass

        top = sorted(set(macros), key=lambda m: -macros.count(m))[:12]
        result = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "output_path": rel_dst,
            "pdf_path": rel_pdf,
            "pdf_sha256": pdf_sha,
            "output_sha256": out_sha,
            "source_id": source_id,
            "pages_extracted": [i + 1 for i in indices],
            "n_pages": n_pages,
            "page_marker": PAGE_MARKER,
            "characters": len(body),
            "math_runs": n_math,
            "tex_macros": len(macros),
            "distinct_macros": len(set(macros)),
            "most_common": ["\\" + m for m in top],
            "note": (
                "Read the file to inspect the content. Each page starts with a "
                "'%% page N' line. Output is TeX-flavoured for reading and is not "
                "guaranteed to compile: \\mathrm and other upright math styles "
                "render in the prose font and cannot be recovered, aligned "
                "environments flatten, and a fraction inside a matrix cell may "
                "read as an extra row. Where an arXiv LaTeX source exists it "
                "remains the authoritative version."
            ),
        }
        request = {"pdf_path": self.pdf_path, "output_path": self.output_path, "pages": self.pages}
        prov = record_call(
            base, "PDFToTeX", request, result,
            api={"name": "heptapod tex_extract", "version": SCHEMA_VERSION},
            source_ids=[source_id],
            files=[file_entry(base, src, pdf_sha), file_entry(base, dst, out_sha)],
        )
        return attach(result, prov)
