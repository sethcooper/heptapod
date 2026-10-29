"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Literature tools: find a paper, then get its text in a form worth reading.

  **Find it.**  ArxivSearchTool searches arXiv (ids, PDF links, abstracts).
                The INSPIRE bundle covers HEP metadata and citations, and has
                no arXiv access or retrieval of its own.

  **Read it.**  ArxivSourceTool fetches the LaTeX e-print, which is the best
                input for reading a paper's equations because they survive
                intact. Where arXiv has no source, PDFToTeXTool reconstructs
                TeX-faithful text from the PDF instead -- symbols,
                sub/superscripts, fractions and radicals come back as TeX
                macros rather than being flattened into characters that no
                longer say what they meant. ArxivPDFTool retrieves the
                PDF that tool reads.

  **Cite it.**  QuoteSourceTool finds a statement in what was fetched and
                returns it verbatim with file, sha256, lines, section,
                equation label and page.

  **Where.**    Everything retrieved for one paper lands together in
                <output_dir>/<arxiv_id>v<N>/ -- the PDF, the extracted source
                tree, and the normalized LaTeX -- so a corpus is one
                directory per paper.

IMPORT COST. The arXiv tools need only `requests`, which is a base
dependency; the PDF path needs pypdfium2, the literature bundle's one pip
dep. tex_extract imports pypdfium2 at module scope, so importing it eagerly
here made `import tools.literature` -- and with it every arXiv tool -- fail
outright on a base install, for a dependency those tools never touch.

The PDF names are therefore resolved LAZILY (PEP 562). On a base install the
package imports, the arXiv tools work, and PDFToTeXTool still resolves (it
imports pypdfium2 inside the method that needs it, so only CALLING it fails);
pdf_to_tex and page_to_tex raise ModuleNotFoundError on first access, naming
what to install. The failure is scoped to the thing that actually needs the
dependency.
"""

from .arxiv_interface import ArxivInterface
from .literature_tools import (
    ArxivSearchTool,
    ArxivSourceTool,
    ArxivPDFTool,
)
from .quote_tool import QuoteSourceTool

__all__ = [
    # find
    "ArxivSearchTool",
    # read
    "ArxivSourceTool",
    "ArxivPDFTool",
    "PDFToTeXTool",
    "pdf_to_tex",
    # cite
    "QuoteSourceTool",
    "page_to_tex",
    # interfaces
    "ArxivInterface",
]

# Names served from the pypdfium2-backed modules, resolved on first access.
_LAZY = {
    "PDFToTeXTool": ".pdf_to_tex_tool",
    "pdf_to_tex": ".tex_extract",
    "page_to_tex": ".tex_extract",
}


def __getattr__(name):
    """Import a PDF-path name on first use (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    mod = import_module(module, __name__)
    value = getattr(mod, name)
    globals()[name] = value  # cache, so this runs once
    return value


def __dir__():
    return sorted(__all__)
