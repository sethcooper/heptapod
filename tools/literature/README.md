# Literature tools

Find a paper, then get its text in a form worth reading.

| stage | tools | bundle |
|---|---|---|
| find it | `ArxivSearchTool` | `arxiv`, `literature` |
| read it | `ArxivSourceTool` (preferred) | `arxiv`, `literature` |
| | `ArxivPDFTool` | `arxiv`, `literature` |
| | `PDFToTeXTool` (fallback) | `literature` |
| cite it | `QuoteSourceTool` | `arxiv`, `literature` |

The INSPIRE bundle covers HEP metadata and citations, and has no arXiv access
or retrieval of its own, so these are additive rather than overlapping.

Most of the document below concerns the PDF→TeX fallback, which is the hardest
part of the bundle.

## Search results on disk

`ArxivSearchTool` returns a digest: the query, `request_url`, `count`, and a
compact `papers` list of `{"id" (versioned), "title", "authors", "published",
"category"}`, with titles cut to 150 characters, authors shortened to the
first one or two, and no abstracts. The full result, every field of every
paper including the abstract, is written to `research/results/<call_id>.json`
under `base_directory` and named in the digest as `results_path` (read it
with Python or `jq '.papers[] | {versioned_id, abstract}'`). The list is cut
from the end so the digest (`provenance` aside) stays under about 6,000 characters; `omitted`
(present only when non-zero) counts the papers left out, all of which are in
the file. If the file cannot be written, `results_path` is null and the note
says so. The helper is `tools/digest.py`.

## Where files land

One directory per paper, shared by both retrieval tools, under `output_dir`
(default `papers/`) inside the tool's `base_directory`:

```
papers/<arxiv_id>v<N>/
  <arxiv_id>v<N>.pdf              ArxivPDFTool
  <arxiv_id>v<N>.eprint.tar.gz    ArxivSourceTool — the e-print as served
  source/                         ArxivSourceTool — the extracted originals, verbatim
  source.tex                      ArxivSourceTool — comments stripped, \input inlined
  retrieval.json                  URL, time and sha256 of each fetch
```

The directory is always a version. An unversioned id means "the current
version": it is pinned to `vN` from the arXiv metadata (or, if that is
unreachable, from the file name arXiv serves) before the cache is consulted,
so a cached v1 is never returned for a paper that now has a v2. Path
separators are flattened, so `hep-ph/9905221v1` becomes `hep-ph_9905221v1`. A
bare PDF URL with no arXiv id in it falls back to a hash of the URL.

`ArxivSourceTool` retries an e-print download that arXiv rate-limits (429,
503) or cuts short (`IncompleteRead`): up to three times, after the server's
`Retry-After` when given and otherwise after 5, 15 and 45 seconds, with at
most 120 seconds of waiting in all. Then the error is returned.

When arXiv has no source for a paper, the e-print endpoint serves the PDF
itself. `ArxivSourceTool` saves those bytes to `pdf_path` in the same
directory rather than discarding them and asking you to fetch the identical
file again through a rate limiter that allows one request every three seconds.

## Why this exists

For papers with an arXiv e-print, the LaTeX source is authoritative and should
be preferred — nothing recovered from a PDF beats the macros the author
actually typed. `ArxivSourceTool` is that preferred path. The PDF→TeX half of
this bundle is the fallback for the cases where no source exists: journal-only
records, older papers, internal notes, theses.

## Two bundles, one directory

`heptapod[arxiv]` installs the retrieval tools and **nothing else** — they need
only `requests`, which is already in the base install.
`heptapod[literature]` adds the PDF→TeX fallback and its one pip dependency,
`pypdfium2`.

Both sets of modules live here. A bundle is a grouping rather than a directory
(the `bsm` bundle likewise spans `tools/analysis/`), and the package defers its
PDF imports, so an `arxiv`-only install imports cleanly and every arXiv tool
works. `pdf_to_tex` and `page_to_tex` raise on first use instead, naming what
to install.

## Why naive PDF text extraction fails on physics

A PDF's text layer maps glyphs to Unicode through the font's ToUnicode CMap.
For a TeX document that mapping is *lossy in exactly the places physics lives*:

| what TeX wrote | what the text layer says | why |
|---|---|---|
| `\epsilon` | `ϵ` U+03F5 | `\varepsilon` also lands near here; NFKC merges them |
| `\mu` | `µ` U+00B5 **micro sign** | not Greek mu |
| `\Delta` | `∆` U+2206 **increment** | not Greek Delta |
| `m` (math italic) | `m` U+006D | plain ASCII — indistinguishable from prose |
| `\mathbb{R}` | `R` | plain ASCII |
| `\sqrt` | `p` | the CMEX extensible radical, slot 0x70 |
| `\frac{1}{4}` | `1`, `4` | the bar is a **drawn rectangle**, not a glyph |
| `\bar{\psi}` | `ψ`, `¯` | accent is a separate spacing glyph |

Round-tripping that Unicode back to TeX produces output that looks
authoritative and is wrong — `\epsilon` returns as `\varepsilon`, and `\sqrt`
returns as the letter p.

## How this module works instead

Two sources of information that survive where Unicode does not:

1. **Font identity per glyph.** TeX segregates math by font: `CMMI` is Computer
   Modern Math Italic, so every glyph in it is a variable; `CMSY`/`CMEX` carry
   operators and extensible delimiters; `MSBM` carries blackboard bold. Keying
   on `(font_family, glyph)` recovers `\epsilon` vs `\varepsilon`, `\mathbb{R}`
   from a plain `R`, and the math/prose boundary.

2. **Path objects.** Fraction bars and radical vinculums are filled rectangles.
   Reading them back gives the structure needed to rebuild `\frac{}{}` and
   `\sqrt{}` by partitioning glyphs above/below/left/right of each rule.

3. **Fence geometry.** A matrix's rows sit on separate baselines, so vertical
   clustering splits them; the enclosing delimiter is what says they form one
   expression. Tall fences are either one large glyph or a vertical stack of
   extensible pieces in the private use area (`U+F8EB`, `U+F8EC`, …), so pieces
   sharing a kind, side and column are merged into one fence, openers matched
   to closers by height, and the enclosed glyphs read off as a grid.

   One trap worth knowing: the extensible vertical bar is `U+000C` (form feed)
   in CMEX, and Python's `str.strip()` counts form feed as whitespace. Any
   filter phrased as "skip whitespace characters" therefore discards every
   `vmatrix` fence before anything can see it. Glyphs are filtered on whether
   they are real positioned glyphs — a font and a non-degenerate box — not on
   what their Unicode mapping looks like.

Script level is measured against each glyph's **baseline origin**, not its
bounding box — a descender like `p` sits below the baseline, which would
otherwise make its own subscript appear higher than its base.

## The glyph table is derived, not hand-written

`build_table.py` compiles a probe document of ~230 macros under four math-font
packages (Computer Modern, Latin Modern, newtx/Times, mathptmx) and records
what each macro actually rendered to. Regenerate after adding macros:

```bash
python build_table.py        # requires pdflatex; writes tex_table.json
```

Current coverage: 688 entries across 22 font families. `pdflatex` is needed
only to regenerate the table, never to use it.

## Usage

```python
from tools.literature import pdf_to_tex

text = pdf_to_tex("paper.pdf")           # whole document
text = pdf_to_tex("paper.pdf", [0, 1])   # first two pages
```

As an agent tool, `PDFToTeXTool` writes to a file under `base_directory`, each
page preceded by a `%% page N` line, and returns JSON with the path, a
structural summary (not the body, so a long paper does not flood the agent's
context), the input PDF's `pdf_sha256` and, for a PDF the arXiv tools fetched,
its `source_id`. A `<output>.provenance.json` sidecar records the same.

## Provenance

Every result carries a top-level `provenance` object:

```
{"call_id": "<Tool>-<yyyymmddTHHMMSSZ>-<8 hex>", "tool", "retrieved_at",
 "request", "request_url", "api": {"name", "version"}, "from_cache",
 "cached_at", "source_ids", "response_sha256", "saved_path"}
```

plus `files` (`[{"path", "sha256", "bytes"}]`) when the call downloaded or
wrote files. The raw response is saved to `research/calls/<call_id>.<ext>`
under `base_directory` (`saved_path`, hashed by `response_sha256`), and one
line with the same fields (`ts` for `retrieved_at`) is appended to
`research/calls.jsonl`. The recorder is `tools/provenance.py`.

`source_ids` are versioned, `arXiv:<id>v<N>`; an unversioned id is never
reported for a downloaded file. `ArxivPDFTool` and `ArxivSourceTool` also
return `sha256` of the downloaded file (recomputed from disk on a cache hit),
the fetched `url` and `retrieved_at`; on a cache hit `from_cache` is true and
`cached_at` is the original fetch time.

`QuoteSourceTool` turns "the paper says X" into a checkable locator. Given a
versioned arXiv id (or a path to a fetched source, file or PDFToTeX output)
and a `pattern` (literal, whitespace-tolerant; `regex=true` for a regular
expression; `context_chars` for surrounding text), it returns

```
{"status": "ok", "source_id", "source_sha256", "file", "line_start",
 "line_end", "char_start", "char_end", "section", "label", "page", "quote",
 "match_mode", "provenance"}
```

`quote` is the verbatim text of the file at `[char_start, char_end)`; lines are
1-based. `section` is the innermost enclosing `\section`/`\subsection` title,
`label` the `\label` of an enclosing equation/align/table (the matched row's
label in a multi-row environment), `page` the last `%% page N` marker above the
match. The original files under `source/` are searched first, so line numbers
are those of the paper as submitted. An absent pattern is an error listing the
files searched; an ambiguous one is an error listing every match's location
(up to 20).

## Known limitations

- **`\mathrm` and other upright math styles are unrecoverable.** They render in
  the same font as prose, so nothing distinguishes them. `\mathrm{fb}` comes
  back as `fb`.
- **Matrices work for `pmatrix`, `bmatrix`, `Bmatrix`, `vmatrix` and `array`**
  with atomic entries. Two cases do not yet:
  - A `\frac` or `\sqrt` **inside a cell** puts its parts on baselines of their
    own and currently reads as extra matrix rows.
  - A big operator inside a tall fence is read as a two-row grid.
- **Multi-line aligned environments** (`align`, `eqnarray`) are not
  reconstructed; their rows become separate output lines.
- **Big-operator limits** (`\sum_{i=0}^{N}` in display style) are read as
  ordinary sub/superscripts rather than limits.
- **Spacing macros** (`\,`, `\quad`) are lost; they leave no glyph.
- **Scanned PDFs have no text layer at all.** This module does no OCR and will
  report that the document is empty.

Output is TeX-*flavoured*, intended for an agent to read and reason over. It is
not guaranteed to compile.

## Testing

```bash
python test_runner.py --component literature
```

Tests compile their fixture from LaTeX source held in the test file, so every
assertion checks against known ground truth. They skip if `pdflatex` is absent.
