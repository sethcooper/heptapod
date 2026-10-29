# INSPIRE tools

Search INSPIRE-HEP for papers, authors, institutions, conferences, journals
and experiments; fetch records, citations and BibTeX; keep a local reading
list and notes. Tools live in `inspire_tools.py`, the API client in
`inspire_interface.py`, and the 24 h response cache in `cache.py` (enabled by
the `cache_enabled` config key, stored in `<base_directory>/.inspire_cache.json`).

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

For INSPIRE, `request_url` is the exact URL sent, query parameters included
(the first one when a tool makes several requests; all are listed in
`request.http_requests`). `from_cache` is true when every response came from
the cache, and `cached_at` is when the cached response was originally fetched.
`source_ids` name each record in the responses with its revision,
`INSPIRE:<recid>/rev<N>`, or its last-updated time,
`INSPIRE:<recid>/<updated>`, when the API does not return a revision
(non-literature records: `INSPIRE:<collection>/<id>/...`). The raw API JSON
(or BibTeX text) is what is saved to `saved_path`.

## Search results on disk

`InspireSearchTool` returns a digest rather than every paper in full. The
complete result (authors, abstracts, DOIs, publication info, keywords) is
written to `research/results/<call_id>.json` under `base_directory`, named by
the same `call_id` as the raw response in `research/calls/`, and the digest
gives its path as `results_path`:

```
{"status": "ok", "schema", "total", "returned", "query_info",
 "results_path": "research/results/<call_id>.json",
 "papers": [{"recid", "arxiv_id", "title", "first_author", "year",
             "citation_count"}],
 "omitted": N, "note", "provenance"}
```

Titles are cut to 150 characters, `first_author` is the collaboration when
there is one (otherwise the first author, with "et al."), and abstracts are
left out. The list is cut from the end so the digest (`provenance` aside) stays under about 6,000
characters; `omitted` (present only when non-zero) counts the papers left
out, all of which are in the file. Read it with Python or `jq`, e.g.
`jq '.papers[] | {recid, abstract}' research/results/<call_id>.json`. If the
file cannot be written, `results_path` is null and the note says so. Errors
are returned as they are. The helper is `tools/digest.py`.

To quote the text of a paper found here, fetch it with the literature bundle's
`ArxivSourceTool` and locate statements with `QuoteSourceTool`.
