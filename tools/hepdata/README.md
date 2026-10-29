# HEPData tools

Find the data records behind a publication on [HEPData](https://www.hepdata.net)
and fetch their tables: exclusion limits, signal-region yields and
backgrounds, efficiencies, likelihood inputs. Tools live in `hepdata_tools.py`
(`hepdata` bundle).

| Tool | Does |
|---|---|
| `HEPDataSearchTool` | search records; each hit with its INSPIRE id, version and table count |
| `HEPDataRecordTool` | one record at a version: metadata, resources (likelihoods, analysis code, web pages) and every table's name, DOI and description |
| `HEPDataTableTool` | one table at a version: saved as `<output_dir>/ins<id>/v<N>/<table>.yaml`, its columns in the results file |

Records are addressed by the publication's INSPIRE record number; find it from
an arXiv id with the `inspire` tools. A record is versioned, and a new version
can change any table, so pin the version once chosen. HEPData holds data for
only some papers; `HEPDataRecordTool` on an INSPIRE id without a record is a
`Not Found` error that says so, and `HEPDataSearchTool` finds the records
there are.

## Digests and results files

Each tool returns a digest, and writes its full structured result to
`research/results/<call_id>.json` under `base_directory`, named by the same
`call_id` as the raw response in `research/calls/`. The digest gives the path
as `results_path`, a `note`, and a compact list:

| Tool | digest | only in the results file |
|---|---|---|
| `HEPDataSearchTool` | `results`: `inspire_id`, `version`, `source_id`, `title`, `collaborations`, `arxiv_id`, `year`, `total_tables` | `journal_info`, `hepdata_doi`, `last_updated`, `tables` (every table title) |
| `HEPDataRecordTool` | record ids, `title`, `collaborations`, `arxiv_id`, `doi`, `resource_count`, `table_count`; `tables`: `name`, `location`, `description` (120 characters) | `abstract`, `resources`, each table's `doi`, `source_id` and longer description |
| `HEPDataTableTool` | ids, `file`, `sha256`, `points`, the variables' names, units and qualifiers; `rows`: the first five points, values only | every column's values (up to `max_points`) with their uncertainties |

Titles are cut to 150 characters. The list is cut from the end so the digest
(`provenance` aside) stays under about 6,000 characters; `omitted` (present only when non-zero)
counts entries left out, all of which are in the file. Read it with Python or
`jq`, e.g. `jq '.results[] | {inspire_id, tables}'` for a search or
`jq '.dependent_variables[0].values'` for a table. If the file cannot be
written, `results_path` is null and the note says so. Errors are returned as
they are. The helper is `tools/digest.py`.

## Tables

The results file of `HEPDataTableTool` holds the table's independent
variables (a number per point, or `[low, high]` for a bin) and dependent
variables with their qualifiers (`SQRT(S)`, `LUMINOSITY`, ...) and per-point
uncertainties, `{"label", "symerror"}` or `{"label", "plus", "minus"}`, with
signs as the record gives them. `max_points` (default 200) caps each column in
that file; the saved YAML always holds the whole table.

## Provenance

Every result carries the `provenance` object of `tools/provenance.py` (see
`tools/inspire/README.md`), and each call is appended to
`research/calls.jsonl` under `base_directory` with its raw response.
`source_ids` name what was read: `HEPData:ins<inspire_id>/v<N>` for a record,
`HEPData:ins<inspire_id>/v<N>/<table>` for a table. For a table, the raw
response saved to `saved_path` is the YAML itself, so `response_sha256` equals
the saved file's `sha256`; `request.http_requests` lists the version lookup,
when the version was not given, and the table download.

HEPData refuses requests whose user agent it does not recognise, so the tools
send the `requests` library's own.
