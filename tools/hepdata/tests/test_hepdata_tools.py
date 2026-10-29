"""
# test_hepdata_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

HEPData tools: versioned source ids, saved tables, columns with their
uncertainties, provenance, digests with the full result on disk, and the
refusals. The HTTP layer is mocked; set
HEPTAPOD_NETWORK_TESTS=1 to also query hepdata.net.
"""

import json
import os
from unittest import mock

import pytest
import requests

from tools.digest import DIGEST_BUDGET
from tools.hepdata import HEPDataRecordTool, HEPDataSearchTool, HEPDataTableTool
from tools.provenance import hepdata_source_id, read_calls, sha256_file

SEARCH = {"total": 1, "results": [{
    "inspire_id": "2964565", "version": 1, "title": "Search for heavy neutral leptons",
    "collaborations": ["ATLAS"], "arxiv_id": "arXiv:2508.20929", "year": 2025,
    "journal_info": "Eur.Phys.J.C 86 (2026) 153", "hepdata_doi": "10.17182/hepdata.167367",
    "last_updated": "2025-12-30T16:17:07", "total_tables": 2,
    "data": [{"title": "Figure 4a"}, {"title": "Table 1"}]}]}

RECORD = {
    "version": 2, "version_count": 2, "recid": 167367,
    "record": {"title": "Search for heavy neutral leptons", "collaborations": ["ATLAS"],
               "arxiv_id": "arXiv:2508.20929", "doi": "10.1140/epjc/x",
               "hepdata_doi": "10.17182/hepdata.167367", "abstract": "<p>HNL &amp; W</p>",
               "resources": [{"description": "Webpage", "type": "html", "url": "https://x"}]},
    "data_tables": [
        {"name": "Figure 4a", "location": "Data from Figure 4a",
         "doi": "10.17182/hepdata.167367.v2/t3",
         "description": "Observed 95&percnt; CL limits on |U<sub>e</sub>|<sup>2</sup>"},
    ],
}

TABLE_YAML = b"""independent_variables:
- header: {name: '$m_{N}$', units: GeV}
  values:
  - {value: 8.0}
  - {low: 10.0, high: 15.0}
  - {value: 20.0}
dependent_variables:
- header: {name: Expected}
  qualifiers:
  - {name: SQRT(S), value: 13000 GeV}
  - {name: LUMINOSITY, value: '140 fb$^{-1}$'}
  values:
  - value: 4.7e-05
    errors:
    - {asymerror: {plus: -7.0e-06, minus: 1.0e-05}, label: '$\\pm1\\sigma$'}
    - {symerror: 2.0e-06, label: syst}
  - {value: 2.1e-05}
  - {value: '-'}
"""


class _Resp:
    def __init__(self, url, params, body, ctype="application/json", status=200):
        self.status_code = status
        self.url = requests.Request("GET", url, params=params).prepare().url
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.text = self.content.decode()
        self.headers = {"Content-Type": ctype}

    def json(self):
        return json.loads(self.text)


def _fake_get(self, url, params=None, timeout=None):
    if url.endswith("/search/"):
        return _Resp(url, params, SEARCH)
    if url.endswith("/record/ins2964565"):
        return _Resp(url, params, RECORD)
    if url.endswith("/record/ins1680021"):
        return _Resp(url, params, b"<html>Not found</html>", "text/html", status=404)
    if "/download/table/ins2964565/Figure%204a/" in url:
        return _Resp(url, params, TABLE_YAML, "application/yaml")
    # HEPData answers a missing table with an HTML error page and status 200
    return _Resp(url, params, b"<html>HEPData Error</html>", "text/html; charset=utf-8")


@pytest.fixture
def http():
    with mock.patch.object(requests.Session, "get", _fake_get):
        yield


def _run(cls, base, **kw):
    return json.loads(cls(base_directory=str(base), **kw)._run())


def _full(r, base):
    """The full result a digest points to, checked to sit at research/results/<call_id>.json."""
    assert r["results_path"] == os.path.join("research", "results",
                                             r["provenance"]["call_id"] + ".json")
    return json.loads((base / r["results_path"]).read_text())


def _digest_size(r):
    return len(json.dumps({k: v for k, v in r.items() if k != "provenance"}, indent=2))


def test_source_ids_need_a_version():
    assert hepdata_source_id("2964565", 1) == "HEPData:ins2964565/v1"
    assert hepdata_source_id("2964565", 1, "Figure 4a") == "HEPData:ins2964565/v1/Figure 4a"
    assert hepdata_source_id("2964565", None) is None


def test_search(http, tmp_path):
    r = _run(HEPDataSearchTool, tmp_path, query="heavy neutral lepton")
    assert r["status"] == "ok" and r["count"] == 1 and r["total"] == 1
    hit = r["results"][0]
    assert hit == {"inspire_id": "2964565", "version": 1, "source_id": "HEPData:ins2964565/v1",
                   "title": "Search for heavy neutral leptons", "collaborations": ["ATLAS"],
                   "arxiv_id": "arXiv:2508.20929", "year": 2025, "total_tables": 2}
    assert "omitted" not in r and "results_path" in r["note"]
    assert r["provenance"]["source_ids"] == ["HEPData:ins2964565/v1"]
    assert "q=heavy+neutral+lepton" in r["provenance"]["request_url"]
    full = _full(r, tmp_path)
    assert full["results"][0]["tables"] == ["Figure 4a", "Table 1"]
    assert full["results"][0]["hepdata_doi"] == "10.17182/hepdata.167367"


def test_record(http, tmp_path):
    r = _run(HEPDataRecordTool, tmp_path, inspire_id="ins2964565")
    assert r["status"] == "ok" and r["version"] == 2
    assert r["table_count"] == 1 and r["resource_count"] == 1
    assert r["tables"] == [{"name": "Figure 4a", "location": "Data from Figure 4a",
                            "description": "Observed 95% CL limits on |Ue|2"}]
    assert "abstract" not in r and "resources" not in r
    full = _full(r, tmp_path)
    assert full["abstract"] == "HNL & W"
    t = full["tables"][0]
    assert t["description"] == "Observed 95% CL limits on |Ue|2"
    assert t["source_id"] == "HEPData:ins2964565/v2/Figure 4a"
    assert full["resources"][0]["url"] == "https://x"


def test_record_without_hepdata(http, tmp_path):
    r = _run(HEPDataRecordTool, tmp_path, inspire_id="1680021")
    assert r["status"] == "error" and r["error"] == "Not Found"
    assert "has no HEPData record" in r["reason"] and "only some papers" in r["reason"]
    assert "HEPDataSearchTool" in r["reason"]
    assert "https://www.hepdata.net/record/ins1680021?format=json" in r["reason"]
    assert "results_path" not in r
    assert not (tmp_path / "research" / "results").exists()


def test_table_saved_and_parsed(http, tmp_path):
    r = _run(HEPDataTableTool, tmp_path, inspire_id="2964565", table="Figure 4a")
    assert r["status"] == "ok"
    assert r["version"] == 2 and r["source_id"] == "HEPData:ins2964565/v2/Figure 4a"
    path = tmp_path / r["file"]
    assert r["file"] == os.path.join("hepdata", "ins2964565", "v2", "Figure_4a.yaml")
    assert path.read_bytes() == TABLE_YAML and r["sha256"] == sha256_file(str(path))
    assert r["points"] == 3 and r["truncated"] is False
    assert r["independent_variables"] == [{"name": "$m_{N}$", "units": "GeV"}]
    assert r["dependent_variables"][0]["qualifiers"]["SQRT(S)"] == "13000 GeV"
    assert r["rows"] == [[8.0, 4.7e-05], [[10.0, 15.0], 2.1e-05], [20.0, "-"]]
    assert "omitted" not in r

    full = _full(r, tmp_path)
    x = full["independent_variables"][0]
    assert x["units"] == "GeV" and x["values"] == [8.0, [10.0, 15.0], 20.0]
    y = full["dependent_variables"][0]
    assert y["qualifiers"]["SQRT(S)"] == "13000 GeV"
    first = y["values"][0]
    assert first["errors"][0] == {"label": "$\\pm1\\sigma$", "plus": -7.0e-06, "minus": 1.0e-05}
    assert first["errors"][1] == {"label": "syst", "symerror": 2.0e-06}
    assert y["values"][2] == {"value": "-"}

    prov = r["provenance"]
    assert prov["files"][0]["sha256"] == r["sha256"]
    assert prov["response_sha256"] == r["sha256"]
    assert prov["saved_path"].endswith(".yaml")
    assert len(prov["request"]["http_requests"]) == 2      # version lookup, then the table
    assert prov["request_url"].endswith("/download/table/ins2964565/Figure%204a/2/yaml")
    calls = read_calls(str(tmp_path))
    assert calls[-1]["source_ids"] == ["HEPData:ins2964565/v2/Figure 4a"]


def test_table_pinned_version_skips_lookup(http, tmp_path):
    r = _run(HEPDataTableTool, tmp_path, inspire_id="2964565", table="Figure 4a", version=1,
             max_points=1)
    assert r["version"] == 1 and r["file"].endswith(os.path.join("v1", "Figure_4a.yaml"))
    assert r["provenance"]["request"]["http_requests"] == [r["provenance"]["request_url"]]
    assert r["truncated"] is True and r["rows"] == [[8.0, 4.7e-05]]
    assert _full(r, tmp_path)["independent_variables"][0]["values"] == [8.0]


def test_missing_table_is_an_error_not_html(http, tmp_path):
    r = _run(HEPDataTableTool, tmp_path, inspire_id="2964565", table="Figure 99", version=1)
    assert r["status"] == "error" and r["error"] == "Not Found"
    assert not (tmp_path / "hepdata").exists()
    assert read_calls(str(tmp_path))[-1]["source_ids"] == []


def _big_search(n_hits=50, n_tables=200):
    title = "Search for heavy neutral leptons decaying to a lepton and two jets " * 4
    return {"total": 400, "results": [{
        "inspire_id": str(2000000 + i), "version": 1, "title": f"{i}: {title}",
        "collaborations": ["CMS"], "arxiv_id": f"arXiv:2501.{i:05d}", "year": 2025,
        "journal_info": "JHEP 01 (2026) 001", "hepdata_doi": f"10.17182/hepdata.{i}",
        "last_updated": "2025-12-30T16:17:07", "total_tables": n_tables,
        "data": [{"title": f"Figure {j}: observed and expected limits on |V_eN|^2"}
                 for j in range(n_tables)]} for i in range(n_hits)]}


def test_search_digest_stays_within_budget(tmp_path):
    big = _big_search()
    with mock.patch.object(requests.Session, "get",
                           lambda self, url, params=None, timeout=None: _Resp(url, params, big)):
        r = _run(HEPDataSearchTool, tmp_path, query="heavy neutrino", max_results=50)
    assert r["status"] == "ok" and r["count"] == 50 and r["total"] == 400
    assert _digest_size(r) <= DIGEST_BUDGET
    shown = r["results"]
    assert 0 < len(shown) < 50 and r["omitted"] == 50 - len(shown)
    assert [h["inspire_id"] for h in shown] == [str(2000000 + i) for i in range(len(shown))]
    assert all(len(h["title"]) <= 150 and h["title"].endswith("\u2026") for h in shown)
    assert all("tables" not in h for h in shown)
    assert f"first {len(shown)} of 50" in r["note"]
    full = _full(r, tmp_path)
    assert len(full["results"]) == 50 and len(full["results"][49]["tables"]) == 200
    assert full["results"][0]["title"].startswith("0: Search")
    assert len(r["provenance"]["source_ids"]) == 50


def test_record_digest_with_many_tables(tmp_path):
    rec = dict(RECORD, data_tables=[
        {"name": f"Table {i}", "location": f"Data from Figure {i}", "doi": f"10.17182/x/t{i}",
         "description": "Observed and expected 95% CL upper limits on the mixing " * 10}
        for i in range(200)])
    with mock.patch.object(requests.Session, "get",
                           lambda self, url, params=None, timeout=None: _Resp(url, params, rec)):
        r = _run(HEPDataRecordTool, tmp_path, inspire_id="2964565")
    assert r["status"] == "ok" and r["table_count"] == 200
    assert _digest_size(r) <= DIGEST_BUDGET
    shown = r["tables"]
    assert [t["name"] for t in shown] == [f"Table {i}" for i in range(len(shown))]
    assert r["omitted"] == 200 - len(shown) > 0
    assert all(len(t["description"]) <= 120 for t in shown)
    assert len(_full(r, tmp_path)["tables"]) == 200


def test_table_digest_shows_first_rows(tmp_path):
    rows = "\n".join(f"  - {{value: {i}.0}}" for i in range(300))
    ys = "\n".join(f"  - {{value: {i}.5}}" for i in range(300))
    body = (f"independent_variables:\n- header: {{name: m, units: GeV}}\n  values:\n{rows}\n"
            f"dependent_variables:\n- header: {{name: Observed}}\n  values:\n{ys}\n").encode()

    def get(self, url, params=None, timeout=None):
        return _Resp(url, params, body, "application/yaml")
    with mock.patch.object(requests.Session, "get", get):
        r = _run(HEPDataTableTool, tmp_path, inspire_id="2964565", table="T", version=1)
    assert r["status"] == "ok" and r["points"] == 300 and r["truncated"] is True
    assert r["rows"][:2] == [[0.0, 0.5], [1.0, 1.5]] and len(r["rows"]) == 5
    assert r["omitted"] == 195                     # 200 points kept in the file (max_points)
    assert _digest_size(r) <= DIGEST_BUDGET
    full = _full(r, tmp_path)
    assert len(full["dependent_variables"][0]["values"]) == 200


def test_unwritable_results_still_return(http, tmp_path):
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "results").write_text("not a directory")
    r = _run(HEPDataSearchTool, tmp_path, query="heavy neutral lepton")
    assert r["status"] == "ok" and r["results_path"] is None
    assert r["results"][0]["inspire_id"] == "2964565"
    assert "could not be written" in r["note"]


def test_refusals(http, tmp_path):
    r = _run(HEPDataTableTool, tmp_path, inspire_id="abc", table="Figure 4a")
    assert r["status"] == "error" and r["error"] == "Invalid Parameter"
    r = _run(HEPDataTableTool, tmp_path, inspire_id="2964565", table="Figure 4a",
             output_dir="../outside")
    assert r["status"] == "error" and r["error"] == "Access Denied"
    assert r["provenance"]["request"]["http_requests"] == []
    r = _run(HEPDataSearchTool, tmp_path, query="  ")
    assert r["status"] == "error" and r["error"] == "Missing Parameter"


@pytest.mark.skipif(not os.environ.get("HEPTAPOD_NETWORK_TESTS"), reason="network")
def test_live_table(tmp_path):
    r = _run(HEPDataTableTool, tmp_path, inspire_id="2964565", table="Figure 4a", version=1)
    assert r["status"] == "ok", r
    assert r["dependent_variables"][0]["name"] == "Observed"
    assert r["independent_variables"][0]["units"] == "GeV"
