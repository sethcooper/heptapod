"""PDG tool results carry provenance with PDG:<pdgid>/<edition> source ids
(the pdg package reads a local SQLite file; no network)."""
import json

import pytest

pdg = pytest.importorskip("pdg")
from tools.pdg.pdg_tool import PDGDatabaseTool, PDGPropertyTool, PDGSearchTool  # noqa: E402
from tools.provenance import read_calls  # noqa: E402


def _run(tool):
    tool._setup()
    return json.loads(tool._run())


def test_width_has_versioned_source_id(tmp_path):
    r = _run(PDGDatabaseTool(particle="Z boson", property="width", base_directory=str(tmp_path)))
    p = r["provenance"]
    ed = r["width"]["edition"]
    assert r["width"]["pdgid"] == "S044W"
    assert p["source_ids"] == [f"PDG:S044W/{ed}"]
    assert p["request"] == {"particle": "Z boson", "property": "width", "branching_limit": 20}
    assert p["request_url"] is None and p["from_cache"] is False
    assert ed in p["api"]["version"]
    (line,) = read_calls(str(tmp_path))
    assert line["call_id"] == p["call_id"] and line["source_ids"] == p["source_ids"]


def test_property_and_search(tmp_path):
    r = _run(PDGPropertyTool(pdgid="S044W", base_directory=str(tmp_path)))
    assert r["provenance"]["source_ids"] == [f"PDG:S044W/{r['property']['edition']}"]
    s = _run(PDGSearchTool(query="Z", base_directory=str(tmp_path)))
    assert any(i.startswith("PDG:S044/") for i in s["provenance"]["source_ids"])


def test_failed_lookup_cites_nothing(tmp_path):
    r = _run(PDGPropertyTool(pdgid="NOPE123", base_directory=str(tmp_path)))
    assert r["status"] == "error" and r["provenance"]["source_ids"] == []
