"""Portable regression checks; optional full SM/MG integration via environment."""

import copy
import json
import os
import shutil
from pathlib import Path
import pytest
from tools.feyngraph.feyngraph_interface import FeynGraphInterface
from tools.feyngraph.catalogue import ProcessModel, enumerate_processes, dump
from tools.feyngraph.catalogue_tools import (
    InspectProcessModelTool,
    EnumerateProcessesTool,
    ValidateProcessCatalogueTool,
)


def toy_model(path):
    path.mkdir()
    (path / "particles.py").write_text("""
from object_library import Particle
import parameters as Param
s = Particle(name='s',antiname='s',pdg_code=900001,spin=1,color=1,mass=Param.MS,width=Param.ZERO,texname='s',antitexname='s',charge=0)
x = Particle(name='x',antiname='x',pdg_code=900002,spin=1,color=1,mass=Param.MX,width=Param.ZERO,texname='x',antitexname='x',charge=0)
""")
    (path / "parameters.py").write_text("""
from object_library import Parameter
ZERO=Parameter(name='ZERO',nature='internal',type='real',value='0.0',texname='0')
MS=Parameter(name='MS',nature='internal',type='real',value='10.0',texname='MS')
MX=Parameter(name='MX',nature='internal',type='real',value='1.0',texname='MX')
""")
    (path / "couplings.py").write_text(
        "from object_library import Coupling\nGC_1=Coupling(name='GC_1',value='1.0',order={'NP':1})\n"
    )
    (path / "lorentz.py").write_text(
        "from object_library import Lorentz\nSSS1=Lorentz(name='SSS1',spins=[1,1,1],structure='1')\n"
    )
    (path / "vertices.py").write_text(
        "from object_library import Vertex\nimport particles as P\nimport couplings as C\nimport lorentz as L\nV_1=Vertex(name='V_1',particles=[P.s,P.x,P.x],color=['1'],lorentz=[L.SSS1],couplings={(0,0):C.GC_1})\n"
    )
    (path / "coupling_orders.py").write_text(
        "from object_library import CouplingOrder\nNP=CouplingOrder(name='NP',expansion_order=99,hierarchy=1)\n"
    )
    return path


def test_builtin_and_alias():
    model = FeynGraphInterface()
    assert model.model_info()["source"] == "builtin"
    assert len(model.enumerate_diagrams(["Z"], ["tau-", "tau+"])) == 1


def test_explicit_ufo_binding_and_portable_ids(tmp_path):
    source = toy_model(tmp_path / "toy")
    other = tmp_path / "copy"
    shutil.copytree(source, other)
    model = ProcessModel(str(source))
    assert model.info["model_sha256"] == ProcessModel(str(other)).info["model_sha256"]
    out = tmp_path / "out"
    out.mkdir()
    data = enumerate_processes(model, out)
    assert len(data["processes"]) == 1
    row = data["processes"][0]
    assert row["initial"] == ["s"] and row["final"] == ["x", "x"]
    assert row["classes"] == [{"id": "c1", "orders": {"NP": 1}, "coupling_monomials": [{"GC_1": 1}],
                               "diagram_ids": [row["diagrams"][0]["id"]]}]
    assert row["diagrams"][0]["graph"]["edges"][0] == {"from": "in:0", "to": "v0", "pdg": 900001, "role": "external"}
    assert (out / row["diagrams"][0]["image"]).stat().st_size > 0
    assert (out / row["diagrams"][0]["tikz"]).stat().st_size > 0
    assert sum(data["coverage"]["outcomes"].values()) == 6
    with pytest.raises(ValueError, match="budget"):
        enumerate_processes(model, out, max_candidates=1)


def test_three_body_chain_with_identical_daughters(tmp_path):
    # s -> x y*, y* -> x x: three diagrams (which x attaches to s), one class.
    source = toy_model(tmp_path / "toy")
    (source / "particles.py").write_text((source / "particles.py").read_text() + """
y = Particle(name='y',antiname='y',pdg_code=900003,spin=1,color=1,mass=Param.MY,width=Param.ZERO,texname='y',antitexname='y',charge=0)
""")
    (source / "parameters.py").write_text((source / "parameters.py").read_text() + "MY=Parameter(name='MY',nature='internal',type='real',value='3.0',texname='MY')\n")
    (source / "vertices.py").write_text((source / "vertices.py").read_text().replace("P.s,P.x,P.x", "P.s,P.x,P.y")
        + "V_2=Vertex(name='V_2',particles=[P.y,P.x,P.x],color=['1'],lorentz=[L.SSS1],couplings={(0,0):C.GC_1})\n")
    out = tmp_path / "out"
    out.mkdir()
    data = enumerate_processes(ProcessModel(str(source)), out, parents=["s"], final_multiplicity=3)
    (row,) = [r for r in data["processes"] if r["final"] == ["x", "x", "x"]]
    assert len(row["diagrams"]) == 3 and len(row["classes"]) == 1
    assert row["classes"][0]["coupling_monomials"] == [{"GC_1": 2}]
    assert data["coverage"]["complete_within_scope"]
    with pytest.raises(ValueError, match="final_multiplicity"):
        enumerate_processes(ProcessModel(str(source)), out, final_multiplicity=1)
    # The model's own truncation: one NP insertion removes the two-vertex chain.
    (source / "coupling_orders.py").write_text((source / "coupling_orders.py").read_text().replace("expansion_order=99", "expansion_order=1"))
    out2 = tmp_path / "out2"
    out2.mkdir()
    data = enumerate_processes(ProcessModel(str(source)), out2, parents=["s"], final_multiplicity=3)
    assert data["scope"]["expansion_orders"] == {"NP": 1} and not data["processes"]


def test_no_silent_fallback(tmp_path):
    interface = FeynGraphInterface(str(tmp_path / "missing"))
    for _ in range(2):
        with pytest.raises(ValueError):
            interface.enumerate_diagrams(["Z"], ["e-", "e+"])


def test_agent_tools_and_receipts(tmp_path):
    tool = InspectProcessModelTool(base_directory=str(tmp_path), output_dir="inspect")
    assert json.loads(tool._run())["status"] == "ok"
    assert (tmp_path / "inspect/manifest.json").exists()
    assert json.loads(tool._run())["status"] == "error"  # No overwrite
    escape = InspectProcessModelTool(
        base_directory=str(tmp_path), output_dir="../escape"
    )
    assert json.loads(escape._run())["status"] == "error"
    toy_model(tmp_path / "toy")
    enum = EnumerateProcessesTool(
        base_directory=str(tmp_path), output_dir="enum", model="toy"
    )
    assert json.loads(enum._run())["processes"] == 1
    bad = EnumerateProcessesTool(
        base_directory=str(tmp_path), output_dir="bad", model="toy", max_candidates=1
    )
    assert json.loads(bad._run())["status"] == "error"
    assert (tmp_path / "bad/response.json").exists()


@pytest.mark.skipif(
    not os.environ.get("CATALOGUE_SM_UFO"),
    reason="Set CATALOGUE_SM_UFO, CATALOGUE_SM_CARD and CATALOGUE_MG5 for integration",
)
def test_sm_independent_validation(tmp_path):
    from tools.feyngraph.catalogue_validation import validate_catalogue

    model = ProcessModel(
        os.environ["CATALOGUE_SM_UFO"], param_card=os.environ["CATALOGUE_SM_CARD"]
    )
    mg = os.environ["CATALOGUE_MG5"]
    # Frozen unrestricted SM benchmark parameter point.
    for n, parents, count in [(3, None, 237), (4, ["H"], 190), (2, None, 40)]:
        out = tmp_path / str(n)
        out.mkdir()
        data = enumerate_processes(model, out, parents=parents, final_multiplicity=n,
                                   max_candidates=10**6, render=n == 2)
        assert len(data["processes"]) == count
        assert validate_catalogue(model, out / "catalogue.json", mg)["status"] == "passed"
    # BSM: MSSM Majorana neutralino and gluino decays, PDG-addressed in MG5.
    mssm = Path(mg) / "models/MSSM_SLHA2"
    bsm = ProcessModel(str(mssm), param_card=str(mssm / "restrict_default.dat"))
    for n in [2, 3]:
        out_bsm = tmp_path / ("mssm" + str(n))
        out_bsm.mkdir()
        enumerate_processes(bsm, out_bsm, parents=["n2", "go"], final_multiplicity=n, max_candidates=10**7, render=False)
        assert validate_catalogue(bsm, out_bsm / "catalogue.json", mg)["status"] == "passed"
    tmp_path = out
    # Keep graph artifacts fixed: corrupt a signed vertex term in the submitted metadata.
    broken = copy.deepcopy(data)
    broken["processes"][0]["diagrams"][0]["graph"]["vertices"][0]["terms"][0]["coupling"] = "GC_0"
    dump(tmp_path / "wrong.json", broken)
    assert validate_catalogue(model, tmp_path / "wrong.json", mg)["status"] == "failed"
    broken = copy.deepcopy(data)
    broken["processes"].pop()
    dump(tmp_path / "missing.json", broken)
    assert any(
        f["code"] == "missing_process"
        for f in validate_catalogue(model, tmp_path / "missing.json", mg)["findings"]
    )


def test_timeout_receipt(tmp_path, monkeypatch):
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    tool = InspectProcessModelTool(base_directory=str(tmp_path), output_dir="timed")
    result = json.loads(tool._run())
    assert result["status"] == "error" and "timeout" in result["error"]
    assert (tmp_path / "timed/manifest.json").exists()


def test_tools_run_through_baseTool_execute(tmp_path):
    """The harness calls tool.execute(**runtime_args), not _run().

    CatalogueTool must keep BaseTool.execute's signature: an override that
    adds a positional argument fails with "missing 1 required positional
    argument" when the harness calls it.
    """
    inspect = InspectProcessModelTool(base_directory=str(tmp_path))
    out = inspect.execute(output_dir="inspect")
    assert "missing 1 required positional argument" not in out
    assert json.loads(out)["status"] == "ok"

    toy_model(tmp_path / "toy")
    enum = EnumerateProcessesTool(base_directory=str(tmp_path))
    out = enum.execute(model="toy", output_dir="enum")
    assert json.loads(out)["processes"] == 1

    val = ValidateProcessCatalogueTool(base_directory=str(tmp_path), mg5_path="/nonexistent")
    out = val.execute(catalogue_path="enum/catalogue.json", expected_scope_json="{}",
                      model="toy", output_dir="val")
    assert "missing 1 required positional argument" not in out
    assert (tmp_path / "val" / "request.json").exists()
