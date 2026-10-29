"""Unit tests for ValidateProcess helpers (toolgrad loops 15/17).

The full _run needs a MadGraph install, so here we unit-test the pure logic:
the colour-flow consistency check (a coloured parent decaying to an all-colourless
final state — compiles but won't shower) and the command-card builder.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.mg5.validate_process import _colour_inconsistent_decays as cic


def test_flags_gluino_to_colourless():
    # gluino (coloured) -> W + chargino (both colour singlets): impossible flow
    bad = cic("generate p p > go go, (go > w+ x1-), (go > w- x1+)")
    assert len(bad) == 2


def test_ok_gluino_with_quarks():
    # gluino -> q q~ + neutralino has coloured products: fine
    assert cic("generate p p > go go, (go > q q~ n1)") == []
    assert cic("generate p p > go go, (go > j j n1)") == []


def test_ok_colourless_parent():
    # chargino -> W + neutralino: colourless parent, not flagged
    assert cic("generate p p > x1+ n2, (x1+ > w+ n1), (n2 > z n1)") == []


def test_ok_squark_to_b():
    # stop -> b + chargino: b is coloured, fine
    assert cic("generate p p > t1 t1~, (t1 > b x1+)") == []


def test_ok_neutralino_to_photon_gravitino():
    # n1 -> photon + gravitino: colourless parent and products, fine
    assert cic("(n1 > a G)") == []


def test_q_and_j_recognised_as_coloured():
    # 'q' (generic quark) and 'j' (jet) are coloured multiparticles
    assert cic("(go > q q~ n1)") == []
    assert cic("(go > j j n1)") == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"[ok] {name}")
    print("all validate_process tests passed")


# --- Sandbox-relative model paths ------------------------------------------ #
# MG5 runs in a temporary directory, so 'import model inputs/model' fails
# with "Path inputs/model is not a valid pathname"; the tool makes the path
# absolute first.

def _vp(base, model):
    from tools.mg5.validate_process import ValidateProcessTool
    return ValidateProcessTool(mg5_path="/nonexistent", base_directory=str(base),
                               process="generate mu- > e- ve~ vm", model=model)


def test_relative_model_path_made_absolute(tmp_path):
    (tmp_path / "inputs" / "model").mkdir(parents=True)
    card = _vp(tmp_path, "inputs/model")._build_card("/x/PROC")
    assert card.splitlines()[0] == f"import model {tmp_path / 'inputs' / 'model'}"


def test_relative_model_path_keeps_restriction(tmp_path):
    (tmp_path / "work" / "sm_ufo").mkdir(parents=True)
    card = _vp(tmp_path, "work/sm_ufo-no_b_mass")._build_card("/x/PROC")
    assert card.splitlines()[0] == f"import model {tmp_path / 'work' / 'sm_ufo'}-no_b_mass"


def test_stock_and_absolute_models_unchanged(tmp_path):
    assert _vp(tmp_path, "sm")._build_card("/x/P").splitlines()[0] == "import model sm"
    assert _vp(tmp_path, "sm-no_b_mass")._build_card("/x/P").splitlines()[0] == "import model sm-no_b_mass"
    assert _vp(tmp_path, "/abs/ufo")._build_card("/x/P").splitlines()[0] == "import model /abs/ufo"
