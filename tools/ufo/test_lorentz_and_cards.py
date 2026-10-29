"""
# test_lorentz_and_cards.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Lorentz structures and SLHA cards, read statically.

The Lorentz fixture is the FeynRules-generated leptoquark UFO the toolkit
already ships; the card tests write their own cards, so nothing here needs
MadGraph.
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.ufo import card_overrides, parse_lorentz, read_slha  # noqa: E402

LQ_UFO = REPO_ROOT / "tools" / "mg5" / "test_files" / "models" / "S1_LQ_RR_UFO"


def test_every_structure_is_read_with_its_spins():
    structures = parse_lorentz(str(LQ_UFO))
    source = (LQ_UFO / "lorentz.py").read_text()
    assert len(structures) == len(re.findall(r"^\w+ = Lorentz\(", source, re.M)) == 19
    for s in structures.values():
        assert s["structure"]
        assert s["spins"] and all(x in (-1, 1, 2, 3, 4, 5) for x in s["spins"])


def test_a_missing_directory_reads_as_empty(tmp_path):
    assert parse_lorentz(str(tmp_path)) == {}


CARD = """\
Block SMINPUTS
    1 1.325070e+02 # aEWM1
Block MASS
   23 9.118800e+01 # MZ
Block yukawa   # lower case: SLHA block names are case-insensitive
    6 1.730000e+02
Block QNUMBERS 9000005  # LQ
    1 -1
DECAY  23 2.441404e+00
#  BR  NDA  ID1 ID2
   0.5   2   11  -11
DECAY   6 0.000000e+00
"""


def test_slha_blocks_and_decays(tmp_path):
    card = tmp_path / "param_card.dat"
    card.write_text(CARD)
    values = read_slha(str(card))
    assert values[("MASS", (23,))] == pytest.approx(91.188)
    assert values[("YUKAWA", (6,))] == pytest.approx(173.0)
    assert values[("DECAY", (23,))] == pytest.approx(2.441404)
    assert values[("DECAY", (6,))] == 0.0
    # Branching-ratio lines and QNUMBERS are not parameters.
    assert not any(k[0] == "QNUMBERS" for k in values)
    assert len(values) == 5


def test_a_malformed_value_names_its_line(tmp_path):
    card = tmp_path / "bad.dat"
    card.write_text("Block MASS\n   23 heavy\n")
    with pytest.raises(ValueError, match="bad.dat:2"):
        read_slha(str(card))


def test_overrides_map_by_block_and_code(tmp_path):
    from tools.ufo import parse_parameters
    params = [p for p in parse_parameters(str(LQ_UFO))
              if p.get("nature") == "external" and "lhablock" in p]
    target = params[0]
    card = tmp_path / "card.dat"
    codes = " ".join(str(c) for c in target["lhacode"])
    card.write_text(f"Block {target['lhablock']}\n  {codes} 42.5 # any comment\n")
    overrides, missing = card_overrides(str(LQ_UFO), str(card))
    assert overrides == {target["var"]: 42.5}
    assert target["var"] not in missing
    assert len(missing) == len(params) - 1


def test_an_antiparticle_names_its_parent_as_antiparticle():
    """`X = Y.anti()` records copied the parent's antiname, so e+ named
    itself as its own antiparticle and anything crossing an incoming e+ to
    an outgoing e- got e+ back."""
    from tools.ufo import parse_particles
    by_name = {p["name"]: p for p in parse_particles(str(LQ_UFO / "particles.py"))}
    for name, rec in by_name.items():
        anti = rec["antiname"]
        assert anti in by_name, name
        assert by_name[anti]["antiname"] == name
    assert by_name["e+"]["antiname"] == "e-"
    assert by_name["e+"]["texname"] == "e+"


def test_coupling_orders_are_read_per_coupling():
    from tools.ufo import parse_coupling_orders, parse_couplings
    orders = parse_coupling_orders(str(LQ_UFO))
    assert set(orders) == set(parse_couplings(str(LQ_UFO)))
    assert all(isinstance(v, int) for o in orders.values() for v in o.values())
    assert any(o for o in orders.values())


def test_width_expressions_may_use_trigonometry_but_nothing_else():
    """MadGraph's MSSM writes Higgs widths with cmath.cos/sin of mixing
    angles; 47 of its partial widths were unreadable before these were
    allowed. Anything outside the whitelist is still refused."""
    import cmath
    from tools.ufo.width_gate import safe_eval_width
    assert safe_eval_width("cmath.cos(a)**2 + cmath.sin(a)**2", {"a": 0.3}) == pytest.approx(1)
    assert safe_eval_width("re(complexconjugate(z)) + sec(0)", {"z": 2 + 1j}) == pytest.approx(3)
    for bad in ("cmath.__dict__", "open('x')", "a.__class__", "cmath.phase(1)"):
        with pytest.raises(ValueError):
            safe_eval_width(bad, {"a": 1.0})
