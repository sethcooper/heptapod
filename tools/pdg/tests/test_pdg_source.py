"""Every PDG value carries its identifier, edition, printed form and link."""
import pytest

pdg = pytest.importorskip("pdg")
from tools.pdg.pdg_interface import get_interface


def test_width_and_lifetime_carry_their_source():
    w = get_interface().get_width("Z boson")
    assert w.pdgid == "S044W" and w.edition and w.url.endswith("node=S044W")
    assert "±" in w.display and w.display.endswith("GeV")
    t = get_interface().get_lifetime("tau")
    assert t.pdgid == "S035T" and "10^-15 s" in t.display


def test_asymmetric_errors_are_kept():
    h = get_interface().get_width("Higgs")
    assert h.error_plus and h.error_minus and h.error_plus != h.error_minus
