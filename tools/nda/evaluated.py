"""
# evaluated.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

The evaluated NDA expression: the kinematics-improved estimate of one
diagram written factor by factor, so that substituting the definitions
reproduces width_gev exactly.

    Gamma = C B c_ps^(n-2) / ((2 s_P + 1) S) * P_P / (2 m_P) * Phi_n
            * prod_v |g_v|^2 E_v^2 * prod_i P_i(E_i) * prod_j N_j(q_j^2) |D_j|^2

- C: colour factor; B: 2 for a two-body decay into two fermions through an
  unprojected vertex, else 1; c_ps = 0.8; S: identical-particle factor;
- P: external polarization sums, 1 (scalar, massless vector), 2E (fermion),
  1 + E^2/m^2 (massive vector), or 2 for a massive final vector whose emitting
  vertex has no fermion or scalar at least as heavy as it; the parent at rest has E = m_P;
- E_i = (m_P - sum_j m_j)/n + m_i;
- Phi_n: massless n-body volume m_P^(2n-4) / (2^(4n-5) pi^(2n-3) (n-1)! (n-2)!),
  or the massive volume (recursive two-body integral) with final-state masses;
- |g_v|: the vertex coupling at the card, sqrt(g_L^2 + g_R^2) for a chiral
  fermion vertex; E_v^2 only at a vertex whose coupling multiplies a momentum
  (three vectors, or two scalars and a vector), E_v the largest energy of its legs;
- line j confined to q_-^2 <= q^2 <= q_+^2, q_-^2 = (sum of its daughters'
  masses)^2, q_+^2 = (m_P - sum of the other final masses)^2, numerator N at
  q_j^2 = (q_-^2 + q_+^2)/2: 1 (scalar, massless vector), q^2 (fermion),
  (1 - q^2/m^2)^2 (massive vector); |D|^2 = 1/((q^2 - m^2)^2 + m^2 Gamma^2)
  at q_j^2, or its average over the interval when the line has a width and
  its pole lies above q_-^2.
"""
import math
import re
from math import factorial
from collections import Counter
from fractions import Fraction
from typing import Any, Dict, List, Optional

C_PS = 0.8

TEX_NAMES = {"a": r"\gamma", "g": "g", "H": "h", "h": "h", "Z": "Z", "W+": "W^+", "W-": "W^-",
             "e-": "e^-", "e+": "e^+", "mu-": r"\mu^-", "mu+": r"\mu^+", "ta-": r"\tau^-", "ta+": r"\tau^+",
             "ve": r"\nu_e", "vm": r"\nu_\mu", "vt": r"\nu_\tau",
             "ve~": r"\bar\nu_e", "vm~": r"\bar\nu_\mu", "vt~": r"\bar\nu_\tau"}
# Coupling and mass subscripts drop charges and bars: g_{hWW}, m_W, m_\mu.
BARE_NAMES = {"a": r"\gamma", "H": "h", "W+": "W", "W-": "W", "e-": "e", "e+": "e", "mu-": r"\mu", "mu+": r"\mu",
              "ta-": r"\tau", "ta+": r"\tau", "ve": r"\nu_e", "vm": r"\nu_\mu", "vt": r"\nu_\tau",
              "ve~": r"\nu_e", "vm~": r"\nu_\mu", "vt~": r"\nu_\tau"}


def subscript_index(name: str) -> str:
    """A model particle's trailing number as a subscript: N1 -> N_{1}."""
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    return "%s_{%s}" % m.groups() if m else name


def tex_name(name: str) -> str:
    if name in TEX_NAMES:
        return TEX_NAMES[name]
    return r"\bar{" + subscript_index(name[:-1]) + "}" if name.endswith("~") else subscript_index(name)


def bare_name(name: str) -> str:
    return BARE_NAMES.get(name, subscript_index(name.rstrip("~")))


def number(x: float, digits: int = 4) -> str:
    """A value in TeX: integers and simple fractions exactly, else 4 significant figures."""
    f = Fraction(x).limit_denominator(64)
    if abs(float(f) - x) <= 1e-12 * abs(x) and (f != 0 or x == 0):
        return str(f.numerator) if f.denominator == 1 else r"\tfrac{%d}{%d}" % (f.numerator, f.denominator)
    mantissa, exponent = ("%.*e" % (digits - 1, x)).split("e")
    exponent = int(exponent)
    if -3 <= exponent <= 4:
        return "%.*f" % (max(0, digits - 1 - exponent), x)
    return r"%s\times10^{%d}" % (mantissa, exponent)


def power(base: str, k: int) -> str:
    if k == 1:
        return base
    simple = re.fullmatch(r"[A-Za-z]+(_\{[^{}]*(\{[^{}]*\}[^{}]*)*\})?|\\[A-Za-z]+", base)
    wrapped = base if simple else r"\left(%s\right)" % base
    return "%s^{%d}" % (wrapped, k)


def massless_phase_space(M: float, n: int) -> float:
    return M ** (2 * n - 4) / (2 ** (4 * n - 5) * math.pi ** (2 * n - 3)
                               * math.factorial(n - 1) * math.factorial(n - 2))


def massless_phase_space_tex(mass: str, n: int) -> str:
    den = 2 ** (4 * n - 5) * math.factorial(n - 1) * math.factorial(n - 2)
    pi = r"\pi" if n == 2 else r"\pi^{%d}" % (2 * n - 3)
    num = "1" if n == 2 else (mass + "^2" if n == 3 else "%s^{%d}" % (mass, 2 * n - 4))
    return r"\frac{%s}{%d%s}" % (num, den, pi)


def polarization(spin: float, energy: float, mass: float) -> float:
    if spin == 0.5:
        return 2.0 * energy
    if spin == 1:
        return 1.0 + energy ** 2 / mass ** 2 if mass > 0 else 1.0
    if spin == 0:
        return 1.0
    raise ValueError(f"spin {spin}")


def numerator(spin: float, q2: float, mass: float) -> float:
    if spin == 0.5:
        return q2
    if spin == 1:
        return (1.0 - q2 / mass ** 2) ** 2 if mass > 0 else 1.0
    if spin == 0:
        return 1.0
    raise ValueError(f"spin {spin}")


def evaluated_formula(details: Dict[str, Any], diagram=None, *, parent: Optional[str] = None,
                      final: Optional[List[str]] = None, lines: Optional[List[str]] = None,
                      couplings: Optional[List[str]] = None,
                      coupling_values: Optional[List[float]] = None) -> Optional[Dict[str, Any]]:
    """The evaluated expression for an improved estimate's details. Labels
    (UFO-style names: "W+", "b~", "mu-") and coupling TeX names come from
    ``diagram`` (a parsed Diagram) unless given. None when the estimate is
    not the propagator-kinematics form (a line without a virtuality interval,
    a spin outside 0, 1/2, 1) or the factors do not reproduce it."""
    try:
        return _evaluated(details, diagram, parent, final, lines, couplings, coupling_values)
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None


def _evaluated(details, diagram, parent, final, lines, couplings, coupling_values):
    n = details["n_body"]
    M = details["mother_mass"]
    masses = details["final_masses_gev"]
    spins = details["final_spins"]
    energies = details["final_energies_gev"]
    props = details["propagator_details"]
    if any(p.get("virtuality") is None for p in props):
        return None
    if diagram is not None:
        parent = parent or diagram.initial[0].label
        final = final or [p.label for p in diagram.final]
        lines = lines or [p.label for p in diagram.propagators]
        if couplings is None:
            couplings = [v.coupling if isinstance(v.coupling, str) else "g_{%d}" % (k + 1)
                         for k, v in enumerate(diagram.vertices)]
    mP = "m_{%s}" % bare_name(parent or "P")
    s_parent = details["mother_spin"]

    # Pure numbers and the parent: C B c^(n-2) P_P / ((2s+1) S 2 m_P).
    mother_pol = polarization(s_parent, M, M) if s_parent in (0, 0.5, 1) else None
    if mother_pol is None:
        return None
    k = details["color_factor"] * details["backtoback_correction"] / ((2 * s_parent + 1) * details["symmetry_factor"])
    # The parent's polarization over 2 m_P: 1/(2 m_P) (scalar), 1 (fermion), 1/m_P (vector).
    lead_value = {0: k / 2, 0.5: k, 1: k}[s_parent]
    mass_power = 0 if s_parent == 0.5 else -1
    value = k * mother_pol / (2 * M)
    parts: List[str] = []
    definitions: List[str] = []
    if n > 2:
        parts.append(r"%s^{%d}" % (C_PS, n - 2) if n > 3 else str(C_PS))
        value *= C_PS ** (n - 2)

    # Phase space.
    phi = details["phase_space_base"]
    massless = massless_phase_space(M, n)
    if all(m == 0 for m in masses):
        phi_tex = massless_phase_space_tex(mP, n)
    else:
        phi_tex = r"\Phi_{%d}" % n
        definitions.append(r"\Phi_{%d}=%s\,%s" % (n, number(phi), "" if n == 2 else (
            r"\mathrm{GeV}^{2}" if n == 3 else r"\mathrm{GeV}^{%d}" % (2 * n - 4))))
    value *= phi if phi_tex.startswith(r"\Phi") else massless

    # Couplings.
    g2 = details["coupling_sq"]
    if couplings:
        counts = Counter(couplings)
        parts.append("".join(power(c, 2 * m) for c, m in counts.items()))
        if coupling_values:
            seen = {}
            for c, v in zip(couplings, coupling_values):
                seen.setdefault(c, v)
            definitions.append(r",\ ".join("|%s|=%s" % (c, number(v)) for c, v in seen.items()))
    value *= g2

    # External polarizations. E_i = Ebar + m_i, Ebar = (m_P - sum m)/n.
    ebar = (M - sum(masses)) / n
    pols = Counter()
    energy_defs: Dict[str, str] = {}
    unsourced = set(details.get("unsourced_vectors") or [])
    for i, (label, spin, mass, energy) in enumerate(zip(final, spins, masses, energies)):
        if spin == 0 or (spin == 1 and mass == 0):
            continue
        if i in unsourced:
            pols["2"] += 1
            value *= 2.0
            continue
        if all(m == 0 for m in masses):
            tex = r"\frac{2%s}{%d}" % (mP, n)
        elif mass == 0:
            tex = r"2\bar E"
        else:
            E = "E_{%s}" % tex_name(label)
            energy_defs.setdefault(E, r"%s=\bar E+m_{%s}=%s\,\mathrm{GeV}" % (E, bare_name(label), number(energy)))
            tex = "2" + E if spin == 0.5 else r"1+\frac{%s^2}{m_{%s}^2}" % (E, bare_name(label))
        pols[tex] += 1
        value *= polarization(spin, energy, mass)
    for tex, m in pols.items():
        parts.append(power(tex, m) if m > 1 else (r"\left(%s\right)" % tex if tex.startswith("1+") else tex))
    if any(m > 0 for m in masses) and pols:
        definitions.append(r"\bar E=\frac{%s-\sum_j m_j}{%d}=%s\,\mathrm{GeV}" % (mP, n, number(ebar)))
        definitions.extend(energy_defs.values())

    # Vertices whose coupling multiplies a momentum.
    for k, v in enumerate(details.get("derivative_vertices") or []):
        E = r"E_{v%s}" % ("" if len(details["derivative_vertices"]) == 1 else k + 1)
        parts.append(E + "^2")
        definitions.append(r"%s=%s\,\mathrm{GeV}" % (E, number(v["energy"])))
        value *= v["energy"] ** 2

    # Propagators.
    names = Counter(bare_name(label) for label in lines)
    used = Counter()
    for label, p in zip(lines, props):
        name = bare_name(label)
        used[name] += 1
        sub = name if names[name] == 1 else "%s,%d" % (name, used[name])
        q2, m, width, spin = p["energy_estimate"] ** 2, p["mass"], p["width"], p["spin"]
        lo, hi = p["virtuality"]
        mass_sym, width_sym, q = "m_{%s}" % name, r"\Gamma_{%s}" % name, "q_{%s}^2" % sub
        num_value = numerator(spin, q2, m)
        num_tex = {0.5: q, 1: (r"\left(1-\frac{%s}{%s^2}\right)^2" % (q, mass_sym)) if m > 0 else ""}.get(spin, "")
        if p["breit_wigner_average"]:
            mg = m * width
            avg = (math.atan((hi - m * m) / mg) - math.atan((lo - m * m) / mg)) / (mg * (hi - lo))
            D = r"\langle|D_{%s}|^2\rangle" % sub
            parts.append(num_tex + D)
            hi_sym, lo_sym = "q_{%s,+}^2" % sub, "q_{%s,-}^2" % sub
            definitions.append(
                r"%s=\frac{\arctan\frac{%s-%s^2}{%s%s}-\arctan\frac{%s-%s^2}{%s%s}}{%s%s\,(%s-%s)}"
                % (D, hi_sym, mass_sym, mass_sym, width_sym, lo_sym, mass_sym, mass_sym, width_sym,
                   mass_sym, width_sym, hi_sym, lo_sym))
            value *= num_value * avg
        else:
            if m == 0:
                den_tex, den_value = r"q_{%s}^4" % sub, q2 ** 2
            elif width > 0:
                den_tex = r"(%s-%s^2)^2+%s^2%s^2" % (q, mass_sym, mass_sym, width_sym)
                den_value = (q2 - m * m) ** 2 + (m * width) ** 2
            else:
                den_tex, den_value = r"(%s-%s^2)^2" % (q, mass_sym), (q2 - m * m) ** 2
            parts.append(r"\frac{%s}{%s}" % (num_tex or "1", den_tex))
            value *= num_value / max(den_value, 1e-300)
        definitions.append(r"q_{%s,\pm}^2=%s,\ %s;\ %s=%s\,\mathrm{GeV}^2"
                           % (sub, number(hi), number(lo), q, number(q2)))

    if not math.isclose(value, details_width(details), rel_tol=1e-9):
        return None
    f = Fraction(lead_value).limit_denominator(64)
    exact = abs(float(f) - lead_value) <= 1e-12 * lead_value
    if mass_power == -1:
        lead = (r"\frac{%d}{%s%s}" % (f.numerator, "" if f.denominator == 1 else f.denominator, mP) if exact
                else r"\frac{%s}{%s}" % (number(lead_value), mP))
    else:
        lead = "" if lead_value == 1 else number(lead_value)
    formula = r"\Gamma=" + r"\,".join([x for x in [lead, phi_tex] if x] + parts)
    factors = dict(details)
    factors["evaluated_width_gev"] = value
    return {"formula": formula, "definitions": definitions, "factors": factors}


def compact_formula(details: Dict[str, Any], *, parent: str, final: List[str], lines: List[str],
                    couplings: List[str]) -> Optional[Dict[str, Any]]:
    """The same estimate in its parameters:

        Gamma ~ kappa prod_v g_v^2 m_P^a prod_j L_j,    kappa = kappa_0 r

    L_j is each internal line's parametric size: 1/(m_j Gamma_j m_P^2)
    (1/(m_j Gamma_j) for a fermion) when the estimate averages its
    Breit-Wigner over an interval holding the pole; 1/m_j^4 (m_P^2/m_j^4 for a
    fermion) when m_j^2 exceeds the line's virtuality; else 1/m_P^4 (1/m_P^2
    for a fermion). A massive vector in the final state contributes
    m_P^2/m_V^2, or the number 2 when its longitudinal growth has no source. The power a gives the width its dimension (a vertex whose
    coupling multiplies a momentum adds m_P^2 through it).

    kappa_0 is the massless limit of the pure numbers, written out: colour,
    B and spin over symmetry, the parent's polarization, 0.8^(n-2), the
    massless phase-space volume over m_P^(2n-4), and (2/n)^k from the k
    fermion energies 2E = 2 m_P/n (1/n^2 per massive vector, 2 per vector without
    a longitudinal source). r = kappa/kappa_0
    is the rest: final-state masses, the propagators at their virtuality
    intervals and the vertex energies over m_P. None when the evaluated form does not apply."""
    n, M = details["n_body"], details["mother_mass"]
    props = details["propagator_details"]
    if any(p.get("virtuality") is None for p in props) or not couplings:
        return None
    s_parent = details["mother_spin"]
    if s_parent not in (0, 0.5, 1):
        return None
    mP = "m_{%s}" % bare_name(parent)
    powers: Dict[str, int] = {}
    values: Dict[str, float] = {}

    def put(symbol, value, k):
        powers[symbol] = powers.get(symbol, 0) + k
        values[symbol] = value

    dimension = 0
    k_fermions, vectors, unsourced = 0, 0, 0
    for i, (label, spin, mass) in enumerate(zip(final, details["final_spins"], details["final_masses_gev"])):
        if spin == 0.5:
            k_fermions += 1
        elif spin == 1 and mass > 0 and i in (details.get("unsourced_vectors") or []):
            unsourced += 1
        elif spin == 1 and mass > 0:
            vectors += 1
            put("m_{%s}" % bare_name(label), mass, -2)
            put(mP, M, 2)
    for label, p in zip(lines, props):
        m, width, spin = p["mass"], p["width"], p["spin"]
        lo, hi = p["virtuality"]
        name = bare_name(label)
        fermion = spin == 0.5
        dimension += -2 if fermion else -4
        if p["breit_wigner_average"] and lo <= m * m <= hi * (1 + 1e-3):
            put("m_{%s}" % name, m, -1)
            put(r"\Gamma_{%s}" % name, width, -1)
            if not fermion:
                put(mP, M, -2)
        elif m * m > p["energy_estimate"] ** 2:
            put("m_{%s}" % name, m, -4)
            if fermion:
                put(mP, M, 2)
        else:
            put(mP, M, -2 if fermion else -4)
    # The parent's polarization, the energies and the phase-space volume are
    # powers of m_P; with the lines' L_j the width needs this one.
    put(mP, M, 1 - dimension)
    rank = lambda sym: (sym != mP, sym.startswith(r"\Gamma"))
    powers = {sym: powers[sym] for sym in sorted(powers, key=rank) if powers[sym]}
    symbolic = math.prod(values[sym] ** k for sym, k in powers.items())
    width = details_width(details)
    kappa = width / (details["coupling_sq"] * symbolic)

    # kappa_0 and its TeX.
    parent_factor = {0: Fraction(1, 2), 0.5: Fraction(1), 1: Fraction(1)}[s_parent]
    lead = (Fraction(details["color_factor"]).limit_denominator(64) * int(details["backtoback_correction"])
            / (Fraction(2 * s_parent + 1).limit_denominator(4) * details["symmetry_factor"]) * parent_factor)
    space = 2 ** (4 * n - 5) * factorial(n - 1) * factorial(n - 2)
    kappa0 = (float(lead) * C_PS ** (n - 2) / (space * math.pi ** (2 * n - 3))
              * (2 / n) ** k_fermions * (1 / n ** 2) ** vectors * 2 ** unsourced)
    pieces = []
    if lead != 1:
        pieces.append(str(lead.numerator) if lead.denominator == 1 else r"\frac{%d}{%d}" % (lead.numerator, lead.denominator))
    space_tex = "8" if n == 2 else r"2^{%d}\,%d!\,%d!" % (4 * n - 5, n - 1, n - 2) if n > 3 else r"2^{%d}\,2!" % (4 * n - 5)
    pi_tex = r"\pi" if n == 2 else r"\pi^{%d}" % (2 * n - 3)
    cps = "" if n == 2 else (r"%s^{%d}" % (C_PS, n - 2) if n > 3 else str(C_PS))
    pieces.append(r"\frac{%s}{%s\,%s}" % (cps or "1", space_tex, pi_tex))
    if k_fermions and n != 2:
        e = Fraction(2, n)
        e_tex = str(e.numerator) if e.denominator == 1 else r"\frac{%d}{%d}" % (e.numerator, e.denominator)
        if e != 1:
            pieces.append(r"\left(%s\right)^{%d}" % (e_tex, k_fermions) if k_fermions > 1 else e_tex)
    if vectors:
        pieces.append(r"\frac{1}{%d^{%d}}" % (n, 2 * vectors))
    if unsourced:
        pieces.append("2" if unsourced == 1 else "2^{%d}" % unsourced)
    r = kappa / kappa0
    counts = Counter(couplings)
    coupling_tex = "".join(power(c, 2 * m) for c, m in counts.items())
    up = "".join(sym if k == 1 else "%s^{%d}" % (sym, k) for sym, k in powers.items() if k > 0)
    down = "".join(sym if -k == 1 else "%s^{%d}" % (sym, -k) for sym, k in powers.items() if k < 0)
    scale = r"\frac{%s}{%s}" % (up or "1", down) if down else up
    formula = r"\Gamma\approx\kappa\,%s%s" % (coupling_tex, r"\," + scale if scale else "")
    if math.isclose(r, 1, rel_tol=1e-9):
        definition = r"\kappa=%s=%s" % (r"\,".join(pieces), number(kappa, 3))
    else:
        definition = r"\kappa=%s\,r=%s,\quad r=%s" % (r"\,".join(pieces), number(kappa, 3), number(r, 3))
    return {"formula": formula, "definitions": [definition], "kappa": kappa, "kappa_0": kappa0, "r": r}


def details_width(details: Dict[str, Any]) -> float:
    """Width of the improved estimate from its details."""
    phi = details["phase_space_improved"]
    return phi * details["matrix_element_sq"] / (2 * details["mother_mass"]) / details["symmetry_factor"]
