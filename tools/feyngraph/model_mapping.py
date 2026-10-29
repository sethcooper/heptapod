"""
# model_mapping.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Particle label and coupling constant mappings for FeynGraph integration.

This module provides centralized mappings between HEPTAPOD particle
conventions and FeynGraph notation, as well as Standard Model coupling
constants and mass hierarchies.
"""

from typing import Optional, Dict
import math

# ============================================================================
# Particle Label Mappings
# ============================================================================

# HEPTAPOD → FeynGraph label mapping
# FeynGraph uses PDG-like notation with some variations
PARTICLE_LABEL_MAP: Dict[str, str] = {
    # Leptons (charged)
    "e": "e-",
    "e-": "e-",
    "e+": "e+",
    "ebar": "e+",
    "mu": "mu-",
    "mu-": "mu-",
    "mu+": "mu+",
    "mubar": "mu+",
    "tau": "tau-",
    "tau-": "tau-",
    "tau+": "tau+",
    "taubar": "tau+",

    # Neutrinos
    "nu_e": "ve",
    "nue": "ve",
    "nu_e_bar": "ve~",
    "nu_ebar": "ve~",
    "nuebar": "ve~",
    "nu_mu": "vm",
    "numu": "vm",
    "nu_mu_bar": "vm~",
    "nu_mubar": "vm~",
    "numubar": "vm~",
    "nu_tau": "vt",
    "nutau": "vt",
    "nu_tau_bar": "vt~",
    "nu_taubar": "vt~",
    "nutaubar": "vt~",

    # Quarks (up-type)
    "u": "u",
    "ubar": "u~",
    "c": "c",
    "cbar": "c~",
    "t": "t",
    "tbar": "t~",

    # Quarks (down-type)
    "d": "d",
    "dbar": "d~",
    "s": "s",
    "sbar": "s~",
    "b": "b",
    "bbar": "b~",

    # Gauge bosons
    "gamma": "a",    # photon
    "g": "g",        # gluon
    "Z": "Z",        # Z boson
    "W+": "W+",      # W+
    "W-": "W-",      # W-

    # Higgs
    "H": "H",
    "h": "H",

    # Composite particles (baryons)
    "p": "p+",       # proton
    "pbar": "p-",    # antiproton
    "n": "n0",       # neutron
    "nbar": "n0~",   # antineutron
}

# Reverse mapping: FeynGraph → HEPTAPOD
FEYNGRAPH_TO_NDA_MAP: Dict[str, str] = {
    # Build reverse mapping automatically
    v: k for k, v in PARTICLE_LABEL_MAP.items()
    if k not in ["ebar", "mubar", "taubar"]  # Prefer canonical forms
}

# Add additional FeynGraph notation variants
FEYNGRAPH_TO_NDA_MAP.update({
    "ta-": "tau-",
    "ta+": "tau+",
    # FeynGraph sometimes uses these forms
    "e": "e-",
    "mu": "mu-",
    "tau": "tau-",
    "ve": "nu_e",
    "vm": "nu_mu",
    "vt": "nu_tau",
    "ve~": "nu_e_bar",
    "vm~": "nu_mu_bar",
    "vt~": "nu_tau_bar",
    # Gauge bosons
    "a": "gamma",
    "g": "g",
    "Z": "Z",
    "W+": "W+",
    "W-": "W-",
    "H": "H",
})

# ============================================================================
# Standard Model Coupling Constants
# ============================================================================

SM_COUPLINGS: Dict[str, float] = {
    # Electromagnetic coupling
    "alpha_em": 1/137.036,         # Fine structure constant
    "e": 0.302822,                 # Electric charge sqrt(4π α_EM)

    # Strong coupling (at M_Z)
    "alpha_s": 0.1179,             # Strong coupling constant
    "g_s": 1.217,                  # Strong coupling sqrt(4π α_S)

    # Weak coupling
    "G_F": 1.166e-5,               # Fermi constant (GeV^-2)
    "g_W": 0.6530,                 # Weak coupling
    "sin_theta_W": 0.2312,         # Weak mixing angle (sin²θ_W)

    # Yukawa couplings (y = sqrt(2) m / v, with v = 246 GeV)
    "y_e": 2.94e-6,                # Electron Yukawa
    "y_mu": 6.07e-4,               # Muon Yukawa
    "y_tau": 1.02e-2,              # Tau Yukawa
    "y_u": 1.29e-5,                # Up Yukawa
    "y_d": 2.71e-5,                # Down Yukawa
    "y_c": 7.36e-3,                # Charm Yukawa
    "y_s": 5.49e-4,                # Strange Yukawa
    "y_t": 0.996,                  # Top Yukawa
    "y_b": 2.42e-2,                # Bottom Yukawa

    # Higgs
    "lambda_H": 0.129,             # Higgs self-coupling
    "v": 246.0,                    # Higgs VEV (GeV)
}

# ---------------------------------------------------------------------------
# Z-boson chiral couplings: gL_f = g_Z (I3_f - Q_f sin²θ_W),
#                            gR_f = g_Z (-Q_f sin²θ_W)
# where g_Z = g_W / cos θ_W.  These are the full vertex coupling factors
# so that |M|² ∝ (gL² + gR²) M² with no additional overall factor.
# ---------------------------------------------------------------------------
_sin2w = SM_COUPLINGS["sin_theta_W"]          # sin²θ_W = 0.2312
_g_Z = SM_COUPLINGS["g_W"] / math.sqrt(1.0 - _sin2w)  # g_W / cos θ_W

# Charged leptons (e, mu, tau): Q = -1, I3 = -1/2
_gL_l = _g_Z * (-0.5 + _sin2w)               # I3 - Q sin²θ_W
_gR_l = _g_Z * _sin2w                         # -Q sin²θ_W

# Up-type quarks (u, c, t): Q = +2/3, I3 = +1/2
_gL_u = _g_Z * (0.5 - (2.0/3.0) * _sin2w)
_gR_u = _g_Z * (-(2.0/3.0) * _sin2w)

# Down-type quarks (d, s, b): Q = -1/3, I3 = -1/2
_gL_d = _g_Z * (-0.5 + (1.0/3.0) * _sin2w)
_gR_d = _g_Z * ((1.0/3.0) * _sin2w)

# Neutrinos (nu_e, nu_mu, nu_tau): Q = 0, I3 = +1/2
_gL_nu = _g_Z * 0.5
_gR_nu = 0.0

SM_COUPLINGS.update({
    # Z gauge coupling
    "g_Z": _g_Z,

    # Z → charged leptons (all flavors identical)
    "gL_e":   _gL_l,   "gR_e":   _gR_l,
    "gL_mu":  _gL_l,   "gR_mu":  _gR_l,
    "gL_tau": _gL_l,   "gR_tau": _gR_l,

    # Z → up-type quarks
    "gL_u": _gL_u,     "gR_u": _gR_u,
    "gL_c": _gL_u,     "gR_c": _gR_u,
    "gL_t": _gL_u,     "gR_t": _gR_u,

    # Z → down-type quarks
    "gL_d": _gL_d,     "gR_d": _gR_d,
    "gL_s": _gL_d,     "gR_s": _gR_d,
    "gL_b": _gL_d,     "gR_b": _gR_d,

    # Z → neutrinos
    "gL_nu_e":   _gL_nu,  "gR_nu_e":   _gR_nu,
    "gL_nu_mu":  _gL_nu,  "gR_nu_mu":  _gR_nu,
    "gL_nu_tau": _gL_nu,  "gR_nu_tau": _gR_nu,

    # W coupling (alias for convenience)
    "g_w": SM_COUPLINGS["g_W"],
})

# ============================================================================
# Particle Masses (GeV)
# ============================================================================

PARTICLE_MASSES: Dict[str, float] = {
    # Leptons
    "e": 0.0005109989,             # Electron
    "mu": 0.1056583745,            # Muon
    "tau": 1.77686,                # Tau
    "nu_e": 0.0,                   # Electron neutrino (massless approx)
    "nu_mu": 0.0,                  # Muon neutrino
    "nu_tau": 0.0,                 # Tau neutrino

    # Quarks (MS-bar masses at 2 GeV for light, pole masses for heavy)
    "u": 0.0023,                   # Up
    "d": 0.0048,                   # Down
    "s": 0.095,                    # Strange
    "c": 1.275,                    # Charm
    "b": 4.18,                     # Bottom
    "t": 172.76,                   # Top

    # Gauge bosons
    "gamma": 0.0,                  # Photon
    "g": 0.0,                      # Gluon
    "Z": 91.1876,                  # Z boson
    "W+": 80.379,                  # W+
    "W-": 80.379,                  # W-

    # Higgs
    "H": 125.10,                   # Higgs boson

    # Baryons
    "p": 0.938272,                 # Proton
    "n": 0.939565,                 # Neutron
}

# ============================================================================
# Coupling Type Mappings
# ============================================================================

INTERACTION_TO_COUPLING: Dict[str, str] = {
    "em": "alpha_em",              # Electromagnetic
    "electromagnetic": "alpha_em",
    "qed": "alpha_em",

    "strong": "alpha_s",           # Strong
    "qcd": "alpha_s",

    "weak": "G_F",                 # Weak
    "electroweak": "G_F",

    "yukawa": "y_t",               # Yukawa (generic, use top as reference)
    "higgs": "lambda_H",           # Higgs self-coupling
}

# ============================================================================
# Public API
# ============================================================================

def nda_to_feyngraph_label(label: str) -> str:
    """
    Convert HEPTAPOD particle label to FeynGraph notation.

    Args:
        label: HEPTAPOD particle label (e.g., "mu", "gamma", "bbar")

    Returns:
        FeynGraph label (e.g., "mu-", "a", "b~")

    Raises:
        ValueError: If label is not recognized

    Examples:
        >>> nda_to_feyngraph_label("gamma")
        "a"
        >>> nda_to_feyngraph_label("bbar")
        "b~"
        >>> nda_to_feyngraph_label("nu_mu")
        "vm"
    """
    if label in PARTICLE_LABEL_MAP:
        return PARTICLE_LABEL_MAP[label]

    # Try case-insensitive lookup
    label_lower = label.lower()
    for key, val in PARTICLE_LABEL_MAP.items():
        if key.lower() == label_lower:
            return val

    raise ValueError(
        f"Unknown particle label: '{label}'. "
        f"Available labels: {sorted(set(PARTICLE_LABEL_MAP.keys()))[:10]}..."
    )


def feyngraph_to_nda_label(label: str) -> str:
    """
    Convert FeynGraph label to HEPTAPOD notation.

    Args:
        label: FeynGraph particle label (e.g., "mu-", "a", "b~")

    Returns:
        HEPTAPOD label (e.g., "mu", "gamma", "bbar")

    Raises:
        ValueError: If label is not recognized

    Examples:
        >>> feyngraph_to_nda_label("a")
        "gamma"
        >>> feyngraph_to_nda_label("b~")
        "bbar"
        >>> feyngraph_to_nda_label("vm")
        "nu_mu"
    """
    if label in FEYNGRAPH_TO_NDA_MAP:
        return FEYNGRAPH_TO_NDA_MAP[label]

    raise ValueError(
        f"Unknown FeynGraph label: '{label}'. "
        f"Available labels: {sorted(set(FEYNGRAPH_TO_NDA_MAP.keys()))[:10]}..."
    )


def get_sm_coupling(coupling_name: str) -> float:
    """
    Get Standard Model coupling constant value.

    Args:
        coupling_name: Coupling name (e.g., "alpha_em", "G_F", "y_b")
                      Can also use interaction type ("weak", "strong", "em")

    Returns:
        Coupling constant value

    Raises:
        ValueError: If coupling is not recognized

    Examples:
        >>> get_sm_coupling("alpha_em")
        0.007297352...
        >>> get_sm_coupling("weak")
        1.166e-05
        >>> get_sm_coupling("y_b")
        0.0242
    """
    # Try direct lookup
    if coupling_name in SM_COUPLINGS:
        return SM_COUPLINGS[coupling_name]

    # Try interaction type mapping
    if coupling_name in INTERACTION_TO_COUPLING:
        return SM_COUPLINGS[INTERACTION_TO_COUPLING[coupling_name]]

    # Try case-insensitive
    coupling_lower = coupling_name.lower()
    for key, val in SM_COUPLINGS.items():
        if key.lower() == coupling_lower:
            return val

    raise ValueError(
        f"Unknown coupling: '{coupling_name}'. "
        f"Available couplings: {sorted(SM_COUPLINGS.keys())}"
    )


def get_particle_mass(label: str) -> float:
    """
    Get particle mass in GeV.

    Args:
        label: HEPTAPOD particle label

    Returns:
        Mass in GeV

    Raises:
        ValueError: If particle is not recognized

    Examples:
        >>> get_particle_mass("mu")
        0.1056583745
        >>> get_particle_mass("H")
        125.10
        >>> get_particle_mass("gamma")
        0.0
    """
    # Normalize label
    label_clean = label.lower()

    # Try direct lookup
    if label in PARTICLE_MASSES:
        return PARTICLE_MASSES[label]

    # Try case-insensitive
    for key, val in PARTICLE_MASSES.items():
        if key.lower() == label_clean:
            return val

    # Antiparticles have same mass
    if label.endswith("bar"):
        base = label[:-3]
        if base in PARTICLE_MASSES:
            return PARTICLE_MASSES[base]

    if label.endswith("+") or label.endswith("-"):
        base = label[:-1]
        if base in PARTICLE_MASSES:
            return PARTICLE_MASSES[base]

    raise ValueError(
        f"Unknown particle for mass lookup: '{label}'. "
        f"Available particles: {sorted(PARTICLE_MASSES.keys())}"
    )


def get_mass_hierarchy_suppression(propagator_mass: float, energy_scale: float) -> float:
    """
    Calculate suppression factor from mass hierarchy.

    For heavy propagators (M >> E), the suppression is ~(E/M)^2.
    For light propagators (M << E), no additional suppression.

    Args:
        propagator_mass: Propagator mass in GeV
        energy_scale: Typical energy scale of process in GeV

    Returns:
        Suppression factor (dimensionless)

    Examples:
        >>> get_mass_hierarchy_suppression(100.0, 10.0)  # Heavy propagator
        0.01
        >>> get_mass_hierarchy_suppression(1.0, 100.0)   # Light propagator
        1.0
    """
    if propagator_mass < 1e-6:  # Massless
        return 1.0

    ratio = energy_scale / propagator_mass

    if ratio < 0.1:  # Heavy propagator regime (M >> E)
        return ratio ** 2
    else:  # Light propagator or comparable masses
        return 1.0
