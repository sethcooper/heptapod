"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Reading UFO model directories.

A UFO is the model format MadGraph consumes; FeynRules is one producer of them
but not the only one, so these modules know nothing about FeynRules and live
apart from it.

Nothing here imports a UFO. ``particles.py`` and ``couplings.py`` execute
arbitrary code on import -- they instantiate objects and import the model
package -- so both modules parse them with ``ast`` instead, which executes
nothing and tolerates the formatting UFO writers actually emit.

Mostly library, plus one tool. InspectUFOTool exposes the reading side,
because the agent that most needs it is the one holding a UFO it just compiled
and no way to look inside; everything else here is called by other tools.

    fold_number               — a numeric literal or BinOp, as an exact Fraction
    parse_particles           — particle declarations, statically
    parse_vertices            — interactions: particle content and orders
    check_particle_properties — declared particles reached the UFO intact
    is_goldstone / is_ghost   — gauge artefact or physical state, by the UFO's
                                own flags rather than by name
    parse_parameters          — external and internal parameter declarations
    evaluate_parameters       — resolved to numbers, in dependency order
    parse_couplings           — GC_* expressions
    parse_coupling_orders     — GC_* coupling orders, {"QCD": 1}
    evaluate_couplings        — those, as complex numbers
    vertex_couplings          — vertices with the (colour, lorentz) key intact
    ufo_gauge                 — unitary or Feynman, from the particle content
    safe_eval                 — one expression, whitelisted namespace
    parse_external_params     — external parameter values
    parse_decays              — FeynRules-computed partial widths
    find_partial_width        — one partial width from a decay table
    safe_eval_width           — evaluate a width in a whitelisted namespace
    analytic_scalar_lq_width  — closed form for a scalar leptoquark width
    compare_width             — a UFO width against a closed-form expectation
    parse_lorentz             — Lorentz structures: spins and tensor string
    read_slha                 — an SLHA card as {(block, codes): value}
    card_overrides            — a card as external-parameter overrides
"""

from .parser import (
    fold_number,
    parse_particles,
    parse_vertices,
    check_particle_properties,
)
from .evaluate import (
    evaluate_couplings,
    evaluate_parameters,
    is_ghost,
    is_goldstone,
    parse_coupling_orders,
    parse_couplings,
    parse_parameters,
    safe_eval,
    ufo_gauge,
    vertex_couplings,
)
from .lorentz import parse_lorentz
from .cards import card_overrides, read_slha
from .width_gate import (
    analytic_scalar_lq_width,
    compare_width,
    find_partial_width,
    parse_decays,
    parse_external_params,
    safe_eval_width,
)

__all__ = [
    "card_overrides",
    "parse_lorentz",
    "read_slha",
    "fold_number",
    "parse_particles",
    "parse_vertices",
    "check_particle_properties",
    "evaluate_couplings",
    "evaluate_parameters",
    "is_ghost",
    "is_goldstone",
    "parse_coupling_orders",
    "parse_couplings",
    "parse_parameters",
    "safe_eval",
    "ufo_gauge",
    "vertex_couplings",
    "analytic_scalar_lq_width",
    "compare_width",
    "find_partial_width",
    "parse_decays",
    "parse_external_params",
    "safe_eval_width",
]
