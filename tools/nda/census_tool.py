"""
# census_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
from pathlib import Path
from typing import List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .census import estimate_catalogue


class EstimateCatalogueNDATool(BaseTool):
    """
    NDA width estimates for every process of a decay census (EnumerateProcesses
    output), by deterministic translation of each census diagram into an
    EstimateDecayWidthNDA request.

    Each diagram's request uses the UFO at the supplied card: masses, widths,
    spins and colours; vertex couplings evaluated at the card (CKM included;
    chiral parts for fermion vertices); the exact SU(3) colour factor; and the
    diagram's connectivity, so internal lines get their allowed virtuality and
    a line that can go on shell is treated in the narrow-width approximation.
    A channel combines its diagrams (coherently among diagrams that differ only
    in internal-line flavour, e.g. GIM; incoherently otherwise), also per
    coupling class.

    Each diagram's formula is its evaluated expression: every factor of the
    estimate with the model's coupling names (g_{hWW}), and definitions giving
    each symbol's value, so that substituting them reproduces the diagram's
    width. A channel's formula is that of its largest diagram, with that
    diagram's share of the channel width (formula_fraction; interference
    makes it differ from 1).

    With lower_estimates (this tool's outputs at lower multiplicities, same
    model and card), two more kinds of channel get an estimate:
    - an external photon or gluon: Gamma(P -> X) times the soft-collinear
      factor sum_i alpha Q_i^2 / (2 pi) ln^2(1/y) per photon (alpha_s C_F for a
      gluon, quarks only) over the final-state fermions of X, n identical
      vectors to the n-th power over n!, with the pair-mass regulator
      m_ij^2 > y M^2 on visible pairs (pair_mass_y);
    - a zero-width particle Y going on shell: Gamma(P -> X Y) Gamma(Y -> S) /
      Gamma_Y, Gamma_Y the sum of Y's lower-multiplicity widths; decays into
      quarks of a particle lighter than the lightest hadron are closed, and a
      channel whose on-shell Y has only such decays is marked unphysical.
    - a photon or gluon split into a massive pair F Fbar: Gamma(P -> X) c_V
      k_F/(3 pi) ln^3(M^2/4m_F^2)/3, k_F = alpha Q_F^2 N_F (photon) or
      alpha_s T_F (gluon), in place of those diagrams' estimates; the other
      diagrams add incoherently.
    Channels exchanging a photon or gluon internally into massless daughters
    get no estimate (requires_regulator) and say why.

    Outputs in output_dir: nda.json (per channel: status, width_gev, formula,
    definitions, classes, methods) and estimates/<process id>.json with every
    diagram's EstimateDecayWidthNDA request and response and its formula request
    and response, each rerunnable on its own. An NDA estimate is a naive
    ballpark of magnitude and scaling, not a calculation.
    """

    catalogue: str = RuntimeField(
        description="EnumerateProcesses output directory (or its catalogue.json), relative to base_directory"
    )
    model: str = RuntimeField(description="The UFO directory the census was built from")
    param_card: Optional[str] = RuntimeField(
        default=None, description="The parameter card the census was built with"
    )
    output_dir: str = RuntimeField(description="New directory for the estimates; must not exist")
    processes: Optional[List[str]] = RuntimeField(
        default=None, description="Optional subset of census process IDs (default: all)"
    )
    lower_estimates: Optional[List[str]] = RuntimeField(
        default=None,
        description="Output directories of this tool at lower multiplicities (same model and card); "
                    "enables the radiative and on-shell sequential estimates",
    )
    pair_mass_y: float = RuntimeField(
        default=0.0025, description="Pair-mass regulator y in m_ij^2 > y M^2 for radiative channels (default 0.05^2)"
    )
    base_directory: str = StateField(description="Workspace root for input and output files")

    def _run(self) -> str:
        base = Path(self.base_directory).resolve()

        def local(value):
            p = (base / value).resolve()
            if not p.is_relative_to(base):
                raise ValueError("Path must be within base_directory")
            return p

        try:
            result = estimate_catalogue(
                local(self.catalogue), local(self.model),
                local(self.param_card) if self.param_card else None,
                local(self.output_dir), base, self.processes,
                [local(p) for p in self.lower_estimates or []], self.pair_mass_y,
            )
        except Exception as exc:
            return self.format_error(error="Census NDA failed", reason=f"{type(exc).__name__}: {exc}")
        return json.dumps(result)
