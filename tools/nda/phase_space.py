"""
# phase_space.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import math
from typing import Optional, List
from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "nda-phasespace-1.0"


def two_body_phase_space(M: float, m1: float, m2: float) -> float:
    """Phi_2 = lambda^(1/2)(M^2, m1^2, m2^2) / (8 pi M^2); 1/(8 pi) when massless."""
    lam = (M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)
    return math.sqrt(max(lam, 0.0)) / (8.0 * math.pi * M**2)


def massive_phase_space(M: float, masses: List[float], points: int = 48) -> float:
    """n-body phase-space volume with final-state masses, by the recursion
    Phi_n(M) = int ds/(2 pi) Phi_2(M; sqrt(s), m_n) Phi_{n-1}(sqrt(s); m_1..m_{n-1})
    over (sum_{i<n} m_i)^2 <= s <= (M - m_n)^2 (Gauss-Legendre). Same
    normalization as the massless formula, which it reproduces when all
    masses vanish."""
    if len(masses) == 2:
        return two_body_phase_space(M, masses[0], masses[1])
    *rest, last = masses
    lo, hi = sum(rest) ** 2, (M - last) ** 2
    if hi <= lo:
        return 0.0
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(points)
    total = 0.0
    for x, w in zip(nodes, weights):
        s = lo + (hi - lo) * (x + 1.0) / 2.0
        inner = massive_phase_space(math.sqrt(s), rest, max(12, points // 2))
        total += w * two_body_phase_space(M, math.sqrt(s), last) * inner
    return total * (hi - lo) / 2.0 / (2.0 * math.pi)


class EstimatePhaseSpaceTool(BaseTool):
    """
    Estimates n-body phase space volume.

    This tool calculates the Lorentz-invariant phase space (LIPS) volume
    for n-body decays. Massless final states use the formula from Eq. A.2
    of arXiv:1402.1178 (with final-state masses, the exact massive volume is
    returned and the massless one reported as phase_space_volume_massless):

    Φ_n(M) = (1 / (2^(4n-5) π^(2n-3))) x (M^(2n-4) / (n-1)!(n-2)!)

    This is purely kinematic - no coupling constants or spin factors.

    Inputs (runtime):
      - mother_mass_gev: Mother particle mass in GeV (energy scale)
      - n_body: Number of final state particles (n >= 2)
      - final_state_masses_gev: (optional) List of final state masses for corrections

    State:
      - base_directory: Sandbox root (required by BaseTool, not used here)

    Behavior:
      1. Validates that n >= 2
      2. Calculates phase space volume using the massless approximation
      3. With final_state_masses, returns the massive volume (recursive 2-body
         factorization) and the suppression relative to massless
      4. Returns LaTeX formula showing the scaling

    Output (JSON):
      {
        "status": "ok",
        "schema": "nda-phasespace-1.0",
        "mother_mass_gev": 125.0,
        "n_body": 2,
        "phase_space_volume": 3.98e-2,
        "formula_latex": "\\Phi_2(M) = \\frac{1}{8\\pi} \\approx 0.0398",
        "scaling": "M^{2n-4} = M^{0}",
        "notes": ["Massless approximation", "Tree-level kinematics"]
      }

    Errors:
      Returns self.format_error() on failures including:
        - Invalid n_body (must be >= 2)
        - Negative or zero mother mass
    """

    # ======================== Runtime fields ======================== #
    mother_mass_gev: float = RuntimeField(
        description="Mother particle mass in GeV (characteristic energy scale)"
    )
    n_body: int = RuntimeField(
        description="Number of final state particles (must be >= 2)"
    )
    final_state_masses_gev: Optional[List[float]] = RuntimeField(
        default=None,
        description="Optional: List of final state masses in GeV for mass corrections"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory (not used for phase space calculations)"
    )
    # ================================================================ #

    def _calculate_phase_space_massless(self, M: float, n: int) -> float:
        """
        Calculate n-body phase space for massless final states.

        Φ_n(M) = (M^(2n-4)) / ((2^(4n-5)) (π^(2n-3)) (n-1)! (n-2)!)

        Args:
            M: Mother particle mass (GeV)
            n: Number of final state particles

        Returns:
            Phase space volume
        """
        # Factorial calculations
        fact_n_minus_1 = math.factorial(n - 1)
        fact_n_minus_2 = math.factorial(n - 2)

        # Numerator: M^(2n-4)
        numerator = M ** (2*n - 4)

        # Denominator: 2^(4n-5) * π^(2n-3) * (n-1)! * (n-2)!
        denominator = (
            (2 ** (4*n - 5)) *
            (math.pi ** (2*n - 3)) *
            fact_n_minus_1 *
            fact_n_minus_2
        )

        return numerator / denominator

    def _generate_latex_formula(self, M: float, n: int, phi_n: float, symbolic: bool = True) -> str:
        """
        Generate LaTeX formula for the phase space volume.

        Args:
            M: Mother mass
            n: Number of particles
            phi_n: Numerical phase space value
            symbolic: If True, use symbolic M; if False, substitute numerical value

        Returns:
            LaTeX string
        """
        power = 2*n - 4
        denom_power_2 = 4*n - 5
        denom_power_pi = 2*n - 3
        fact1 = n - 1
        fact2 = n - 2

        # Calculate numerical coefficient
        import math
        coeff_numeric = 1.0 / ((2 ** denom_power_2) * (math.pi ** denom_power_pi) *
                               math.factorial(fact1) * math.factorial(fact2))

        # Special cases for cleaner expressions
        if n == 2:
            # 2-body: Φ_2 = 1/(8π) ≈ 0.0398
            if symbolic:
                return r"\Phi_2 = \frac{1}{8\pi}"
            else:
                return rf"\Phi_2 = \frac{{1}}{{8\pi}} \approx {coeff_numeric:.4f}"

        elif n == 3:
            # 3-body: Φ_3 = M^2/(32π^3)
            if symbolic:
                return r"\Phi_3 = \frac{M^2}{32\pi^3}"
            else:
                return rf"\Phi_3 = \frac{{M^2}}{{32\pi^3}} \approx \frac{{M^2}}{{{1/(coeff_numeric):.1f}}}"

        elif n == 4:
            # 4-body: Φ_4 = M^4/(2048 π^5 × 12)
            denom_val = int((2 ** denom_power_2) * math.factorial(fact1) * math.factorial(fact2))
            if symbolic:
                return rf"\Phi_4 = \frac{{M^4}}{{{denom_val} \pi^5}}"
            else:
                return rf"\Phi_4 = \frac{{M^4}}{{{denom_val} \pi^5}} \approx \frac{{M^4}}{{{1/coeff_numeric:.1e}}}"

        else:
            # General case
            denom_val = int((2 ** denom_power_2) * math.factorial(fact1) * math.factorial(fact2))
            if symbolic:
                if power == 0:
                    return rf"\Phi_{{{n}}} = \frac{{1}}{{{denom_val} \pi^{{{denom_power_pi}}}}}"
                else:
                    return rf"\Phi_{{{n}}} = \frac{{M^{{{power}}}}}{{{denom_val} \pi^{{{denom_power_pi}}}}}"
            else:
                return (
                    rf"\Phi_{{{n}}} = \frac{{M^{{{power}}}}}{{{denom_val} \pi^{{{denom_power_pi}}}}} "
                    rf"\approx {phi_n:.4e}"
                )

    def _run(self) -> str:
        """
        Main execution method.

        Returns:
            JSON string with phase space estimate
        """
        # Validate inputs
        if self.n_body < 2:
            return self.format_error(
                error="Invalid n_body",
                reason=f"n_body must be >= 2, got {self.n_body}",
                suggestion="Set n_body to 2 for two-body decays, 3 for three-body, etc."
            )

        if self.mother_mass_gev <= 0:
            return self.format_error(
                error="Invalid Mass",
                reason=f"mother_mass_gev must be positive, got {self.mother_mass_gev}",
                suggestion="Provide a positive mass in GeV"
            )

        # Check mass threshold if final state masses provided
        if self.final_state_masses_gev:
            if len(self.final_state_masses_gev) != self.n_body:
                return self.format_error(
                    error="Mass List Mismatch",
                    reason=f"Provided {len(self.final_state_masses_gev)} masses but n_body={self.n_body}",
                    suggestion="Ensure final_state_masses_gev has exactly n_body elements"
                )

            total_final_mass = sum(self.final_state_masses_gev)
            if self.mother_mass_gev < total_final_mass:
                return self.format_error(
                    error="Kinematically Forbidden",
                    reason=f"M_mother ({self.mother_mass_gev:.3f} GeV) < Σm_final ({total_final_mass:.3f} GeV)",
                    suggestion="This decay is kinematically impossible"
                )

        # Calculate phase space
        try:
            phi_n = self._calculate_phase_space_massless(self.mother_mass_gev, self.n_body)
        except Exception as e:
            return self.format_error(
                error="Calculation Error",
                reason=str(e),
                suggestion="Check input parameters"
            )

        # Generate both symbolic and numerical LaTeX formulas
        latex_formula_symbolic = self._generate_latex_formula(self.mother_mass_gev, self.n_body, phi_n, symbolic=True)
        latex_formula_numeric = self._generate_latex_formula(self.mother_mass_gev, self.n_body, phi_n, symbolic=False)

        # Scaling with mass
        power = 2 * self.n_body - 4
        if power == 0:
            scaling = "M^{0} = 1"
        else:
            scaling = f"M^{{{power}}}"

        # Build streamlined result
        result = {
            "status": "ok",
            "phase_space_volume": phi_n,
            "formula": latex_formula_symbolic,  # Clean symbolic form
            "scaling": scaling,                 # M^(2n-4)
            "n_body": self.n_body,
        }
        # With final-state masses, the volume is the massive one; the massless
        # formula and the threshold suppression are reported alongside.
        if self.final_state_masses_gev and any(m > 0 for m in self.final_state_masses_gev):
            massive = massive_phase_space(self.mother_mass_gev, list(self.final_state_masses_gev))
            result["phase_space_volume"] = massive
            result["phase_space_volume_massless"] = phi_n
            result["mass_suppression"] = massive / phi_n

        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
