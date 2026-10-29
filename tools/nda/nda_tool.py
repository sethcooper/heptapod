"""
# nda_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Naive Dimensional Analysis (NDA) decay width estimation tool.

Provides order-of-magnitude estimates for decay widths using dimensional analysis
and simple diagram specifications.
"""
import json
import math
import os
from collections import Counter
from typing import Optional, Dict, Any
from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .simple_diagram import (
    parse_diagram, Diagram, Particle, Vertex, Propagator,
    spin_to_ufo, infer_color_factor
)
from .phase_space import EstimatePhaseSpaceTool
from .matrix_element import (
    EstimateMatrixElementTool, INTERACTION_TYPES, resolve_nda_interaction_type
)
from .topology import validate_graph_theory_constraint
from . import kinematics
from .evaluated import evaluated_formula
from .summary_formatting import format_decay_width_summary
from tools.logging.findings import append_finding

# Phase space fudge factor from arXiv:1402.1178 Eq. (10)
# This improves the rectangular approximation of the phase space integral
C_PS_FUDGE = 0.8


class EstimateDecayWidthNDATool(BaseTool):
    """
    Estimates decay width using Naive Dimensional Analysis - Diagram-First Interface.

    This is a complete rewrite with clean diagram-based input.

    Input Format:
    {
      "diagram": {
        "topology": "tree_2body",  // or auto-inferred
        "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
        "final": [
          {"label": "b", "spin": "1/2", "mass": 4.2},
          {"label": "bbar", "spin": "1/2", "mass": 4.2}
        ],
        "vertices": [
          {"type": "yukawa", "coupling": "y_b"}
        ],
        "couplings": {"y_b": 0.03},
        "color_factor": 3.0  // optional, can be inferred
      }
    }

    Minimal Generic Example:
    {
      "diagram": {
        "topology": "tree_2body",
        "initial": [{"spin": 0}],
        "final": [{"spin": "1/2"}, {"spin": "1/2"}],
        "vertices": [{"type": "yukawa", "coupling": 0.1}],
        "energy_scale": 100.0
      }
    }

    Formula: Γ = (1/2M) × Φ_n × <|M|^2> × [propagator factors] × [loop factors]

    Workflow:
      - Hand-written diagram: provide the diagram dict directly with masses, spins,
        couplings, and vertices.
      - FeynGraph workflow: call `EnumerateDiagrams` first to enumerate all diagrams
        for a process, then pass each diagram from the output's `diagrams[i].diagram`
        field to this tool for individual NDA estimates.
      - For branching ratios across multiple diagram classes, use `EstimateBranchingRatioNDA`
        which takes a list of diagram classes and a reference total width.

    Returns:
      {
        "status": "ok",
        "width_gev": 0.02686,          // best estimate (improved if available)
        "width_gev_nda_raw": 0.03358,  // raw NDA estimate
        "method": "improved (arXiv:1402.1178)",
        "formula": "\\Gamma \\sim ..."
      }
    """

    # ======================== Runtime fields ======================== #
    diagram: Dict[str, Any] = RuntimeField(
        description=(
            "Single diagram specification dict with keys: 'topology', 'initial', "
            "'final', 'vertices', and optionally 'propagators', 'couplings'. "
            "Particle labels: nu_X for neutrinos, nu_X_bar for antineutrinos (ν̄). "
            "Use this OR diagram_classes OR initial/final, not multiple."
        )
    )
    include_summary: bool = RuntimeField(
        description="Include formatted markdown summary in output (default: True)",
        default=True
    )
    process_label: Optional[str] = RuntimeField(
        description="Process label for summary (e.g., 'H → bb̄'). Note: bar notation (X̄) = antiparticle = X_bar",
        default=None
    )
    reference_width: Optional[float] = RuntimeField(
        description="Reference width in GeV for comparison (e.g., experimental value)",
        default=None
    )
    reference_label: str = RuntimeField(
        description="Label for reference value (default: 'Experimental')",
        default="Experimental"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    record_finding: bool = StateField(
        default=True,
        description="Append the estimate to the findings ledger (off for batch callers)"
    )
    # ================================================================ #

    def _setup(self):
        """Validate and initialize."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _infer_propagator_spin(self, label: str) -> float:
        """
        Infer propagator spin from particle label.

        This is a fallback when spin is not explicitly specified.
        Uses common particle naming conventions.

        Args:
            label: Particle label (e.g., "W", "Z", "gamma", "t", "H")

        Returns:
            Inferred spin value
        """
        label_lower = label.lower()

        # Scalars (spin 0)
        scalar_labels = {"h", "higgs", "phi", "s", "a0", "h0", "h+", "h-", "charged_higgs"}
        if label_lower in scalar_labels or label_lower.startswith("scalar"):
            return 0.0

        # Vectors (spin 1)
        vector_labels = {"w", "w+", "w-", "z", "gamma", "photon", "g", "gluon", "a", "z'", "w'"}
        if label_lower in vector_labels or label_lower.startswith("vector"):
            return 1.0

        # Fermions (spin 1/2) - quarks, leptons, and common BSM fermions
        fermion_labels = {
            # Quarks
            "u", "d", "s", "c", "b", "t", "top", "bottom", "charm", "strange",
            "ubar", "dbar", "sbar", "cbar", "bbar", "tbar",
            # Leptons
            "e", "mu", "tau", "electron", "muon", "tauon",
            "nu_e", "nu_mu", "nu_tau", "nue", "numu", "nutau",
            "e+", "e-", "mu+", "mu-", "tau+", "tau-",
            # BSM fermions
            "chi", "chi0", "chi+", "chi-", "neutralino", "chargino",
            "n1", "n2", "n3", "n4", "c1", "c2",  # MSSM naming
        }
        if label_lower in fermion_labels or label_lower.startswith("fermion"):
            return 0.5

        # Gravitino (spin 3/2)
        if label_lower in {"gravitino", "psi_3/2"}:
            return 1.5

        # Graviton (spin 2)
        if label_lower in {"graviton", "g_mu_nu"}:
            return 2.0

        # Default: assume fermion (most common in BSM)
        return 0.5

    def _calculate_propagator_suppression(
        self,
        propagators: list,
        mother_mass: float,
        n_body: int,
        final_masses: Optional[list] = None,
    ) -> tuple[float, str, list, list]:
        """
        Calculate propagator suppression factor.

        Args:
            propagators: List of Propagator objects
            mother_mass: Mother particle mass in GeV
            n_body: Number of final state particles

        Returns:
            (suppression_factor, latex_string, regime_info, regime_types)
            - regime_types: List of regime strings ("heavy", "light", "intermediate")
              for use in formula building

        Note on loop propagators:
            For loop diagrams, the propagator suppression is handled differently.
            The loop integral (giving 1/16π²) already accounts for the loop structure.
            We use a single effective mass suppression ~1/M² for the dominant loop mass,
            not 1/M² per propagator. This matches the QuickNDA approach and standard NDA.
        """
        if not propagators:
            return (1.0, "", [], [])

        # Separate tree and loop propagators
        tree_props = [p for p in propagators if not getattr(p, 'is_loop_propagator', False)]
        loop_props = [p for p in propagators if getattr(p, 'is_loop_propagator', False)]

        # Estimate typical momentum
        n_tree = len(tree_props)
        q_sq_typical = (mother_mass / math.sqrt(n_tree + 1))**2 if n_tree > 0 else mother_mass**2

        total_suppression = 1.0
        latex_parts = []
        regime_info = []
        regime_types = []

        # Handle loop propagators: use single effective mass suppression
        if loop_props:
            # Use the dominant (heaviest) loop mass for effective suppression
            loop_masses = [p.mass for p in loop_props if p.mass > 0]
            if loop_masses:
                M_loop_eff = max(loop_masses)  # Dominant mass scale
                loop_prop_factor = 1.0 / (M_loop_eff**2)
                total_suppression *= loop_prop_factor
                latex_parts.append(f"M_{{{M_loop_eff:.0f}}}^{{-2}}")
                regime_info.append(f"loop (M_eff={M_loop_eff:.1f}, {len(loop_props)} propagators)")
                regime_types.append("loop")

        # Handle tree propagators individually
        for prop in tree_props:
            M_prop = prop.mass
            regime = prop.regime
            Gamma_prop = prop.width or 0.0

            # Known far-side daughters: the allowed virtuality interval fixes
            # |P(q^2)|^2 (a Breit-Wigner average for a line with a width,
            # otherwise the typical virtuality).
            if prop.final_legs is not None and final_masses and None not in final_masses:
                lo, hi = kinematics.virtuality_range(mother_mass, final_masses, prop.final_legs)
                prop_factor = kinematics.propagator_factor(M_prop, Gamma_prop, lo, hi)
                total_suppression *= prop_factor
                latex_parts.append(f"|q^2 - M_{{{M_prop:.0f}}}^2|^{{-2}}")
                regime_info.append(f"virtuality {math.sqrt(lo):.3g} <= sqrt(q^2) <= {math.sqrt(hi):.3g} GeV (M={M_prop:.1f})")
                regime_types.append("virtuality")
                continue

            # Determine regime if auto
            if regime == "auto":
                ratio = M_prop / mother_mass if mother_mass > 0 else 0
                if ratio > 5.0:
                    regime = "heavy"
                elif ratio < 0.2:
                    regime = "light"
                else:
                    regime = "intermediate"

            # Calculate suppression
            # NOTE: This is the propagator contribution to |M|² (squared amplitude).
            # For a propagator 1/(q² - M²) in the amplitude, the squared amplitude gets
            # |1/(q² - M²)|² = 1/(q² - M²)². For heavy propagators (M >> q), this is 1/M⁴.
            if regime == "heavy":
                # Far off-shell: |propagator|² ~ 1/M⁴
                prop_factor = 1.0 / (M_prop**4)
                latex_prop = f"M_{{{M_prop:.0f}}}^{{-4}}"
                regime_str = f"heavy (M={M_prop:.1f} >> {mother_mass:.1f})"

            elif regime == "light":
                # Massless/light limit: |propagator|² ~ 1/q⁴
                prop_factor = 1.0 / (q_sq_typical**2)
                latex_prop = f"q^{{-4}}"
                regime_str = f"light (M={M_prop:.1f} << {mother_mass:.1f})"

            else:  # intermediate
                # Full propagator: |1/(q² - M² + iMΓ)|²
                denominator_sq = (q_sq_typical - M_prop**2)**2 + (Gamma_prop * M_prop)**2
                prop_factor = 1.0 / max(denominator_sq, 1e-20)
                latex_prop = f"|q^2 - M_{{{M_prop:.0f}}}^2|^{{-2}}"
                regime_str = f"intermediate (M={M_prop:.1f} ~ {mother_mass:.1f})"

            total_suppression *= prop_factor
            latex_parts.append(latex_prop)
            regime_info.append(regime_str)
            regime_types.append(regime)

        latex_string = " \\times ".join(latex_parts)
        return (total_suppression, latex_string, regime_info, regime_types)

    def _calculate_loop_factor(self, diagram: Diagram) -> tuple[float, Dict[str, Any]]:
        """
        Calculate loop suppression factor.

        Args:
            diagram: Diagram object

        Returns:
            (loop_factor, loop_info_dict)
        """
        # Check for loop propagators
        loop_props = [p for p in diagram.propagators if p.is_loop_propagator]

        if not loop_props:
            return (1.0, {})

        # For now: simple 1/(16π²) per loop
        # TODO: More sophisticated loop analysis
        n_loops = 1  # Infer from propagator structure
        loop_factor = (1.0 / (16.0 * math.pi**2)) ** n_loops

        loop_info = {
            "n_loops": n_loops,
            "loop_factor": loop_factor,
            "loop_propagators": len(loop_props),
            "note": "Applied 1/(16π²) loop suppression"
        }

        return (loop_factor, loop_info)

    def _calculate_kinematics_improved_estimate(
        self,
        diagram: Diagram,
        mother_mass: float,
        total_coupling_sq: float,
        n_body: int,
        color_factor: float,
        nda_width: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate kinematics-improved width estimate using arXiv:1402.1178 methods.

        This method implements more accurate kinematics from the MadWidth paper:
        - Energy distribution: E_i = (M - Σm_j)/N + m_i (kinetic energy equally shared)
        - Spin-dependent propagator factors (Table 1)
        - Spin-dependent polarization sums (Table 1)
        - Phase space fudge factor c_ps = 0.8 (Eq. 10)

        Args:
            diagram: Parsed diagram object
            mother_mass: Mother particle mass M in GeV
            total_coupling_sq: |coupling|² (squared coupling product)
            n_body: Number of final state particles
            color_factor: Color factor N_c

        Returns:
            Dictionary with improved estimate details
        """
        final_particles = diagram.final
        propagators = diagram.propagators
        mother_spin = diagram.initial[0].spin

        # ========================================================================
        # SPECIAL CASE: Contact interactions (no propagators)
        # For 4-fermion contacts, the matrix element has angular correlations
        # like (p_μ · p_ν)(p_e · p_ν') that aren't captured by simple 2E products.
        # In this case, just apply the phase space fudge factor to NDA.
        # ========================================================================
        vertex_types = [v.type.lower() for v in diagram.vertices]
        is_contact = len(propagators) == 0
        is_4fermion = any("4fermion" in vt or "fermi" in vt for vt in vertex_types)

        if is_contact and is_4fermion:
            # For contact 4-fermion operators, use simplified approach:
            # Γ_improved = Γ_NDA × c_ps^(n-2)
            # This accounts for phase space corrections without trying to
            # "improve" the matrix element kinematics (which don't factorize).
            fudge_power = max(0, n_body - 2)
            correction_factor = C_PS_FUDGE ** fudge_power

            # Use the passed NDA width directly
            width_improved_contact = nda_width * correction_factor

            return {
                "width_gev_improved": width_improved_contact,
                "note": "Contact interaction: phase space fudge factor applied to NDA",
                "method": "contact_simplified",
                "details": {
                    "fudge_factor": C_PS_FUDGE,
                    "fudge_power": fudge_power,
                    "correction_factor": correction_factor,
                    "nda_reference": nda_width
                }
            }

        # ========================================================================
        # STEP 1: Calculate final state energies using paper's prescription
        # E_i = (M - Σm_j)/N + m_i  (kinetic energy equally distributed)
        # ========================================================================
        final_masses = [p.mass if p.mass is not None else 0.0 for p in final_particles]
        total_final_mass = sum(final_masses)
        available_kinetic = mother_mass - total_final_mass

        if available_kinetic <= 0:
            # Kinematically forbidden or at threshold
            return {
                "width_gev_improved": 0.0,
                "note": "Kinematically forbidden or at threshold",
                "details": {}
            }

        # Energy for each final state particle
        final_energies = []
        for m_i in final_masses:
            E_i = available_kinetic / n_body + m_i
            final_energies.append(E_i)

        # ========================================================================
        # STEP 2: Calculate polarization tensor factors for external particles
        # From Table 1: P(E) for external particles (spin-dependent)
        # ========================================================================
        def polarization_factor(spin: float, energy: float, mass: float) -> float:
            """
            Calculate polarization tensor factor from Table 1.

            For external particles, this is the spin sum factor.
            """
            if spin == 0:
                # Scalar: polarization = 1
                return 1.0
            elif spin == 0.5:
                # Fermion: polarization = 2E
                return 2.0 * energy
            elif spin == 1:
                # Vector: polarization = f(E,M) = 1 + E²/M² (massive), 1 (massless)
                if mass > 0:
                    return 1.0 + (energy**2) / (mass**2)
                else:
                    return 1.0  # Massless vector (photon, gluon)
            elif spin == 1.5:
                # Spin-3/2: polarization = 2E × f(E,M)
                f_EM = 1.0 + (energy**2) / (mass**2) if mass > 0 else 1.0
                return 2.0 * energy * f_EM
            elif spin == 2:
                # Graviton: polarization = f(E,M)²
                f_EM = 1.0 + (energy**2) / (mass**2) if mass > 0 else 1.0
                return f_EM ** 2
            else:
                # Default: assume 2s+1 degrees of freedom
                return 2.0 * spin + 1.0

        # External polarization product (includes ALL external particles)
        # For decay: mother + all final state particles
        ext_polarization = 1.0

        # Mother particle contribution (at rest, E = M)
        mother_spin_val = mother_spin if mother_spin is not None else 0
        mother_mass_val = diagram.initial[0].mass if diagram.initial[0].mass else mother_mass
        ext_polarization *= polarization_factor(mother_spin_val, mother_mass, mother_mass_val)

        # A massive final vector's longitudinal growth E^2/M^2 needs a source:
        # by the equivalence theorem the longitudinal state couples as the
        # Goldstone boson, in proportion to the masses at the vertex that
        # emits it. Without a fermion or scalar at that vertex at least as
        # heavy as the vector (a gauge self-coupling, or light fermions), the
        # growth cancels between diagrams and the factor keeps its value at
        # rest, 2.
        def leg_spin_mass(name):
            kind, _, k = name.partition(":")
            k = int(k)
            if kind == "in":
                return (mother_spin if mother_spin is not None else 0), mother_mass
            if kind == "out":
                q = final_particles[k]
                return (q.spin if q.spin is not None else 0.5), final_masses[k]
            q = diagram.propagators[k]
            return (q.spin if q.spin is not None else 0), (q.mass or 0.0)

        unsourced_vectors = []
        for i, p in enumerate(final_particles):
            if p.spin != 1 or not final_masses[i]:
                continue
            emitting = [v for v in diagram.vertices if v.legs and f"out:{i}" in v.legs]
            if len(emitting) != 1:
                continue
            others = [leg_spin_mass(x) for x in emitting[0].legs if x != f"out:{i}"]
            if not any(sp in (0, 0.5) and m >= final_masses[i] for sp, m in others):
                unsourced_vectors.append(i)

        # Final state particles
        for i, p in enumerate(final_particles):
            spin_i = p.spin if p.spin is not None else 0.5
            E_i = final_energies[i]
            m_i = final_masses[i]
            ext_polarization *= 2.0 if i in unsourced_vectors else polarization_factor(spin_i, E_i, m_i)

        # Mother particle spin averaging (1/N_s where N_s = 2s+1)
        mother_spin_states = int(2 * mother_spin_val + 1)
        spin_avg = 1.0 / mother_spin_states

        # ========================================================================
        # STEP 3: Calculate propagator factors with spin-dependent numerators
        # From Table 1: Propa(E) for internal lines
        # ========================================================================
        def propagator_factor_improved(
            prop_spin: float,
            E_prop: float,
            M_prop: float,
            Gamma_prop: float,
            averaged_denominator: Optional[float] = None,
        ) -> float:
            """
            Calculate |Propagator(E)|² with spin-dependent numerator from Table 1.

            The propagator structure is: numerator / (E² - M² + iMΓ)
            |P|² = |numerator|² / [(E² - M²)² + (MΓ)²]
            """
            # Denominator: |(E² - M² + iMΓ)|² = (E² - M²)² + (MΓ)²
            denom = (E_prop**2 - M_prop**2)**2 + (M_prop * Gamma_prop)**2

            # For very small denominator (near resonance), use minimum
            if denom < 1e-20:
                denom = 1e-20
            if averaged_denominator is not None:
                denom = 1.0 / averaged_denominator

            # Spin-dependent numerator from Table 1
            if prop_spin == 0:
                # Scalar: numerator = 1
                numerator_sq = 1.0
            elif prop_spin == 0.5:
                # Fermion: numerator = E
                numerator_sq = E_prop**2
            elif prop_spin == 1:
                # Vector: numerator = (1 - E²/M²) for massive
                if M_prop > 0:
                    numerator_sq = (1.0 - E_prop**2 / M_prop**2)**2
                else:
                    # Massless vector: 1/E² scaling
                    numerator_sq = 1.0
            elif prop_spin == 1.5:
                # Spin-3/2: numerator = (2/3)E(1 - E²/M²)
                if M_prop > 0:
                    numerator_sq = ((2.0/3.0) * E_prop * (1.0 - E_prop**2 / M_prop**2))**2
                else:
                    numerator_sq = ((2.0/3.0) * E_prop)**2
            elif prop_spin == 2:
                # Graviton: numerator = (7/6 - 4E²/3M² + 2E⁴/3M⁴)
                if M_prop > 0:
                    x = E_prop**2 / M_prop**2
                    numerator_sq = (7.0/6.0 - (4.0/3.0)*x + (2.0/3.0)*x**2)**2
                else:
                    numerator_sq = (7.0/6.0)**2
            else:
                # Default: scalar-like
                numerator_sq = 1.0

            return numerator_sq / denom

        # Calculate propagator energies and factors
        prop_factor_improved = 1.0
        prop_details = []

        for prop in propagators:
            M_prop = prop.mass
            Gamma_prop = prop.width if prop.width is not None else 0.0

            # Use propagator spin if specified, otherwise infer from label or default
            prop_spin = prop.spin if prop.spin is not None else self._infer_propagator_spin(prop.label)

            # Estimate propagator energy from final state energies it connects to
            # Simple approximation: use average of final state energies
            E_prop = sum(final_energies) / n_body if n_body > 0 else mother_mass / 2
            averaged = None
            interval = None
            if prop.final_legs is not None:
                lo, hi = kinematics.virtuality_range(mother_mass, final_masses, prop.final_legs)
                E_prop = math.sqrt((lo + hi) / 2)
                averaged = kinematics.propagator_factor(M_prop, Gamma_prop, lo, hi)
                interval = [lo, hi]

            # For heavy propagators (M_prop >> mother_mass), the propagator is far off-shell
            # The relevant energy scale is the mother mass (typical momentum transfer), NOT M_prop
            # Using E_prop ~ M_prop would wrongly zero out vector propagator numerators
            # Keep E_prop at the mother mass scale for heavy propagators
            # (Note: the 1/M^4 suppression is handled in the denominator correctly)

            pf = propagator_factor_improved(prop_spin, E_prop, M_prop, Gamma_prop, averaged)
            prop_factor_improved *= pf

            prop_details.append({
                "label": prop.label,
                "mass": M_prop,
                "width": Gamma_prop,
                "spin": prop_spin,
                "energy_estimate": E_prop,
                "virtuality": interval,
                "breit_wigner_average": interval is not None and kinematics.averages(M_prop, Gamma_prop, *interval),
                "factor": pf
            })

        # ========================================================================
        # STEP 4: Calculate phase space with fudge factor
        # Apply c_ps = 0.8 to improve estimate (Eq. 10)
        # ========================================================================
        # Use existing phase space calculation but apply fudge factor
        # For n-body: Φ_n with c_ps correction
        ps_tool = EstimatePhaseSpaceTool(
            mother_mass_gev=mother_mass,
            n_body=n_body,
            final_state_masses_gev=final_masses if any(m > 0 for m in final_masses) else None,
            base_directory=self.base_directory
        )
        ps_result = json.loads(ps_tool._run())

        if ps_result.get("status") != "ok":
            return {
                "width_gev_improved": 0.0,
                "note": "Phase space calculation failed",
                "details": {}
            }

        phi_n_base = ps_result["phase_space_volume"]

        # Apply fudge factor: for n > 2 body, apply c_ps^(n-2) factor
        # From Eq. (10), each recursive step gets c_ps
        if n_body > 2:
            fudge_power = n_body - 2
            phi_n_improved = phi_n_base * (C_PS_FUDGE ** fudge_power)
        else:
            phi_n_improved = phi_n_base

        # ========================================================================
        # STEP 5: Combine into improved width estimate
        # Γ = (1/2M) × Φ_n × <|M|²>
        # <|M|²> = (N_color/N_s) × |∏Propa|² × ∏P_ext × |Σ C_i Lorentz_i|²
        #
        # From arXiv:1402.1178 Eq. (11):
        # - N_color = color factor
        # - N_s = mother spin states (for averaging)
        # - |∏Propa|² = product of squared propagators (spin-dependent)
        # - ∏P_ext = product of external polarization tensors (Table 1)
        # - |Σ C_i Lorentz_i|² = squared vertex factor (coupling × simplified Lorentz)
        #
        # For renormalizable vertices, Lorentz structures (γ^μ, etc.) → 1
        # so vertex factor = coupling, except where the coupling multiplies a
        # momentum (below). Other energy dependence is in ∏P_ext.
        #
        # For non-renormalizable (dim > 4), vertex has momentum factors:
        # - dim-5: vertex ∝ E (e.g., magnetic moment)
        # - dim-6: vertex ∝ E² (e.g., 4-fermion contact from EFT)
        # ========================================================================

        # Determine operator dimension for energy scaling in vertex factor
        n_vertices = len(diagram.vertices)
        vertex_types = [v.type.lower() for v in diagram.vertices]

        is_4fermion_contact = any("4fermion" in vt or "fermi" in vt for vt in vertex_types)

        # A renormalizable vertex is constant except where the coupling
        # multiplies a momentum: three vectors (the gauge self-coupling) or two
        # scalars and a vector. Each such vertex gives |M|^2 a factor E_v^2,
        # E_v the largest energy among its legs (the parent's mass, a final
        # energy, or an internal line's sqrt of mean virtuality). 4-fermion
        # contact operators carry their E^4 in the external polarizations.
        vertex_energy_factor = 1.0
        derivative_vertices = []
        if not is_4fermion_contact:
            def leg(name):
                kind, _, k = name.partition(":")
                k = int(k)
                if kind == "in":
                    return mother_spin_val, mother_mass
                if kind == "out":
                    p = final_particles[k]
                    return (p.spin if p.spin is not None else 0.5), final_energies[k]
                return prop_details[k]["spin"], prop_details[k]["energy_estimate"]
            for v in diagram.vertices:
                if not v.legs or len(v.legs) != 3:
                    continue
                legs = [leg(x) for x in v.legs]
                spins = sorted(sp for sp, _ in legs)
                if spins in ([1, 1, 1], [0, 0, 1]):
                    energy = max(e for _, e in legs)
                    derivative_vertices.append({"legs": v.legs, "energy": energy})
                    vertex_energy_factor *= energy ** 2

        # ====================================================================
        # Back-to-back kinematic correction for 2-body fermionic decays
        # ====================================================================
        # The NDA polarization factor for fermions is P(E) = 2E, so for two
        # final-state fermions the product is P₁P₂ = 4E₁E₂. The actual
        # spin-summed trace depends on the vertex Lorentz structure:
        #
        #   Unprojected (yukawa, vector, va, ...):
        #     Tr(p̸₁ p̸₂) = 4 p₁·p₂.  For back-to-back massless particles
        #     p₁·p₂ = 2E₁E₂, so the trace = 8E₁E₂ = 2× NDA.  Needs ×2.
        #
        #   Chiral-projected (left-handed, right-handed, chiral):
        #     Tr(γᵘ PL p̸₁ γᵛ PL p̸₂) = 2 p₁·p₂ terms.  For back-to-back
        #     this gives 4E₁E₂ = exactly the NDA estimate.  No correction.
        #
        # The PL/PR projector halves the trace, which exactly cancels the
        # back-to-back kinematic enhancement.
        #
        # For n ≥ 3 body decays, final-state momenta aren't back-to-back,
        # and the average ⟨p_i·p_j⟩ over phase space is closer to E_iE_j,
        # so no correction is needed.
        _CHIRAL_VERTEX_TYPES = {
            "left-handed", "right-handed",
            "chiral", "vector-chiral", "yukawa-chiral", "scalar-chiral",
            "tensor-chiral", "dipole-chiral",
        }

        backtoback_correction = 1.0
        if n_body == 2:
            n_final_fermions = sum(
                1 for p in final_particles
                if p.spin is not None and p.spin == 0.5
            )
            if n_final_fermions == 2:
                # Only apply correction for unprojected trace structures.
                # Chiral vertices (PL/PR projected) already match the NDA.
                # Resolve first: a valence suffix ("left-handed-3pt", as
                # EnumerateDiagrams writes) must not hide a chiral vertex.
                all_chiral = all(
                    resolve_nda_interaction_type(vt) in _CHIRAL_VERTEX_TYPES for vt in vertex_types
                )
                if not all_chiral:
                    backtoback_correction = 2.0

        me_sq_improved = (
            color_factor *
            spin_avg *
            ext_polarization *
            total_coupling_sq *
            vertex_energy_factor *
            prop_factor_improved *
            backtoback_correction
        )

        # Final width, with the identical-particle factor (as the raw estimate;
        # the contact path above starts from the raw width, which has it)
        from tools.nda.simple_diagram import compute_symmetry_factor
        symmetry = compute_symmetry_factor(diagram)
        width_improved = (1.0 / (2.0 * mother_mass)) * phi_n_improved * me_sq_improved / symmetry

        return {
            "width_gev_improved": width_improved,
            "note": "Kinematics-improved estimate using arXiv:1402.1178 methods",
            "details": {
                "final_energies_gev": final_energies,
                "external_polarization": ext_polarization,
                "spin_averaging": spin_avg,
                "vertex_energy_factor": vertex_energy_factor,
                "derivative_vertices": derivative_vertices,
                "unsourced_vectors": unsourced_vectors,
                "propagator_factor": prop_factor_improved,
                "propagator_details": prop_details,
                "phase_space_base": phi_n_base,
                "phase_space_improved": phi_n_improved,
                "fudge_factor": C_PS_FUDGE,
                "backtoback_correction": backtoback_correction,
                "matrix_element_sq": me_sq_improved,
                "n_body": n_body,
                "mother_mass": mother_mass,
                "mother_spin": mother_spin_val,
                "final_masses_gev": final_masses,
                "final_spins": [p.spin if p.spin is not None else 0.5 for p in final_particles],
                "color_factor": color_factor,
                "coupling_sq": total_coupling_sq,
                "symmetry_factor": symmetry,
            }
        }

    def _narrow_width(self, diagram: Dict[str, Any], parsed: Diagram, mother_mass: float,
                      final_masses: list) -> Optional[Dict[str, Any]]:
        """Narrow-width estimate: for each outermost line R that can go on shell,
        Gamma(parent -> rest + R) Gamma(R -> S) / Gamma_R, each factor estimated
        by this tool (inner resonances recursively); separate outermost lines
        (e.g. either Z in H -> Z Z*) add, one on shell at a time. None when no
        line resonates or a factor cannot be estimated."""
        lines = kinematics.resonant_lines(diagram, mother_mass, final_masses)
        if not lines:
            return None
        sides = kinematics.far_side(diagram)
        outermost = [j for j in lines
                     if not any(k != j and set(sides[j]) < set(sides[k]) for k in lines)]
        from tools.nda.simple_diagram import compute_symmetry_factor
        symmetry = compute_symmetry_factor(parsed)
        terms = []
        for j in outermost:
            R = parsed.propagators[j]
            production, decay = kinematics.split_at(diagram, j)
            decay_color = R.decay_color_factor
            if decay_color is None:
                decay_color = infer_color_factor([parsed.final[i] for i in sides[j]])
            decay["color_factor"] = decay_color
            production["color_factor"] = parsed.color_factor / decay_color
            widths = []
            for sub in (production, decay):
                out = json.loads(type(self)(diagram=sub, include_summary=False, record_finding=False,
                                            base_directory=self.base_directory)._run())
                if out.get("status") != "ok" or not out.get("width_gev"):
                    return None
                widths.append(out)
            terms.append({
                "width_gev": widths[0]["width_gev"] * widths[1]["width_gev"] / R.width / symmetry,
                "resonance": {"label": R.label, "mass": R.mass, "width": R.width, "final_legs": sides[j]},
                "production": widths[0],
                "decay": widths[1],
            })
        return {
            "width_gev": sum(t["width_gev"] for t in terms),
            "resonances": terms,
            "formula": r"\Gamma \approx \sum_R \Gamma(\text{parent} \to X\,R)\,\Gamma(R \to S)/\Gamma_R",
        }

    def _resolve_coupling_value(self, vertex: Vertex, couplings: Dict[str, float]) -> float:
        """
        Resolve coupling value from vertex specification.

        Args:
            vertex: Vertex object
            couplings: Dictionary of coupling values

        Returns:
            Numerical coupling value

        Raises:
            ValueError: If coupling cannot be resolved
        """
        coupling = vertex.coupling

        if isinstance(coupling, (int, float)):
            return float(coupling)

        if isinstance(coupling, str):
            if coupling in couplings:
                return couplings[coupling]
            else:
                raise ValueError(f"Coupling '{coupling}' not found in couplings dict")

        if isinstance(coupling, dict):
            resolved = []
            for key, val in coupling.items():
                if isinstance(val, (int, float)):
                    resolved.append(float(val))
                elif isinstance(val, str):
                    if val in couplings:
                        resolved.append(couplings[val])
                    else:
                        raise ValueError(
                            f"Dict coupling '{key}': '{val}' not found in couplings dict"
                        )
                else:
                    raise ValueError(
                        f"Invalid dict coupling component '{key}': {val}"
                    )
            return math.sqrt(sum(v**2 for v in resolved))

        raise ValueError(f"Invalid coupling specification: {coupling}")

    def _run(self) -> str:
        """
        Main execution method.

        Returns:
            JSON string with decay width estimate
        """
        try:
            self._setup()
        except Exception as e:
            return self.format_error(
                error="Setup Error",
                reason=str(e)
            )

        # Validate diagram input
        if self.diagram is None:
            return self.format_error(
                error="Missing Input",
                reason="'diagram' parameter is required",
                suggestion=(
                    "Provide a diagram dict with 'initial', 'final', 'vertices', "
                    "and optionally 'propagators', 'couplings'. "
                    "Use EnumerateDiagrams to auto-generate diagram dicts from particle labels."
                )
            )

        # Unwrap EnumerateDiagrams output format if needed
        diagram = self.diagram
        if "diagram" in diagram and "initial" not in diagram:
            diagram = diagram["diagram"]

        # Parse diagram
        try:
            parsed_diagram = parse_diagram(diagram)
        except Exception as e:
            return self.format_error(
                error="Diagram Parse Error",
                reason=str(e),
                suggestion="Check diagram follows simple format (see docs)"
            )

        # Validate diagram (physics consistency)
        is_valid, warnings = parsed_diagram.validate()
        if not is_valid:
            return self.format_error(
                error="Diagram Validation Failed",
                reason="; ".join(warnings),
                suggestion="Fix diagram structure or missing fields"
            )

        # CRITICAL: Validate graph theory constraint I = (Σₙ n·Vₙ - E)/2
        # This is a hard mathematical requirement for any valid Feynman diagram
        is_valid_graph, graph_error = validate_graph_theory_constraint(self.diagram)
        if not is_valid_graph:
            return self.format_error(
                error="Graph Theory Violation",
                reason=graph_error,
                suggestion=(
                    "The diagram structure violates fundamental graph theory. "
                    "This typically means the number of propagators doesn't match "
                    "the vertex structure and external lines. Check the error message "
                    "for the required number of propagators."
                )
            )

        # Extract key parameters
        mother = parsed_diagram.initial[0]
        mother_mass = mother.mass or parsed_diagram.energy_scale
        if mother_mass is None:
            return self.format_error(
                error="Missing Energy Scale",
                reason="Either initial particle mass or energy_scale must be specified"
            )

        mother_spin_value = mother.spin
        if mother_spin_value is None:
            return self.format_error(
                error="Missing Mother Spin",
                reason="Initial particle must have spin specified"
            )

        n_body = len(parsed_diagram.final)
        final_spins = [p.spin for p in parsed_diagram.final]
        if None in final_spins:
            return self.format_error(
                error="Missing Final State Spin",
                reason="All final state particles must have spin specified"
            )

        final_masses = [p.mass for p in parsed_diagram.final]

        # Connectivity (optional): each propagator's far-side daughters, and a
        # narrow-width factorization when a line can go on shell.
        narrow = None
        if kinematics.has_connectivity(diagram) and None not in final_masses:
            try:
                for prop, legs in zip(parsed_diagram.propagators, kinematics.far_side(diagram)):
                    if prop.final_legs is None:
                        prop.final_legs = legs
                narrow = self._narrow_width(diagram, parsed_diagram, mother_mass, final_masses)
            except ValueError as e:
                return self.format_error(error="Connectivity Error", reason=str(e))

        # ====================================================================
        # STEP 1: Phase Space
        # ====================================================================
        ps_tool = EstimatePhaseSpaceTool(
            mother_mass_gev=mother_mass,
            n_body=n_body,
            final_state_masses_gev=final_masses if all(m is not None for m in final_masses) else None,
            base_directory=self.base_directory
        )

        ps_result_str = ps_tool._run()
        try:
            ps_result = json.loads(ps_result_str)
        except json.JSONDecodeError:
            return ps_result_str  # Return error

        if ps_result.get("status") != "ok":
            return ps_result_str  # Propagate error

        phi_n = ps_result["phase_space_volume"]

        # ====================================================================
        # STEP 2: Matrix Element
        # ====================================================================

        # Combine coupling values from all vertices. The raw estimate counts
        # energies by vertex, so a dimensionful coupling enters in units of the
        # decaying mass; the improved estimate carries the energy dependence in
        # its polarizations and propagators, so it takes couplings as given.
        total_coupling = 1.0
        total_coupling_physical = 1.0
        interaction_types = []

        for vertex in parsed_diagram.vertices:
            try:
                coupling_val = self._resolve_coupling_value(vertex, parsed_diagram.couplings)
                # A dimensionful coupling counts in units of the decaying mass.
                total_coupling *= coupling_val / mother_mass ** vertex.coupling_dimension
                total_coupling_physical *= coupling_val
                interaction_types.append(resolve_nda_interaction_type(vertex.type))
            except ValueError as e:
                return self.format_error(
                    error="Coupling Resolution Error",
                    reason=str(e)
                )

        # Use first vertex type as representative, resolved to NDA name
        representative_interaction = resolve_nda_interaction_type(parsed_diagram.vertices[0].type)

        # Get diagram structure info
        n_vertices = len(parsed_diagram.vertices)
        n_propagators = len(parsed_diagram.propagators)

        # Convert spins to UFO codes
        mother_spin_ufo = spin_to_ufo(mother_spin_value)
        final_spin_ufos = [spin_to_ufo(s) for s in final_spins]

        # Map float spins to names for matrix element tool
        spin_map = {0: "scalar", 0.5: "fermion", 1: "vector", 2: "graviton"}
        mother_spin_name = spin_map.get(mother_spin_value, "scalar")
        final_spin_names = [spin_map.get(s, "fermion") for s in final_spins]

        # Infer operator dimension from vertex type scaling data
        interaction_info = INTERACTION_TYPES.get(representative_interaction, {})
        operator_dim = interaction_info.get("op_dim", 4)

        # Calculate matrix element
        # For multi-vertex diagrams, we need to account for proper energy scaling
        if n_vertices == 1:
            # Single vertex: use matrix element tool as-is
            me_tool = EstimateMatrixElementTool(
                interaction_type=representative_interaction,
                mother_spin=mother_spin_name,
                final_state_spins=final_spin_names,
                coupling_value=total_coupling,
                energy_scale_gev=mother_mass,
                operator_dimension=operator_dim,
                cutoff_scale_gev=1000.0,
                color_factor=parsed_diagram.color_factor,
                base_directory=self.base_directory
            )

            me_result_str = me_tool._run()
            try:
                me_result = json.loads(me_result_str)
            except json.JSONDecodeError:
                return me_result_str  # Return error

            if me_result.get("status") != "ok":
                return me_result_str  # Propagate error

            me_sq = me_result["matrix_element_sq"]
            coupling_sym = me_result.get("coupling", "g")
            # Build coupling counts for single-vertex case
            _coupling_counts = {coupling_sym: 2}
        else:
            # Multi-vertex: calculate directly with correct energy scaling
            # We already have total_coupling = product of all couplings

            # Spin averaging/summing
            mother_spin_ufo = spin_to_ufo(mother_spin_value)
            spin_avg_factor = 1.0 / mother_spin_ufo
            spin_sum_factor = 1.0
            for s in final_spins:
                spin_sum_factor *= spin_to_ufo(s)

            # Check if this is a loop diagram
            has_loop_propagators = any(
                getattr(p, 'is_loop_propagator', False)
                for p in parsed_diagram.propagators
            )

            # Energy factor depends on diagram type
            if has_loop_propagators:
                # For LOOP diagrams:
                # The amplitude A ~ (couplings) × (loop form factor)
                # The loop form factor contains energy dependence from the loop integral
                # This is captured by the 1/(16π²) loop factor and propagator suppression
                # Matrix element: |M|² ~ (couplings)² × M² (standard dimensional analysis)
                energy_power = 2
            elif operator_dim == 4:
                # For TREE-LEVEL multi-vertex: |M|² ~ g^(2*n) × E^(2*n)
                energy_power = 2 * n_vertices
            elif operator_dim == 6:
                energy_power = 4  # dim-6 effective operator
            else:
                energy_power = 2 * operator_dim - 4

            energy_factor = mother_mass ** energy_power

            # Matrix element squared
            me_sq = (
                spin_avg_factor *
                spin_sum_factor *
                parsed_diagram.color_factor *
                (total_coupling ** 2) *
                energy_factor
            )

            # Collect distinct coupling symbols from all vertices
            _coupling_counts: dict[str, int] = {}
            for _v in parsed_diagram.vertices:
                _c = _v.coupling
                if isinstance(_c, str):
                    _coupling_counts[_c] = _coupling_counts.get(_c, 0) + 2
                elif isinstance(_c, dict):
                    _rep = list(_c.values())[0] if _c else "g"
                    _rep = _rep if isinstance(_rep, str) else "g"
                    _coupling_counts[_rep] = _coupling_counts.get(_rep, 0) + 2
                else:
                    _coupling_counts["g"] = _coupling_counts.get("g", 0) + 2
            # For backward compat: single symbol used by ME formula line
            coupling_sym = list(_coupling_counts.keys())[0] if _coupling_counts else "g"

        # ====================================================================
        # STEP 3: Propagator Suppression
        # ====================================================================
        try:
            propagator_suppression, propagator_latex, propagator_regimes, propagator_regime_types = \
                self._calculate_propagator_suppression(
                    parsed_diagram.propagators,
                    mother_mass,
                    n_body,
                    final_masses,
                )
        except kinematics.OnShellStableLine as e:
            return self.format_error(
                error="On-Shell Intermediate Without Width",
                reason=str(e) + "; this final state is a lower-multiplicity decay followed by the decay of a particle the inputs treat as stable",
                suggestion="Give the line its width, or estimate the lower-multiplicity decay instead",
            )

        # ====================================================================
        # STEP 4: Loop Suppression
        # ====================================================================
        loop_factor, loop_info = self._calculate_loop_factor(parsed_diagram)

        # ====================================================================
        # STEP 5: Combine: Γ = (1/2M) × Φ_n × |M|² × [props] × [loops] / S
        # ====================================================================
        from tools.nda.simple_diagram import compute_symmetry_factor
        sym_factor = compute_symmetry_factor(parsed_diagram)
        width_gev = (1.0 / (2.0 * mother_mass)) * phi_n * me_sq * propagator_suppression * loop_factor / sym_factor

        # ====================================================================
        # STEP 6: Kinematics-Improved Estimate (arXiv:1402.1178)
        # ====================================================================
        # Calculate improved estimate using more accurate kinematics
        improved_estimate = self._calculate_kinematics_improved_estimate(
            diagram=parsed_diagram,
            mother_mass=mother_mass,
            total_coupling_sq=total_coupling_physical**2,
            n_body=n_body,
            color_factor=parsed_diagram.color_factor,
            nda_width=width_gev  # Pass NDA width for contact interactions
        )

        # ====================================================================
        # Build NDA Formula (unified symbolic expression)
        # ====================================================================
        # coupling_sym already set above in matrix element calculation
        ps_formula_symbolic = ps_result.get("formula", "")

        # Build symbolic matrix element formula
        if n_vertices == 1:
            # For single vertex, get from matrix element tool result
            me_formula_symbolic = me_result.get("formula", "")
        else:
            # For multi-vertex, construct it
            coupling_power_me = 2 * n_vertices
            energy_power_me = 2 * n_vertices if operator_dim == 4 else (4 if operator_dim == 6 else 2 * operator_dim - 4)
            me_formula_symbolic = rf"|M|^2 \sim {coupling_sym}^{coupling_power_me} M^{{{energy_power_me}}}"

        # n_vertices and n_propagators already defined above

        # Build unified NDA scaling formula
        # Basic structure: Γ ~ (coupling^(2*n_vertices)) * (phase_space) * (propagator_suppression)

        # Phase space scaling
        ps_power = 2*n_body - 4  # M^(2n-4) from phase space

        # Matrix element scaling (depends on operator dimension AND number of vertices)
        operator_dim = interaction_info.get("op_dim", 4)
        if operator_dim == 4:
            # Renormalizable: |M|² ~ g^(2*n_vertices) * M^(2*n_vertices)
            # Each vertex contributes g² M²
            me_mass_power = 2 * n_vertices
        elif operator_dim == 6:
            # Dimension-6 (e.g., Fermi): |M|² ~ G_F^2 * M^4
            # This is for a SINGLE effective vertex
            me_mass_power = 4
        else:
            # Generic higher-dimensional
            me_mass_power = 2 * operator_dim - 4

        # Propagator suppression scaling
        # Handle different regimes appropriately:
        # - Heavy propagators: show M_prop^4 in denominator explicitly
        # - Light/massless propagators: 1/q^4 ~ 1/M^4 (mother mass), absorbed into mass power
        # - Intermediate: treat like heavy for formula display
        prop_symbols = []  # Only for heavy/intermediate propagators
        light_prop_power = 0  # Power contribution from light propagators (absorbed into M)
        if n_propagators > 0:
            for i, prop in enumerate(parsed_diagram.propagators):
                M_prop = prop.mass
                regime = propagator_regime_types[i] if i < len(propagator_regime_types) else "heavy"

                if regime == "light":
                    # Light/massless propagators: 1/q^4 where q ~ M (mother mass)
                    # This contributes M^(-4) to the formula, absorbed into total mass power
                    light_prop_power += 4
                else:
                    # Heavy or intermediate: show explicit propagator mass scale
                    # Use particle label if mass is zero or very small (avoids M_{0})
                    if M_prop < 1.0:
                        # Use label for clearer formula (e.g., "M_γ" instead of "M_0")
                        prop_symbols.append(f"M_{{{prop.label}}}")
                    else:
                        prop_symbols.append(f"M_{{{M_prop:.0f}}}")

        # Total mass power in NUMERATOR: (1/M) * M^(ps_power) * M^(me_mass_power)
        # Light propagators contribute M^(-4) each (from 1/q^4 ~ 1/M^4)
        total_mass_power = -1 + ps_power + me_mass_power - light_prop_power

        # Build formula components
        # Canonical phase space factors
        if n_body == 2:
            pi_factor = r"16\pi"
        elif n_body == 3:
            pi_factor = r"64\pi^3"
        else:
            pi_power = 2*n_body - 3
            pi_factor = rf"\pi^{{{pi_power}}}"

        # Coupling term — preserve distinct labels from each vertex
        coupling_power = 2 * n_vertices
        _coupling_parts = []
        for _name, _power in _coupling_counts.items():
            if _power == 2:
                _coupling_parts.append(f"{_name}^2")
            else:
                _coupling_parts.append(f"{_name}^{{{_power}}}")
        coupling_term = " ".join(_coupling_parts) if _coupling_parts else f"g^{{{coupling_power}}}"

        # Mass term
        if total_mass_power == 1:
            mass_term = "M"
        elif total_mass_power == 0:
            mass_term = "1"
        elif total_mass_power < 0:
            mass_term = f"M^{{{total_mass_power}}}"
        else:
            mass_term = f"M^{{{total_mass_power}}}"

        # Propagator term (goes in denominator as M_prop^4 from |P(E)|²)
        # Combine like factors: e.g., two W propagators -> M_W^8 instead of M_W^4 M_W^4
        if prop_symbols:
            # Count occurrences of each unique symbol
            symbol_counts = Counter(prop_symbols)

            # Build combined terms
            prop_terms = []
            for sym, count in symbol_counts.items():
                power = 4 * count  # Each propagator contributes ^4
                if power == 4:
                    prop_terms.append(f"{sym}^4")
                else:
                    prop_terms.append(f"{sym}^{{{power}}}")

            prop_term = " ".join(prop_terms)
        else:
            prop_term = None

        # Construct unified formula
        if prop_term:
            formula = rf"\Gamma \sim \frac{{{coupling_term} {mass_term}}}{{{pi_factor} {prop_term}}}"
        else:
            formula = rf"\Gamma \sim \frac{{{coupling_term} {mass_term}}}{{{pi_factor}}}"

        # ====================================================================
        # Build Breakdown (symbolic formulas + numerical values)
        # ====================================================================
        breakdown = {
            "phase_space": {
                "formula": ps_formula_symbolic,
                "value": phi_n,
                "scaling": ps_result.get("scaling", "")
            },
            "matrix_element": {
                "formula": me_formula_symbolic,
                "value": me_sq,
                "interaction": representative_interaction
            }
        }

        # Add propagator info if present
        if n_propagators > 0:
            breakdown["propagators"] = {
                "formula": propagator_latex,
                "value": propagator_suppression,
                "regimes": propagator_regimes,
                "count": n_propagators
            }

        # Add loop info if present
        if loop_info:
            breakdown["loops"] = {
                "formula": rf"\frac{{1}}{{(16\pi^2)^{{{loop_info['n_loops']}}}}}",
                "value": loop_factor,
                "n_loops": loop_info["n_loops"]
            }

        # ====================================================================
        # Build full result (used for summary formatting and findings)
        # ====================================================================
        full_result = {
            "status": "ok",
            "width_gev": width_gev,
            "formula": formula,
            "breakdown": breakdown,
            "diagram": {
                "topology": parsed_diagram.topology,
                "n_vertices": n_vertices,
                "n_propagators": n_propagators,
                "loop_order": loop_info.get("n_loops", 0) if loop_info else 0,
                "interactions": interaction_types
            }
        }

        # Add kinematics-improved estimate if calculation succeeded
        if improved_estimate.get("width_gev_improved", 0) > 0:
            full_result["improved_estimate"] = {
                "width_gev": improved_estimate["width_gev_improved"],
                "method": improved_estimate.get("method", "arXiv:1402.1178 kinematics"),
                "note": improved_estimate.get("note", ""),
                "details": improved_estimate.get("details", {})
            }
            if width_gev > 0:
                full_result["improved_estimate"]["ratio_to_nda"] = (
                    improved_estimate["width_gev_improved"] / width_gev
                )

        if warnings:
            full_result["warnings"] = warnings

        # ====================================================================
        # Build lean return (what the agent gets back)
        # ====================================================================
        # Determine best estimate
        improved_width = improved_estimate.get("width_gev_improved", 0)
        if narrow is not None:
            best_width = narrow["width_gev"]
            method = "narrow width"
        elif improved_width > 0:
            best_width = improved_width
            method = "improved (arXiv:1402.1178)"
        else:
            best_width = width_gev
            method = "raw NDA"

        result = {
            "status": "ok",
            "width_gev": best_width,
            "width_gev_nda_raw": width_gev,
            "method": method,
            "formula": formula,
            "formula_type": "scaling",
            "formula_note": (
                "Formula shows parametric scaling only (coupling/mass/pi dependence). "
                "The numerical width_gev includes additional NDA prefactors "
                "(spin sums, phase space fudge factors, polarization tensors) "
                "not shown in the formula. Compare widths numerically, not via formulas."
            ),
        }
        if method.startswith("improved"):
            # The evaluated expression: every factor of the improved estimate,
            # which multiplies out to width_gev.
            evaluated = evaluated_formula(
                improved_estimate["details"], parsed_diagram,
                coupling_values=[self._resolve_coupling_value(v, parsed_diagram.couplings)
                                 for v in parsed_diagram.vertices])
            if evaluated is not None:
                result.update(formula=evaluated["formula"], formula_type="evaluated",
                              definitions=evaluated["definitions"], factors=evaluated["factors"],
                              formula_scaling=formula)
                result.pop("formula_note")
        if narrow is not None:
            result["narrow_width"] = narrow
            result["formula"] = narrow["formula"]
            result["formula_type"] = "narrow_width"
            for key in ("definitions", "factors", "formula_scaling"):
                result.pop(key, None)
        if warnings:
            result["warnings"] = warnings

        # Generate and save summary; append to findings ledger
        if self.include_summary:
            summary_text = format_decay_width_summary(
                nda_result=full_result,
                process_label=self.process_label,
                reference_width=self.reference_width,
                reference_label=self.reference_label,
                include_breakdown=True
            )

            # Save summary to file
            if self.base_directory:
                process_safe = (self.process_label or "single").replace(" ", "_").replace("→", "to").replace("->", "to")
                summary_filename = f"nda_estimate_{process_safe}.md"
                summary_filepath = os.path.join(self.base_directory, summary_filename)
                try:
                    with open(summary_filepath, 'w') as f:
                        f.write(summary_text)
                    result["summary_file"] = summary_filepath
                except Exception:
                    pass

        # Append findings (best-effort, lightweight index)
        try:
            if not self.record_finding:
                return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            process_label = self.process_label or "decay"
            entries = [f"Method: {method}"]
            entries.append(f"Width: {best_width:.4e} GeV")
            if improved_estimate.get("width_gev_improved", 0) > 0 and width_gev > 0:
                entries.append(f"Raw NDA: {width_gev:.4e} GeV")
            if self.reference_width:
                ratio = best_width / self.reference_width if self.reference_width > 0 else 0
                entries.append(
                    f"Reference ({self.reference_label}): {self.reference_width:.4e} GeV "
                    f"(ratio: {ratio:.2e})"
                )
            entries.append(f"Formula: ${formula}$")
            append_finding(self.base_directory, f"NDA: {process_label}", entries)
        except Exception:
            pass

        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
