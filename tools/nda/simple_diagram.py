"""
# simple_diagram.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Simple diagram parser and validator for NDA estimates.

Uses concise diagram syntax:
{
  "topology": "tree_2body",
  "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
  "final": [{"label": "b", "spin": "1/2", "mass": 4.2}, ...],
  "vertices": [{"type": "yukawa", "coupling": "y_b"}],
  "couplings": {"y_b": 0.03},
  "color_factor": 3.0
}
"""
import json
import math
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

try:
    from .topology import TOPOLOGIES, infer_topology, validate_topology
except ImportError:
    from topology import TOPOLOGIES, infer_topology, validate_topology


# Spin parsing
SPIN_NAMES = {
    "scalar": 0,
    "fermion": 0.5,
    "vector": 1,
    "graviton": 2,
}

SPIN_TO_UFO = {
    0: 1,      # scalar
    0.5: 2,    # fermion
    1: 3,      # vector
    2: 5,      # graviton
}


def parse_spin(spin: Union[str, int, float]) -> float:
    """
    Parse spin specification to numerical value.

    Args:
        spin: Spin spec ("scalar", "fermion", "1/2", 0, 0.5, etc.)

    Returns:
        Spin value as float

    Raises:
        ValueError: If spin is invalid
    """
    if isinstance(spin, (int, float)):
        return float(spin)

    if isinstance(spin, str):
        spin_lower = spin.lower().strip()

        # Check named spins
        if spin_lower in SPIN_NAMES:
            return SPIN_NAMES[spin_lower]

        # Check fractions
        if "/" in spin_lower:
            parts = spin_lower.split("/")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    pass

        # Try direct conversion
        try:
            return float(spin_lower)
        except ValueError:
            pass

    raise ValueError(f"Invalid spin specification: {spin}")


def spin_to_ufo(spin: float) -> int:
    """Convert spin value to UFO code (2s+1)."""
    if spin in SPIN_TO_UFO:
        return SPIN_TO_UFO[spin]
    # General formula
    return int(2 * spin + 1)


@dataclass
class Particle:
    """Represents a particle in the diagram."""
    label: Optional[str] = None
    spin: Optional[float] = None
    mass: Optional[float] = None
    color: Optional[str] = None
    # Quantum numbers (optional for backward compatibility)
    charge: Optional[float] = None              # Electric charge (units of e)
    lepton_e: Optional[float] = None            # Electron lepton number
    lepton_mu: Optional[float] = None           # Muon lepton number
    lepton_tau: Optional[float] = None          # Tau lepton number
    baryon_number: Optional[float] = None       # Baryon number
    custom_quantum_numbers: Optional[Dict[str, float]] = None  # BSM extensions

    def to_dict(self, compact: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values.

        Args:
            compact: If True, only include fields used by NDA (label, spin,
                     mass, color). Quantum numbers are omitted to reduce
                     token cost in MCP responses.
        """
        result = {}
        if self.label is not None:
            result["label"] = self.label
        if self.spin is not None:
            result["spin"] = self.spin
        if self.mass is not None:
            result["mass"] = self.mass
        if self.color is not None:
            result["color"] = self.color
        if compact:
            return result
        # Add quantum numbers (full mode only)
        if self.charge is not None:
            result["charge"] = self.charge
        if self.lepton_e is not None:
            result["lepton_e"] = self.lepton_e
        if self.lepton_mu is not None:
            result["lepton_mu"] = self.lepton_mu
        if self.lepton_tau is not None:
            result["lepton_tau"] = self.lepton_tau
        if self.baryon_number is not None:
            result["baryon_number"] = self.baryon_number
        if self.custom_quantum_numbers is not None:
            result["custom_quantum_numbers"] = self.custom_quantum_numbers
        return result


@dataclass
class Vertex:
    """Represents a vertex in the diagram."""
    type: str
    coupling: Union[str, float, Dict[str, float]]  # Coupling name, value, or dict (e.g., chiral {"gL": ..., "gR": ...})
    fields: Optional[List[str]] = None
    #: Dirac structure on each bilinear of a contact operator, e.g. ["V", "A"].
    structures: Optional[List[str]] = None
    #: Which external legs form each bilinear, e.g. [[0, 2], [1, 3]].
    pairing: Optional[List[List[int]]] = None
    #: Optional connectivity: the lines meeting here, as "in:0", "out:<i>"
    #: (index into final) and "prop:<j>" (index into propagators).
    legs: Optional[List[str]] = None
    #: Mass dimension of the coupling (1 for a trilinear scalar or a
    #: scalar-vector-vector vertex); it is made dimensionless with the
    #: decaying particle's mass.
    coupling_dimension: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"type": self.type, "coupling": self.coupling}
        if self.fields:
            result["fields"] = self.fields
        if self.legs:
            result["legs"] = self.legs
        if self.coupling_dimension:
            result["coupling_dimension"] = self.coupling_dimension
        return result


@dataclass
class Propagator:
    """Represents an internal propagator."""
    label: str
    mass: float
    width: Optional[float] = None
    spin: Optional[float] = None  # Spin for spin-dependent propagator factors (Table 1 of arXiv:1402.1178)
    regime: str = "auto"  # auto, heavy, light, intermediate
    is_loop_propagator: bool = False
    #: Optional: indices of the final-state particles this line produces
    #: (its momentum is their sum), which fixes its allowed virtuality.
    final_legs: Optional[List[int]] = None
    #: Optional: colour factor of this line's decay into final_legs, used to
    #: split the diagram's colour factor in the narrow-width approximation.
    decay_color_factor: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "label": self.label,
            "mass": self.mass,
            "regime": self.regime,
        }
        if self.width is not None:
            result["width"] = self.width
        if self.spin is not None:
            result["spin"] = self.spin
        if self.is_loop_propagator:
            result["is_loop_propagator"] = True
        if self.final_legs is not None:
            result["final_legs"] = self.final_legs
        if self.decay_color_factor is not None:
            result["decay_color_factor"] = self.decay_color_factor
        return result


@dataclass
class Diagram:
    """
    Diagram specification for NDA estimates.

    This is the clean, user-friendly format.
    """
    topology: str
    initial: List[Particle]
    final: List[Particle]
    vertices: List[Vertex]
    couplings: Dict[str, float] = field(default_factory=dict)
    propagators: List[Propagator] = field(default_factory=list)
    color_factor: float = 1.0
    energy_scale: Optional[float] = None  # Override scale (default: mother mass)
    symmetry_factor: Optional[int] = None  # Override identical-particle factor
    #: False when conservation was established upstream (e.g. a census built
    #: from the model's own vertices), where label-based lookups do not apply.
    check_quantum_numbers: bool = True

    def to_dict(self, compact: bool = False) -> Dict[str, Any]:
        """Convert to dictionary format.

        Args:
            compact: If True, omit quantum numbers from particles.
                     Produces smaller dicts suitable for NDA and BR tools.
        """
        result = {
            "topology": self.topology,
            "initial": [p.to_dict(compact=compact) for p in self.initial],
            "final": [p.to_dict(compact=compact) for p in self.final],
            "vertices": [v.to_dict() for v in self.vertices],
        }
        if self.couplings:
            result["couplings"] = self.couplings
        if self.propagators:
            result["propagators"] = [p.to_dict() for p in self.propagators]
        if self.color_factor != 1.0:
            result["color_factor"] = self.color_factor
        if self.energy_scale is not None:
            result["energy_scale"] = self.energy_scale
        if self.symmetry_factor is not None:
            result["symmetry_factor"] = self.symmetry_factor
        if not self.check_quantum_numbers:
            result["check_quantum_numbers"] = False
        return result

    def validate(self, check_quantum_numbers: Optional[bool] = None) -> tuple[bool, List[str]]:
        """
        Validate diagram for physical consistency.

        Args:
            check_quantum_numbers: If True, validate quantum number conservation

        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        if check_quantum_numbers is None:
            check_quantum_numbers = self.check_quantum_numbers

        # Validate topology
        is_valid, msg = validate_topology(self.to_dict(), self.topology)
        if not is_valid:
            warnings.append(msg)

        # Check initial state
        if len(self.initial) != 1:
            warnings.append(f"Expected 1 initial state for decay, got {len(self.initial)}")

        # Check kinematic thresholds
        mother_mass = self.initial[0].mass
        if mother_mass is not None:
            final_masses = [p.mass for p in self.final if p.mass is not None]
            if final_masses and len(final_masses) == len(self.final):
                total_final = sum(final_masses)
                if mother_mass < total_final:
                    warnings.append(
                        f"Kinematic threshold violated: "
                        f"M_initial={mother_mass:.3f} < Σm_final={total_final:.3f}"
                    )

        # Check couplings are defined
        for vertex in self.vertices:
            if isinstance(vertex.coupling, str):
                if vertex.coupling not in self.couplings:
                    warnings.append(
                        f"Coupling '{vertex.coupling}' used in vertex but not defined in couplings dict"
                    )
            elif isinstance(vertex.coupling, dict):
                for key, val in vertex.coupling.items():
                    if isinstance(val, str) and val not in self.couplings:
                        warnings.append(
                            f"Coupling '{val}' (in dict key '{key}') used in vertex but not defined in couplings dict"
                        )

        # Check spins are specified
        for i, p in enumerate(self.initial):
            if p.spin is None:
                warnings.append(f"Initial particle {i} (label={p.label}) missing spin")

        for i, p in enumerate(self.final):
            if p.spin is None:
                warnings.append(f"Final particle {i} (label={p.label}) missing spin")

        # NEW: Quantum number validation (opt-in)
        # Only add actual conservation violations as warnings (not
        # informational messages about undefined quantum numbers,
        # which are expected for generic/BSM particle labels).
        if check_quantum_numbers:
            try:
                from .quantum_validation import validate_quantum_numbers
                qn_valid, qn_warnings = validate_quantum_numbers(self, strict=False)
                if not qn_valid:
                    warnings.extend(qn_warnings)
            except ImportError:
                # Graceful fallback if quantum_validation not available
                pass

        return (len(warnings) == 0, warnings)


def parse_particle(particle_dict: Dict[str, Any]) -> Particle:
    """Parse particle specification."""
    spin = particle_dict.get("spin")
    if spin is not None:
        spin = parse_spin(spin)

    return Particle(
        label=particle_dict.get("label"),
        spin=spin,
        mass=particle_dict.get("mass"),
        color=particle_dict.get("color"),
        charge=particle_dict.get("charge"),
        lepton_e=particle_dict.get("lepton_e"),
        lepton_mu=particle_dict.get("lepton_mu"),
        lepton_tau=particle_dict.get("lepton_tau"),
        baryon_number=particle_dict.get("baryon_number"),
        custom_quantum_numbers=particle_dict.get("custom_quantum_numbers")
    )


def parse_vertex(vertex_dict: Dict[str, Any]) -> Vertex:
    """Parse vertex specification."""
    return Vertex(
        type=vertex_dict["type"],
        coupling=vertex_dict.get("coupling", 1.0),
        fields=vertex_dict.get("fields"),
        legs=vertex_dict.get("legs"),
        coupling_dimension=int(vertex_dict.get("coupling_dimension", 0)),
    )


def parse_propagator(prop_dict: Dict[str, Any]) -> Propagator:
    """Parse propagator specification."""
    # Handle spin - convert string to float if needed (e.g., "1/2" -> 0.5)
    spin_raw = prop_dict.get("spin")
    spin = None
    if spin_raw is not None:
        if isinstance(spin_raw, (int, float)):
            spin = float(spin_raw)
        elif isinstance(spin_raw, str):
            # Handle fractional notation like "1/2", "3/2"
            if "/" in spin_raw:
                num, denom = spin_raw.split("/")
                spin = float(num) / float(denom)
            else:
                spin = float(spin_raw)

    return Propagator(
        label=prop_dict["label"],
        mass=prop_dict["mass"],
        width=prop_dict.get("width"),
        spin=spin,
        regime=prop_dict.get("regime", "auto"),
        is_loop_propagator=prop_dict.get("is_loop_propagator", False),
        final_legs=prop_dict.get("final_legs"),
        decay_color_factor=prop_dict.get("decay_color_factor"),
    )


def compute_symmetry_factor(diagram: Diagram) -> int:
    """Compute identical-particle symmetry factor for final state.

    If ``diagram.symmetry_factor`` is set explicitly, return it.
    Otherwise count identical labels in the final state and return n!
    for each group of n identical particles.

    Returns:
        Product of factorials of identical-particle multiplicities.
    """
    if diagram.symmetry_factor is not None:
        return diagram.symmetry_factor
    from collections import Counter
    counts = Counter(p.label for p in diagram.final if p.label)
    factor = 1
    for count in counts.values():
        factor *= math.factorial(count)
    return factor


def parse_diagram(diagram_dict: Dict[str, Any]) -> Diagram:
    """
    Parse diagram specification.

    Args:
        diagram_dict: Dictionary with diagram specification

    Returns:
        Diagram object

    Raises:
        ValueError: If diagram is malformed
    """
    # Parse initial state
    initial_list = diagram_dict.get("initial", [])
    if not initial_list:
        raise ValueError("Diagram must have 'initial' field with at least one particle")
    initial = [parse_particle(p) for p in initial_list]

    # Parse final state
    final_list = diagram_dict.get("final", [])
    if not final_list:
        raise ValueError("Diagram must have 'final' field with at least one particle")
    final = [parse_particle(p) for p in final_list]

    # Parse vertices
    vertices_list = diagram_dict.get("vertices", [])
    if not vertices_list:
        raise ValueError("Diagram must have 'vertices' field with at least one vertex")
    vertices = [parse_vertex(v) for v in vertices_list]

    # Parse propagators (optional)
    propagators_list = diagram_dict.get("propagators", [])
    propagators = [parse_propagator(p) for p in propagators_list]

    # Get topology (or infer it)
    topology = diagram_dict.get("topology")
    if topology is None:
        topology = infer_topology({
            "initial": initial_list,
            "final": final_list,
            "vertices": vertices_list,
            "propagators": propagators_list
        })

    # Get couplings dictionary
    couplings = diagram_dict.get("couplings", {})

    # Get color factor
    color_factor = diagram_dict.get("color_factor", 1.0)

    # Get energy scale override
    energy_scale = diagram_dict.get("energy_scale")

    # Get symmetry factor override
    symmetry_factor = diagram_dict.get("symmetry_factor")

    return Diagram(
        topology=topology,
        initial=initial,
        final=final,
        vertices=vertices,
        couplings=couplings,
        propagators=propagators,
        color_factor=color_factor,
        energy_scale=energy_scale,
        symmetry_factor=symmetry_factor,
        check_quantum_numbers=bool(diagram_dict.get("check_quantum_numbers", True)),
    )


def infer_color_factor(particles: List[Particle]) -> float:
    """
    Infer color factor from particle content.

    Args:
        particles: List of particles

    Returns:
        Color factor (1 for colorless, 3 for QCD triplet pair, etc.)
    """
    color_types = [p.color for p in particles if p.color is not None]

    if not color_types:
        return 1.0

    # Count color representations
    n_triplet = color_types.count("triplet")
    n_antitriplet = color_types.count("antitriplet")
    n_octet = color_types.count("octet")

    # Simple cases
    if n_triplet == 1 and n_antitriplet == 1 and n_octet == 0:
        # q qbar → color factor 3
        return 3.0
    elif n_triplet == 2 and n_antitriplet == 0:
        # qq → color factor 3 (antisymmetric)
        return 3.0
    elif n_octet > 0:
        # Gluons present → more complex
        return 8.0

    return 1.0
