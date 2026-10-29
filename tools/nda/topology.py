"""
# topology.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Topology library for NDA decay width estimates.

Defines standard topologies with their properties and validation rules.

Key concepts:
- Loop number: L = I - V + 1 (I = internal lines, V = vertices)
- Tree-level: L = 0
- One-loop: L = 1
"""
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass


def calculate_loop_number(n_internal_propagators: int, n_vertices: int) -> int:
    """
    Calculate the number of loops using the formula L = I - V + 1.

    Args:
        n_internal_propagators: Number of internal propagators (I)
        n_vertices: Number of vertices (V)

    Returns:
        Number of loops (L)

    Examples:
        - Tree 2-body (1 vertex, 0 propagators): L = 0 - 1 + 1 = 0
        - Tree 3-body with W (2 vertices, 1 propagator): L = 1 - 2 + 1 = 0
        - H -> gamma gamma triangle (3 vertices, 3 internal): L = 3 - 3 + 1 = 1
        - Box diagram (4 vertices, 4 internal): L = 4 - 4 + 1 = 1
    """
    return n_internal_propagators - n_vertices + 1


def validate_graph_theory_constraint(diagram: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate fundamental graph theory constraint: I = (Σₙ n·Vₙ - E)/2

    This validates the relationship between internal propagators, vertex
    valences, and external lines for any Feynman diagram.

    Args:
        diagram: Diagram dictionary with vertices, initial, final states

    Returns:
        (is_valid, error_message)

    The constraint comes from counting field contractions:
    - Each vertex of valence n has n half-edges
    - Each external line uses 1 half-edge
    - Each internal propagator uses 2 half-edges
    - Total half-edges: Σₙ n·Vₙ = E + 2I

    Examples:
        μ→eνν via W: E=4, one 3-pt initial vertex + one 3-pt final vertex
            I = (3 + 3 - 4)/2 = 1 ✓
        μ→3e2ν (5 final): E=6, need V₃=4 for chain
            I = (3×4 - 6)/2 = 3 propagators
        H→bb: E=3, one 3-pt Yukawa vertex
            I = (3 - 3)/2 = 0 ✓
    """
    vertices = diagram.get("vertices", [])
    initial = diagram.get("initial", [])
    final = diagram.get("final", [])
    propagators = diagram.get("propagators", [])

    E = len(initial) + len(final)  # External lines
    I = len(propagators)  # Internal propagators

    # Calculate Σₙ n·Vₙ by inferring vertex valences
    # For now, we need to infer valence from vertex types
    # Most common: gauge vertices are 3-point, Yukawa are 3-point,
    # quartic vertices are 4-point (λφ⁴, gauge⁴)

    total_half_edges = 0
    vertex_valences = []

    for i, vertex in enumerate(vertices):
        vtype = vertex.get("type", "").lower()

        # Infer vertex valence from type
        # First, try to parse explicit valence from type string (e.g., "weak-3pt", "gauge-4pt")
        match = re.search(r'(\d+)pt', vtype)
        if match:
            valence = int(match.group(1))
        elif "quartic" in vtype or vtype in ["higgs-quartic", "gauge4"]:
            valence = 4
        elif "4fermion" in vtype or "4-fermion" in vtype or (vtype.startswith("fermi") and "gauge" not in vtype):
            # 4-fermion operators (dim-6 effective, Fermi theory, etc.)
            # Note: "fermi" check excludes "gauge-fermion" which is a 3-point vertex
            valence = 4
        elif "triple" in vtype:
            valence = 3
        else:
            # Default: 3-point vertex (most common for renormalizable theories)
            valence = 3

        vertex_valences.append(valence)
        total_half_edges += valence

    # Apply constraint: Σₙ n·Vₙ = E + 2I
    expected_half_edges = E + 2 * I

    if total_half_edges != expected_half_edges:
        # Calculate what I should be
        if (total_half_edges - E) % 2 != 0:
            return False, (
                f"Graph theory violation: impossible configuration. "
                f"Σₙ n·Vₙ = {total_half_edges} (vertices: {vertex_valences}), "
                f"E = {E} external lines. "
                f"For consistency need (Σₙ n·Vₙ - E) to be even, but got {total_half_edges - E}. "
                f"Check vertex structure."
            )

        expected_I = (total_half_edges - E) // 2
        return False, (
            f"Graph theory violation: I = (Σₙ n·Vₙ - E)/2. "
            f"Given {len(vertices)} vertices with valences {vertex_valences}: "
            f"Σₙ n·Vₙ = {total_half_edges}, E = {E}. "
            f"Required propagators: I = ({total_half_edges} - {E})/2 = {expected_I}, "
            f"but diagram specifies {I} propagator(s). "
            f"{'Add ' + str(expected_I - I) if expected_I > I else 'Remove ' + str(I - expected_I)} "
            f"propagator(s) to satisfy the constraint."
        )

    return True, ""


@dataclass
class TopologySpec:
    """Specification for a diagram topology."""
    name: str
    n_initial: int
    n_final_min: int
    n_final_max: Optional[int]
    n_vertices_min: int
    n_vertices_max: Optional[int]
    n_propagators_min: int  # Minimum internal propagators
    n_propagators_max: Optional[int]  # Maximum internal propagators (None = unlimited)
    loop_order: int  # Expected loop order (0=tree, 1=one-loop, etc.)
    description: str
    # When True the vertex ceiling is set by the external legs instead of
    # n_vertices_max: a connected tree with E external legs has at most E - 2
    # internal vertices (all 3-point), i.e. a 1 -> n decay has up to n - 1.
    vertices_bounded_by_legs: bool = False

    def validate(self, diagram: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate diagram matches topology specification.

        Checks:
        1. Initial/final state counts
        2. Vertex counts
        3. Propagator counts
        4. Loop number consistency (using L = I - V + 1)
        5. Graph theory constraint (I = (Σₙ n·Vₙ - E)/2)

        Returns:
            (is_valid, error_message)
        """
        initial = diagram.get("initial", [])
        final = diagram.get("final", [])
        vertices = diagram.get("vertices", [])
        propagators = diagram.get("propagators", [])

        # Check initial state count
        if len(initial) != self.n_initial:
            return False, f"{self.name} requires {self.n_initial} initial state, got {len(initial)}"

        # Check final state count
        if len(final) < self.n_final_min:
            return False, f"{self.name} requires at least {self.n_final_min} final states, got {len(final)}"
        if self.n_final_max and len(final) > self.n_final_max:
            return False, f"{self.name} requires at most {self.n_final_max} final states, got {len(final)}"

        # Check vertex count
        if len(vertices) < self.n_vertices_min:
            return False, f"{self.name} requires at least {self.n_vertices_min} vertices, got {len(vertices)}"
        if self.n_vertices_max and len(vertices) > self.n_vertices_max:
            return False, f"{self.name} requires at most {self.n_vertices_max} vertices, got {len(vertices)}"
        if self.vertices_bounded_by_legs:
            n_legs = len(initial) + len(final)
            v_max = max(1, n_legs - 2)
            if len(vertices) > v_max:
                return False, (
                    f"{self.name}: a tree with {n_legs} external legs has at most "
                    f"{v_max} vertices (E - 2), got {len(vertices)}"
                )

        # Check propagator count
        if len(propagators) < self.n_propagators_min:
            return False, f"{self.name} requires at least {self.n_propagators_min} propagators, got {len(propagators)}"
        if self.n_propagators_max is not None and len(propagators) > self.n_propagators_max:
            return False, f"{self.name} requires at most {self.n_propagators_max} propagators, got {len(propagators)}"

        # FUNDAMENTAL: Validate graph theory constraint I = (Σₙ n·Vₙ - E)/2
        # This is a hard constraint that must hold for ANY valid Feynman diagram
        is_valid_graph, graph_error = validate_graph_theory_constraint(diagram)
        if not is_valid_graph:
            return False, graph_error

        # Calculate and check loop number using L = I - V + 1
        # Skip for custom topology (user-defined, minimal validation)
        if self.name != "custom":
            calculated_loops = calculate_loop_number(len(propagators), len(vertices))
            if calculated_loops != self.loop_order:
                return False, (
                    f"{self.name} expects loop order {self.loop_order}, but got {calculated_loops} "
                    f"(from L = I - V + 1 = {len(propagators)} - {len(vertices)} + 1). "
                    f"This indicates a mismatch in diagram structure."
                )

        return True, ""


# ============================================================================
# TOPOLOGY LIBRARY
# ============================================================================
#
# Naming convention:
#   tree_Nbody[_Mprop]  - Tree-level, N final particles, M propagators (optional)
#   loop_Nbody[_Lloop]  - Loop-level, N final particles, L loops (optional)
#
# Key insight: For LLM agents:
#   - "tree_3body" (no suffix) → effective operator (non-renormalizable allowed)
#   - "tree_3body_1prop" → renormalizable with W/Z propagator
#
# Loop counting: All topologies automatically validated using L = I - V + 1
# ============================================================================

TOPOLOGIES = {
    # ========================================================================
    # TREE-LEVEL: Direct vertices (no internal propagators)
    # ========================================================================
    # These can be either renormalizable (e.g., H→bb via Yukawa) or
    # non-renormalizable effective operators (e.g., Fermi 4-fermion)

    "tree_2body": TopologySpec(
        name="tree_2body",
        n_initial=1,
        n_final_min=2,
        n_final_max=2,
        n_vertices_min=1,
        n_vertices_max=1,
        n_propagators_min=0,
        n_propagators_max=0,
        loop_order=0,  # L = 0 - 1 + 1 = 0 ✓
        description="Tree 2-body via direct vertex (e.g., H→bb, H→WW)"
    ),

    # "tree_Nbody" without a _Mprop suffix names the multiplicity only: any
    # connected tree with N final states (contact vertex, or an internal
    # line such as the W in mu -> e nu nu).  Generated for every N by
    # resolve_topology(); tree_3body / tree_nbody are listed explicitly.
    "tree_3body": TopologySpec(
        name="tree_3body",
        n_initial=1,
        n_final_min=3,
        n_final_max=3,
        n_vertices_min=1,
        n_vertices_max=None,
        n_propagators_min=0,
        n_propagators_max=None,
        loop_order=0,  # I = V - 1
        description="Tree 3-body: contact vertex (Fermi operator) or 1 propagator (μ→eνν via W)",
        vertices_bounded_by_legs=True,
    ),

    "tree_nbody": TopologySpec(
        name="tree_nbody",
        n_initial=1,
        n_final_min=2,
        n_final_max=None,
        n_vertices_min=1,
        n_vertices_max=None,
        n_propagators_min=0,
        n_propagators_max=None,
        loop_order=0,  # I = V - 1
        description="Tree n-body, any connected tree (contact vertex or internal propagators)",
        vertices_bounded_by_legs=True,
    ),

    # ========================================================================
    # TREE-LEVEL: With internal propagators (renormalizable)
    # ========================================================================

    "tree_2body_1prop": TopologySpec(
        name="tree_2body_1prop",
        n_initial=1,
        n_final_min=2,
        n_final_max=2,
        n_vertices_min=2,
        n_vertices_max=2,
        n_propagators_min=1,
        n_propagators_max=1,
        loop_order=0,  # L = 1 - 2 + 1 = 0 ✓
        description="Tree 2-body with 1 propagator (e.g., H→ff via Z*/γ*)"
    ),

    "tree_3body_1prop": TopologySpec(
        name="tree_3body_1prop",
        n_initial=1,
        n_final_min=3,
        n_final_max=3,
        n_vertices_min=2,
        n_vertices_max=2,
        n_propagators_min=1,
        n_propagators_max=1,
        loop_order=0,  # L = 1 - 2 + 1 = 0 ✓
        description="Tree 3-body with 1 propagator (e.g., μ→eνν via W, τ→μνν via W)"
    ),

    "tree_nbody_1prop": TopologySpec(
        name="tree_nbody_1prop",
        n_initial=1,
        n_final_min=2,
        n_final_max=None,
        n_vertices_min=2,
        n_vertices_max=2,
        n_propagators_min=1,
        n_propagators_max=1,
        loop_order=0,  # L = 1 - 2 + 1 = 0 ✓
        description="Tree n-body with 1 propagator (NOTE: only valid for n≤3 with 3-point vertices due to graph theory constraint I=(3V-E)/2)"
    ),

    "tree_nbody_nprop": TopologySpec(
        name="tree_nbody_nprop",
        n_initial=1,
        n_final_min=3,
        n_final_max=None,
        n_vertices_min=2,
        n_vertices_max=None,
        n_propagators_min=2,
        n_propagators_max=None,
        loop_order=0,  # L = I - V + 1 = 0, so I = V - 1 (chain/tree structure)
        description="Tree n-body with multiple propagators (e.g., cascade decays)",
        vertices_bounded_by_legs=True,
    ),

    # ========================================================================
    # ONE-LOOP TOPOLOGIES
    # ========================================================================

    "loop_2body_triangle": TopologySpec(
        name="loop_2body_triangle",
        n_initial=1,
        n_final_min=2,
        n_final_max=2,
        n_vertices_min=3,
        n_vertices_max=3,
        n_propagators_min=3,
        n_propagators_max=3,
        loop_order=1,  # L = 3 - 3 + 1 = 1 ✓
        description="Triangle loop (e.g., H→γγ, H→gg via fermion/W loop)"
    ),

    "loop_2body_box": TopologySpec(
        name="loop_2body_box",
        n_initial=1,
        n_final_min=2,
        n_final_max=2,
        n_vertices_min=4,
        n_vertices_max=4,
        n_propagators_min=4,
        n_propagators_max=4,
        loop_order=1,  # L = 4 - 4 + 1 = 1 ✓
        description="Box loop (e.g., 4-point 1-loop corrections)"
    ),

    "loop_nbody_1loop": TopologySpec(
        name="loop_nbody_1loop",
        n_initial=1,
        n_final_min=2,
        n_final_max=None,
        n_vertices_min=3,
        n_vertices_max=None,
        n_propagators_min=3,
        n_propagators_max=None,
        loop_order=1,  # L = I - V + 1 = 1, so I = V (loop structure)
        description="Generic 1-loop diagram (flexible n-body)"
    ),

    # ========================================================================
    # CUSTOM
    # ========================================================================

    "custom": TopologySpec(
        name="custom",
        n_initial=1,
        n_final_min=1,
        n_final_max=None,
        n_vertices_min=1,
        n_vertices_max=None,
        n_propagators_min=0,
        n_propagators_max=None,
        loop_order=0,  # Flexible, won't validate loop consistency
        description="Custom topology (user-defined, minimal validation)"
    ),
}


_TREE_NAME = re.compile(r"^tree_(\d+)body(?:_(\d+)prop)?$")
_LOOP_NAME = re.compile(r"^loop_(\d+)body$")


def resolve_topology(name: str) -> Optional[TopologySpec]:
    """
    Return the TopologySpec for ``name``, generating tree specs on demand.

    Explicit entries in TOPOLOGIES win.  Otherwise:

    - ``tree_Nbody``        any connected tree with N final states:
                            1 <= V <= N - 1 vertices, I = V - 1 propagators.
    - ``tree_Nbody_Mprop``  a tree with exactly M propagators (V = M + 1),
                            M <= N - 2 for 3-point vertices.
    - ``loop_Nbody``        a one-loop diagram with N final states.

    Returns None for names that match none of these.
    """
    if name in TOPOLOGIES:
        return TOPOLOGIES[name]

    m = _TREE_NAME.match(name)
    if m:
        n_final = int(m.group(1))
        if n_final < 2:
            return None
        if m.group(2) is None:
            return TopologySpec(
                name=name, n_initial=1,
                n_final_min=n_final, n_final_max=n_final,
                n_vertices_min=1, n_vertices_max=None,
                n_propagators_min=0, n_propagators_max=None,
                loop_order=0,
                description=f"Tree {n_final}-body, any connected tree",
                vertices_bounded_by_legs=True,
            )
        n_prop = int(m.group(2))
        return TopologySpec(
            name=name, n_initial=1,
            n_final_min=n_final, n_final_max=n_final,
            n_vertices_min=n_prop + 1, n_vertices_max=n_prop + 1,
            n_propagators_min=n_prop, n_propagators_max=n_prop,
            loop_order=0,
            description=f"Tree {n_final}-body with {n_prop} propagator(s)",
            vertices_bounded_by_legs=True,
        )

    m = _LOOP_NAME.match(name)
    if m:
        n_final = int(m.group(1))
        return TopologySpec(
            name=name, n_initial=1,
            n_final_min=n_final, n_final_max=n_final,
            n_vertices_min=1, n_vertices_max=None,
            n_propagators_min=1, n_propagators_max=None,
            loop_order=1,
            description=f"One-loop {n_final}-body",
        )

    return None


def infer_topology(diagram: Dict[str, Any]) -> str:
    """
    Infer topology from diagram structure if not specified.

    Uses the new naming scheme:
    - tree_Nbody: Direct vertex (no propagators)
    - tree_Nbody_Mprop: Tree with M propagators
    - loop_Nbody_*: Loop diagrams

    Args:
        diagram: Diagram dictionary

    Returns:
        Inferred topology name (using new naming convention)
    """
    initial = diagram.get("initial", [])
    final = diagram.get("final", [])
    vertices = diagram.get("vertices", [])
    propagators = diagram.get("propagators", [])

    n_initial = len(initial)
    n_final = len(final)
    n_vertices = len(vertices)
    n_propagators = len(propagators)

    # Calculate loop number
    n_loops = calculate_loop_number(n_propagators, n_vertices)

    # ========================================================================
    # LOOP DIAGRAMS (L >= 1)
    # ========================================================================
    if n_loops >= 1:
        # One-loop specific topologies
        if n_loops == 1:
            if n_final == 2 and n_vertices == 3:
                return "loop_2body_triangle"
            elif n_final == 2 and n_vertices == 4:
                return "loop_2body_box"
            else:
                return "loop_nbody_1loop"
        else:
            # Multi-loop (not yet supported in detail)
            return "custom"

    # ========================================================================
    # TREE-LEVEL DIAGRAMS (L = 0)
    # ========================================================================
    # No propagators → direct vertex
    if n_propagators == 0:
        if n_final == 2:
            return "tree_2body"
        elif n_final == 3:
            return "tree_3body"
        else:
            return "tree_nbody"

    # With propagators → renormalizable interaction
    elif n_propagators == 1:
        if n_final == 2:
            return "tree_2body_1prop"
        elif n_final == 3:
            return "tree_3body_1prop"
        else:
            return "tree_nbody_1prop"

    # Multiple propagators
    else:
        return "tree_nbody_nprop"


def validate_topology(diagram: Dict[str, Any], topology: Optional[str] = None) -> tuple[bool, str]:
    """
    Validate diagram against its topology.

    Args:
        diagram: Diagram dictionary
        topology: Topology name (if None, will infer)

    Returns:
        (is_valid, error_message)
    """
    if topology is None:
        topology = diagram.get("topology")

    if topology is None:
        topology = infer_topology(diagram)

    spec = resolve_topology(topology)
    if spec is None:
        return False, (
            f"Unknown topology: {topology}. Use tree_<N>body (any tree with N "
            f"final states), tree_<N>body_<M>prop, loop_<N>body, one of "
            f"{sorted(TOPOLOGIES)}, or omit 'topology' to infer it."
        )
    return spec.validate(diagram)
