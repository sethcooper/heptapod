# NDA (Naive Dimensional Analysis) Tool

Order-of-magnitude decay width estimates using dimensional analysis.

## Overview

The NDA tool provides quick estimates for particle decay widths without requiring full event generation. It uses **Naive Dimensional Analysis** - a power-counting scheme based on dimensional analysis and coupling hierarchies.

**Use cases:**
- Preliminary phenomenology studies
- BSM model exploration
- Quick cross-checks before detailed simulations
- Teaching and learning particle physics

**Not a replacement for:** Full event generation (MadGraph, Pythia). NDA provides estimates to 1-2 orders of magnitude accuracy.

## Quick Start

```python
from tools.nda import EstimateDecayWidthNDATool
import json

# Define decay process using simple diagram format
diagram = {
    "topology": "tree_2body",
    "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
    "final": [
        {"label": "b", "spin": "1/2", "mass": 4.2},
        {"label": "bbar", "spin": "1/2", "mass": 4.2}
    ],
    "vertices": [{"type": "yukawa", "coupling": 0.03}],
    "color_factor": 3.0
}

# Create and run tool
tool = EstimateDecayWidthNDATool(
    diagram=diagram,
    base_directory="/path/to/working/dir"
)

tool._setup()
result_str = tool._run()
result = json.loads(result_str)

print(f"Γ(H→bb̄) ≈ {result['width_mev']:.1f} MeV")
# Output: Γ(H→bb̄) ≈ 26.9 MeV
```

## Tool Interface

### Inputs (RuntimeField)

**`diagram`** (Dict[str, Any], required)
- Diagram specification in simple format
- See [Diagram Format](#diagram-format) below

### State (StateField)

**`base_directory`** (str, required)
- Base sandbox directory for file operations
- All file paths are resolved relative to this directory

### Output (JSON)

```json
{
  "status": "ok",
  "width_gev": 0.02686,
  "width_mev": 26.86,
  "formula": "\\Gamma = \\frac{1}{16\\pi M} \\times y^2 M^2",
  "scaling": "\\Gamma \\sim \\frac{y^2 M}{16\\pi}",
  "diagram": {
    "topology": "tree_2body",
    "n_vertices": 1,
    "n_propagators": 0,
    "loop_order": 0
  },
  "phase_space": 0.0398,
  "matrix_element": 168.75,
  "coupling": "y",
  "interaction": "yukawa"
}
```

## Diagram Format

The diagram is specified as a dictionary with the following structure:

```python
diagram = {
    # Topology (optional - can be auto-inferred)
    "topology": "tree_2body",  # tree_2body, tree_3body, s_channel_1prop,
                               # triangle_loop, box_loop, three_loop

    # Initial state (required)
    "initial": [
        {"label": "H", "spin": 0, "mass": 125.0}
    ],

    # Final state (required)
    "final": [
        {"label": "b", "spin": "1/2", "mass": 4.2},
        {"label": "bbar", "spin": "1/2", "mass": 4.2}
    ],

    # Vertices (required)
    "vertices": [
        {"type": "yukawa", "coupling": 0.03}
    ],

    # Propagators (optional - for processes with intermediate particles)
    "propagators": [
        {"label": "W", "mass": 80.4, "width": 2.1, "regime": "heavy"}
    ],

    # Couplings (optional - for named couplings)
    "couplings": {
        "y_b": 0.03
    },

    # Color factor (optional - auto-inferred if not provided)
    "color_factor": 3.0,

    # Energy scale (optional - for generic processes without masses)
    "energy_scale": 100.0
}
```

### Supported Topologies

**Tree-level:**
- `tree_2body` - 1→2 decay
- `tree_3body` - 1→3 decay
- `tree_nbody` - 1→n decay (n ≥ 4)
- `s_channel_1prop` - Process with s-channel propagator

**Loop diagrams:**
- `triangle_loop` - 1-loop triangle
- `box_loop` / `two_loop` - 2-loop box
- `three_loop` / `hexagon_loop` - 3-loop hexagon

### Supported Vertex Types

- `yukawa` - Yukawa coupling (scalar-fermion-fermion)
- `gauge-vector` - Gauge boson coupling (vector)
- `gauge-axial` - Gauge boson coupling (axial)
- `scalar-4pt` - Scalar 4-point interaction
- `dim6-4fermion` - Dimension-6 four-fermion operator
- `dim8-4fermion` - Dimension-8 four-fermion operator

### Optional connectivity

A diagram may say which lines meet at each vertex: `vertices[i].legs`, a list
of `"in:0"`, `"out:<i>"` (index into `final`) and `"prop:<j>"` (index into
`propagators`). With it, and final-state masses:

- each internal line carries the momentum of the final-state particles on its
  far side, so its virtuality is confined to
  `(sum_S m)^2 <= q^2 <= (M - sum_rest m)^2`; an off-shell line with a
  `width` uses `|P(q^2)|^2` averaged over that interval (a Breit-Wigner
  average, which keeps the enhancement just below a threshold) when the pole
  lies above the interval; one without a width, or whose pole lies at or below
  the interval's lower end (`tau* -> tau nu nubar`), uses it at the interval's
  midpoint;
- a line that can go on shell (and has a `width`) is treated in the
  narrow-width approximation, `Gamma = Gamma(parent -> rest + R) Gamma(R -> S) / Gamma_R`,
  each factor estimated by the tool itself (nested resonances recursively;
  separate lines that can resonate, such as either Z in `H -> Z Z*`, add);
- a zero-width line that can go on shell is reported, not estimated: the
  final state is then a lower-multiplicity decay followed by that particle's
  decay.

Other optional fields: `vertices[i].coupling_dimension` (mass dimension of a
coupling such as a trilinear scalar or `h W W`; the raw estimate counts it in
units of the decaying mass), `propagators[j].decay_color_factor` (to split the
colour factor in the narrow-width approximation) and
`check_quantum_numbers: false` (conservation already established, e.g. by a
census built from the model's vertices).

With final-state masses, phase space is the massive volume (exact recursive
two-body factorization), not the massless formula.

### From a census: EstimateCatalogueNDA

`EstimateCatalogueNDA` takes an `EnumerateProcesses` output, the UFO and the
card, and writes one request per census diagram in the format above
(UFO masses, widths, spins and colours at the card; couplings evaluated at the
card, chiral parts for fermion vertices; exact SU(3) colour factors;
connectivity), runs this tool on each, and combines a channel's diagrams:
coherently, with their coupling phases, among diagrams that differ only in
internal-line flavour (so CKM unitarity cancels as in GIM), incoherently
otherwise. Each request is saved with its response and reruns on its own.

Each diagram's response carries its evaluated expression (`formula_type:
"evaluated"`): the improved estimate written factor by factor,

```
Gamma = C B c_ps^(n-2) / ((2 s_P + 1) S) * P_P / (2 m_P) * Phi_n
        * prod_v |g_v|^2 E_v^2 * prod_i P_i(E_i) * prod_j N_j(q_j^2) |D_j|^2
```

with `definitions` giving every symbol's value (energies `E_i = Ebar + m_i`,
line virtualities and their intervals, couplings, the massive phase-space
volume), so that substituting them reproduces `width_gev` (`factors` holds the
numbers). `E_v^2` enters only at a vertex whose coupling multiplies a momentum
(three vectors, or two scalars and a vector), with `E_v` the largest energy
among its legs: the parent's mass, a final energy, or an internal line's
square root of mean virtuality (`factors.derivative_vertices`).

A massive final vector's polarization sum `P = 1 + E^2/m_V^2` grows with energy
through its longitudinal state. By the equivalence theorem that state couples as
the vector's Goldstone boson, in proportion to the masses at the vertex that
emits it, so the growth needs a source there: a fermion or scalar at least as
heavy as the vector (`N -> l W`, `t -> b W`, `h -> W W`). A vector emitted at a
vertex without one (a gauge self-coupling, or a pair of light fermions) keeps its
value at rest, `P = 2`: its growth cancels between diagrams (as in
`e+ e- -> W+ W-`), which an incoherent sum of diagrams cannot see. These
vectors are listed in `factors.unsourced_vectors`. A diagram
estimated in the narrow-width approximation has the factorized form
`Gamma(P -> X R) Gamma(R -> S) / Gamma_R` instead.

The census writes each diagram's estimate twice, with couplings named after
the particles at each vertex (`g_{hWW}`, `g_{We\nu_e}`). `formula` and
`definitions` give it in its parameters:

```
Gamma ~ kappa prod_v g_v^2 m_P^a prod_j L_j,    kappa = kappa_0 r
```

`L_j` is each line's parametric size: `1/(m_j Gamma_j m_P^2)` (`1/(m_j
Gamma_j)` for a fermion) when the estimate averages its Breit-Wigner over an
interval holding the pole, `1/m_j^4` (`m_P^2/m_j^4` for a fermion) when
`m_j^2` exceeds the line's virtuality, else `1/m_P^4` (`1/m_P^2` for a
fermion); a massive final vector gives `m_P^2/m_V^2` (the number 2 in `kappa_0`
when its growth has no source); `a` makes the dimension.
The one definition writes `kappa_0`, the massless limit of the pure numbers
(colour, B and spin over symmetry, the parent's polarization, `0.8^(n-2)`, the
massless volume over `m_P^(2n-4)`, `(2/n)^k` from the fermion energies), and
`r = kappa/kappa_0`, which holds the final-state masses, the propagators at
their intervals and the vertex energies; `r` is left out when it is 1.
`formula_evaluated` and `definitions_evaluated` give the factor-by-factor
expression. A channel carries the expressions of its largest diagram and that
diagram's share of the channel width (`formula_fraction`), and so does each
coupling class.

A channel with an external massless vector, or with a photon or gluon exchange
whose far-side final-state particles are all massless, in a diagram whose
estimate is at least 1e-6 of the channel's largest diagram, is
`requires_regulator`: its uncut width diverges and is defined only with cuts.
A photon or gluon whose far side carries mass (gamma* -> tau+ tau-,
g* -> b bbar) has virtuality of at least that mass squared, and the channel is
estimated like any other. An exchange suppressed by small couplings
(h -> u ubar gamma* through y_u) is kept.

Given `lower_estimates` (this tool's outputs at lower multiplicities):

- external photons and gluons get the soft-collinear estimate with the
  pair-mass regulator `m_ij^2 > y M^2` on visible pairs (`pair_mass_y`,
  default `0.05^2`): `Gamma(P -> X) * prod_V (c_V ln^2(1/y))^(n_V) / n_V!`,
  `c_gamma = sum_i alpha Q_i^2 / (2 pi)`, `c_g = sum_i alpha_s C_F / (2 pi)`,
  over the final-state fermions of X (quarks only for a gluon); without an
  emitting fermion (h -> W+ W- gamma) there is no collinear logarithm and no
  estimate;
- a photon or gluon that splits into a massive pair F Fbar (its far side
  carries mass, so it is not `requires_regulator`) gets
  `Gamma(P -> X) * c_V * k_F/(3 pi) * ln^3(M^2/4 m_F^2) / 3` in place of its
  diagrams' estimates, with `c_V` the soft-collinear coefficient of X above,
  `k_F = alpha Q_F^2 N_F` for a photon and `alpha_s T_F` for a gluon: the
  emission's `ln^2(M^2/q^2)` integrated against the splitting's `dq^2/q^2`
  from `4 m_F^2` to `M^2`. The midpoint virtuality would miss the pole at
  `q^2 -> 4 m_F^2`. Distinct `(V, F, X)` add; the channel's other diagrams keep
  their estimates and add incoherently (`Gamma_other`). Against MG5 (no cut;
  no width on internal W and top lines in W and top decays) at the PDG 2026
  card, the 129 such four-body channels have NDA/MG median 5.1, 10-90%
  1.3-12;
- a zero-width particle Y that goes on shell gives
  `Gamma(P -> X Y) Gamma(Y -> S) / Gamma_Y`, with `Gamma_Y` the sum of Y's
  lower-multiplicity widths; decays into quarks of a particle lighter than the
  lightest hadron (pi0) are closed, and a channel whose on-shell Y has only
  such decays (e -> nu u dbar with massless quarks) is `unphysical`.

Internal photon and gluon exchange into massless daughters stays
`requires_regulator`.

On the unrestricted SM at its default card, against exact tree-level widths:
two-body channels 0.72-1.0; all 175 finite three-body channels 0.40-0.75;
`H -> e+ e- mu+ mu-` 0.42.

## Examples

### Example 1: Standard Model Higgs

```python
# H → bb̄
diagram = {
    "topology": "tree_2body",
    "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
    "final": [
        {"label": "b", "spin": "1/2", "mass": 4.2},
        {"label": "bbar", "spin": "1/2", "mass": 4.2}
    ],
    "vertices": [{"type": "yukawa", "coupling": 0.03}],
    "color_factor": 3.0
}

tool = EstimateDecayWidthNDATool(diagram=diagram, base_directory="/tmp")
result = json.loads(tool._run())
# Γ ≈ 27 MeV (experimental: ~2.4 MeV)
```

### Example 2: Muon Decay (EFT)

```python
# μ → eνν̄ (4-fermion contact interaction)
diagram = {
    "topology": "tree_3body",
    "initial": [{"label": "mu", "spin": "1/2", "mass": 0.106}],
    "final": [
        {"label": "e", "spin": "1/2"},
        {"label": "nu_mu", "spin": "1/2"},
        {"label": "nu_e", "spin": "1/2"}
    ],
    "vertices": [{"type": "dim6-4fermion", "coupling": "G_F"}],
    "couplings": {"G_F": 1.166e-5}  # GeV^-2
}
```

### Example 3: Generic BSM Process

```python
# Generic scalar → fermion pair (no specific particle labels)
diagram = {
    "topology": "tree_2body",
    "initial": [{"spin": 0}],
    "final": [{"spin": "1/2"}, {"spin": "1/2"}],
    "vertices": [{"type": "yukawa", "coupling": 0.1}],
    "energy_scale": 100.0  # GeV
}
```

### Example 4: Loop Process

```python
# H → γγ via top quark loop
diagram = {
    "topology": "triangle_loop",
    "initial": [{"label": "H", "spin": 0, "mass": 125.0}],
    "final": [
        {"label": "gamma", "spin": 1},
        {"label": "gamma", "spin": 1}
    ],
    "vertices": [
        {"type": "yukawa", "coupling": 1.0},
        {"type": "gauge-vector", "coupling": 0.3},
        {"type": "gauge-vector", "coupling": 0.3}
    ],
    "propagators": [
        {"label": "t", "mass": 173.0, "is_loop_propagator": True},
        {"label": "t", "mass": 173.0, "is_loop_propagator": True},
        {"label": "t", "mass": 173.0, "is_loop_propagator": True}
    ]
}
# Includes 1/(16π²) loop suppression automatically
```

## Physics Methodology

NDA estimates decay widths using dimensional analysis:

**Tree-level 2-body:**
```
Γ ~ g² M / (16π)
```

**Tree-level 3-body:**
```
Γ ~ g² M³ / (64π³)
```

**1-loop suppression:**
```
Γ_loop ~ Γ_tree × 1/(16π²)
```

**Heavy propagator suppression:**
```
Factor ~ 1/M_X²  (for M_X >> E)
```

**Accuracy:** Typically 1-2 orders of magnitude for a hand-written diagram;
with connectivity and masses (as `EstimateCatalogueNDA` writes them), within a
factor of a few across the SM decay tables above. Not a replacement for a
calculation.

## Testing

Run tests:
```bash
python tools/nda/test_nda.py
```

Run example calculations:
```bash
python examples/nda_decay_widths.py
```

## Files

- `nda_tool.py` - Main tool implementation (`EstimateDecayWidthNDATool`)
- `phase_space.py` - Phase space factor calculations
- `matrix_element.py` - Matrix element NDA estimation
- `topology.py` - Topology definitions
- `simple_diagram.py` - Diagram data structures
- `test_nda.py` - Test suite
- `__init__.py` - Package exports

## Dependencies

- Python 3.7+
- NumPy
- Orchestral AI framework (BaseTool)

## References

- NDA methodology: arXiv:1804.01954
- Phase space formulas: Peskin & Schroeder, "An Introduction to QFT"
- Decay width formulas: PDG (Particle Data Group)

## License

Part of the HEPTAPOD package.
Copyright (C) 2026 HEPTAPOD authors.
Licensed under GNU GPL v3 or later.
