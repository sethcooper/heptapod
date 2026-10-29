# Many-diagram processes: restricting the integration

MadEvent integrates with roughly one phase-space channel per diagram, and every
channel carries the full interfering |M|^2. A high-multiplicity final state with
many diagrams therefore means many channels, slow survey/refine and a hard-to-trust
error. When only part of the amplitude matters, or when you want to know which
part matters, generate less.

## First, know the classes

Organize the diagrams before integrating: which coupling monomials and orders
each class carries, which propagators can go on shell, which classes are
suppressed (small Yukawas, extra powers of a weak coupling). An NDA estimate per
class, with phase-space suppression, tells you what to expect from each piece.

## Ways to restrict

**Amplitude coupling orders** limit which diagrams are generated:

```
generate h > e+ e- mu+ mu- QED<=3
generate p p > e+ e- mu+ mu- QCD=0
generate ta- > mu- mu- mu- mu+ mu+ NP=4     # custom model orders
```

**Squared orders** select terms of |M|^2, so a single interference term between
classes of different order can be integrated on its own:

```
generate p p > e+ e- j j QED^2==8           # exactly that order in |M|^2
generate p p > e+ e- j j QCD^2<=2
```

Orders only separate classes the model tags differently. In the SM UFO Yukawa
and gauge vertices are both QED=1, so h -> 4l has every diagram at QED=3 and
orders cannot split it; use filters or a restriction instead.

**Diagram filters** act on propagators:

```
generate h > e+ e- mu+ mu- / a              # no photon propagator anywhere
generate h > e+ e- mu+ mu- $$ a             # no s-channel photon
generate h > e+ e- mu+ mu- $ z              # s-channel Z kept, but never near on shell
```

Removing diagrams can break gauge invariance; MG5 warns when it notices. Treat a
filtered run as a component, not a physical rate, unless the removed set is
separately gauge invariant.

**A restriction card** (`restrict_<name>.dat` next to the UFO's
`restrict_default.dat`, loaded with `import model <UFO>-<name>`) sets parameters to
zero and drops the couplings and diagrams that vanish with them, e.g. a Yukawa.
This is gauge safe when the parameter is a genuine independent input, but it
changes the model: use it for diagnostics, never for a result that must use the
model unrestricted.

## Using restricted runs

- Run the dominant class first: it converges fast and fixes the scale of the rate
  and the precision you can expect.
- A class's size relative to the full rate tells you whether it can be dropped,
  but only if it, and its interference with the rest, is well below the
  precision target. Small couplings can still matter where a propagator is
  singular (a soft gamma* -> l+ l-, a near on-shell resonance).
- Restricted rates do not add up to the coherent rate. Cross terms belong to
  neither class. Either integrate the full process, or integrate each class
  and each interference term (squared orders) separately and add them with
  their errors.
- The coherent rate of the full process is the answer when every contribution
  must be kept. Restricted runs tell you which channels are hard, and whether
  the full run's quoted error is believable.
