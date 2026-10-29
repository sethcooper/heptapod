---
name: mg5
bundle: mg5
description: Generate events with MadGraph5 from a UFO model — decay chains, widths, order constraints, and reading the errors it actually prints. Use whenever the user mentions MadGraph, MG5, mg5_aMC, `import model`, `generate`, `compute_widths`, MadSpin, LHE output, or BSM event generation. Also use proactively before writing a `.mg5` command card from scratch — cascade syntax and width handling both have failure modes that produce plausible-looking but wrong events.
---

# MadGraph5: generating events that are what you asked for

Two of MG5's failure modes are quiet. A deep decay chain can write an LHE that
is missing the inner decay, and placeholder widths can smear a reconstructed
peak by hundreds of keV. Neither raises an error. This skill covers both, plus
how to read the errors MG5 does raise.

For authoring the model itself, see the `feynrules` skill — several errors that
surface here are caused there.

```
UFO/ directory ──[import model]──▶ matrix element ──[launch]──▶ unweighted_events.lhe.gz
```

## Two rules that cover most first-run failures

### 1. The comma decay-chain syntax is shallow — prefer an explicit n-body ME

```
generate ta- > mu- phid, phid > Ap Ap, Ap > mu+ mu-
```

looks like it should give a five-body final state. For cascades **three levels
deep** MG5 often unfolds only two into the written LHE: the inner
`Ap > mu+ mu-` is generated as a separate matrix element but not stitched into
the event record, and the A′s appear as `status=1` rather than decayed.

The robust alternative is the explicit n-body matrix element with an
interaction-order constraint:

```
generate ta- > mu- mu- mu- mu+ mu+ NP=4
```

`NP=4` selects exactly the cascade topology, counted vertex by vertex: one LFV,
one $\phi A'A'$, two $A'\bar\mu\mu$ kinetic mixings. It carries the correct
spin correlations and Breit–Wigner shapes and writes the full final state.

When multiplicity makes the n-body ME blow up and a chain is unavoidable: group
with parentheses per decaying particle, run `compute_widths` *before*
`generate` (rule 2), and check the LHE has the expected particle count per
event before going further.

### 2. Set widths deliberately

A placeholder such as `Width -> {Wphid, 1.0*^-9}` survives into the param card,
and MG5's internal numerical floors near very small widths show up as a fake
~0.4 MeV spread in reconstructed peaks. For real propagator widths in the param
card, run `compute_widths` **before** `generate`:

```
import model UFO_dir
compute_widths phid Ap     # writes UFO_dir/param_card.dat
generate ...
output ...                 # copies that param_card into the process Cards/
launch
```

Do **not** put `compute_widths` after `output` or `launch`. MG5 reloads the
model when computing widths, discarding the generated process, and `launch`
then fails with `No processes generated. Please generate a process first.`
(pitfall **P12**).

For a deliberate narrow-width approximation, set widths explicitly to something
like 1e-15 GeV — but know the numerical limits you are working against.

### 3. A partial width you report is an integration of that channel

A partial width (or cross section) that you report or use as a check is a
Monte Carlo integration of the channel's squared amplitude: `generate` the
exact final state, `output`, `launch`, and read the width from the run (the
LHE header or the log). Restrict coupling orders (`QED<=2`, `NP=1`, ...) only
where the excluded diagrams are negligible, to save time without changing the
answer.

`compute_widths` (MadWidth) is an estimate, not this integration. Above two
bodies it assembles each channel from chained two-body decays and keeps fewer
diagrams than `generate`, so it can drop identical-particle exchange and
gauge-cancelling diagrams: a three-body partial width can come out wrong by
factors of several while its neighbours agree. It also zeroes partial widths of
coloured parents below its QCD scale. Use it for the propagator widths in the
param card (rule 2), not as the number you report.

```
import model UFO_dir
generate n1 > ve z z        # the channel itself, every diagram
output n1_vezz
launch n1_vezz
set nevents 5000            # the width is the run's integral; events are a by-product
```

## Many diagrams: restrict before you wait

A large final state can have dozens of interfering diagrams, and MadEvent
integrates roughly one channel per diagram. Organize the diagrams into coupling
classes first, then use amplitude orders (`QED<=3`, model orders like `NP=4`),
squared orders (`QED^2==8`, for single interference terms), diagram filters
(`/ a`, `$$ a`, `$ z`) or a restriction card to integrate the pieces that matter
quickly. Restricted runs are components: they do not sum to the coherent rate,
and dropping a class needs evidence that it is below your precision. See
`references/many_diagrams.md`.

## When MG5 fails, read the right log

The heptapod `MadGraphFromRunCard` tool returns terse errors, and the message
you see is often **misleading**: MG5 auto-converts UFOs, and the conversion can
mask the real problem behind something like
`invalid syntax (object_library.py, line 268)`.

To surface the actual error, run mg5_aMC from the shell — not through the tool
— on a minimal import:

```bash
echo "import model /full/path/to/UFO_dir" > /tmp/_t.txt
$MG5_PATH/bin/mg5_aMC -f /tmp/_t.txt 2>&1 | tail -30
```

That shows what MG5 hits *after* the auto-converter. Cross-reference
`references/error_decoder.md`.

## Before writing a `.mg5` card

1. **What is the target process, and what NP order does the cascade carry?**
   Count the BSM vertices on the target diagram — that count is your `NP=N`
   constraint.
2. **Do widths matter here?** Plan for `compute_widths` for propagators, or
   accept the narrow-width approximation deliberately. A partial width you
   report comes from launching that channel (rule 3).

If the model itself is in doubt — missing order tags, duplicate mass
parameters — check the UFO first; see the `feynrules` skill.

## Running

```bash
# 1. Write the MG5 card (see references/mg5_card_template.mg5). For a batch of
#    launches or a scan, start it with `set automatic_html_opening False`:
#    MG5 otherwise opens a browser page per launch. A handful of interactive
#    runs can leave it on.
# 2. Run MG5 — the heptapod MadGraphFromRunCard tool, or mg5_aMC directly.
#    For another run of a compiled process (more events, another seed, changed
#    cuts) pass process_dir=<existing process dir> instead of rebuilding it.
# 3. Verify the LHE before histogramming: check particle count per event (P11)
# 4. Convert LHE → JSONL → numpy via heptapod LHEToJSONL / EventJSONLToNumpy
# 5. Truth-level cross-check on a single event before processing the bulk
```

## Reference files

- **`references/pitfalls.md`** — the generation-side pitfalls in full: shallow
  decay chains, placeholder widths, the auto-conversion error mask, choosing
  between MadSpin / chains / explicit ME, verifying the LHE, and the
  `compute_widths` ordering trap.
- **`references/mg5_card_template.mg5`** — a minimal tested card with
  `compute_widths`, an NP-order constraint, and notes on each line.
- **`references/decay_chains.md`** — decay-chain syntax in depth: a decision
  tree for commas, parentheses, `NP=N`, MadSpin and narrow-width.
- **`references/error_decoder.md`** — error message → real cause → fix.
- **`references/many_diagrams.md`** — coupling orders, squared orders, diagram
  filters and restriction cards for high-multiplicity processes; how restricted
  runs relate to the coherent rate.
