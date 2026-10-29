---
name: sherpa
bundle: sherpa
description: Write Sherpa 3 YAML run cards that initialise on the first try and generate the events a comparison needs. Use whenever the user mentions Sherpa, a Sherpa.yaml or Run.dat, Comix, Amegic, AHADIC, CKKW merging, MC@NLO, SCALE_VARIATIONS, or generating events with SherpaFromRunCardTool. Also use proactively before writing a Sherpa card from scratch — Sherpa ignores a misspelt setting without a word and reports a bad value only after the process integration has started, so a single typo costs minutes and a wrong sample looks like a right one.
---

# Sherpa 3: run cards that initialise on the first try

Sherpa reads a YAML card and takes from it only the keys it knows. A key it
does not know is dropped silently: `BEAM_ENERGIEZ: 6500` leaves the beams at
zero energy and the run continues. A value it cannot use, a selector or
process it cannot build, ends initialisation with a fatal error, after the
process integration has started. **Validate before you generate**, so the
first costs nothing and the second costs seconds instead of minutes.

```
Sherpa.yaml ──[ValidateSherpaRunCard]──▶ ok ──[SherpaFromRunCard]──▶ events.root
                     │                                   │
                     └─▶ "BEAM_ENERGIEZ is not a         └─▶ rivet_analyses=[...]
                          Sherpa setting; did you mean          ──▶ rivet.yoda
                          BEAM_ENERGIES?"
```

## The loop that works

1. **Write the card.** Start from `references/card_template.yaml`. Beams and
   their energies (per beam, in GeV), the process list, the coupling order,
   the cuts as selectors. Sherpa 3 is YAML; the `Run.dat` syntax of Sherpa 2
   is not read.
2. **Validate it.** `ValidateSherpaRunCardTool`. The static checks catch
   misspelt keys against every setting Sherpa 3.0.3 reads, Pythia settings
   that strayed in, and the shape of the beams, energies and processes.
   `dry_run_init=True` then initialises Sherpa on the card without an
   event: Sherpa's own fatal error if there is one, the keys it never read,
   and the effective configuration. Pass the planned `data_dir` as
   `work_dir` and the integration it performs is reused by the run.
3. **Generate.** `SherpaFromRunCardTool`. Read the `summary` it returns
   rather than the events, and read `xsec` and `weighted` before analysing
   anything (see below).
4. **Compare.** The dataset is the one the Pythia tools write, so
   everything downstream is shared: `JetClusterSlowJetTool`,
   `ConvertEventsTool`, Delphes, and Rivet through `rivet_analyses` or
   `RunRivetTool`. A Pythia and a Sherpa sample of the same process meet in
   `RivetCompareTool` and `RivetPlotTool` as two labelled predictions.

## What the numbers mean

- **Cross sections are in pb.** Sherpa reports pb; the result gives
  `sigmaGen_pb` and, beside it, `sigmaGen_mb` for comparison with Pythia.
- **The event weight is a cross section.** In Sherpa's unweighted mode
  every event carries the same weight, the total cross section in pb, and
  the sum of weights over the trials is the cross section. Do not expect
  weights of 1. `weighted` is true when the weights *vary*, as they do in
  weighted and partially unweighted modes; then every histogram must use
  the per-event `weight`.
- **Variations are named weights.** `SCALE_VARIATIONS`, `PDF_VARIATIONS`
  and the rest become extra weights per event, named as Sherpa names them
  (`MUR=0.5__MUF=0.5__LHAPDF=-1`). They reach the dataset only through
  `SherpaCPPFromRunCardTool`: the Python interface behind
  `SherpaFromRunCardTool` does not expose them, and that tool says so when
  it had to drop them.
- **The integration is the slow part.** Sherpa integrates the process at
  initialisation and keeps the result in `Results/` under `data_dir`. A
  second run in the same directory starts at once; a changed process wants
  a fresh directory.

## The two run-card tools

Sherpa cannot be initialised twice in one process, so both tools run it in
a process of their own. `SherpaFromRunCardTool` runs Sherpa's Python
interface in a worker: the `sherpa-mc` wheel, or the module of a build with
Python support named by `sherpa_path`. `SherpaCPPFromRunCardTool` runs
`heptapod-sherpa`, a small C++ driver compiled once against a Sherpa install
(`sherpa_path`, or the wheel's own libraries, which ship with their headers)
and needs a C++ compiler on `PATH`; it is the one that stores the weight
variations. Same card, same dataset otherwise. `SherpaCPPFromProgramTool`
compiles and runs your own C++ program against the install:
`references/main_template.cc` is one that reaches the event record and the
weights map directly.

## Reference files

- **`references/pitfalls.md`** -- the failure modes in full, each with the
  symptom you actually observe, the cause, and the fix.
- **`references/card_template.yaml`** -- an annotated card for a hadron
  collider and a lepton collider, with the blocks a typical run needs.
- **`references/main_template.cc`** -- a C++ main program against a Sherpa
  install, for what a card cannot express.
- **`references/example_session.md`** -- an end-to-end example: the card,
  the validator with and without a typo, the dry run, the generation with
  variations, and Rivet beside Pythia.
