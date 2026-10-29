---
name: rivet
bundle: rivet
description: 'Compare generator predictions with published collider measurements using Rivet — find the analysis for a measurement, generate matching events with Pythia, run the analysis, test the agreement and plot it. Use whenever the user mentions Rivet, YODA, a Rivet analysis name (ATLAS_2012_I1204784, MC_ZINC, ...), comparing a generator or tune to data, validating a tune, or reproducing a published distribution. Also use proactively when a Pythia run is meant to be compared with an experiment: the analysis dictates the beams, energy and process, and Rivet accepts events at the wrong energy without complaint.'
---

# Rivet: predictions against measurements

Rivet runs a published analysis on generator events, binned as the paper
binned them, and ships the paper's numbers alongside. The comparison is only
meaningful if the events are the ones the paper measured: same beams, same
centre-of-mass energy, a process that produces what the analysis selects.
Rivet checks the beam particles, drops a mismatched analysis with a warning,
and **does not check the energy**: a 7 TeV analysis run on 13 TeV events fills
every histogram with nonsense. The tools check all three before anything runs.

```
RivetFindAnalyses ──▶ analysis, its run_info, beams, √s
        │
        ▼
PythiaFromRunCard(store="full", rivet_analyses=[...]) ──▶ rivet.yoda   (streamed, no event file)
   or   PythiaFromRunCard(output_format="hepmc3", store="full") ──▶ RunRivet ──▶ rivet.yoda
        │
        ▼
RivetCompare ──▶ χ²/ndf, p-values, pulls        RivetPlot ──▶ data vs prediction, ratio, gallery
```

## The loop that works

1. **Find the analysis.** `RivetFindAnalysesTool` with `query` (words from the
   measurement: `"Z transverse momentum"`, `"jet shapes"`, an INSPIRE id) and
   the run you can make: `beams="p+ p+"`, `energy_gev=13000`. Then call it
   again with `name=` to read the full record: `run_info` says what to
   generate, `options` what the analysis can be told, `histograms` what it
   fills, `cautions` whether to trust it.
2. **Write the card to match.** `run_info` is the process; `beams` and
   `energies_gev` are `Beams:idA/idB` and `Beams:eCM`. Follow the `pythia`
   skill for the card itself. Apply the analysis's own cuts where it says so
   (`PhaseSpace:mHatMin` for a Z analysis, `PhaseSpace:pTHatMin` for jets)
   -- generating what the analysis will reject only burns time.
3. **Generate straight into Rivet.** `PythiaFromRunCardTool` with
   `store="full"` and `rivet_analyses=["ANA", "MC_ZINC:LMODE=MU"]`. The
   events stream through a pipe; nothing but `rivet.yoda` and the manifest
   is written. A complete event is ~100 KiB of HepMC3, so for anything
   beyond a test run this is the only sensible route. For an existing file
   (yours or another generator's), `RunRivetTool`.
4. **Compare.** `RivetCompareTool` on the YODA file: per histogram χ²/ndf and
   p-value against the data, the same for the shape alone, the worst
   histograms first. Then `RivetPlotTool` for the pictures: data points,
   your prediction with its statistical band and weight-variation
   envelope, the ratio panel, an `index.html` gallery.
5. **Iterate on statistics before physics.** The run tool warns below 1000
   events; a published distribution needs 10^5 to 10^6. Read
   `mc_empty_bins` and the prediction's own band before concluding anything
   about the model. Independent seeds combine with `RivetMergeTool`.

## What must be true of the events

- **The complete record.** Rivet's projections for prompt leptons, b- and
  c-hadrons, taus and unstable particles walk the decay history. A
  final-state-only HepMC3 (the `finals` stores) makes every lepton prompt and
  every hadron stable. `store="full"` is required with `rivet_analyses` and
  is what `RunRivetTool` expects; the HepMC3 written is the one Pythia's own
  converter writes.
- **Beams and energy of the analysis.** Checked before the run from the
  events (or from Pythia after `init`); a mismatch is an error naming the
  analysis's energies. `ignore_beams=True` runs anyway, for a deliberate
  study, and the result then says the comparison to data is meaningless.
- **A cross section, when the analysis needs one.** Analyses with
  `needs_cross_section` normalise to it. Pythia's HepMC3 output and the
  stream carry the running cross section with every event; another
  generator's file may not, and `cross_section_pb` supplies it.

## Reading the comparison

The statistics are the plain ones: data and prediction uncertainties in
quadrature per bin, no correlations (the publications' correlation matrices
are not in the YODA files). So:

- `p_value >= 0.05`: **consistent** within the uncertainties shown.
- `shape consistent, normalisation off`: the shape agrees once the prediction
  is scaled to the data's integral; the cross section, the PDF, or a
  missing higher-order correction is the question, not the shower.
- `mc_empty_bins > 0`: generated events never reached those bins. More
  events, or the process genuinely does not populate them.
- Few events make everything look consistent: the prediction's own error
  dominates. Judge the model only when the band in the ratio panel is narrow
  next to the data's.
- `MC_*` analyses have no data. They are generator validation plots
  (`MC_ZINC`, `MC_JETS`, ...); the plot tool draws them prediction-only.

## Names, options and weights

- Names are `EXPERIMENT_YEAR_IINSPIREID` (`ATLAS_2012_I1204784`). Old
  `S`-numbers are aliases Rivet accepts with a warning; the catalogue
  suggests the canonical name on a typo.
- Options go on the name: `MC_ZINC:LMODE=MU`. They change the histogram
  paths (`/MC_ZINC:LMODE=MU/Z_pT`); the compare and plot tools follow.
- Weight variations in the Pythia card (`UncertaintyBands:List`,
  `VariationFrag:List`) reach Rivet as named weights; every histogram is
  booked once per variation and the plot tool draws their envelope.
  `skip_weights=True` runs the nominal only, several times faster.
- Status: `VALIDATED` is the only one to build a conclusion on;
  `UNVALIDATED`, `PRELIMINARY`, `OBSOLETE` come back as `cautions`.

## From your own Pythia build or another generator

With `pythia_path` configured, `PythiaCPPFromRunCardTool` generates through
that build and `rivet_analyses` streams its events exactly as
`PythiaFromRunCardTool` does from the wheel; nothing changes in the loop
above. Any HepMC3 file works with
`RunRivetTool`, gzipped or not. A C++ main program against a local Pythia
build can write HepMC3 with
`Pythia8ToHepMC` (`Pythia8Plugins/HepMC3.h`) for `RunRivetTool`, or call
Rivet directly through `Pythia8Plugins/Pythia8Rivet.h` when Pythia was
configured `--with-rivet`: `references/main_rivet_template.cc` is a working
program for either. Its YODA output goes to `RivetCompareTool` and
`RivetPlotTool` like any other.

## Installing Rivet

Rivet is C++ with Python bindings and is not on PyPI. The quickest working
installation is a conda-forge environment,
`conda create -n rivet -c conda-forge rivet`, whose prefix is `rivet_path`
(`tb config set heptapod rivet_path <prefix>`). `rivet-bootstrap` builds and
LCG releases work the same way: the tools need only `bin/rivet-config`.

## Reference files

- **`references/pitfalls.md`** -- the failure modes in full, each with the
  symptom you actually observe, the cause, and the fix.
- **`references/main_rivet_template.cc`** -- a C++ main program that runs
  Pythia and Rivet together, or writes HepMC3 for `RunRivetTool`.
- **`references/example_session.md`** -- an end-to-end example: finding
  the analysis, the card, the streamed run, the comparison and plots, with
  the tool outputs as they come back.
