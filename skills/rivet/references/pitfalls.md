# Rivet pitfalls

Each entry: the symptom you observe, why it happens, what to do. The tools
catch most of these before a run; this is what the message means when they do,
and what to look for when they cannot.

---

## R1 — An analysis at the wrong energy runs and fills everything

**Symptom.** A 13 TeV analysis on 7 TeV events (or the reverse) runs to the
end, every histogram is filled, and the comparison is terrible everywhere.
Rivet printed nothing.

**Cause.** Rivet checks the beam *particles* against the analysis and removes a
mismatch with a warning. It does not check the beam *energies*; the analysis
runs on whatever arrives.

**Fix.** `RunRivetTool` and `PythiaFromRunCardTool(rivet_analyses=...)` read
the beams and √s and refuse an analysis whose `Energies` do not include them
(within 1%). The error names the analysis's energies; generate at one of
them, or pick an analysis for your energy with `RivetFindAnalysesTool`
(`beams=`, `energy_gev=`). `ignore_beams=True` overrides, knowingly.

---

## R2 — Every lepton is prompt, no hadron ever decays

**Symptom.** Non-prompt backgrounds vanish; b-tagging analyses find no
b-hadrons; `UnstableParticles`-based analyses (J/ψ, Λ, D mesons, τ) see
nothing; Rivet may warn about missing beam particles or decay vertices.

**Cause.** The HepMC3 file holds the final state only: two beams and the
stable particles as their direct products. Rivet's projections read the
decay history, which is not there. This is what a HepMC3 made from a
`finals` or `finals+kinematics` dataset looks like (fine for Delphes, not for
Rivet).

**Fix.** `store="full"`: the HepMC3 then carries the complete record, as
Pythia's own converter writes it (statuses 1, 2, 4 and the shower codes,
with the vertex genealogy). `rivet_analyses` requires it; `ConvertEventsTool`
writes it from a `full` dataset.

---

## R3 — "is not a known Rivet analysis"

**Symptom.** Rivet exits with that line and prints all two thousand names.

**Cause.** A misspelt name, or an old `S`-number alias that this Rivet
version no longer carries, or a user analysis whose plugin library is not
on the analysis path.

**Fix.** The tools check names first and suggest close matches. Search with
`RivetFindAnalysesTool(query=...)` rather than typing from memory; the
INSPIRE id alone finds it. For your own analysis, `analysis_path=[dir]`
with the compiled `RivetANA.so` and its `.info/.plot/.yoda` beside it.

---

## R4 — Histograms are filled with zero events

**Symptom.** `raw_entries: 0` for an analysis, every histogram `empty`, and a
warning that no event entered any histogram.

**Cause.** The events do not contain what the analysis selects: a Z
analysis given a QCD sample, a muon-channel analysis given electrons, a
`PhaseSpace:mHatMin` far above the analysis's mass window, or the analysis
option set to the wrong channel (`LMODE=EL` with Z → μμ events).

**Fix.** Read the analysis's `run_info` (`RivetFindAnalysesTool(name=...)`)
and make the card produce exactly that. Check `options`: many analyses need
`LMODE`, `PTMIN` or a scheme set to match the sample.

---

## R5 — Everything is "consistent" after 500 events

**Symptom.** p-values near 1 for every histogram; the ratio panel's band is
±50%.

**Cause.** With few events the prediction's own statistical uncertainty is
far larger than the data's, and χ² is small by construction. The test
compares two uncertain numbers; it cannot tell you about the model yet.

**Fix.** More events. Published distributions typically need 10^5 to 10^6;
the run tool warns below 1000. Stream with `rivet_analyses` rather than
writing a file; run independent seeds and combine them with
`RivetMergeTool`. Judge only when the prediction's band is narrow next to
the data's.

---

## R6 — "Cross Section Needed"

**Symptom.** The run refuses before starting: an analysis needs the cross
section and the events carry none.

**Cause.** Analyses with `NeedCrossSection: yes` normalise to σ. Pythia's
HepMC3 output and the stream record the running σ with every event; a file
from another generator, or a hand-written one, may not.

**Fix.** `cross_section_pb=<σ in pb>` on `RunRivetTool`, or regenerate
with `PythiaFromRunCardTool`. Note the unit: Pythia reports mb, Rivet wants
pb (1 mb = 10^9 pb).

---

## R7 — The comparison says "disagreement" with one bin at 60σ

**Symptom.** χ²/ndf in the hundreds, driven by a bin where the prediction is
exactly zero while the data is not.

**Cause.** An empty prediction bin has value 0 and uncertainty 0: no
generated event reached it. Treated as a prediction, its pull is the data
divided by the data's tiny error. It says nothing about the model, only
about the sample size.

**Fix.** `RivetCompareTool` leaves such bins out of χ² and reports them as
`mc_empty_bins`; the verdict says "more events needed". Generate more, or
accept that the tail is beyond the sample.

---

## R8 — A good shape with the wrong normalisation

**Symptom.** p-value ≈ 0, but `chi2_shape` is fine and
`normalisation_mc_over_data` is 0.8 or 1.3.

**Cause.** The distribution's shape is right and its integral is not: a
leading-order cross section against an NNLO-normalised measurement, a PDF
choice, a branching ratio, a missing fiducial factor.

**Fix.** This is a physics finding, not an error. Report both numbers.
If the analysis is normalised (`1/σ dσ/dx`), a normalisation offset there
instead means the fiducial selection differs — check the cuts in `run_info`.

---

## R9 — Weight variations make the run several times slower

**Symptom.** A card with `UncertaintyBands:List` or `VariationFrag:List`
runs Rivet much slower than the same card without; the YODA file holds many
copies of every histogram (`[fsrup]`, `[fsrdown]`, ...).

**Cause.** Rivet books and fills every histogram once per weight. Pythia
also emits a weight per individual variation and per group, so a list of
two variations can mean four extra weights.

**Fix.** Keep the variations when you want the envelope the plot tool
draws. Otherwise `skip_weights=True` (the run tool) or drop the list from
the card for the Rivet run.

---

## R10 — Rivet removed an analysis

**Symptom.** A warning "Rivet removed X: its beams do not match the events",
and X missing from the output; or the run fails with "every analysis was
incompatible with the first event's beams".

**Cause.** The beam particles differ from the analysis's `Beams` (e+e-
events into a pp analysis, p-Pb into pp). Rivet's own check.

**Fix.** The tools catch this before the run when the events are readable;
when Rivet reports it anyway (a user analysis, or a file without beam
information), fix the beams or choose the analysis for them.

---

## R11 — The HepMC3 file is gigabytes

**Symptom.** `max_output_gb` stops the generation; the disk fills.

**Cause.** A complete hadron-level event is ~100 KiB of HepMC3 ASCII; a
million events is 100 GB.

**Fix.** Stream: `rivet_analyses` on `PythiaFromRunCardTool` runs Rivet on a
pipe and writes no event file. If a file is needed (another tool reads it
too), `gzip_output=True` cuts it by about four; Rivet reads `.hepmc.gz`
directly.

---

## R12 — Status UNVALIDATED, PRELIMINARY, OBSOLETE

**Symptom.** `cautions` on the analysis record; the same text in the run
warnings.

**Cause.** Rivet's own validation of the analysis against the paper is
incomplete, or a newer analysis supersedes it. `SINGLEWEIGHT` means it
handles only the nominal weight.

**Fix.** Prefer a `VALIDATED` analysis of the same measurement. If none
exists, say so when reporting the comparison.

---

## R13 — A plot title or axis label looks wrong

**Symptom.** A label rendered as plain words with the backslashes gone, or a
macro left as text.

**Cause.** Rivet's `.plot` files are LaTeX for its own LaTeX-based plotting;
matplotlib renders a subset (mathtext). The plot tool translates the common
Rivet macros (`\pT`, `\TeV`, `\text`) and falls back to plain text when
mathtext refuses a label.

**Fix.** Cosmetic. The histogram's path and the gallery caption identify it;
for publication-quality typography, `rivet-mkhtml` with a LaTeX
installation reads the same YODA files.
