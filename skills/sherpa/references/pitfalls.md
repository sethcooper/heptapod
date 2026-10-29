# Sherpa pitfalls

Each entry: the symptom you observe, why it happens, what to do. The tools
catch most of these before a run; this is what the message means when they
do, and what to look for when they cannot.

---

## S1 — A misspelt key changes nothing, and nothing says so

**Symptom.** The card sets `BEAM_ENERGIEZ: 6500` or `SHOWER: {FS_PT2MINN:
2}`; the run proceeds; the beams are at zero energy, or the shower at its
default. No message anywhere.

**Cause.** Sherpa reads the keys it knows and ignores the rest. Only with
`CHECK_SETTINGS: true` does it list, at the end of the run, the keys nothing
asked for.

**Fix.** `ValidateSherpaRunCardTool` checks every key against the settings
Sherpa 3.0.3 reads and suggests the intended spelling. `dry_run_init=True`
runs Sherpa with `CHECK_SETTINGS` on and reports what it never read.

---

## S2 — The collider is at half the energy

**Symptom.** A "13 TeV" sample has a cross section several times too small
and no high-mass tail.

**Cause.** `BEAM_ENERGIES` is the energy *per beam*. `BEAM_ENERGIES: 13000`
is a 26 TeV collider; 6500 is 13 TeV. Pythia's `Beams:eCM` is the
centre-of-mass energy, which is where the habit comes from.

**Fix.** Halve it. The validator warns above 100 TeV per beam, which is the
only unambiguous case; below that only you know what was meant.

---

## S3 — The weights are not 1

**Symptom.** Every event weight is some number like 43552.7, so a histogram
normalised "per event" is off by that factor, or `weighted` is false while
the weights are plainly not unity.

**Cause.** Sherpa's event weight is a cross section in pb. In unweighted
mode every event carries the total cross section, so the sum of weights
over the number of trials is σ. `weighted` reports whether the weights
*vary*, not whether they are 1.

**Fix.** Normalise by `xsec.weightSum`, or by σ over the number of events,
not by the event count alone. For a comparison with Pythia the result
carries `sigmaGen_mb` beside `sigmaGen_pb`.

---

## S4 — The variations are gone

**Symptom.** The card has `SCALE_VARIATIONS` or `PDF_VARIATIONS`; the
dataset holds one weight per event; the result warns that the variations
were not stored.

**Cause.** The `sherpa-mc` wheel's Python interface returns the nominal
weight only. The variations live in Sherpa's weights map, which the C++
driver reads.

**Fix.** Run the card with `SherpaCPPFromRunCardTool`. With a C++ compiler
on `PATH` it builds the driver against the wheel's own libraries, once, or
against the install `sherpa_path` names, and the variations arrive as named
weights.

---

## S5 — The run spends minutes before the first event

**Symptom.** Nothing happens for a long time after the tool starts; the log
shows "Integrating ..." lines.

**Cause.** Sherpa integrates the process at initialisation, which for a
multi-leg or NLO process takes minutes to hours. The result is kept in
`Results/` under `data_dir`.

**Fix.** Expected. Validate first so a typo does not cost the integration.
Run the dry run with `work_dir` set to the planned `data_dir`, and the
generation there reuses it. Keep one `data_dir` per process; a changed
process in an old directory is Sherpa's to detect and sometimes is not.

---

## S6 — A selector that cuts nothing

**Symptom.** `SELECTORS: [[Masss, 11, -11, 66, 116]]` (or any name Sherpa
does not have) leaves the sample uncut. No message.

**Cause.** An unknown selector is dropped silently, like an unknown key
(S1). Sherpa lists the selectors it has only on request
(`SHOW_SELECTOR_SYNTAX: 1`).

**Fix.** The validator checks every selector name, and the matrix-element
and shower generator names, against the lists Sherpa 3.0.3 prints, and
suggests the intended one. A value Sherpa cannot parse (`FS_PT2MIN: abc`)
is different: that is a fatal error at initialisation, which the dry run
reports with Sherpa's own message.

---

## S7 — The card is in Sherpa 2 syntax

**Symptom.** A card with `(run){ ... }(run)` or `BEAM_1 2212; BEAM_ENERGY_1
6500;` fails to parse, or parses as one nonsense key.

**Cause.** Sherpa 3 reads YAML. The `Run.dat` dialect of Sherpa 2 is not
accepted, and `-f Run.dat` on the command line is deprecated.

**Fix.** Rewrite as YAML (`references/card_template.yaml`). Most settings
kept their names; the structure changed.

---

## S8 — A PDF set the install does not have

**Symptom.** Fatal error at initialisation naming a PDF set; or a run that
uses a different set from the one the card asked for.

**Cause.** The `sherpa-mc` wheel ships its own PDF grids (NNPDF 3.1 NNLO
and CT14 among them) and no LHAPDF. A set LHAPDF would provide needs an
install built with LHAPDF and the set downloaded.

**Fix.** Use a shipped set, or a local Sherpa built with LHAPDF
(`sherpa_path`). The dry run's `effective_settings` show which `PDF_SET`
the run would use.

---

## S9 — Top quarks or W bosons that never decay

**Symptom.** Final states with a bare top or W in them; or a fatal error
about the width of an external particle.

**Cause.** A massive particle produced in the matrix element is stable to
Sherpa unless `HARD_DECAYS` is enabled, and Sherpa requires a zero width for
particles external to the matrix element.

**Fix.** `HARD_DECAYS: {Enabled: true}` and `PARTICLE_DATA: {6: {Width:
0}}` (likewise 24, 23, 25 as appropriate), as in the template.

---

## S10 — Multi-leg final states without merging

**Symptom.** `93 93 -> 11 -11 93{2}` runs, but the jet multiplicities make no
sense, or Sherpa complains about a missing merging scale.

**Cause.** The `{n}` suffix adds up to n extra partons at matrix-element
level; combining those samples with the shower is CKKW merging, which needs
its scale (`CKKW: 20` in the process options) and a jet criterion.

**Fix.** Either give `CKKW` with the merging scale in GeV, or drop the
`{n}` and let the shower make the jets.

---

## S11 — The analysis is at the wrong energy

**Symptom.** A Rivet analysis fills every histogram and agrees with nothing.

**Cause.** Rivet does not check the beam energy; see the `rivet` skill. With
Sherpa the energy mistake is usually S2.

**Fix.** The tools check the beams and √s against the analysis before the
run, from `BEAMS` and `BEAM_ENERGIES`, and refuse a mismatch.

---

## S12 — A second run in the same process hangs

**Symptom.** A Python program that constructs `Sherpa.Sherpa(...)` a second
time never returns from the constructor; a tool server that ran Sherpa
in-process hangs on its second call.

**Cause.** Sherpa keeps its state in statics and is initialised once per
process.

**Fix.** One process per run. The tools do this for you: the wheel runs in
a worker process, the driver is a process anyway. In your own programs,
loop over events inside one Sherpa object, not over Sherpa objects.

---

## S13 — Output in the terminal, or none in the log

**Symptom.** Sherpa's banner and progress appear where the tool's result
should be; or `sherpa.log` is empty after a failure.

**Cause.** Sherpa writes to the C-level stdout, which under the MCP
transport is the tool's own channel. The tools redirect it to `sherpa.log`
in `data_dir`; `-O0` keeps the rest quiet. A fatal error at construction,
before the redirect, can reach stderr.

**Fix.** Read `sherpa.log`; the tool quotes the lines after "Fatal error
thrown" in its message. Nothing needs changing in the card for this.
