# Pythia run-card pitfalls

Each entry: the symptom you actually observe, why it happens, and the fix.
Behaviour here was checked against Pythia 8.317 and heptapod's
`PythiaFromRunCardTool`, not inferred from the manual.

---

## P1 — "Pythia Init Failed"

**Symptom.** `PythiaFromRunCardTool` returns:

```
Error: Pythia Init Failed
- Reason: Initialization returned false
- Context: Error in ProcessLevel::init: no process switched on
           Abort from Pythia::init: processLevel initialization failed
```

**Cause.** Pythia reports bad input by printing and returning `false` from
`init()`; it never raises. The tool cannot let that output reach stdout --
under the MCP stdio transport stdout is the JSON-RPC channel -- so it writes
it to `data_dir/pythia.log` and puts Pythia's error and abort lines in the
error's context. (Pythia prints nothing as it happens under `Print:quiet =
on`; the tool calls `stat()` on failure, which always lists them.)

**Fix.** Read the context, then `pythia.log` for the full story. Better,
catch it before a run: `ValidatePythiaRunCardTool` checks every line, and
with `dry_run_init=True` it initialises Pythia and returns its messages:

```json
{"init_ok": false,
 "messages": ["Error in ProcessLevel::init: all processes have vanishing cross sections",
              "Abort from Pythia::init: processLevel initialization failed"]}
```

Outside the tools, the same thing by hand:

```python
import pythia8mc
p = pythia8mc.Pythia("", printBanner=False)
p.readFile("run.cmnd")     # prints the offending line
print(p.init())
p.stat()                   # lists every error and warning, even under Print:quiet
```

---

## P2 — A misspelled boolean silently disables the setting

**Symptom.** The run completes, event counts look normal, and the physics is
wrong — no hadrons, or MPI missing.

**Cause.** `boolString` in the Pythia source returns true only for
`true`, `1`, `on`, `yes`, `ok`. **Every other string is false**, with no
warning and no effect on `init()`:

```
HadronLevel:all = onn        ! reads as off
PartonLevel:MPI = enabled    ! reads as off
```

**Fix.** Use `on`/`off`. The validator reports any boolean outside the
recognised spellings as an error, because Pythia will not.

---

## P3 — An LHE card that only works inside the tool

**Symptom.** A card showers its LHE file under `PythiaFromRunCardTool` but
generates internal events -- or fails -- in a standalone program.

**Cause.** Given `lhe_path`, the tool appends `Beams:frameType = 4` and
`Beams:LHEF = <absolute path>` to the end of the card it runs (Pythia's last
assignment wins, so this holds however the card spells either line) and
comments out any `Beams:LHEF` already there. The card itself need not carry
either line -- but then nothing else reads it as an LHE card. Without
`frameType = 4`, Pythia ignores `Beams:LHEF` altogether.

**Fix.** For the tool, pass `lhe_path` with `shower_lhe=True`. For a card
meant to run anywhere, write both lines. Either way, `data_dir/run.cmnd` is
the card exactly as it ran. An LHE file holding fewer events than `n_events`
ends the run early; the result reports `lhe_exhausted` rather than failures.

---

## P4 — Seeds

**Symptom.** Two runs meant to differ produce identical events, or a run
meant to be reproducible is not.

**Cause.** The tool's `seed`, when given, is appended after the card and
wins over any `Random:seed` in it; without one, the card's seed applies, and
without that, Pythia's fixed default -- so every unseeded run is identical.
`Random:seed` does nothing unless `Random:setSeed = on`. The tool accepts
seeds in [1, 900000000]: Pythia rejects larger ones, and 0 seeds from the
clock, which no one can reproduce.

**Fix.** Pass `seed` to the tool, a different one per statistically
independent run, and record it -- the manifest does.

---

## P5 — `Main:numberOfEvents` has no effect

**Symptom.** The card says 100000, the tool produces 1000.

**Cause.** `PythiaFromRunCardTool` loops over its own `n_events` runtime field
and never reads `Main:numberOfEvents`. The setting is real and does work in a
standalone main program.

**Fix.** Pass `n_events` to the tool.

---

## P6 — The hard process is missing from the output

**Symptom.** No resonance in the dataset; nothing to reconstruct against.

**Cause.** The default `store="finals+kinematics"` keeps only final-state
particles (status > 0). Intermediates — the Z, the top, any BSM state — are
not final.

**Fix.** Use `store="full"` for every particle with its mother/daughter
indices. In ROOT or JSONL the record is stored as Pythia produced it; as HepMC3
it becomes the vertex genealogy, exactly as Pythia's own converter writes it.
Decide before generating: the history is not recoverable afterwards. To keep
only some species, `select_pids` filters what is stored (a filtered record has
no genealogy, so it reaches HepMC3 as its final state).

---

## P7 — `HardQCD:all` without a pT cut

**Symptom.** Enormous cross section, every event at the lowest available pT,
the sample useless for anything hard.

**Cause.** The QCD 2→2 cross section diverges as pT → 0. Pythia will happily
generate there.

**Fix.** Always pair `HardQCD:*` with `PhaseSpace:pTHatMin`. For a steeply
falling spectrum where you need reach at high pT, use
`PhaseSpace:bias2Selection = on` with `PhaseSpace:bias2SelectionPow` — and then
weight every event by its `weight` field (see P13). Biased events are *not*
equally probable, and a biased sample analysed unweighted is wrong by orders of
magnitude, not by a normalisation factor.

`SoftQCD:*` processes take no such cut — they *are* the low-pT region and are
regularised internally.

---

## P8 — Particle properties are not settings

**Symptom.** Searching for "top mass" returns nothing useful;
`ParticleData:m0 = 172.5` is rejected as an unknown setting.

**Cause.** Two separate databases. `Settings::readString` routes any line whose
first non-blank character is a **digit** to `ParticleData` instead:

```
6:m0 = 172.5                 ! top mass
6:mWidth = 1.32              ! top width
15:mayDecay = off            ! stable taus
23:onMode = off              ! all Z channels off
23:onIfAny = 13 -13          ! Z -> mu mu back on
```

An id absent from the table is an error (`input particle not found in Particle
Data Table`) and aborts init, so a BSM state must be introduced first, with an
`<id>:new = ...` line above any line that modifies it. An SLHA file or an LHE
header cannot do this for the card: both are read during `init()`, after the
card, so the card's lines for that id have already been rejected.

**Fix.** Use the `<pdgid>:<property>` form. `PythiaSettingsSearchTool` does not
index these; the validator does check the ids and property names.

---

## P9 — A value containing a space is truncated

**Symptom.** A path or a name is cut off at the first space, with no error.

**Cause.** Pythia reads exactly two whitespace-separated tokens per line. This
is also *why* a trailing `! comment` is harmless — but it means

```
Beams:LHEF = /path/with a space/events.lhe
```

sets the value to `/path/with`.

**Fix.** Avoid spaces in paths used from run cards.

---

## P10 — xmldoc version mismatch

**Symptom.**

```
PYTHIA Abort from Pythia::checkVersion: unmatched version numbers :
  in code 8.312 but in XML 8.317
PYTHIA Abort from Pythia::init: constructor initialization failed
```

**Cause.** The compiled Pythia library and `share/Pythia8/xmldoc` come from
different releases. This happens readily when two Pythia packages share one
environment — e.g. a conda `pythia8` package and the `pythia8mc` wheel, where
whichever installed last owns `$PREFIX/share/Pythia8/xmldoc`.

**Fix.** Point the mismatched build at its own xmldoc via `PYTHIA8DATA`:

```bash
PYTHIA8DATA=/path/to/matching/share/Pythia8/xmldoc ./my_main
```

Or install the two into separate environments. `pythia8mc` is affected too: it
reads `$PREFIX/share/Pythia8/xmldoc`, the same shared directory, and
`PYTHIA8DATA` redirects every build in the environment at once — so the
separate environment is the fix that holds.

---

## P11 — A card that passes every static check still fails

**Symptom.** The validator reports `valid: true`, and `init()` still returns
false.

**Cause.** Static checks verify each line in isolation. They cannot know that
`PhaseSpace:pTHatMin = 1e9` at `Beams:eCM = 13000.` leaves no phase space, or
that a process group and a set of cuts are jointly empty. Pythia only discovers
this while computing cross sections.

**Fix.** Use `dry_run_init=True`. It costs one initialisation, no events, and
returns the real reason:

```
PYTHIA Error in ProcessLevel::init: all processes have vanishing cross sections
```

---

## P12 — A tune silently discards parameters set before it

**Symptom.** The card sets a tuned parameter, the run completes, and the
parameter has its default value. No warning, and the card is entirely valid.

**Cause.** `Tune:pp` and `Tune:ee` are not passive labels — reading the line
*applies* the tune, resetting every parameter the tune owns. Order therefore
decides the outcome:

```
MultipartonInteractions:pT0Ref = 3.14
Tune:pp = 14                          ! -> pT0Ref is now 2.28
```

```
Tune:pp = 14
MultipartonInteractions:pT0Ref = 3.14 ! -> pT0Ref stays 3.14
```

Note that `Tune:pp = 14` is the *default* value, so the line does not even show
up as a changed setting — but it still applies the tune.

**Fix.** Put the `Tune:` line **first**, then override individual parameters
below it. `ValidatePythiaRunCardTool` with `dry_run_init=True` catches this: it
compares what the card assigns against Pythia's effective configuration and
reports anything that did not survive:

```
warning: MultipartonInteractions:pT0Ref = 3.14 did not survive initialisation;
         the effective value is 2.28
```

Related: overriding `PDF:pSet` away from what a tune assumes is physically
inconsistent even though nothing complains — MPI parameters like `pT0Ref` are
tuned against a specific PDF.

---

## P13 — Weighted events analysed as if unweighted

**Symptom.** Cross sections and distributions off by orders of magnitude, with
the shape wrong too — not a constant normalisation error.

**Cause.** Several perfectly ordinary settings produce *weighted* events:
cross-section biasing (`PhaseSpace:bias2Selection`), ME/PS merging, and shower
uncertainty variations. With biasing on, weights within a single sample span
many orders of magnitude:

```
0.0183, 0.00771, 0.00134, 0.000844, 2.83e-08, 0.00184, 1.27e-05, 9.24e-07
```

**Fix.** Every event in the dataset carries a `weight` field. Use it in every
histogram, efficiency, and yield, and normalise by `sigmaGen / weightSum`
rather than `sigmaGen / nAccepted`. `PythiaFromRunCardTool` reports

```json
{"weighted": true, "weight_range": [1.46e-06, 5.66e-02], "warnings": ["..."]}
```

whenever any event weight differs from 1.0, and the manifest records
`weight_sum`, `weight_min`, `weight_max`, `weight_mean` under
`outputs.weights`. If `weighted` is false, every weight is exactly 1.0 and
plain counting is correct. A run with weight variations also stores each
event's variation weights as `weights`, named in `outputs.weights.names`.

---

## P14 — Generated samples are large, and nothing is meant to be opened

**Symptom.** A generation request fills the disk, or produces a file nothing
can open.

**Cause.** A hadron-level event holds ~500 final-state particles out of
~1500 in its full record. Per event, the default store spec costs about
7 KiB in ROOT and 60 KiB in JSONL; `store="full"` costs about 60 KiB even in
ROOT.

| events | ROOT, default | JSONL, default |
|--------|---------------|----------------|
| 1,000 | 0.007 GB | 0.06 GB |
| 100,000 | 0.7 GB | 6 GB |
| 1,000,000 | 7 GB | 60 GB |

**Fix.** Do not read the events to check a run. Read the `summary` in the
tool result, or `outputs.summary` in the manifest, which is accumulated
during generation:

```json
{"n_events": 20,
 "particles_per_event": {"mean": 498.9, "min": 235, "max": 744},
 "leading_pt_mean_gev": 8.55,
 "top_species": [{"id": 22, "name": "gamma", "per_event": 231.2},
                 {"id": 211, "name": "pi+", "per_event": 99.7}]}
```

That is enough to tell whether the card produced the process intended.

`max_output_gb` (default 2.0) stops generation before the file runs away,
sets `truncated: true`, and keeps everything already written. Raise it
deliberately if a large sample really is wanted, and keep the default ROOT
output and store spec when it is.

---

## P15 — `valid: true` is not "the physics is right"

The validator checks that Pythia will read and initialise the card. It cannot
check that you selected the process you meant, cut where you intended, or left
the right particles stable. Read back `data_dir/run.cmnd` — the card exactly
as it ran — and check a few events with `store="full"` before generating the
bulk sample.
