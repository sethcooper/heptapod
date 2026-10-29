---
name: pythia
bundle: event_gen
description: Write Pythia 8 `.cmnd` run cards and standalone C++/Python main programs that initialise on the first try. Use whenever the user mentions Pythia, a `.cmnd` file, showering LHE events, hadronization, parton showers, `Beams:eCM`, `HardQCD`, `SoftQCD`, minimum bias, or generating events with PythiaFromRunCardTool. Also use proactively before writing a run card from scratch — Pythia accepts some mistakes silently (a misspelled `on` reads as `off`) and reports the rest only by failing `init()`, so a single typo costs a whole run.
---

# Pythia 8: run cards that initialise on the first try

Pythia does not raise on a bad card. It prints a complaint and makes `init()`
return `false` -- `PythiaFromRunCardTool` then reports Pythia's error lines and
keeps its full output in `data_dir/pythia.log` -- and some mistakes it accepts
without a word: a misspelled `on` reads as `off`, a non-numeric mass as 0.
**Validate before you generate**, so neither costs a run.

```
.cmnd card ──[ValidatePythiaRunCard]──▶ ok ──[PythiaFromRunCard]──▶ events.root
                     │
                     └─▶ "line 4: unknown setting 'PhaseSpace:pTHatMinn';
                          did you mean PhaseSpace:pTHatMin?"
```

## The loop that works

1. **Find the settings.** `PythiaSettingsSearchTool` searches all ~2000 of them
   by physics phrase — `"beam energy"` → `Beams:eCM`, `"pt cut"` →
   `PhaseSpace:pTHatMin` — and returns type, default, and allowed range. Do not
   guess names from memory; the naming is inconsistent enough that plausible
   guesses (`PhaseSpace:pTMin`, `Beams:sqrtS`) do not exist.
2. **Write the card.** Start from `references/card_template.cmnd`.
3. **Validate it.** `ValidatePythiaRunCardTool` with `dry_run_init=True`. The
   line checks catch typos, bad types, out-of-range values, and unknown PDG
   ids, against the installed Pythia's own database and parser. The dry run catches combinations that are individually valid but
   collectively impossible, returns Pythia's own message, and reports the
   **effective configuration** — what Pythia actually holds after
   initialisation, which is not always what the card asked for (see below).
4. **Generate.** `PythiaFromRunCardTool`. Read the `summary` it returns rather
   than the events file, and check `weighted` before analysing anything.
   The tool runs Pythia's Python interface (the `pythia8mc` wheel). When a
   Pythia build of your own is configured as `pythia_path`,
   `PythiaCPPFromRunCardTool` runs that build through a compiled driver
   from the same card; the dataset, summary and manifest are the same, and
   the manifest records which Pythia ran.

## Do not read the events to check a run

A hadron-level event is ~7 KiB in the default ROOT output and ~60 KiB as JSONL,
so a real sample is gigabytes. The result and manifest instead carry a
`summary` accumulated during generation — particles per event, mean leading
pT, most common species by name. That is what tells you whether the card
produced the process you meant. `max_output_gb` (default 2.0) stops a runaway
request and keeps what it wrote.

## Six things that fail silently

### 1. A flag typo does not fail — it turns the flag off

Pythia's `boolString` returns true only for `true/1/on/yes/ok`. Every other
string is false, with **no warning at all**:

```
HadronLevel:all = onn        ! reads as OFF. No error. Parton-level events.
```

Nothing in the log, nothing in the return value. You get a full run of events
with no hadronization. The validator flags any boolean outside the recognised
spellings for exactly this reason.

### 2. LHE showering: pass `lhe_path`, and read `run.cmnd`

Given `lhe_path` (with `shower_lhe=True`), `PythiaFromRunCardTool` appends
`Beams:frameType = 4` and `Beams:LHEF = <absolute path>` to the end of the
card, where Pythia's last-assignment-wins applies them however the card is
written. The card needs neither line for the tool -- but without both, it does
not read an LHE file anywhere else. `data_dir/run.cmnd` is the card exactly as
it ran.

### 3. The tool's seed beats the card's

The tool's `seed` is appended after the card, so it overrides any
`Random:seed` there; without it the card's seed applies, and without that
every run uses Pythia's same default. Pass a different `seed` per independent
run. `Random:seed` does nothing unless `Random:setSeed = on`.

### 4. A tune discards everything you set above it

`Tune:pp` and `Tune:ee` are not labels — reading the line *applies* the tune
and resets every parameter it owns. So this loses your value:

```
MultipartonInteractions:pT0Ref = 3.14
Tune:pp = 14                  ! pT0Ref is now 2.28, silently
```

and this keeps it:

```
Tune:pp = 14
MultipartonInteractions:pT0Ref = 3.14
```

**Put the tune line first.** `dry_run_init=True` catches the wrong order by
diffing the card against Pythia's effective configuration.

### 5. Weighted events analysed as unweighted

`PhaseSpace:bias2Selection`, ME/PS merging, and shower uncertainty variations
all produce weighted events, with weights spanning many orders of magnitude
within one sample. Every event carries a `weight` field; the generation result
reports `"weighted": true` with the observed range whenever any weight differs
from 1.0. Ignoring it does not rescale a histogram — it produces a wrong shape.
Normalise by `sigmaGen / weightSum`, not `sigmaGen / nAccepted`.

### 6. The default store spec hides the hard process

The default `store="finals+kinematics"` records only final-state particles.
Any reconstruction that needs the resonance itself, or mother/daughter links,
needs `store="full"`. Decide this before generating — regenerating is the only
fix.

## Particle properties are not settings

Masses, widths, lifetimes and decay switches do not live in the settings
database and `PythiaSettingsSearchTool` will not find them. They are addressed
by PDG id, on lines that Pythia routes to the particle data table instead:

```
6:m0 = 172.5                 ! top mass
15:mayDecay = off            ! make taus stable
23:onMode = off              ! turn off all Z decays ...
23:onIfAny = 13 -13          ! ... then re-enable Z → μμ
4900101:m0 = 10.0            ! a Hidden Valley quark, already in the table
9900001:new = X void 1 0 0 500.  ! a new state: define it before any line that uses it
```

The rule is mechanical: a line whose first non-blank character is a **digit**
goes to `ParticleData`; a line starting with a **letter** goes to `Settings`.
Everything else is a comment — which is why both `!` and `#` work.

## Before writing a card

1. **Internal process or external LHE?** Internal means a process group
   (`HardQCD:all`, `Top:gg2ttbar`, …) plus `Beams:eCM`. External means
   `lhe_path` passed to the tool (which sets `Beams:frameType = 4` and
   `Beams:LHEF`), and *no* process group.
2. **What has to survive to the event record?** Set decay switches and the
   tool's `store` spec now, not after the run.
3. **Where do the events go next?** The default ROOT output suits analysis in
   Python. For Delphes or Rivet, generate with `output_format="hepmc3"`, or
   convert an existing dataset with `ConvertEventsTool`. The HepMC3 file
   carries the weights and cross section, and either the final state alone
   (enough for Delphes) or, with `store="full"`, the complete event
   genealogy Pythia's own HepMC3 output has -- which Rivet needs whenever an
   analysis asks about decay history (prompt leptons, b-hadrons, unstable
   particles). For Rivet, use `store="full"`, and prefer `rivet_analyses`
   (the `rivet` skill): the events then stream into Rivet through a pipe and
   no multi-gigabyte file is written.
4. **Is a cut needed?** `HardQCD:all = on` without `PhaseSpace:pTHatMin`
   diverges into soft junk. `SoftQCD` processes take no such cut.

## Standalone main programs

For work that does not fit the run-card tool — custom `UserHooks`, per-event
analysis in the generation loop, parallel runs — write a main program instead.
Templates: `references/main_template.cc` and `references/main_template.py`.
A C++ program runs against a Pythia build of your own through
`PythiaCPPFromProgramTool` (`pythia_cpp` bundle), which compiles it with the
install's own flags and runs it in the sandbox; a Python program uses the
`pythia8mc` wheel.

One API difference matters. In the `pythia8mc` wheel that heptapod uses,
`pythia.info` **does not exist**; it is `pythia.infoPython()`. C++ examples
using `pythia.info` translate to Python only with that substitution.

`references/example_index.md` maps physics topics to the ~150 `mainNNN`
examples in the Pythia distribution — the fastest route to a working starting
point for anything unusual.

## Reference files

- **`references/pitfalls.md`** — the failure modes in full, each with the
  symptom you actually observe, the cause, and the fix.
- **`references/card_template.cmnd`** — an annotated card covering both the
  standalone and the LHE-showering shapes.
- **`references/main_template.cc`** / **`references/main_template.py`** —
  minimal working main programs with the event loop and cross-section
  bookkeeping in place.
- **`references/example_index.md`** — keyword → `mainNNN` index of the official
  examples, with the file layout of the Pythia distribution.
