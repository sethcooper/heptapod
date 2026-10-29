# Example: Sherpa 3 at the Z pole, validated, generated, compared

An end-to-end session with the tool outputs abridged to what matters. The
numbers are from the `sherpa-mc` 3.0.3 wheel on a laptop; the sample is
small so the run takes seconds after the integration.

## 1. The card

```yaml
BEAMS: [11, -11]
BEAM_ENERGIES: 45.6
PDF_LIBRARY: None
PROCESSES:
- 11 -11 -> 93 93: {}
SCALE_VARIATIONS: [[0.25, 0.25], [4.0, 4.0]]
```

`BEAM_ENERGIES` is per beam: 45.6 GeV each is √s = 91.2 GeV.

## 2. Validate

```
ValidateSherpaRunCardTool(card_path="lep.yaml")
→ {"valid": true, "counts": {"error": 0, "warning": 0}, "findings": [],
   "variations": true, "sherpa_version": "3.0.3"}
```

With a typo the same call says what Sherpa never would:

```
ValidateSherpaRunCardTool(card_text="BEAMS: [11, -11]\nBEAM_ENERGIEZ: 45.6\n...")
→ {"valid": false, "findings": [
     {"severity": "error", "line": 2, "key": "BEAM_ENERGIEZ",
      "message": "'BEAM_ENERGIEZ' is not a Sherpa setting; it would be ignored without a word",
      "suggestion": "did you mean BEAM_ENERGIES?"},
     {"severity": "error", "key": "BEAM_ENERGIES",
      "message": "BEAM_ENERGIES is not set; Sherpa's default (none) ends initialisation with an error"}]}
```

A selector Sherpa does not have is caught the same way (`'Masss' is not a
Sherpa selector; did you mean Mass?`), since Sherpa would apply no cut and
say nothing.

The dry run initialises Sherpa on the card in the directory the generation
will use, so the integration is done once:

```
ValidateSherpaRunCardTool(card_path="lep.yaml", dry_run_init=True, work_dir="zpole")
→ {"valid": true, "dry_run": {"ok": true, "fatal": [], "unused_settings": {},
   "effective_settings": [{"setting": "BEAMS", "default": "0, 0", "card": "11, -11", "final": "11, -11"},
                          {"setting": "BEAM_ENERGIES", "default": "0, 0", "card": "45.6", "final": "45.6"},
                          {"setting": "MI_HANDLER", "default": "Amisic", "final": "None"}, ...]}}
```

The `effective_settings` are what Sherpa used: here it switched multiple
interactions off by itself for a lepton collider.

## 3. Generate

```
SherpaCPPFromRunCardTool(data_dir="zpole", card_path="lep.yaml", n_events=10000, seed=3)
→ {"status": "ok", "events": "zpole/events.root", "format": "root",
   "store": "finals+kinematics", "accepted": 10000, "failed": 0, "backend": "cpp",
   "xsec": {"sigmaGen_pb": 39414.2, "sigmaErr_pb": 861.1,
            "sigmaGen_mb": 3.94e-08, "sigmaErr_mb": 8.6e-10, "weightSum": 435527182.3},
   "weighted": false,
   "weight_variations": ["MUR=0.5__MUF=0.5__LHAPDF=-1", "MUR=2__MUF=2__LHAPDF=-1"],
   "summary": {"n_events": 10000, "particles_per_event": {"mean": 20.8, ...},
               "top_species": [{"name": "gamma", ...}, {"name": "pi+", ...}, ...]}}
```

Three things to read here. The cross section is in pb, 39.4 nb, as Sherpa
reports it, with the mb figure beside it for Pythia. `weighted` is false
because every event carries the same weight; that weight is 43 552.7 pb,
not 1, and `weightSum` is the weight times the event count. The variations
are present because the card has `SCALE_VARIATIONS` and the run went
through `SherpaCPPFromRunCardTool`: with a C++ compiler on `PATH` and no
`sherpa_path`, the driver was built against the wheel's own libraries, once.
`SherpaFromRunCardTool` on the same card reports only the nominal weight
and says so in `warnings`.

Generation runs at several hundred events per second for this process;
the first run in `zpole/` spent its first seconds on the integration the
dry run had already done.

## 4. Into Rivet, beside Pythia

```
SherpaFromRunCardTool(data_dir="zpole_rivet", card_path="lep.yaml", n_events=10000,
                      store="full", rivet_analyses=["ALEPH_1996_I428072"])
PythiaFromRunCardTool(data_dir="zpole_pythia", cmnd_path="lep.cmnd", n_events=10000,
                      store="full", rivet_analyses=["ALEPH_1996_I428072"])
RivetCompareTool(yoda_paths=["zpole_rivet/rivet.yoda", "zpole_pythia/rivet.yoda"],
                 labels=["Sherpa 3.0.3", "Pythia 8.317"], analyses=["ALEPH_1996_I428072"])
RivetPlotTool(yoda_paths=[...], labels=[...], output_dir="zpole/plots")
```

The same analysis, the same data, two labelled predictions per plot. The
beams and √s of both runs are checked against the analysis before either
starts; a 13 TeV analysis offered here is refused with the analysis's own
energies in the message.
