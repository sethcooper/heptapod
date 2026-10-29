# Example: Pythia 8 against ATLAS φ*η in Drell-Yan at 7 TeV

An end-to-end session, with the tool outputs abridged to what matters. The
sample is small (3000 events) so the run takes a minute; the comparison
numbers are what such a sample gives.

## 1. Find the analysis

```
RivetFindAnalysesTool(query="drell-yan phi*", beams="p+ p+", energy_gev=7000)
→ {"n_matches": 1, "analyses": [{"name": "ATLAS_2012_I1204784",
    "summary": "Measurement of angular correlations in Drell-Yan lepton pairs to probe Z/gamma* boson transverse momentum",
    "experiment": "ATLAS", "year": 2012, "status": "VALIDATED",
    "beams": ["p+ p+"], "energies_gev": [7000.0], "has_reference_data": true}]}

RivetFindAnalysesTool(name="ATLAS_2012_I1204784")
→ {"run_info": "Z/gamma* production with decays to electrons and/or muons.",
   "beams": ["p+ p+"], "energies_gev": [7000.0], "options": [],
   "histograms": [{"path": "/ATLAS_2012_I1204784/d01-x01-y01", "title": "phi*_eta spectrum, Z -> ee (bare)"},
                  {"path": "/ATLAS_2012_I1204784/d01-x01-y02", "title": "phi*_eta spectrum, Z -> mumu (bare)"}, ... 16 in all],
   "cautions": []}
```

The run_info fixes the card: Z/γ* at 7 TeV decaying to e or μ. `MC_ZINC`
(a validation analysis with no data) adds the Z mass and pT for a sanity
check; its `LMODE` option selects the channel.

## 2. The card

```
Beams:eCM = 7000.
WeakSingleBoson:ffbar2gmZ = on
23:onMode = off
23:onIfAny = 11 13
PhaseSpace:mHatMin = 40.
UncertaintyBands:doVariations = on
UncertaintyBands:List = {fsrup fsr:muRfac=0.5,fsrdown fsr:muRfac=2.0}
```

`ValidatePythiaRunCardTool(dry_run_init=True)` first, as always.

## 3. Generate into Rivet

```
PythiaFromRunCardTool(data_dir="zll7", cmnd_path="zll7.cmnd", n_events=3000, seed=1,
                      store="full", rivet_analyses=["ATLAS_2012_I1204784", "MC_ZINC:LMODE=MU"])
→ {"status": "ok", "format": "hepmc3", "store": "full", "accepted": 3000,
   "rivet": {"output_yoda": "zll7/rivet.yoda", "n_events": 3000.0,
             "analyses": {"ATLAS_2012_I1204784": {"histograms": 16, "filled": 16, "raw_entries": 4764},
                          "MC_ZINC:LMODE=MU":   {"histograms": 7,  "filled": 7,  "raw_entries": 7335}},
             "weight_variations": ["fsr.murfac=0.5", "fsr.murfac=2.0", "fsrdown", "fsrup"]},
   "xsec": {"sigmaGen_mb": 1.568e-06, ...},
   "next": "RivetCompareTool compares the histograms with the measured data; RivetPlotTool draws them together"}
```

No events file: `zll7/` holds `rivet.yoda`, the manifest, `run.cmnd` and the
logs. With a wrong card the tool would have stopped before Pythia ran:

```
PythiaFromRunCardTool(..., rivet_analyses=["ATLAS_2019_I1736531"])    # a 13 TeV analysis
→ Error: Incompatible Analyses
  - Reason: ATLAS_2019_I1736531: sqrt(s) = 7000 GeV is not among the analysis's 13000 GeV
```

## 4. Compare

```
RivetCompareTool(yoda_paths=["zll7/rivet.yoda"], labels=["Pythia 8.317"])
→ {"files": [{"label": "Pythia 8.317", "analyses": {
     "ATLAS_2012_I1204784": {"histograms_compared": 16, "consistent_at_5pct": 14, "mean_chi2_ndf": 0.91,
        "worst": [{"path": "/ATLAS_2012_I1204784/d02-x01-y01", "chi2_ndf": 1.655, "p_value": 0.0104,
                   "max_abs_pull": 4.1, "mean_ratio_mc_over_data": 1.028, "mc_empty_bins": 1,
                   "verdict": "1 empty prediction bins (more events needed)"}, ...]},
     "MC_ZINC:LMODE=MU": {"histograms_compared": 0, "without_reference_data": 7}}}],
   "report": "rivet/comparison.json"}
```

Fourteen of sixteen distributions agree at 5% with 3000 events, and the two
that do not are each missing their last bin for want of events. This says
the generator is in the right place; it does not yet test the model. At
10^5 events the prediction's uncertainty drops below the data's and the
same numbers become a statement about the tune.

## 5. Plot

```
RivetPlotTool(yoda_paths=["zll7/rivet.yoda"], labels=["Pythia 8.317"], output_dir="zll7/plots")
→ {"n_plots": 23, "index_html": "zll7/plots/index.html",
   "analyses": {"ATLAS_2012_I1204784": [{"histogram": "d01-x01-y01", "image": "zll7/plots/ATLAS_2012_I1204784/d01-x01-y01.png",
                                          "chi2_ndf": ["38.0/33"], "p_value": [0.251]}, ...]}}
```

Each image: the ATLAS points with their errors, the Pythia prediction as a
stepped line with its statistical band and the hatched envelope of the FSR
variations, and the prediction/data ratio beneath on the analysis's own
ratio scale (0.8 to 1.2 here). The legend carries χ²/ndf.

## 6. More statistics

Two more seeds, then one merged file:

```
PythiaFromRunCardTool(data_dir="zll7_s2", ..., seed=2, store="full", rivet_analyses=[...])
PythiaFromRunCardTool(data_dir="zll7_s3", ..., seed=3, store="full", rivet_analyses=[...])
RivetMergeTool(yoda_paths=["zll7/rivet.yoda", "zll7_s2/rivet.yoda", "zll7_s3/rivet.yoda"],
               output_path="zll7_all.yoda")
→ {"n_events": 9000.0, "cross_section_pb": 1568.4, ...}
```

`equivalent=True` (the default) is right for the same process with
different seeds; the analyses' `finalize` is rerun on the combined
histograms, so normalised distributions come out right.
