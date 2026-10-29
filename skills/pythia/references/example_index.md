# Pythia 8 example index

The Pythia distribution ships ~150 worked main programs. Almost anything
unusual you need has a working starting point among them, and reading the
right one is faster than deriving the setup from the manual.

Numbers here are the 8.3xx three-digit scheme (`mainNNN.cc`). Programs are
numbered by theme, so the leading digits alone narrow the search:

| Range | Theme |
|-------|-------|
| 1xx | Getting started, LHE input, HepMC/ROOT/Rivet output, matching & merging |
| 2xx | Tooling: PDFs, jet finders, parallelism, user hooks, reweighting, **Python** |
| 3xx | Physics by topic: QCD, DIS/photon, heavy flavour, Standard Model |
| 4xx | Showers (Vincia), heavy ions, hadronization variations, rescattering, cosmic rays |
| 5xx | BSM |

## Where things are

```
$PYTHIA/examples/          mainNNN.cc, mainNNN.py, mainNNN.cmnd, Makefile
$PYTHIA/share/Pythia8/xmldoc/     the manual, and the settings database itself
$PYTHIA/share/Pythia8/htmldoc/    the same manual, browsable
    SampleMainPrograms.html         prose description of every example
    ExampleKeywords.html            keyword -> example cross-reference
```

Build and run one with `make mainNNN && ./mainNNN`, from the `examples`
directory. A few need external libraries (HepMC, ROOT, Rivet, FastJet, LHAPDF)
configured at Pythia build time.

## By task

### Starting from nothing
| | |
|---|---|
| `main101` | The minimal program: beams, process, event loop, one histogram |
| `main102` | Z production at the Tevatron |
| `main103` | e+e- collisions |
| `main111` | Reading settings from a `.cmnd` file — the shape closest to heptapod's run-card tool |
| `main113` | Process selection driven entirely by a command file |

### Showering external (LHE) events
| | |
|---|---|
| `main121` | The basic LHE-input program |
| `main123` | LHE input via a command file, with subruns |
| `main124`, `main125`, `main127` | LHE variants |
| `main132`, `main133`, `main134` | LHE in, HepMC out |
| `main136` | HDF5/LHEH5 input |
| `main502` | LHE input where Pythia only hadronizes |

### Matching and merging
| | |
|---|---|
| `main151` | MC@NLO / aMC@NLO |
| `main152`, `main154` | POWHEG |
| `main153`, `main164` | MadGraph5_aMC@NLO |
| `main161` | Leading-order merging with jet finding |
| `main162` | CKKW-L, UMEPS, NL3, UNLOPS |
| `main163` | MLM matching |

### Jets
| | |
|---|---|
| `main211`, `main214`, `main216` | SlowJet — the clusterer heptapod's `JetClusterSlowJetTool` wraps |
| `main212`, `main215` | FastJet |
| `main213` | SlowJet and FastJet compared |
| `main242` | Anti-kT inside a user hook, with process veto |

### Python
| | |
|---|---|
| `main291` | The Python equivalent of `main101` |
| `main292`, `main297` | Parallelism from Python |
| `main293` | User hooks, jet finding, process veto |
| `main294` | Parsing the particle database *without* the bindings |
| `main295` | MadGraph / aMC@NLO from Python |
| `main296` | Total and partial cross sections |

Note these import `pythia8` (the source-built binding) and prepend `../lib` to
`sys.path`. With heptapod's `pythia8mc` wheel, drop that preamble and import
`pythia8mc` instead — and use `pythia.infoPython()` where the C++ examples use
`pythia.info`.

### Extending Pythia
| | |
|---|---|
| `main242`, `main243`, `main246`, `main248` | User hooks |
| `main244` | External resonance / resonance decay |
| `main245` | An external derived class for PDFs |
| `main261` | Cross-section biasing through a user hook |
| `main232` | Event filtering |
| `main233` | Reading an event record from elsewhere |

### Performance
| | |
|---|---|
| `main221`–`main225` | Parallelism (threads, OpenMP, Rivet) |
| `main482`, `main484`–`main487` | Switching beams/energies while reusing MPI initialisation |

### Physics topics
| | |
|---|---|
| QCD in pp | `main321` (min bias, cross sections), `main322`/`main323` (biasing), `main326` (MPI, double parton scattering), `main325` (diffraction) |
| DIS, photons | `main341` (DIS), `main342` (hard diffraction), `main343`–`main345` (photoproduction, UPC) |
| Heavy flavour, onia | `main361`/`main364` (B decays, EvtGen), `main363`/`main365` (onia), `main366` (charm, fixed target) |
| Top | `main362` (top mass and colour reconnection), `main367` (R-hadrons), `main368`–`main372` (toponium, spin correlations) |
| Higgs | `main381` |
| Showers | `main401`–`main407` (Vincia, weak showers, dark matter) |
| Heavy ions | `main421`–`main427` (Angantyr, centrality, Glauber) |
| Hadronization | `main441` (string shoving), `main442` (rope hadronization), `main443` (thermal model), `main234`, `main329` |
| Rescattering | `main461`–`main469` |
| Cosmic rays | `main424`, `main484`, `main485`, `main487` |

### BSM
| | |
|---|---|
| `main501`, `main508` | SUSY, driven by a command file |
| `main503`, `main504` | Extra dimensions |
| `main505` | R-hadrons and displaced vertices |
| `main507` | Dark matter with jet finding |
| `main509`, `main510`, `main511` | Hidden Valley — dark showers and dark hadronization |
| `main233` | Reading a SUSY event record |

## Finding more

Every example declares its own keywords on line 6:

```bash
grep -l "Keywords:.*displaced" $PYTHIA/examples/*.cc
grep -h "Keywords:" $PYTHIA/examples/main5*.cc
```

`share/Pythia8/htmldoc/ExampleKeywords.html` is the same index, pre-built.
