# HepMC3 reference conversions

Each `<stem>.jsonl.gz` holds raw Pythia event records, one JSON object per
event, and `<stem>.hepmc.gz` the HepMC3 Pythia's own converter
(`Pythia8ToHepMC3`, `include/Pythia8Plugins/HepMC3.h`) wrote for the same
events. `tools/tests/test_event_io.py` rebuilds the HepMC3 events from the
records with `event_io.hepmc_event_from_record` and requires the same
particles, HepMC statuses, vertices and vertex positions.

The records were produced with Pythia 8.312 and HepMC3 3.3.1 from the
`<stem>.cmnd` cards here, with `Random:setSeed = on` and `Random:seed = 5`,
by a main program that wrote each event through `Pythia8ToHepMC` and, in
parallel, every particle as

    [id, status, mother1, mother2, daughter1, daughter2,
     px, py, pz, e, m, xProd, yProd, zProd, tProd, statusHepMC()]

| stem      | events | what it exercises                                      |
|-----------|--------|--------------------------------------------------------|
| softqcd   | 3      | diffraction and non-diffractive pp, beam remnants      |
| zll       | 1      | Drell-Yan with ISR/FSR and shower weight variations    |
| ee        | 2      | e+e- -> Z -> hadrons, string fragmentation ranges      |
| bbbar     | 1      | heavy-flavour hadrons, decay chains, displaced vertices|
