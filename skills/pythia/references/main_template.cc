// main_template.cc is a part of the HEPTAPOD package.
// Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
// HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// A minimal, complete Pythia 8 main program.
//
// Use a standalone main program (rather than a .cmnd card through
// PythiaFromRunCardTool) when you need something the card interface cannot
// express: UserHooks, per-event analysis inside the generation loop,
// custom cross-section reweighting, multiple subruns, or parallel
// generation.
//
// Build (with pythia8-config on PATH, e.g. from a conda env):
//   g++ main_template.cc -o main_template \
//       $(pythia8-config --cxxflags --libs)
// or inside the Pythia examples/ directory:
//   make main_template
//
// Run:
//   ./main_template main_template.cmnd
//
// Everything the card interface supports works here too: prefer readFile
// on a separate .cmnd over a wall of readString calls, so the same card
// can be validated with ValidatePythiaRunCardTool and reused by the
// heptapod tool.

#include "Pythia8/Pythia.h"

using namespace Pythia8;

int main(int argc, char* argv[]) {

  Pythia pythia;

  // Settings. Either read a card ...
  if (argc > 1) {
    if (!pythia.readFile(argv[1])) {
      // readFile returns false on an unknown setting or an out-of-range
      // value, having already printed which line was at fault. Stopping
      // here rather than pressing on is what turns a silent
      // misconfiguration into a visible one.
      cout << " Aborting: run card " << argv[1] << " was not accepted."
           << endl;
      return 1;
    }
  // ... or configure inline.
  } else {
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13600.");
    pythia.readString("HardQCD:all = on");
    pythia.readString("PhaseSpace:pTHatMin = 20.");
  }

  // Reproducibility. Without setSeed the seed value is ignored entirely.
  // Set after the card, this overrides any Random:seed in it.
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 12345");

  // init() returns false rather than throwing. Ignoring it means the event
  // loop below runs against an uninitialised generator.
  if (!pythia.init()) {
    cout << " Aborting: Pythia failed to initialise." << endl;
    return 1;
  }

  // Histograms come with Pythia; no external plotting library needed.
  Hist nCharged("charged multiplicity", 100, -0.5, 799.5);
  Hist pTlead("leading visible particle pT", 100, 0., 500.);

  const int nEvent = 1000;
  int nFailed = 0;

  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {

    // next() returns false on a failed event. Individual failures are
    // normal; counting them is how you notice when they stop being rare.
    if (!pythia.next()) {
      ++nFailed;
      continue;
    }

    // Every event carries a weight. It is 1 in an unweighted run, but not
    // under cross-section biasing, merging or shower variations, and a
    // histogram filled without it is then wrong in shape.
    const double weight = pythia.info.weight();

    // Final-state particles have status > 0. Neutrinos carry pT no
    // detector sees, hence isVisible for the leading particle.
    int    nCh   = 0;
    double pTmax = 0.;
    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (!p.isFinal()) continue;
      if (p.isCharged()) ++nCh;
      if (p.isVisible() && p.pT() > pTmax) pTmax = p.pT();
    }
    nCharged.fill(nCh, weight);
    pTlead.fill(pTmax, weight);
  }

  // Cross-section bookkeeping. sigmaGen is in mb; multiply by 1e9 for pb.
  // Normalise a weighted histogram by sigmaGen / weightSum, which reduces
  // to sigmaGen / nAccepted only when every weight is 1. Divide by the
  // bin width as well for a differential cross section.
  pythia.stat();
  cout << "\n Generated " << pythia.info.nAccepted() << " events, "
       << nFailed << " failed."
       << "\n sigma = " << pythia.info.sigmaGen() << " +- "
       << pythia.info.sigmaErr() << " mb" << endl;

  // Scale to pb per unit weight.
  const double scale = pythia.info.sigmaGen() * 1e9
    / pythia.info.weightSum();
  nCharged *= scale;
  pTlead   *= scale;
  cout << nCharged << pTlead;

  return 0;
}
