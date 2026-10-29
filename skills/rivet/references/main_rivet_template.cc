// main_rivet_template.cc is a part of the HEPTAPOD package.
// Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
// HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Pythia 8 and Rivet in one C++ program.
//
// Two ways to get Pythia events into Rivet from a local Pythia build:
//
//   (A) call Rivet directly, through Pythia8Plugins/Pythia8Rivet.h.
//       Needs a Pythia configured with --with-rivet (and HepMC3). Nothing
//       is written but the YODA file; this is the C++ equivalent of
//       PythiaFromRunCardTool(rivet_analyses=...).
//
//   (B) write HepMC3 with Pythia8Plugins/HepMC3.h and run RunRivetTool on
//       the file. Needs a Pythia configured with --with-hepmc3. The file
//       holds the complete record, so every Rivet projection works.
//
// Build (A):
//   g++ -std=c++17 -DPYTHIA8_WITH_RIVET main_rivet_template.cc \
//       -o main_rivet $(pythia8-config --cxxflags --libs) \
//       $(rivet-config --cppflags --ldflags --libs) -lHepMC3
// Build (B): the same without -DPYTHIA8_WITH_RIVET and the rivet-config
// part. With heptapod's PythiaCPPFromProgramTool, pass the extra flags as
// extra_cxxflags and extra_ldflags.
//
// Run:
//   ./main_rivet card.cmnd 100000 out.yoda ATLAS_2012_I1204784 \
//       MC_ZINC:LMODE=MU
//   ./main_rivet card.cmnd 100000 out.hepmc
// Mode (B) is chosen when the output name ends in .hepmc.
//
// The card is an ordinary Pythia .cmnd. The analysis names come from
// RivetFindAnalysesTool, and the beams and energy in the card must be the
// analysis's own: Rivet does not check the energy.

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#ifdef PYTHIA8_WITH_RIVET
#include "Pythia8Plugins/Pythia8Rivet.h"
#endif

#include <iostream>
#include <string>
#include <vector>

using namespace Pythia8;

int main(int argc, char* argv[]) {

  // Command line: card, event count, output, analyses.
  if (argc < 4) {
    cerr << "usage: " << argv[0]
         << " card.cmnd nEvents out.yoda|out.hepmc [ANALYSIS ...]\n";
    return 2;
  }
  const string card = argv[1];
  const int nEvents = atoi(argv[2]);
  const string output = argv[3];
  vector<string> analyses(argv + 4, argv + argc);
  const bool toFile = output.size() > 6
    && output.compare(output.size() - 6, 6, ".hepmc") == 0;

  // Generator.
  Pythia pythia;
  pythia.readFile(card);
  if (!pythia.init()) {
    cerr << "Pythia initialisation failed; see the messages above\n";
    return 3;
  }

  // (B) HepMC3 output: Pythia8ToHepMC writes the complete record, the
  // weights with their names, and the running cross section in pb.
  Pythia8ToHepMC* toHepMCPtr = toFile ? new Pythia8ToHepMC(output) : nullptr;

#ifdef PYTHIA8_WITH_RIVET
  // (A) Rivet in-process: one Pythia8Rivet object, every analysis added
  // before the first event, histograms written by done(). ignoreBeams(true)
  // would run analyses whose beams mismatch, for a deliberate study only.
  Pythia8Rivet* rivetPtr = nullptr;
  if (!toFile) {
    if (analyses.empty()) {
      cerr << "name at least one Rivet analysis\n";
      return 2;
    }
    rivetPtr = new Pythia8Rivet(pythia, output);
    for (const string& ana : analyses) rivetPtr->addAnalysis(ana);
    rivetPtr->init();
  }
#else
  if (!toFile) {
    cerr << "built without Rivet: give a .hepmc output and use RunRivetTool\n";
    return 2;
  }
#endif

  // Event loop. A failed event is skipped, not counted.
  int nAccepted = 0;
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!pythia.next()) continue;
    ++nAccepted;
    if (toHepMCPtr) toHepMCPtr->writeNextEvent(pythia);
#ifdef PYTHIA8_WITH_RIVET
    if (rivetPtr) (*rivetPtr)();
#endif
  }

  // The cross-section table to stdout, then the outputs are finalised.
  pythia.stat();
#ifdef PYTHIA8_WITH_RIVET
  if (rivetPtr) {
    rivetPtr->done();
    delete rivetPtr;
  }
#endif
  delete toHepMCPtr;

  // What RunRivetTool reports from the file is here for the in-process
  // run: sigmaGen in mb (Rivet works in pb), the weight sum for weighted
  // events.
  cout << "accepted " << nAccepted << " events, sigmaGen = "
       << pythia.info.sigmaGen() << " +- " << pythia.info.sigmaErr()
       << " mb, weightSum = " << pythia.info.weightSum() << "\n";
  return 0;
}
