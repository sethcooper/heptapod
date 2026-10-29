// main_template.cc is a part of the HEPTAPOD package.
// Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
// HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// A minimal, complete Sherpa 3 main program.
//
// Use a main program (rather than a card through SherpaFromRunCardTool)
// when you need the event record or the weights map inside the generation
// loop: a custom observable per event, a veto, the matrix-element weight
// of a configuration. Everything a card can say still goes in the card;
// the program reads it and adds the loop.
//
// Build against a Sherpa install with heptapod's SherpaCPPFromProgramTool,
// which uses the install's own flags, or by hand:
//   c++ -std=c++17 main_template.cc -o main_template \
//       $(Sherpa-config --cppflags --ldflags) -lSherpaMain
// (the sherpa-mc wheel's Sherpa-config reports the wrong paths; the tool
// knows the right ones.)
//
// Run, in the directory where Sherpa may keep its Process and Results:
//   ./main_template -e1000 -O0 Sherpa.yaml

#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Org/Exception.H"
#include "ATOOLS/Phys/Blob_List.H"
#include "ATOOLS/Phys/Particle.H"
#include "ATOOLS/Phys/Weights.H"

#include <iostream>
#include <string>

using std::string;

int main(int argc, char* argv[]) {

  // Sherpa takes its command line as Sherpa itself does: -e for the event
  // count, -R for the seed, -O for the output level, the card last.
  SHERPA::Sherpa* sherpaPtr = new SHERPA::Sherpa(argc, argv);
  try {
    if (!sherpaPtr->InitializeTheRun()
      || !sherpaPtr->InitializeTheEventHandler()) {
      std::cerr << "Sherpa initialisation failed; see the messages above\n";
      return 3;
    }
  } catch (const ATOOLS::Exception& e) {
    std::cerr << e << "\n";
    return 3;
  }

  // Event loop. The blob list is the event: one blob per vertex, from the
  // beams through the hard process, the showers and hadronisation to the
  // hadron decays. A final-state particle is active and goes into no blob.
  const long nEvents = sherpaPtr->NumberOfEvents();
  long nAccepted = 0;
  double weightSum = 0.;
  int nCharged = 0;
  for (long iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!sherpaPtr->GenerateOneEvent()) continue;
    ++nAccepted;
    const ATOOLS::Blob_List& blobs = sherpaPtr->GetBlobList();

    // The nominal weight is a cross section in pb; the variations of the
    // "All" group are factors on it, named as Sherpa names them.
    const double weight = blobs.Weight();
    weightSum += weight;
    const ATOOLS::Weights_Map weightsMap = blobs.WeightsMap();
    if (iEvent == 0) {
      std::map<string, ATOOLS::Weights>::const_iterator all
        = weightsMap.find("All");
      if (all != weightsMap.end())
        for (int k = 1; k < int(all->second.Size()); ++k)
          std::cout << "variation " << all->second.Name(k) << "\n";
    }

    // Final-state charged particles.
    for (int iBlob = 0; iBlob < int(blobs.size()); ++iBlob) {
      const ATOOLS::Blob* blobPtr = blobs[iBlob];
      for (int j = 0; j < blobPtr->NOutP(); ++j) {
        const ATOOLS::Particle* prtPtr = blobPtr->ConstOutParticle(j);
        const bool isFinal = prtPtr->Status() == ATOOLS::part_status::active
          && prtPtr->DecayBlob() == nullptr;
        if (isFinal && prtPtr->Flav().IntCharge() != 0) ++nCharged;
      }
    }
  }

  // Sherpa's own summary goes to its output; the cross section is in pb.
  sherpaPtr->SummarizeRun();
  std::cout << "accepted " << nAccepted << " events, sigma = "
            << sherpaPtr->TotalXS() << " +- " << sherpaPtr->TotalErr()
            << " pb, weight sum " << weightSum
            << " pb, mean charged multiplicity "
            << (nAccepted ? double(nCharged) / nAccepted : 0.) << "\n";
  delete sherpaPtr;
  return 0;
}
