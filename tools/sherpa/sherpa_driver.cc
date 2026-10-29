// sherpa_driver.cc is a part of the HEPTAPOD package.
// Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
// HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// The driver behind SherpaCPPFromRunCardTool: run a Sherpa .yaml card with
// a Sherpa install (or the sherpa-mc wheel's own libraries) and stream the
// event record to stdout, one JSON object per line, in the layout
// tools/sherpa/record.py turns back into the dataset batch the shared
// generation loop writes. The rows are the ones record.py builds from the
// Python interface, so the dataset does not depend on the tool; this
// driver adds what that interface cannot reach: the named weight
// variations. Built against the install by tools/cpp/build.py.
//
//   heptapod-sherpa --card run.yaml --log sherpa.log --nevents N --chunk C
//                   [--seed S] [--store full|finals|finals+kinematics]
//                   [--select-pids 11,13]
//
// Lines written, each a JSON object with one top-level key:
//   {"header": {"version": "3.0.3"}}                            after init
//   {"names": ["Weight", "MUR=0.5__MUF=0.5__LHAPDF=-1", ...]} first event
//   {"p": [[...], ...], "i": [...], "w": [...]}                 one event
//   {"f": 1}                        one event Sherpa failed to generate
//   {"xsec": {...}}                 every C accepted events, and at the end
//   {"end": {"init_ok": true, "accepted": A, "failed": F, ...}} last line
//
// A particle row is id, status, info, prod_vertex, end_vertex, px, py, pz,
// e, m, vx, vy, vz, vt (record.py's ROW): the status is the HepMC status,
// negative for a particle that is not final, and the vertices index the
// event's blobs. The scalars are the trials behind the event, the ME
// weight and the squared renormalisation and factorisation scales. The
// weights are the nominal weight in pb followed by every variation of the
// "All" group as an absolute weight. Everything Sherpa itself prints goes
// to --log. SIGTERM ends the run cleanly after the current event. Exit
// status: 0 ok, 2 bad arguments, 3 Sherpa init failed, 4 I/O error.

#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Org/Exception.H"
#include "ATOOLS/Phys/Blob.H"
#include "ATOOLS/Phys/Blob_List.H"
#include "ATOOLS/Phys/Particle.H"
#include "ATOOLS/Org/CXXFLAGS.H"
#include "ATOOLS/Phys/Weights.H"

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace {

//==========================================================================

// A SIGTERM or SIGINT sets this flag, which the event loop reads between
// events. It is the one global in the program: a signal handler can reach
// nothing else.

volatile std::sig_atomic_t stopRequested = 0;

void onTerm(int) { stopRequested = 1; }

//--------------------------------------------------------------------------

// Point cout at the run's log for the lifetime of the object and restore it
// after, so the stream teardown at exit finds the buffer it started with.

class CoutToLog {
public:
  explicit CoutToLog(std::streambuf* logBuf)
    : savedBuf(std::cout.rdbuf(logBuf)) {}
  ~CoutToLog() { std::cout.rdbuf(savedBuf); }
  std::streambuf* saved() const { return savedBuf; }
private:
  std::streambuf* savedBuf;
};

//--------------------------------------------------------------------------

// Shortest of %.15g to %.17g that round-trips, as close to Python's float
// repr as printf gets. Non-finite values are written the way Python's json
// module writes them (NaN, Infinity), which its parser also reads.

string num(double x) {
  if (std::isnan(x)) return "NaN";
  if (std::isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.15g", x);
  for (int prec = 16; prec <= 17 && std::strtod(buf, nullptr) != x; ++prec)
    std::snprintf(buf, sizeof buf, "%.*g", prec, x);
  string s(buf);
  if (s.find_first_of(".eE") == string::npos) s += ".0";
  return s;
}

//--------------------------------------------------------------------------

// JSON true and false.

const char* boolStr(bool b) { return b ? "true" : "false"; }

//--------------------------------------------------------------------------

// A JSON string literal: quotes and backslashes escaped, newlines encoded.

string quoted(const string& s) {
  string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    if (c == '\n') {
      out += "\\n";
      continue;
    }
    out += c;
  }
  return out + "\"";
}

//--------------------------------------------------------------------------

// Report a command-line error and return the exit status for it.

int usage(const char* msg) {
  std::cerr << "heptapod-sherpa: " << msg << "\n";
  return 2;
}

//--------------------------------------------------------------------------

// Whether a blob gives out a quark, gluon or diquark: a beam hadron
// breaking up into its partons, not a decay.

bool breaksUp(const ATOOLS::Blob* blobPtr) {
  for (int j = 0; j < blobPtr->NOutP(); ++j) {
    const ATOOLS::Flavour flav = blobPtr->ConstOutParticle(j)->Flav();
    if (flav.IsQuark() || flav.IsGluon() || flav.IsDiQuark()) return true;
  }
  return false;
}

//--------------------------------------------------------------------------

// The HepMC status of a particle, signed as the dataset stores it, by the
// rule record.py applies to the Python interface: a particle from no blob
// is a beam (-4); one that is active and goes into no blob is final (1);
// a documentation entry keeps that meaning (-3); a hadron, muon or tau that
// is the only particle entering its decay blob, and does not break up into
// partons there, decayed (-2); everything else is intermediate (-11).

int signedStatus(const ATOOLS::Particle* prtPtr) {
  const ATOOLS::Blob* prodPtr = prtPtr->ProductionBlob();
  const ATOOLS::Blob* decPtr = prtPtr->DecayBlob();
  if (prodPtr == nullptr) return -4;
  const int stat = int(prtPtr->Status());
  if (decPtr == nullptr && stat == ATOOLS::part_status::active) return 1;
  if (stat == ATOOLS::part_status::documentation) return -3;
  const long kf = prtPtr->Flav().Kfcode();
  const bool unstable = prtPtr->Flav().IsHadron() || kf == 13 || kf == 15;
  if (decPtr != nullptr && decPtr->NInP() == 1 && unstable
    && !breaksUp(decPtr)) return -2;
  return -11;
}

//--------------------------------------------------------------------------

// The running cross section as one JSON line, in pb as Sherpa reports it.

string xsecJSON(const SHERPA::Sherpa& sherpa, int nAccepted,
  double nTried, double weightSum) {
  return "{\"xsec\":{\"sigma_pb\":" + num(sherpa.TotalXS())
    + ",\"sigma_err_pb\":" + num(sherpa.TotalErr())
    + ",\"weightSum\":" + num(weightSum)
    + ",\"n_accepted\":" + std::to_string(nAccepted)
    + ",\"n_tried\":" + num(nTried) + "}}\n";
}

//--------------------------------------------------------------------------

// One particle as a JSON array, in record.py's row order.

void appendParticle(string& line, const ATOOLS::Particle* prtPtr,
  int status, int prodVertex, int endVertex) {
  const ATOOLS::Flavour flav = prtPtr->Flav();
  const long kf = flav.Kfcode();
  const ATOOLS::Vec4D p = prtPtr->Momentum();
  const ATOOLS::Vec4D x = prtPtr->XProd();
  line += '[';
  line += std::to_string(flav.IsAnti() ? -kf : kf);
  line += "," + std::to_string(status);
  line += "," + std::to_string(int(prtPtr->Info()));
  line += "," + std::to_string(prodVertex);
  line += "," + std::to_string(endVertex);
  line += "," + num(p[1]) + "," + num(p[2]) + "," + num(p[3]) + ","
    + num(p[0]) + "," + num(p.Mass());
  line += "," + num(x[1]) + "," + num(x[2]) + "," + num(x[3]) + ","
    + num(x[0]);
  line += ']';
}

//--------------------------------------------------------------------------

// A blob-data entry as a double, or the default when the blob lacks it.

double blobValue(ATOOLS::Blob* blobPtr, const string& name, double dflt) {
  if (blobPtr == nullptr) return dflt;
  ATOOLS::Blob_Data_Base* dataPtr = (*blobPtr)[name];
  if (dataPtr == nullptr) return dflt;
  return dataPtr->Get<double>();
}

//==========================================================================

} // end anonymous namespace

//==========================================================================

int main(int argc, char* argv[]) {

  // Command-line options.
  string card;
  string log;
  string seed;
  string store = "finals+kinematics";
  int nEvents = -1;
  int chunk = 10000;
  std::set<long> selectIds;
  for (int i = 1; i < argc; ++i) {
    string opt = argv[i];
    if (i + 1 >= argc) return usage(("missing value for " + opt).c_str());
    string val = argv[++i];
    if (opt == "--card") card = val;
    else if (opt == "--log") log = val;
    else if (opt == "--seed") seed = val;
    else if (opt == "--store") store = val;
    else if (opt == "--nevents") nEvents = std::atoi(val.c_str());
    else if (opt == "--chunk") chunk = std::max(1, std::atoi(val.c_str()));
    else if (opt == "--select-pids") {
      std::stringstream ss(val);
      string tok;
      while (std::getline(ss, tok, ','))
        if (!tok.empty()) selectIds.insert(std::labs(std::atol(tok.c_str())));
    } else {
      return usage(("unknown option " + opt).c_str());
    }
  }
  if (card.empty() || log.empty() || nEvents < 0)
    return usage("--card, --log and --nevents are required");
  if (store != "full" && store != "finals" && store != "finals+kinematics")
    return usage(("unknown --store " + store).c_str());
  const bool allParticles = store == "full";

  // Sherpa writes to cout. Under the tool, stdout is the event stream, so
  // cout is pointed at the run's log before Sherpa is constructed and the
  // stream keeps the original buffer.
  std::ofstream logFile(log, std::ios::app);
  if (!logFile) return usage(("cannot open log " + log).c_str());
  CoutToLog redirect(logFile.rdbuf());
  std::ostream data(redirect.saved());

  // The command line the sherpa-mc backend hands Sherpa: quiet output, the
  // event count, the seed, the card.
  vector<string> args = {"Sherpa", "-O0", "-e" + std::to_string(nEvents)};
  if (!seed.empty()) args.push_back("-R" + seed);
  args.push_back(card);
  vector<char*> cArgs;
  for (string& arg : args) cArgs.push_back(&arg[0]);
  cArgs.push_back(nullptr);

  // Initialise Sherpa; it reports failure by return value or exception.
  SHERPA::Sherpa* sherpaPtr = nullptr;
  try {
    sherpaPtr = new SHERPA::Sherpa(int(args.size()), cArgs.data());
    if (!sherpaPtr->InitializeTheRun()
      || !sherpaPtr->InitializeTheEventHandler()) {
      data << "{\"end\":{\"init_ok\":false}}\n" << std::flush;
      return 3;
    }
  } catch (const ATOOLS::Exception& e) {
    std::cout << e << "\n";
    data << "{\"end\":{\"init_ok\":false}}\n" << std::flush;
    return 3;
  } catch (const std::exception& e) {
    std::cout << "Fatal error thrown: " << e.what() << "\n";
    data << "{\"end\":{\"init_ok\":false}}\n" << std::flush;
    return 3;
  }

  // Stop cleanly after the current event on SIGTERM or SIGINT. Installed
  // after Sherpa's own handlers, so that these are the ones that run.
  std::signal(SIGTERM, onTerm);
  std::signal(SIGINT, onTerm);

  // The header: Sherpa's version, from the version code (3000003 is 3.0.3).
  const int code = SHERPA_VERSION_CODE;
  const string version = std::to_string(code / 1000000) + "."
    + std::to_string((code / 1000) % 1000) + "." + std::to_string(code % 1000);
  data << "{\"header\":{\"version\":" << quoted(version) << "}}\n"
       << std::flush;

  // Event loop. A failed event is reported and not counted; a whole chunk
  // of consecutive failures ends the run.
  int nAccepted = 0;
  int nFailed = 0;
  int nConsecutive = 0;
  double nTried = 0.;
  double weightSum = 0.;
  bool namesSent = false;
  string line;
  line.reserve(1 << 16);
  try {
    while (nAccepted < nEvents && !stopRequested) {
      if (!sherpaPtr->GenerateOneEvent()) {
        ++nFailed;
        data << "{\"f\":1}\n";
        if (++nConsecutive >= chunk) break;
        continue;
      }
      nConsecutive = 0;
      const ATOOLS::Blob_List& blobs = sherpaPtr->GetBlobList();
      const double weight = blobs.Weight();
      const ATOOLS::Weights_Map weightsMap = blobs.WeightsMap();
      std::map<string, ATOOLS::Weights>::const_iterator all
        = weightsMap.find("All");

      // Weight names exist once an event has been generated: the nominal
      // weight, then the variations of the "All" group.
      if (!namesSent) {
        data << "{\"names\":[\"Weight\"";
        if (all != weightsMap.end())
          for (int k = 1; k < int(all->second.Size()); ++k)
            data << "," << quoted(all->second.Name(k));
        data << "]}\n";
        namesSent = true;
      }

      // Blobs are the vertices; a particle is identified by its object,
      // since colour reconnection copies one under the same number.
      std::map<const ATOOLS::Blob*, int> vertexOf;
      for (int k = 0; k < int(blobs.size()); ++k) vertexOf[blobs[k]] = k;
      std::set<const ATOOLS::Particle*> seen;
      line.assign("{\"p\":[");
      int nWritten = 0;
      for (int k = 0; k < int(blobs.size()); ++k) {
        ATOOLS::Blob* blobPtr = blobs[k];
        const int nIn = blobPtr->NInP();
        const int nOut = blobPtr->NOutP();
        for (int j = 0; j < nIn + nOut; ++j) {
          const ATOOLS::Particle* prtPtr = j < nIn
            ? blobPtr->ConstInParticle(j) : blobPtr->ConstOutParticle(j - nIn);
          if (!seen.insert(prtPtr).second) continue;
          const int status = signedStatus(prtPtr);
          if (!allParticles && status <= 0) continue;
          if (!selectIds.empty()
            && !selectIds.count(prtPtr->Flav().Kfcode())) continue;
          std::map<const ATOOLS::Blob*, int>::const_iterator it;
          int prodVertex = -1;
          int endVertex = -1;
          it = vertexOf.find(prtPtr->ProductionBlob());
          if (it != vertexOf.end()) prodVertex = it->second;
          it = vertexOf.find(prtPtr->DecayBlob());
          if (it != vertexOf.end()) endVertex = it->second;
          if (nWritten > 0) line += ',';
          appendParticle(line, prtPtr, status, prodVertex, endVertex);
          ++nWritten;
        }
      }

      // The scalars from the signal blob, and the weights.
      ATOOLS::Blob* signalPtr = blobs.FindFirst(ATOOLS::btp::Signal_Process);
      const double trials = blobValue(signalPtr, "Trials", 1.);
      nTried += trials;
      weightSum += weight;
      line += "],\"i\":[" + num(trials)
        + "," + num(blobValue(signalPtr, "MEWeight", 0.))
        + "," + num(blobValue(signalPtr, "Renormalization_Scale", 0.))
        + "," + num(blobValue(signalPtr, "Factorization_Scale", 0.))
        + "],\"w\":[" + num(weight);
      if (all != weightsMap.end())
        for (int k = 1; k < int(all->second.Size()); ++k)
          line += "," + num(weight * all->second[k]);
      line += "]}\n";
      data << line;
      if (!data) {
        std::cerr << "heptapod-sherpa: writing the event stream failed\n";
        return 4;
      }

      // The running cross section after every chunk of accepted events.
      ++nAccepted;
      if (nAccepted % chunk == 0)
        data << xsecJSON(*sherpaPtr, nAccepted, nTried, weightSum)
             << std::flush;
    }
  } catch (const ATOOLS::Exception& e) {
    std::cout << e << "\n";
    data << "{\"end\":{\"init_ok\":true,\"accepted\":" << nAccepted
         << ",\"failed\":" << nFailed << ",\"error\":"
         << quoted("Sherpa exception during generation") << "}}\n"
         << std::flush;
    return 5;
  } catch (const std::exception& e) {
    data << "{\"end\":{\"init_ok\":true,\"accepted\":" << nAccepted
         << ",\"failed\":" << nFailed << ",\"error\":" << quoted(e.what())
         << "}}\n" << std::flush;
    return 5;
  }

  // Sherpa's own run summary to the log, then the final cross section and
  // the end line.
  sherpaPtr->SummarizeRun();
  data << xsecJSON(*sherpaPtr, nAccepted, nTried, weightSum)
       << "{\"end\":{\"init_ok\":true,\"accepted\":" << nAccepted
       << ",\"failed\":" << nFailed
       << ",\"stopped\":" << boolStr(stopRequested != 0) << "}}\n"
       << std::flush;
  delete sherpaPtr;
  return data ? 0 : 4;

}
