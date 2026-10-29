// pythia_driver.cc is a part of the HEPTAPOD package.
// Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
// HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// The driver behind PythiaCPPFromRunCardTool: run a .cmnd card with a
// Pythia 8 install and stream the event record to stdout, one JSON object
// per line, in the layout tools/pythia/driver.py turns back into the
// Awkward Array Pythia.nextBatch would have produced. From there the
// tool's own code applies the store spec, summarises, and writes ROOT,
// HepMC3 or JSONL, so the driver needs nothing of the install but Pythia.
// Built against the install by tools/cpp/build.py.
//
//   heptapod-pythia --card run.cmnd --log pythia.log --nevents N --chunk C
//                   [--store full|finals|finals+kinematics]
//                   [--select-pids 11,13] [--xmldoc DIR]
//
// Lines written, each a JSON object with one top-level key:
//   {"header": {"version": "8.317", "beams": [{...}, {...}]}} after init
//   {"names": ["Weight", ...]}      weight names, after the first event
//   {"p": [[...], ...], "i": [...], "w": [...]}               one event
//   {"f": 1}                        one event Pythia failed to generate
//   {"xsec": {...}}                 every C accepted events, and at the end
//   {"end": {"init_ok": true, "accepted": A, "failed": F, ...}} last line
//
// Everything Pythia itself prints goes to --log, where the pythia8mc
// backend also keeps it. SIGTERM ends the run cleanly after the current
// event, with the xsec and end lines written. Exit status: 0 ok, 2 bad
// arguments, 3 Pythia init failed, 4 I/O error.

#include "Pythia8/Pythia.h"

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace Pythia8;

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
  if (isnan(x)) return "NaN";
  if (isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
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
  cerr << "heptapod-pythia: " << msg << "\n";
  return 2;
}

//--------------------------------------------------------------------------

// The two beams as four-momenta, as PythiaFromRunCardTool records them: A
// along +z and B along -z, except for Beams:frameType = 3, where the card
// gives each beam's momentum. Masses are Pythia's own.

string beamsJSON(Pythia& pythia) {
  const Info& info = pythia.info;
  Settings& settings = pythia.settings;
  const bool useMomenta = settings.mode("Beams:frameType") == 3;
  string out = "[";
  for (int side = 0; side < 2; ++side) {
    const int id = side == 0 ? info.idA() : info.idB();
    double e = side == 0 ? info.eA() : info.eB();
    const double m = pythia.particleData.m0(id);
    double px = 0.;
    double py = 0.;
    double pz = 0.;
    if (useMomenta) {
      const string ab = side == 0 ? "A" : "B";
      px = settings.parm("Beams:px" + ab);
      py = settings.parm("Beams:py" + ab);
      pz = settings.parm("Beams:pz" + ab);
      e = sqrt(px * px + py * py + pz * pz + m * m);
    } else {
      pz = (side == 0 ? 1. : -1.) * sqrt(max(e * e - m * m, 0.));
    }
    if (side == 1) out += ',';
    out += "{\"id\":" + to_string(id) + ",\"px\":" + num(px)
      + ",\"py\":" + num(py) + ",\"pz\":" + num(pz)
      + ",\"e\":" + num(e) + ",\"m\":" + num(m) + "}";
  }
  return out + "]";
}

//--------------------------------------------------------------------------

// The running cross section as one JSON line.

string xsecJSON(const Info& info) {
  return "{\"xsec\":{\"sigmaGen_mb\":" + num(info.sigmaGen())
    + ",\"sigmaErr_mb\":" + num(info.sigmaErr())
    + ",\"weightSum\":" + num(info.weightSum())
    + ",\"nAccepted\":" + to_string(info.nAccepted())
    + ",\"nTried\":" + to_string(info.nTried()) + "}}\n";
}

//--------------------------------------------------------------------------

// One particle as a JSON array: id and status; with allFields the lineage
// and colours; the four-momentum and mass; and with allFields again the
// scale, polarisation, lifetime and production vertex (zeros if unset).

void appendParticle(string& line, const Particle& p, bool allFields) {
  line += '[';
  line += to_string(p.id()) + "," + to_string(p.status());
  if (allFields)
    line += "," + to_string(p.mother1()) + "," + to_string(p.mother2())
      + "," + to_string(p.daughter1()) + "," + to_string(p.daughter2())
      + "," + to_string(p.col()) + "," + to_string(p.acol());
  line += "," + num(p.px()) + "," + num(p.py()) + "," + num(p.pz())
    + "," + num(p.e()) + "," + num(p.m());
  if (allFields) {
    line += "," + num(p.scale()) + "," + num(p.pol()) + "," + num(p.tau());
    if (p.hasVertex())
      line += "," + num(p.xProd()) + "," + num(p.yProd())
        + "," + num(p.zProd()) + "," + num(p.tProd());
    else
      line += ",0.0,0.0,0.0,0.0";
  }
  line += ']';
}

//--------------------------------------------------------------------------

// The per-event scalars, in the order INFO_FIELDS in driver.py names
// them.

string infoJSON(const Info& info) {
  return to_string(info.id1()) + "," + to_string(info.id2())
    + "," + num(info.x1()) + "," + num(info.x2())
    + "," + num(info.pdf1()) + "," + num(info.pdf2())
    + "," + num(info.alphaS()) + "," + num(info.alphaEM())
    + "," + num(info.Q2Fac()) + "," + num(info.Q2Ren())
    + "," + num(info.mHat()) + "," + num(info.sHat())
    + "," + num(info.tHat()) + "," + num(info.uHat())
    + "," + num(info.pT2Hat());
}

//==========================================================================

} // end anonymous namespace

//==========================================================================

int main(int argc, char* argv[]) {

  // Command-line options.
  string card;
  string log;
  string xmldoc;
  string store = "finals+kinematics";
  int nEvents = -1;
  int chunk = 10000;
  set<int> selectIds;
  for (int i = 1; i < argc; ++i) {
    string opt = argv[i];
    if (i + 1 >= argc) return usage(("missing value for " + opt).c_str());
    string val = argv[++i];
    if (opt == "--card") card = val;
    else if (opt == "--log") log = val;
    else if (opt == "--xmldoc") xmldoc = val;
    else if (opt == "--store") store = val;
    else if (opt == "--nevents") nEvents = atoi(val.c_str());
    else if (opt == "--chunk") chunk = max(1, atoi(val.c_str()));
    else if (opt == "--select-pids") {
      stringstream ss(val);
      string tok;
      while (getline(ss, tok, ','))
        if (!tok.empty()) selectIds.insert(abs(atoi(tok.c_str())));
    } else {
      return usage(("unknown option " + opt).c_str());
    }
  }
  if (card.empty() || log.empty() || nEvents < 0)
    return usage("--card, --log and --nevents are required");
  if (store != "full" && store != "finals" && store != "finals+kinematics")
    return usage(("unknown --store " + store).c_str());
  const bool allParticles = store == "full";
  const bool allFields = store != "finals+kinematics";

  // Pythia writes to cout. Under the tool, stdout is the event stream, so
  // cout is pointed at the run's log before Pythia is constructed and the
  // stream keeps the original buffer.
  ofstream logFile(log, std::ios::app);
  if (!logFile) return usage(("cannot open log " + log).c_str());
  CoutToLog redirect(logFile.rdbuf());
  ostream data(redirect.saved());

  // Stop cleanly after the current event on SIGTERM or SIGINT.
  std::signal(SIGTERM, onTerm);
  std::signal(SIGINT, onTerm);

  // Same settings, in the same order, as the pythia8mc backend. The card
  // is read after them, so anything it sets wins; the tool has already
  // appended its seed and LHE overrides to the card.
  Pythia pythia(xmldoc.empty() ? "../share/Pythia8/xmldoc" : xmldoc, false);
  pythia.readString("Print:quiet = on");
  pythia.readString("Init:showProcesses = off");
  pythia.readString("Init:showMultipartonInteractions = off");
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showChangedParticleData = off");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");
  const bool initOK = pythia.readFile(card) && pythia.init();
  if (!initOK) {
    // Print:quiet hides the errors as they happen; stat() lists them.
    pythia.stat();
    data << "{\"end\":{\"init_ok\":false}}\n" << std::flush;
    return 3;
  }

  // The header: Pythia's version and the beams.
  char version[16];
  std::snprintf(version, sizeof version, "%.3f",
    pythia.settings.parm("Pythia:versionNumber"));
  data << "{\"header\":{\"version\":\"" << version << "\",\"beams\":"
       << beamsJSON(pythia) << "}}\n" << std::flush;

  // Event loop. A failed event is reported and not counted; a whole chunk
  // of consecutive failures ends the run, as nextBatch(want, "skip") does.
  const Info& info = pythia.info;
  int nAccepted = 0;
  int nFailed = 0;
  int nConsecutive = 0;
  bool lheExhausted = false;
  bool namesSent = false;
  string line;
  line.reserve(1 << 16);
  while (nAccepted < nEvents && !stopRequested) {
    if (!pythia.next()) {
      if (info.atEndOfFile()) {
        lheExhausted = true;
        break;
      }
      ++nFailed;
      data << "{\"f\":1}\n";
      if (++nConsecutive >= chunk) break;
      continue;
    }
    nConsecutive = 0;

    // Weight names exist once an event has been generated.
    if (!namesSent) {
      data << "{\"names\":[";
      const vector<string> names = info.weightNameVector();
      for (int k = 0; k < int(names.size()); ++k)
        data << (k ? "," : "") << quoted(names[k]);
      data << "]}\n";
      namesSent = true;
    }

    // The event: the selected particles, the scalars and the weights.
    const Event& event = pythia.event;
    line.assign("{\"p\":[");
    int nWritten = 0;
    for (int i = 0; i < event.size(); ++i) {
      const Particle& p = event[i];
      if (!allParticles && !p.isFinal()) continue;
      if (!selectIds.empty() && !selectIds.count(abs(p.id()))) continue;
      if (nWritten > 0) line += ',';
      appendParticle(line, p, allFields);
      ++nWritten;
    }
    line += "],\"i\":[" + infoJSON(info) + "],\"w\":[";
    const vector<double> weights = info.weightValueVector();
    for (int k = 0; k < int(weights.size()); ++k)
      line += (k ? "," : "") + num(weights[k]);
    line += "]}\n";
    data << line;
    if (!data) {
      cerr << "heptapod-pythia: writing the event stream failed\n";
      return 4;
    }

    // The running cross section after every chunk of accepted events.
    ++nAccepted;
    if (nAccepted % chunk == 0) data << xsecJSON(info) << std::flush;
  }

  // stat() writes the cross-section table to the log, as the pythia8mc
  // backend does at the end of its run; then the final cross section and
  // the end line.
  pythia.stat();
  data << xsecJSON(info)
       << "{\"end\":{\"init_ok\":true,\"accepted\":" << nAccepted
       << ",\"failed\":" << nFailed
       << ",\"lhe_exhausted\":" << boolStr(lheExhausted)
       << ",\"stopped\":" << boolStr(stopRequested != 0) << "}}\n"
       << std::flush;
  return data ? 0 : 4;

}
