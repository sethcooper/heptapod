"""
MadGraphFromRunCard reports the final (refine) results.

When the card's ``output``/``launch`` name an absolute process directory
outside data_dir, the tool must find the LHE file there and report the
refine summary (``Width : 3.006e-19 +- 2.841e-22``, 20000 events), not
the survey summary (``Width : 3.017e-19``, ``Nb of events : 0``) as
``lhe_file=null, note="nevents=0"``.

A fake mg5_aMC replays that log, so no MadGraph install is needed.
"""
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.mg5.mg5 import (  # noqa: E402
    MadGraphFromRunCardTool,
    _parse_run_summary,
    _parse_width_from_log,
)

# Verbatim structure of results/work/mg_run/mg5_run.log in muon-gpt56-3
# (ANSI colour codes included, as MG5 writes them).
LOG = """\
Generating 20000 events with run name run_01
survey  run_01
INFO: Running Survey
  === Results Summary for run: run_01 tag: tag_1 ===

     Width :   3.017e-19 +- 3.406e-21 GeV
     Nb of events :  0

INFO: End survey
refine 20000
\x1b[1mINFO: Refine results to 20000 \x1b[0m
- Current estimate of cross-section: 3.017e-19 +- 3.4062e-21
INFO: finish refine
refine 20000 --treshold=0.9
No need for second refine due to stability of cross-section
INFO: Combining Events
  === Results Summary for run: run_01 tag: tag_1 ===

     Width :   3.006e-19 +- 2.841e-22 GeV
     Nb of events :  20000

store_events
INFO: Done
quit
INFO:
more information in {procdir}/index.html
quit
"""

FAKE_MG5 = textwrap.dedent('''\
    #!{python}
    import os, sys, gzip
    card = open(sys.argv[1]).read().split("\\n")
    proc = [l.split()[1] for l in card if l.startswith("output ")][0]
    proc = proc if os.path.isabs(proc) else os.path.join(os.getcwd(), proc)
    ev = os.path.join(proc, "Events", "run_01")
    os.makedirs(ev, exist_ok=True)
    with gzip.open(os.path.join(ev, "unweighted_events.lhe.gz"), "wt") as f:
        f.write("<LesHouchesEvents version=\\"3.0\\">\\n</LesHouchesEvents>\\n")
    sys.stdout.write(open(os.environ["FAKE_MG5_LOG"]).read().replace("{{procdir}}", proc))
''')


def _fake_mg5(tmp_path: Path) -> Path:
    root = tmp_path / "MG5_fake"
    (root / "bin").mkdir(parents=True)
    exe = root / "bin" / "mg5_aMC"
    exe.write_text(FAKE_MG5.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    logf = tmp_path / "replay.log"
    logf.write_text(LOG)
    os.environ["FAKE_MG5_LOG"] = str(logf)
    return root


def test_parse_run_summary_takes_final_block(tmp_path):
    log = tmp_path / "mg5_run.log"
    log.write_text(LOG.format(procdir="/x/y"))
    info = _parse_run_summary(str(log))
    assert info == {
        "width_gev": 3.006e-19,
        "width_gev_error": 2.841e-22,
        "nevents": 20000,
        "run_name": "run_01",
    }
    assert _parse_width_from_log(str(log)) == {"width_gev": 3.006e-19, "width_gev_error": 2.841e-22}


def test_parse_run_summary_cross_section(tmp_path):
    log = tmp_path / "mg5_run.log"
    log.write_text(
        "  === Results Summary for run: run_01 tag: tag_1 ===\n"
        "     Cross-section :   1.0e+02 +- 5.0e+00 pb\n     Nb of events :  0\n"
        "  === Results Summary for run: run_01 tag: tag_1 ===\n"
        "     Cross-section :   9.87e+01 +- 1.2e-01 pb\n     Nb of events :  10000\n"
    )
    info = _parse_run_summary(str(log))
    assert info["cross_section_pb"] == 98.7
    assert info["cross_section_pb_error"] == 0.12
    assert info["nevents"] == 10000


def test_tool_reports_refine_results_and_lhe_outside_data_dir(tmp_path):
    base = tmp_path / "workspace"
    (base / "cards").mkdir(parents=True)
    procdir = base / "results" / "work" / "mg5_mudecay"  # outside data_dir
    (base / "cards" / "mu_decay.mg5").write_text(
        "import model sm\n"
        "generate mu- > ve~ e- vm\n"
        f"output {procdir}\n"
        f"launch {procdir}\n"
        "set nevents 20000\n"
        "done\n"
    )
    tool = MadGraphFromRunCardTool(mg5_path=str(_fake_mg5(tmp_path)), base_directory=str(base))
    out = json.loads(tool.execute(data_dir="results/work/mg_run", command_card="cards/mu_decay.mg5"))

    assert out["status"] == "ok"
    assert out["lhe_file"] == "results/work/mg5_mudecay/Events/run_01/unweighted_events.lhe.gz"
    assert out["nevents"] == 20000
    assert out["width_gev"] == 3.006e-19
    assert out["width_gev_error"] == 2.841e-22
    assert "note" not in out

    manifest = json.loads((base / out["manifest_json"]).read_text())
    assert manifest["results"]["nevents"] == 20000
    assert manifest["outputs"]["lhe_path"] == out["lhe_file"]


def test_tool_relative_output_inside_data_dir_still_works(tmp_path):
    base = tmp_path / "workspace"
    (base / "cards").mkdir(parents=True)
    (base / "cards" / "c.mg5").write_text("generate mu- > ve~ e- vm\noutput proc\nlaunch proc\ndone\n")
    tool = MadGraphFromRunCardTool(mg5_path=str(_fake_mg5(tmp_path)), base_directory=str(base))
    out = json.loads(tool.execute(data_dir="run", command_card="cards/c.mg5"))
    assert out["lhe_file"] == "run/proc/Events/run_01/unweighted_events.lhe.gz"
    assert out["nevents"] == 20000
