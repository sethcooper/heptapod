#!/usr/bin/env python3
"""
# test_rivet.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

"""
Rivet tools test suite.

Most of it runs without Rivet. The analysis catalogue is read from metadata
files written here in Rivet's own formats; the comparison statistics are
checked against known values; the tools are driven against a fake
installation whose `rivet` is a small script that reads the events (a file
or the streaming pipe) and writes a marker file, with the YODA reader
replaced by canned histograms. That pins the tools' contract -- the checks
before a run, the arguments passed, the parsing of Rivet's messages, the
shape of the results -- independently of a Rivet build.

With RIVET_PATH pointing at a real installation (and pythia8mc and pyhepmc
installed), the full chain also runs: generate a small Z sample, run two
analyses on it from a file and through the pipe, compare with the data,
draw the plots and merge two runs.
"""

import gzip
import json
import math
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

yaml = pytest.importorskip("yaml", reason="pyyaml not installed")

from tools.rivet import catalogue as cat  # noqa: E402
from tools.rivet import install as inst  # noqa: E402
from tools.rivet import yoda_io  # noqa: E402
from tools.rivet.compare import chi2_survival, compare_histograms, verdict  # noqa: E402
from tools.rivet.plot import mathtext, plain_text  # noqa: E402
from tools.rivet.run import (  # noqa: E402
    AnalysisCheckError,
    check_analyses,
    parse_rivet_log,
    summarise_output,
)
from tools.rivet import (  # noqa: E402
    RivetCompareTool,
    RivetFindAnalysesTool,
    RivetMergeTool,
    RivetPlotTool,
    RunRivetTool,
)

try:
    import matplotlib  # noqa: F401

    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False
try:
    import pyhepmc  # noqa: F401

    HAVE_PYHEPMC = True
except ImportError:
    HAVE_PYHEPMC = False
try:
    import pythia8mc  # noqa: F401

    HAVE_PYTHIA = True
except ImportError:
    HAVE_PYTHIA = False

_REAL_RIVET = os.environ.get("RIVET_PATH", "")
HAVE_REAL_RIVET = bool(_REAL_RIVET) and os.path.isfile(
    os.path.join(os.path.expanduser(_REAL_RIVET), "bin", "rivet-config")
)


# --------------------------------------------------------------------- #
# A fake Rivet data directory and installation
# --------------------------------------------------------------------- #

_INFO_PP = """\
Name: FAKE_2012_I1
Year: 2012
Summary: $Z/\\gamma^*$ boson transverse momentum via $\\phi^*_\\eta$
Experiment: FAKE
Collider: LHC
InspireID: 1
Status: VALIDATED
Authors:
 - Someone <someone@example.org>
References:
 - arXiv:1211.0000 [hep-ex]
RunInfo:
  $Z/\\gamma^*$ production with decays to electrons and/or muons.
NumEvents: 1000000
Beams: [p+, p+]
Energies: [7000]
Options:
 - LMODE=EL,MU
NeedCrossSection: yes
Description:
  A measurement of angular correlations in Drell-Yan lepton pairs.
Keywords: [Z, DrellYan]
BibKey: Someone:2012aa
BibTeX: '@article{Someone:2012aa,
      title          = "{Measurement of 'angular' correlations}",
}'
"""

_INFO_EE = """\
Name: FAKE_EE
Summary: Charged particle multiplicity at the Z pole
Experiment: FAKEEE
Collider: LEP
Status: VALIDATED NOHEPDATA SINGLEWEIGHT
Beams: [[e+, e-], [e-, e+]]
Energies: [[45.6, 45.6]]
RunInfo: Hadronic Z decays.
"""

_INFO_MC = """\
Name: MC_FAKE
Summary: Monte Carlo validation observables for jets
Status: VALIDATED
RunInfo: Any jet process.
Options:
 - PTMIN=#
"""

_INFO_HEAVY = """\
Name: FAKE_ION
Summary: Charged multiplicity in p-Pb and Pb-Pb collisions
Experiment: FAKEION
Status: UNVALIDATED
Beams:
    1:[p+,Pb]
    2:[Pb,Pb]
Energies:
    1:[5020]
    2:[2760, 5020]
"""

_PLOT_PP = """\
# BEGIN PLOT /FAKE_2012_I1/d..-x..-y..
XLabel=$\\phi^*_\\eta$
YLabel=$\\frac{1}{\\sigma}\\frac{\\mathrm{d}\\sigma}{\\mathrm{d}\\phi^*_\\eta}$
LogX=1
RatioPlotYMin=0.8
RatioPlotYMax=1.2
# END PLOT

# BEGIN PLOT /FAKE_2012_I1/d01-x01-y01
Title=$\\phi^*_\\eta$ spectrum, Z$\\rightarrow$ ee (bare)
# END PLOT

# BEGIN PLOT /FAKE_2012_I1/d01-x01-y02
Title=$\\phi^*_\\eta$ spectrum, Z$\\rightarrow\\mu\\mu$ (bare)
LogY=0
# END PLOT
"""

_EDGES = [0.0, 0.004, 0.008, 0.012, 0.016, 0.02]
_REF_VALUES = [9.7, 9.59, 9.38, 9.07, 8.77]
_REF_ERRORS = [0.056, 0.052, 0.051, 0.054, 0.048]


def _yoda_estimate_text(path, edges, values, errors):
    rows = [f"{v:e}\t{-e:e}\t{e:e}\t" for v, e in zip(values, errors)]
    lines = [
        f"BEGIN YODA_ESTIMATE1D_V3 {path}",
        "IsRef: 1",
        f"Path: {path}",
        "Title:",
        "Type: Estimate1D",
        "---",
        f"Edges(A1): [{', '.join(f'{e:e}' for e in edges)}]",
        'ErrorLabels: [""]',
        "# value      \ttotalDn      \ttotalUp",
        "nan          \t---          \t---",
        *rows,
        "nan          \t---          \t---",
        "END YODA_ESTIMATE1D_V3",
        "",
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """A Rivet data directory with four analyses in Rivet's own file
    formats."""
    root = tmp_path_factory.mktemp("rivet_data")
    (root / "FAKE_2012_I1.info").write_text(_INFO_PP)
    (root / "FAKE_2012_I1.plot").write_text(_PLOT_PP)
    with gzip.open(root / "FAKE_2012_I1.yoda.gz", "wt") as handle:
        handle.write(
            _yoda_estimate_text(
                "/REF/FAKE_2012_I1/d01-x01-y01",
                _EDGES,
                _REF_VALUES,
                _REF_ERRORS,
            )
        )
        handle.write(
            _yoda_estimate_text(
                "/REF/FAKE_2012_I1/d01-x01-y02",
                _EDGES,
                _REF_VALUES,
                _REF_ERRORS,
            )
        )
    (root / "FAKE_EE.info").write_text(_INFO_EE)
    (root / "MC_FAKE.info").write_text(_INFO_MC)
    (root / "FAKE_ION.info").write_text(_INFO_HEAVY)
    return root


_FAKE_RIVET = '''#!{python}
"""A stand-in for the rivet program: reads the events, writes a marker file."""
import os, sys
args = sys.argv[1:]
mode = os.environ.get("FAKE_RIVET_MODE", "ok")
out = args[args.index("-o") + 1]
analyses = [args[i + 1] for i, a in enumerate(args) if a == "-a"]
valued = {{"-o", "-a", "-x", "-n", "--nskip", "--nominal-weight",
          "--analysis-path-append", "-p", "-m"}}
inputs = []
skip = False
for i, a in enumerate(args):
    if skip:
        skip = False
        continue
    if a in valued:
        skip = True
        continue
    if not a.startswith("-"):
        inputs.append(a.split(":=")[0].split(":x")[0])
args_path = os.path.join(os.path.dirname(out), "fake_rivet_args.json")
with open(args_path, "w") as f:
    f.write(repr(args))
if mode == "unknown":
    print(f"'{{analyses[0]}}' is not a known Rivet analysis! "
          "Do you need to set RIVET_ANALYSIS_PATH or use the --pwd switch?")
    sys.exit(1)
if mode == "all_removed":
    print(f"Rivet.AnalysisHandler: WARN  Analysis '{{analyses[0]}}' "
          "is incompatible with the provided beams: removing", file=sys.stderr)
    print("Error in run initialisation: All analyses were incompatible "
          "with the first event's beams")
    sys.exit(1)
n = 0
for path in inputs:
    if path.endswith(".yoda"):
        continue
    with open(path, "r") as handle:
        for line in handle:
            if line.startswith("E "):
                n += 1
                if n % 100 == 0:
                    print(f"Event {{n}} (0:00:01 elapsed)")
                if mode == "crash_midway" and n >= 20:
                    sys.exit(3)
print(f"Event {{n}} (0:00:02 elapsed)")
print("Cross-section = 1.000000e+03 pb")
if mode == "warn":
    warn = "Rivet.Analysis.FAKE_2012_I1: WARN  Something odd happened"
    print(warn, file=sys.stderr)
    print(warn, file=sys.stderr)
with open(out, "w") as f:
    f.write(f"FAKEYODA events={{n}} analyses={{','.join(analyses)}} "
            f"inputs={{len(inputs)}}\\n")
'''


def _canned_dump(install, paths, patterns=(), timeout=600):
    """Histograms for the fake rivet's marker files and the fake data
    directory."""
    files = []
    for path in paths:
        if path.endswith(".yoda.gz"):
            name = os.path.basename(path)[: -len(".yoda.gz")]
            objects = [
                _estimate(
                    f"/REF/{name}/d01-x01-y01", _REF_VALUES, _REF_ERRORS
                ),
                _estimate(
                    f"/REF/{name}/d01-x01-y02", _REF_VALUES, _REF_ERRORS
                ),
            ]
            files.append({"path": path, "objects": objects})
            continue
        with open(path) as handle:
            header = handle.readline()
        fields = dict(part.split("=", 1) for part in header.split()[1:])
        n = int(fields["events"])
        objects = [
            {
                "path": "/RAW/_EVTCOUNT",
                "type": "Counter",
                "title": "~",
                "annotations": {},
                "total": {
                    "sumw": float(n),
                    "sumw2": float(n),
                    "entries": float(n),
                },
            },
            {
                "path": "/RAW/_XSEC",
                "type": "Estimate0D",
                "title": "~",
                "annotations": {},
                "values": [1000.0],
                "err_down": [10.0],
                "err_up": [10.0],
            },
        ]
        for analysis in fields["analyses"].split(","):
            scale = 1.0 + 0.02 * n**0.5 / 10
            values = [v * scale for v in _REF_VALUES]
            errors = [v * (n**-0.5 if n else 0.0) for v in values]
            for hist in ("d01-x01-y01", "d01-x01-y02"):
                objects.append(
                    _estimate(f"/{analysis}/{hist}", values, errors)
                )
                objects.append(
                    _estimate(
                        f"/{analysis}/{hist}[fsrup]",
                        [v * 1.1 for v in values],
                        errors,
                    )
                )
                entries = [float(n) / 5] * 5 if n else [0.0] * 5
                objects.append(
                    {
                        "path": f"/RAW/{analysis}/{hist}",
                        "type": "Histo1D",
                        "title": "~",
                        "annotations": {},
                        "edges": _EDGES,
                        "sumw": entries,
                        "sumw2": entries,
                        "entries": entries,
                        "masked": [],
                        "total": {
                            "sumw": float(n),
                            "sumw2": float(n),
                            "entries": float(n),
                        },
                    }
                )
        files.append({"path": path, "objects": objects})
    return {"yoda_version": "fake", "files": files}


def _estimate(path, values, errors, edges=_EDGES):
    return {
        "path": path,
        "type": "Estimate1D",
        "title": "~",
        "annotations": {},
        "edges": list(edges),
        "values": list(values),
        "err_down": list(errors),
        "err_up": list(errors),
        "masked": [],
    }


@pytest.fixture
def fake_install(tmp_path, data_dir, monkeypatch):
    """A RivetInstall whose programs are stand-ins and whose YODA reader is
    canned."""
    prefix = tmp_path / "rivet_prefix"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("rivet", "rivet-merge"):
        script = bin_dir / name
        script.write_text(_FAKE_RIVET.format(python=sys.executable))
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    install = inst.RivetInstall(
        prefix=str(prefix),
        version="fake-4",
        bin_dir=str(bin_dir),
        data_dir=str(data_dir),
        lib_dir=str(prefix / "lib"),
        python=sys.executable,
        yoda_version="fake-2",
        env=dict(os.environ),
    )
    monkeypatch.setattr("tools.rivet.base.resolve_rivet", lambda path: install)
    monkeypatch.setattr(
        "tools.rivet.install.resolve_rivet", lambda path: install
    )
    monkeypatch.setattr("tools.rivet.yoda_io.dump_yoda", _canned_dump)
    monkeypatch.delenv("FAKE_RIVET_MODE", raising=False)
    return install


def _hepmc_text(n_events, sqrt_s=7000.0, pid=2212):
    """A minimal HepMC3 ASCII file: two beams and one final pion per event."""
    half = sqrt_s / 2
    lines = [
        "HepMC::Version 3.02.05",
        "HepMC::Asciiv3-START_EVENT_LISTING",
        "W Weight",
    ]
    for i in range(n_events):
        lines += [
            f"E {i} 1 3",
            "U GEV MM",
            "W 1.0",
            "A 0 GenCrossSection 1000.0 10.0 -1 -1",
            f"P 1 0 {pid} 0 0 {half} {half} 0.938 4",
            f"P 2 0 {pid} 0 0 {-half} {half} 0.938 4",
            "V -1 0 [1,2]",
            "P 3 -1 211 1 0 0 1.01 0.14 1",
        ]
    lines.append("HepMC::Asciiv3-END_EVENT_LISTING")
    return "\n".join(lines) + "\n"


def _json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"status": "error", "text": text}


# --------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------- #


def test_beam_names_cover_spellings_and_pdg_ids():
    assert (
        cat.beam_name("p")
        == cat.beam_name("p+")
        == cat.beam_name(2212)
        == "p+"
    )
    assert cat.beam_name("pbar-") == cat.beam_name(-2212) == "p-"
    assert cat.beam_name(-11) == "e+" and cat.beam_name("e-") == "e-"
    assert cat.beam_name(1000822080) == "Pb" and cat.beam_name("Pb") == "Pb"
    assert cat.beam_name(1000030070) == "ion(Z=3)"


def test_beams_are_normalised_from_every_metadata_form():
    assert cat.normalise_beams(["p+", "p+"]) == [("p+", "p+")]
    assert cat.normalise_beams([["e+", "e-"], ["e-", "e+"]]) == [("e+", "e-")]
    assert cat.normalise_beams({1: ["p+", "p+"], 2: ["p+", "Pb"]}) == [
        ("p+", "p+"),
        ("Pb", "p+"),
    ]
    assert cat.normalise_beams("1:[p+,p+] 2:[Pb,Pb]") == [
        ("p+", "p+"),
        ("Pb", "Pb"),
    ]
    assert cat.normalise_beams("p+ p+") == [("p+", "p+")]
    assert cat.normalise_beams([2212, -2212]) == [("p+", "p-")]
    assert cat.normalise_beams(None) == []
    with pytest.raises(ValueError):
        cat.normalise_beams(["p+"])


def test_energies_are_normalised_from_every_metadata_form():
    assert cat.normalise_energies([7000]) == [7000.0]
    assert cat.normalise_energies([[3500, 3500]]) == pytest.approx([7000.0])
    assert cat.normalise_energies([[328000, 4000]]) == pytest.approx(
        [2 * math.sqrt(328000 * 4000)]
    )
    assert cat.normalise_energies({1: [5020], 2: [2760, 5020]}) == [
        5020.0,
        2760.0,
        5020.0,
    ]
    assert cat.normalise_energies("1:[5020,7000] 2:[4000]") == [
        5020.0,
        7000.0,
        4000.0,
    ]
    assert cat.normalise_energies([]) == []


def test_info_files_are_parsed_including_the_odd_ones(data_dir):
    catalogue = cat.load_catalogue(str(data_dir))
    assert len(catalogue) == 4
    pp = catalogue.get("FAKE_2012_I1")
    assert (
        pp.experiment == "FAKE"
        and pp.year == 2012
        and pp.status == "VALIDATED"
    )
    assert pp.beams == [("p+", "p+")] and pp.energies_gev == [7000.0]
    assert pp.needs_cross_section is True and pp.options == ["LMODE=EL,MU"]
    assert pp.has_reference_data and pp.has_plot_info
    assert "angular correlations" in pp.description and pp.keywords == [
        "Z",
        "DrellYan",
    ]
    assert pp.references == ["arXiv:1211.0000 [hep-ex]"]
    ee = catalogue.get("FAKE_EE")
    assert ee.beams == [("e+", "e-")] and ee.energies_gev == pytest.approx(
        [91.2]
    )
    assert ee.status == "VALIDATED" and ee.status_flags == [
        "NOHEPDATA",
        "SINGLEWEIGHT",
    ]
    assert not ee.has_reference_data
    ion = catalogue.get("FAKE_ION")
    assert ion.beams == [("Pb", "p+"), ("Pb", "Pb")] and ion.energies_gev == [
        5020.0,
        2760.0,
        5020.0,
    ]
    mc = catalogue.get("MC_FAKE")
    assert (
        mc.beams == []
        and mc.energies_gev == []
        and mc.needs_cross_section is None
    )
    # An option string on the name is ignored when looking an analysis up.
    assert catalogue.get("FAKE_2012_I1:LMODE=MU") is pp


def test_a_file_pyyaml_rejects_still_yields_its_essentials(tmp_path):
    (tmp_path / "ODD_1.info").write_text(
        "Name: ODD_1\nSummary: Measurement: of: the odd thing\nBibKey: "
        "Odd:1995mt\n"
        "Status: VALIDATED\nBeams: [e+, e-]\nEnergies: [91.2]\nReferences:\n "
        "- Z.Phys. C69 (1996) 393\n"
    )
    analysis = cat.parse_info(str(tmp_path / "ODD_1.info"))
    assert analysis.parse_error
    assert analysis.beams == [("e+", "e-")] and analysis.energies_gev == [91.2]
    assert analysis.status == "VALIDATED" and analysis.references == [
        "Z.Phys. C69 (1996) 393"
    ]


def test_search_matches_words_without_markup_and_filters_by_run(data_dir):
    catalogue = cat.load_catalogue(str(data_dir))
    assert [a.name for a in catalogue.search("phi* transverse momentum")] == [
        "FAKE_2012_I1"
    ]
    assert [a.name for a in catalogue.search("drell-yan")] == ["FAKE_2012_I1"]
    assert catalogue.search("z pole") and catalogue.search(
        "Z"
    )  # whole words only
    assert [
        a.name for a in catalogue.search("Z", beams="p+ p+", energy_gev=7000)
    ] == ["FAKE_2012_I1"]
    assert catalogue.search("Z", beams="p+ p+", energy_gev=13000) == []
    # Analyses that list no beams or energies accept any run.
    names = {
        a.name
        for a in catalogue.search("", beams=[2212, 2212], energy_gev=7000)
    }
    assert names == {"FAKE_2012_I1", "MC_FAKE"}
    assert {
        a.name for a in catalogue.search("", beams="Pb Pb", energy_gev=2760)
    } == {"FAKE_ION", "MC_FAKE"}
    assert [a.name for a in catalogue.search("", experiment="fakeee")] == [
        "FAKE_EE"
    ]
    assert [a.name for a in catalogue.search("", status="unvalidated")] == [
        "FAKE_ION"
    ]
    assert [
        a.name for a in catalogue.search("", with_reference_data=True)
    ] == ["FAKE_2012_I1"]
    assert [a.name for a in catalogue.search("", keyword="drellyan")] == [
        "FAKE_2012_I1"
    ]
    assert catalogue.suggest("FAKE_2012_I")[0] == "FAKE_2012_I1"


def test_compatibility_explains_itself(data_dir):
    pp = cat.load_catalogue(str(data_dir)).get("FAKE_2012_I1")
    assert pp.compatible("p+ p+", 7000.0) == (True, "")
    assert pp.compatible("p+ p+", 7050.0)[0]  # within 1%
    ok, why = pp.compatible("p+ p+", 13000.0)
    assert not ok and "13000" in why and "7000" in why
    ok, why = pp.compatible("e+ e-", 7000.0)
    assert not ok and "e+ e-" in why
    assert pp.compatible(None, None) == (True, "")


def test_reference_paths_and_plot_styles_come_from_the_data_files(data_dir):
    assert cat.reference_paths(str(data_dir), "FAKE_2012_I1:LMODE=MU") == [
        ("/REF/FAKE_2012_I1/d01-x01-y01", "ESTIMATE1D"),
        ("/REF/FAKE_2012_I1/d01-x01-y02", "ESTIMATE1D"),
    ]
    assert cat.reference_paths(str(data_dir), "MC_FAKE") == []
    styles = cat.plot_styles(str(data_dir), "FAKE_2012_I1")
    first = cat.plot_style(styles, "/FAKE_2012_I1/d01-x01-y01")
    assert first["LogX"] == "1" and first["Title"].startswith(
        "$\\phi^*_\\eta$ spectrum, Z$\\rightarrow$ ee"
    )
    second = cat.plot_style(styles, "/FAKE_2012_I1/d01-x01-y02")
    assert (
        second["LogY"] == "0" and "RatioPlotYMin" in second
    )  # later blocks add to earlier
    assert cat.plot_style(styles, "/OTHER/d01-x01-y01") == {}


def test_cautions_flag_what_a_reader_should_know(data_dir):
    catalogue = cat.load_catalogue(str(data_dir))
    assert cat.caution(catalogue.get("FAKE_2012_I1")) == []
    notes = " ".join(cat.caution(catalogue.get("FAKE_EE")))
    assert "SINGLEWEIGHT" in notes and "no reference data" in notes
    assert any(
        "UNVALIDATED" in n for n in cat.caution(catalogue.get("FAKE_ION"))
    )
    assert (
        cat.caution(catalogue.get("MC_FAKE")) == []
    )  # MC_ analyses have no data by design


# --------------------------------------------------------------------- #
# Comparison statistics
# --------------------------------------------------------------------- #


def test_chi2_survival_matches_the_tables():
    assert chi2_survival(3.841, 1) == pytest.approx(0.05, abs=2e-4)
    assert chi2_survival(18.307, 10) == pytest.approx(0.05, abs=2e-4)
    assert chi2_survival(0.0, 5) == 1.0
    assert chi2_survival(1000.0, 5) < 1e-100
    assert chi2_survival(5.0, 5) == pytest.approx(0.4159, abs=1e-3)
    assert math.isnan(chi2_survival(1.0, 0))


def test_identical_histograms_agree_perfectly():
    ref = _estimate("/REF/A/h", _REF_VALUES, _REF_ERRORS)
    mc = _estimate("/A/h", _REF_VALUES, [0.3] * 5)
    result = compare_histograms(mc, ref)
    assert (
        result["comparable"]
        and result["chi2"] == 0
        and result["p_value"] == 1.0
    )
    assert result["ndf"] == 5 and result["mc_empty_bins"] == 0
    assert verdict(result) == "consistent"


def test_pulls_use_the_uncertainty_facing_the_other_value():
    ref = _estimate("/REF/A/h", [10.0] * 5, [1.0] * 5)
    ref["err_up"] = [2.0] * 5  # asymmetric data errors
    mc = _estimate("/A/h", [12.0] * 5, [0.0] * 5)
    result = compare_histograms(mc, ref)
    # Prediction above the data: the data's upper error (2.0) applies.
    assert result["bins"][0]["pull"] == pytest.approx(1.0)
    assert result["chi2"] == pytest.approx(5.0)


def test_a_normalisation_offset_is_told_from_a_wrong_shape():
    ref = _estimate("/REF/A/h", _REF_VALUES, [0.05] * 5)
    scaled = _estimate("/A/h", [v * 1.2 for v in _REF_VALUES], [0.01] * 5)
    result = compare_histograms(scaled, ref)
    assert result["p_value"] < 1e-3
    assert result["normalisation_mc_over_data"] == pytest.approx(1.2)
    assert result["chi2_shape"] == pytest.approx(0.0, abs=1e-9)
    assert verdict(result) == "shape consistent, normalisation off"


def test_empty_prediction_bins_count_rather_than_blow_up():
    ref = _estimate("/REF/A/h", _REF_VALUES, _REF_ERRORS)
    mc = _estimate(
        "/A/h", _REF_VALUES[:3] + [0.0, 0.0], [0.3] * 3 + [0.0, 0.0]
    )
    result = compare_histograms(mc, ref)
    assert result["ndf"] == 3 and result["mc_empty_bins"] == 2
    assert result["chi2"] == 0
    assert "empty prediction bins" in verdict(result)
    nothing = _estimate("/A/h", [0.0] * 5, [0.0] * 5)
    result = compare_histograms(nothing, ref)
    assert not result["comparable"] and result["mc_empty_bins"] == 5


def test_mismatched_binning_is_not_compared():
    ref = _estimate("/REF/A/h", _REF_VALUES, _REF_ERRORS)
    mc = _estimate("/A/h", _REF_VALUES[:4], _REF_ERRORS[:4], edges=_EDGES[:5])
    result = compare_histograms(mc, ref)
    assert not result["comparable"] and "binning" in result["reason"]


# --------------------------------------------------------------------- #
# Rivet's log and output
# --------------------------------------------------------------------- #

_LOG = (
    "Rivet 4.1.4 running on machine host (arm64) at 2026-10-19 12:35:10\n"
    "Reading events from 'events.hepmc'\n"
    "Rivet.AnalysisLoader: WARN  Instantiating analysis 'ALEPH_1996_I428072'"
    " via alias 'ALEPH_1996_S3486095'. Using the canonical name is"
    " recommended\n"
    "Rivet.AnalysisHandler: WARN  Analysis 'ALEPH_1996_I428072' is"
    " incompatible with the provided beams: removing\n"
    "Rivet.Analysis.X: WARN  thing\n"
    "Rivet.Analysis.X: WARN  thing\n"
    "Event 100 (0:00:01 elapsed)\n"
    "Event 200 (0:00:02 elapsed)\n"
    "Finished event loop at 2026-10-19 12:35:21\n"
    "Cross-section = 1.568382e+03 pb\n"
    "The MCnet usage guidelines apply to Rivet: see"
    " http://www.montecarlonet.org/GUIDELINES\n"
)


def test_rivets_log_is_read_for_what_matters():
    log = parse_rivet_log(_LOG)
    assert log["removed"] == ["ALEPH_1996_I428072"]
    assert log["last_event"] == 200 and log[
        "cross_section_pb"
    ] == pytest.approx(1568.382)
    assert any(w.endswith("thing (x2)") for w in log["warnings"])
    assert (
        not log["all_removed"] and log["unknown"] == [] and log["errors"] == []
    )
    failed = parse_rivet_log(
        "'FOO' is not a known Rivet analysis! Do you need to set "
        "RIVET_ANALYSIS_PATH?\n"
    )
    assert failed["unknown"] == ["FOO"]
    dead = parse_rivet_log(
        "Error in run initialisation: All analyses were incompatible with the "
        "first event's beams\n"
    )
    assert dead["all_removed"] and dead["errors"]


def test_output_summary_counts_histograms_events_and_variations():
    objects = (
        {
            o["path"]: o
            for o in _canned_dump(None, ["x"], ())["files"][0]["objects"]
        }
        if False
        else None
    )
    # Build the objects by hand rather than through a marker file.
    objects = {
        "/RAW/_EVTCOUNT": {
            "path": "/RAW/_EVTCOUNT",
            "type": "Counter",
            "total": {"entries": 500.0, "sumw": 500.0},
        },
        "/RAW/_XSEC": {
            "path": "/RAW/_XSEC",
            "type": "Estimate0D",
            "values": [12.0],
            "err_up": [1.0],
            "err_down": [1.0],
        },
        "/A/h1": _estimate("/A/h1", _REF_VALUES, _REF_ERRORS),
        "/A/h1[v1]": _estimate("/A/h1[v1]", _REF_VALUES, _REF_ERRORS),
        "/A/h2": _estimate("/A/h2", [0.0] * 5, [0.0] * 5),
        "/RAW/A/h1": {
            "path": "/RAW/A/h1",
            "type": "Histo1D",
            "edges": _EDGES,
            "entries": [10.0] * 5,
            "sumw": [10.0] * 5,
            "sumw2": [10.0] * 5,
            "total": {"entries": 50.0, "sumw": 50.0},
        },
        "/RAW/A/h2": {
            "path": "/RAW/A/h2",
            "type": "Histo1D",
            "edges": _EDGES,
            "entries": [0.0] * 5,
            "sumw": [0.0] * 5,
            "sumw2": [0.0] * 5,
            "total": {"entries": 0.0, "sumw": 0.0},
        },
        "/B:OPT=1/h": _estimate("/B:OPT=1/h", [0.0] * 5, [0.0] * 5),
        "/TMP/_BEAMPZ": _estimate("/TMP/_BEAMPZ", [1.0], [0.0], edges=[0, 1]),
    }
    summary = summarise_output(objects)
    assert summary["n_events"] == 500.0 and summary["cross_section_pb"] == (
        12.0,
        1.0,
    )
    assert summary["weight_variations"] == ["v1"]
    assert summary["analyses"]["A"] == {
        "histograms": 2,
        "filled": 1,
        "empty": ["h2"],
        "raw_entries": 50,
    }
    assert summary["analyses"]["B:OPT=1"]["raw_entries"] == 0
    assert "_BEAMPZ" not in str(summary)


def test_yoda_path_conventions():
    assert yoda_io.split_path("/ATLAS_2012_I1204784/d01-x01-y01") == (
        "ATLAS_2012_I1204784",
        "d01-x01-y01",
    )
    assert yoda_io.split_path("/MC_ZINC:LMODE=MU/Z_pT[fsrup]") == (
        "MC_ZINC:LMODE=MU",
        "Z_pT[fsrup]",
    )
    assert yoda_io.analysis_name("/RAW/MC_ZINC:LMODE=MU/Z_pT") == "MC_ZINC"
    assert (
        yoda_io.reference_path("/MC_ZINC:LMODE=MU/Z_pT[fsrup]")
        == "/REF/MC_ZINC/Z_pT"
    )
    assert (
        yoda_io.nominal_path("/A/h[v]") == "/A/h"
        and yoda_io.variation_name("/A/h[v]") == "v"
    )
    assert yoda_io.is_internal("/RAW/_EVTCOUNT") and yoda_io.is_internal(
        "/TMP/_BEAMPZ"
    )
    assert not yoda_io.is_internal("/A/d01-x01-y01")


def test_analysis_checks_before_a_run(data_dir):
    catalogue = cat.load_catalogue(str(data_dir))
    run = {
        "beams": ["p+", "p+"],
        "sqrt_s_gev": 7000.0,
        "cross_section_pb": 1000.0,
    }
    known, warnings = check_analyses(
        catalogue, ["FAKE_2012_I1:LMODE=MU", "MC_FAKE:PTMIN=20"], [], run
    )
    assert set(known) == {"FAKE_2012_I1", "MC_FAKE"} and warnings == []
    with pytest.raises(AnalysisCheckError) as caught:
        check_analyses(catalogue, ["FAKE_2012_I"], [], run)
    assert (
        caught.value.error == "Unknown Analysis"
        and "FAKE_2012_I1" in caught.value.reason
    )
    with pytest.raises(AnalysisCheckError) as caught:
        check_analyses(
            catalogue, ["FAKE_2012_I1"], [], {**run, "sqrt_s_gev": 13000.0}
        )
    assert (
        caught.value.error == "Incompatible Analyses"
        and "13000" in caught.value.reason
    )
    _, warnings = check_analyses(
        catalogue,
        ["FAKE_2012_I1"],
        [],
        {**run, "sqrt_s_gev": 13000.0},
        ignore_beams=True,
    )
    assert any("13000" in w for w in warnings)
    with pytest.raises(AnalysisCheckError) as caught:
        check_analyses(
            catalogue,
            ["FAKE_2012_I1"],
            [],
            {"beams": ["p+", "p+"], "sqrt_s_gev": 7000.0},
        )
    assert caught.value.error == "Cross Section Needed"
    check_analyses(
        catalogue,
        ["FAKE_2012_I1"],
        [],
        {"beams": ["p+", "p+"], "sqrt_s_gev": 7000.0},
        cross_section_pb=5.0,
    )
    _, warnings = check_analyses(
        catalogue,
        ["FAKE_2012_I1:PTMIN=3", "FAKE_EE"],
        [],
        {"beams": ["p+", "p+"], "sqrt_s_gev": 7000.0, "cross_section_pb": 1.0},
        ignore_beams=True,
    )
    assert any("option 'PTMIN'" in w for w in warnings) and any(
        "SINGLEWEIGHT" in w for w in warnings
    )
    # A user analysis in a custom directory is left for Rivet to judge.
    custom = data_dir.parent / "custom"
    custom.mkdir(exist_ok=True)
    (custom / "RivetMINE.so").write_bytes(b"")
    known, _ = check_analyses(catalogue, ["MINE"], [str(custom)], run)
    assert known == {}


def test_mathtext_translation_and_plain_fallback():
    assert mathtext(r"$\pT$ [\GeV]") == r"$p_\mathrm{T}$ [\,\mathrm{GeV}]"
    assert mathtext(r"\text{jets}") == r"\mathrm{jets}"
    assert plain_text(r"$\phi^*_\eta$ spectrum") == "phi eta spectrum"


# --------------------------------------------------------------------- #
# The tools, against the fake installation
# --------------------------------------------------------------------- #


def test_find_tool_searches_and_describes(tmp_path, fake_install):
    tool = RivetFindAnalysesTool(base_directory=str(tmp_path))
    found = _json(
        tool.execute(
            query="transverse momentum", beams="p+ p+", energy_gev=7000
        )
    )
    assert (
        found["n_matches"] == 1
        and found["analyses"][0]["name"] == "FAKE_2012_I1"
    )
    assert found["analyses"][0]["needs_cross_section"] is True
    detail = _json(tool.execute(name="FAKE_2012_I1"))
    assert detail["run_info"].startswith("$Z/")
    assert [h["path"] for h in detail["histograms"]] == [
        "/FAKE_2012_I1/d01-x01-y01",
        "/FAKE_2012_I1/d01-x01-y02",
    ]
    assert detail["histograms"][1]["title"].endswith("(bare)")
    assert detail["cautions"] == [] and detail["options"] == ["LMODE=EL,MU"]
    assert "not an analysis" in tool.execute(name="FAKE_2012_I")
    empty = _json(tool.execute(query="nothing matches this"))
    assert empty["n_matches"] == 0 and "note" in empty


def test_run_tool_refuses_what_rivet_would_get_wrong(tmp_path, fake_install):
    (tmp_path / "events.hepmc").write_text(_hepmc_text(10))
    tool = RunRivetTool(base_directory=str(tmp_path))
    assert "hepmc_path is required" in tool.execute(analyses=["FAKE_2012_I1"])
    assert "analyses is required" in tool.execute(hepmc_path="events.hepmc")
    assert "does not exist" in tool.execute(
        hepmc_path="missing.hepmc", analyses=["FAKE_2012_I1"]
    )
    assert "escapes" in tool.execute(
        hepmc_path="../events.hepmc", analyses=["FAKE_2012_I1"]
    )
    out = tool.execute(hepmc_path="events.hepmc", analyses=["FAKE_2012_I"])
    assert "Unknown Analysis" in out and "FAKE_2012_I1" in out
    if HAVE_PYHEPMC:
        (tmp_path / "ee.hepmc").write_text(
            _hepmc_text(10, sqrt_s=91.2, pid=11)
        )
        out = tool.execute(hepmc_path="ee.hepmc", analyses=["FAKE_2012_I1"])
        assert "Incompatible Analyses" in out and "e- e-" in out
        (tmp_path / "hi.hepmc").write_text(_hepmc_text(10, sqrt_s=13000.0))
        out = tool.execute(hepmc_path="hi.hepmc", analyses=["FAKE_2012_I1"])
        assert "Incompatible Analyses" in out and "13000" in out


def test_run_tool_runs_rivet_and_reads_its_output(tmp_path, fake_install):
    (tmp_path / "events.hepmc").write_text(_hepmc_text(250))
    tool = RunRivetTool(base_directory=str(tmp_path))
    result = _json(
        tool.execute(
            hepmc_path="events.hepmc",
            analyses=["FAKE_2012_I1:LMODE=MU", "MC_FAKE"],
            output_path="out/run.yoda",
            max_events=200,
            skip_weights=True,
        )
    )
    assert result["status"] == "ok", result
    assert (
        result["output_yoda"] == "out/run.yoda" and result["n_events"] == 250.0
    )
    assert result["cross_section_pb"] == 1000.0
    assert result["analyses"]["FAKE_2012_I1:LMODE=MU"] == {
        "histograms": 2,
        "filled": 2,
        "raw_entries": 500,
        "reference_data": True,
    }
    assert result["analyses"]["MC_FAKE"]["reference_data"] is False
    assert result["weight_variations"] == ["fsrup"]
    if HAVE_PYHEPMC:
        assert result["events"]["beams"] == ["p+", "p+"] and result["events"][
            "sqrt_s_gev"
        ] == pytest.approx(7000.0)
    args = (tmp_path / "out" / "fake_rivet_args.json").read_text()
    assert (
        "'-n', '200'" in args
        and "'--skip-weights'" in args
        and "'-a', 'FAKE_2012_I1:LMODE=MU'" in args
    )
    assert (tmp_path / "out" / "rivet.stdout.log").exists()
    assert any("only 250 events" in w for w in result["warnings"])


def test_run_tool_reports_rivets_own_failures(
    tmp_path, fake_install, monkeypatch
):
    (tmp_path / "events.hepmc").write_text(_hepmc_text(5))
    tool = RunRivetTool(base_directory=str(tmp_path))
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "RivetMINE.so").write_bytes(b"")
    fake_install.env["FAKE_RIVET_MODE"] = "unknown"
    out = tool.execute(
        hepmc_path="events.hepmc", analyses=["MINE"], analysis_path=["custom"]
    )
    assert "Rivet Run Failed" in out and "unknown analysis: MINE" in out
    fake_install.env["FAKE_RIVET_MODE"] = "all_removed"
    out = tool.execute(hepmc_path="events.hepmc", analyses=["MC_FAKE"])
    assert "Rivet Run Failed" in out and "incompatible" in out
    fake_install.env["FAKE_RIVET_MODE"] = "warn"
    result = _json(
        tool.execute(hepmc_path="events.hepmc", analyses=["MC_FAKE"])
    )
    assert result["status"] == "ok" and any(
        "Something odd happened (x2)" in w for w in result["warnings"]
    )


def test_compare_tool_writes_a_report_and_a_summary(tmp_path, fake_install):
    (tmp_path / "a.yoda").write_text(
        "FAKEYODA events=400 analyses=FAKE_2012_I1,MC_FAKE inputs=1\n"
    )
    (tmp_path / "b.yoda").write_text(
        "FAKEYODA events=10000 analyses=FAKE_2012_I1 inputs=1\n"
    )
    tool = RivetCompareTool(base_directory=str(tmp_path))
    result = _json(
        tool.execute(
            yoda_paths=["a.yoda", "b.yoda"],
            labels=["small", "large"],
            output_path="cmp/report.json",
        )
    )
    assert result["status"] == "ok", result
    small, large = result["files"]
    assert (
        small["label"] == "small"
        and small["analyses"]["FAKE_2012_I1"]["histograms_compared"] == 2
    )
    assert small["analyses"]["MC_FAKE"]["without_reference_data"] == 2
    assert small["analyses"]["MC_FAKE"]["histograms_compared"] == 0
    assert (
        large["analyses"]["FAKE_2012_I1"]["consistent_at_5pct"] < 2
    )  # 20% high with tiny errors
    worst = large["analyses"]["FAKE_2012_I1"]["worst"][0]
    assert worst["verdict"] == "shape consistent, normalisation off"
    assert worst["title"].startswith("$\\phi^*_\\eta$ spectrum")
    report = json.loads((tmp_path / "cmp" / "report.json").read_text())
    bins = report["files"][1]["analyses"]["FAKE_2012_I1"]["histograms"][0][
        "bins"
    ]
    assert len(bins) == 5 and "pull" in bins[0]
    assert "labels" in tool.execute(yoda_paths=["a.yoda"], labels=["x", "y"])


@pytest.mark.skipif(not HAVE_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_tool_draws_every_histogram_with_a_gallery(
    tmp_path, fake_install
):
    (tmp_path / "a.yoda").write_text(
        "FAKEYODA events=400 analyses=FAKE_2012_I1,MC_FAKE inputs=1\n"
    )
    (tmp_path / "b.yoda").write_text(
        "FAKEYODA events=1600 analyses=FAKE_2012_I1 inputs=1\n"
    )
    tool = RivetPlotTool(base_directory=str(tmp_path))
    result = _json(
        tool.execute(
            yoda_paths=["a.yoda", "b.yoda"],
            labels=["tune A", "tune B"],
            output_dir="plots",
        )
    )
    assert result["status"] == "ok", result
    assert result["n_plots"] == 4
    images = [p["image"] for a in result["analyses"].values() for p in a]
    assert all((tmp_path / image).stat().st_size > 1000 for image in images)
    assert (tmp_path / "plots" / "index.html").exists()
    index = json.loads((tmp_path / "plots" / "index.json").read_text())
    assert index["labels"] == ["tune A", "tune B"] and len(index["plots"]) == 4
    with_data = [p for p in index["plots"] if p["has_data"]]
    assert len(with_data) == 2 and all(
        c["comparable"] for p in with_data for c in p["comparisons"]
    )
    assert result["analyses"]["MC_FAKE"][0]["data"] == "none"
    only = _json(
        tool.execute(
            yoda_paths=["a.yoda"], histograms="y02$", image_format="svg"
        )
    )
    assert only["n_plots"] == 2 and all(
        p["image"].endswith(".svg")
        for a in only["analyses"].values()
        for p in a
    )


def test_merge_tool_combines_runs(tmp_path, fake_install):
    (tmp_path / "a.yoda").write_text(
        "FAKEYODA events=400 analyses=FAKE_2012_I1 inputs=1\n"
    )
    (tmp_path / "b.yoda").write_text(
        "FAKEYODA events=600 analyses=FAKE_2012_I1 inputs=1\n"
    )
    tool = RivetMergeTool(base_directory=str(tmp_path))
    assert "at least two" in tool.execute(yoda_paths=["a.yoda"])
    result = _json(
        tool.execute(
            yoda_paths=["a.yoda", "b.yoda"],
            output_path="m/merged.yoda",
            cross_sections_pb=[1.0, 2.0],
        )
    )
    assert result["status"] == "ok", result
    assert (
        result["output_yoda"] == "m/merged.yoda"
        and result["n_inputs"] == 2
        and result["equivalent"]
    )
    args = (tmp_path / "m" / "fake_rivet_args.json").read_text()
    assert "'-e'" in args and ":=1.0'" in args and ":=2.0'" in args


@pytest.mark.skipif(
    not (HAVE_PYTHIA and HAVE_PYHEPMC), reason="pythia8mc and pyhepmc needed"
)
def test_generation_streams_events_to_rivet(
    tmp_path, fake_install, monkeypatch
):
    from tools.pythia.generate import PythiaFromRunCardTool

    (tmp_path / "z.cmnd").write_text(
        "Beams:eCM = 7000.\nWeakSingleBoson:ffbar2gmZ = on\n23:onMode = "
        "off\n23:onIfAny = 13\n"
        "PhaseSpace:mHatMin = 60.\nPartonLevel:MPI = off\n"
    )
    tool = PythiaFromRunCardTool(base_directory=str(tmp_path))
    assert "complete event record" in tool.execute(
        data_dir="s",
        cmnd_path="z.cmnd",
        n_events=5,
        rivet_analyses=["MC_FAKE"],
    )
    assert "Incompatible Analyses" in tool.execute(
        data_dir="s",
        cmnd_path="z.cmnd",
        n_events=5,
        store="full",
        rivet_analyses=["FAKE_EE"],
    )
    result = _json(
        tool.execute(
            data_dir="s",
            cmnd_path="z.cmnd",
            n_events=30,
            seed=5,
            store="full",
            rivet_analyses=["FAKE_2012_I1:LMODE=MU", "MC_FAKE"],
        )
    )
    assert result["status"] == "ok", result
    assert "events" not in result and result["format"] == "hepmc3"
    assert (
        result["rivet"]["output_yoda"] == "s/rivet.yoda"
        and result["rivet"]["n_events"] == 30.0
    )
    assert result["rivet"]["analyses"]["MC_FAKE"]["filled"] == 2
    assert not (tmp_path / "s" / "events.hepmc.fifo").exists()
    assert not any(
        name.startswith("events.") for name in os.listdir(tmp_path / "s")
    )
    manifest = json.loads((tmp_path / "s" / "manifest.json").read_text())
    assert (
        "events" not in manifest["outputs"]
        and manifest["outputs"]["rivet"]["n_events"] == 30.0
    )
    # Rivet dying mid-run stops the generation with its log, not a hang.
    fake_install.env["FAKE_RIVET_MODE"] = "crash_midway"
    out = tool.execute(
        data_dir="s2",
        cmnd_path="z.cmnd",
        n_events=60,
        seed=6,
        store="full",
        chunk_size=10,
        rivet_analyses=["MC_FAKE"],
    )
    assert "Rivet" in out and "status 3" in out
    assert not (tmp_path / "s2" / "events.hepmc.fifo").exists()


# --------------------------------------------------------------------- #
# A real Rivet installation
# --------------------------------------------------------------------- #


@pytest.mark.skipif(
    not HAVE_REAL_RIVET, reason="RIVET_PATH not set to a Rivet installation"
)
def test_real_installation_resolves():
    install = inst.resolve_rivet(_REAL_RIVET)
    assert install.version and install.yoda_version
    assert os.path.isdir(install.data_dir) and os.access(
        install.executable("rivet"), os.X_OK
    )
    catalogue = cat.load_catalogue(install.data_dir)
    assert len(catalogue) > 1000
    assert catalogue.get("ATLAS_2012_I1204784").compatible("p+ p+", 7000.0)[0]
    found = catalogue.search("drell-yan phi*", beams="p+ p+", energy_gev=7000)
    assert "ATLAS_2012_I1204784" in [a.name for a in found]


@pytest.fixture(scope="module")
def real_sample(tmp_path_factory):
    """A small Z -> mu mu sample at 7 TeV as a complete-record HepMC3 file."""
    from tools.pythia.generate import PythiaFromRunCardTool

    base = tmp_path_factory.mktemp("real")
    (base / "z.cmnd").write_text(
        "Beams:eCM = 7000.\nWeakSingleBoson:ffbar2gmZ = on\n23:onMode = "
        "off\n23:onIfAny = 13\n"
        "PhaseSpace:mHatMin = 60.\nUncertaintyBands:doVariations = on\n"
        "UncertaintyBands:List = {fsrup fsr:muRfac=0.5,fsrdown "
        "fsr:muRfac=2.0}\n"
    )
    tool = PythiaFromRunCardTool(base_directory=str(base))
    result = _json(
        tool.execute(
            data_dir="gen",
            cmnd_path="z.cmnd",
            n_events=80,
            seed=11,
            store="full",
            output_format="hepmc3",
            gzip_output=True,
        )
    )
    assert result["status"] == "ok", result
    return base


@pytest.mark.skipif(
    not (HAVE_REAL_RIVET and HAVE_PYTHIA and HAVE_PYHEPMC and HAVE_MATPLOTLIB),
    reason="RIVET_PATH, pythia8mc, pyhepmc and matplotlib needed",
)
def test_real_chain_from_file_and_pipe(real_sample, monkeypatch):
    from tools.pythia.generate import PythiaFromRunCardTool

    monkeypatch.setenv("RIVET_PATH", _REAL_RIVET)
    base = real_sample
    run = _json(
        RunRivetTool(base_directory=str(base)).execute(
            hepmc_path="gen/events.hepmc.gz",
            analyses=["ATLAS_2012_I1204784", "MC_ZINC:LMODE=MU"],
            output_path="rivet/file.yoda",
        )
    )
    assert run["status"] == "ok", run
    assert (
        run["n_events"] == 80.0
        and run["analyses"]["ATLAS_2012_I1204784"]["histograms"] == 16
    )
    assert run["analyses"]["MC_ZINC:LMODE=MU"]["raw_entries"] > 0
    # Pythia 8.312 prefixes shower variations with AUX_; 8.317 does not. Either
    # backend may be running, so the names are compared without it.
    assert {v.removeprefix("AUX_") for v in run["weight_variations"]} >= {
        "fsrup",
        "fsrdown",
    }
    assert run["cross_section_pb"] > 0

    bad = RunRivetTool(base_directory=str(base)).execute(
        hepmc_path="gen/events.hepmc.gz",
        analyses=["ATLAS_2019_I1736531"],
        output_path="rivet/bad.yoda",
    )
    assert "Incompatible Analyses" in bad and "13000" in bad

    streamed = _json(
        PythiaFromRunCardTool(base_directory=str(base)).execute(
            data_dir="stream",
            cmnd_path="z.cmnd",
            n_events=40,
            seed=12,
            store="full",
            rivet_analyses=["ATLAS_2012_I1204784"],
        )
    )
    assert streamed["status"] == "ok", streamed
    # Only the muon histograms can fill: the sample is Z -> mu mu.
    assert (
        streamed["rivet"]["n_events"] == 40.0
        and streamed["rivet"]["analyses"]["ATLAS_2012_I1204784"]["filled"] >= 8
    )
    assert not (base / "stream" / "events.hepmc.fifo").exists()

    compared = _json(
        RivetCompareTool(base_directory=str(base)).execute(
            yoda_paths=["rivet/file.yoda", "stream/rivet.yoda"],
            labels=["file", "pipe"],
            analyses=["ATLAS_2012_I1204784"],
            output_path="rivet/cmp.json",
        )
    )
    assert compared["status"] == "ok", compared
    summary = compared["files"][0]["analyses"]["ATLAS_2012_I1204784"]
    # The eight electron histograms hold no prediction, and say so.
    assert (
        summary["histograms_compared"] == 8
        and len(summary["not_compared"]) == 8
    )
    assert summary["worst"][0]["title"]

    plotted = _json(
        RivetPlotTool(base_directory=str(base)).execute(
            yoda_paths=["rivet/file.yoda", "stream/rivet.yoda"],
            labels=["file", "pipe"],
            histograms="d01-x01-y02$",
            output_dir="rivet/plots",
        )
    )
    assert plotted["status"] == "ok", plotted
    assert (
        plotted["n_plots"] == 1
        and (base / "rivet" / "plots" / "index.html").exists()
    )
    assert len(plotted["analyses"]["ATLAS_2012_I1204784"][0]["chi2_ndf"]) == 2

    merged = _json(
        RivetMergeTool(base_directory=str(base)).execute(
            yoda_paths=["rivet/file.yoda", "stream/rivet.yoda"],
            output_path="rivet/merged.yoda",
        )
    )
    assert merged["status"] == "ok", merged
    assert merged["n_events"] == 120.0


def run_all_tests() -> bool:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    return pytest.main([__file__, "-v" if verbose else "-q"]) == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
