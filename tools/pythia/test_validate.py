"""
# test_validate.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for the Pythia settings database and run-card validator.

The behavioural assertions here are grounded in Pythia's own source
(`Settings::readString`, `ParticleData::readString`, `boolString`) rather than
in what the manual says, because the validator's whole value is predicting what
Pythia will actually do with a card.
"""

import json
import sys
from pathlib import Path

import pytest

# Repository root on the path, so this file works both under pytest and when
# test_runner.py executes it directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pythia.card import parse_card_line
from tools.pythia.search import PythiaSettingsSearchTool
from tools.pythia.validate import ValidatePythiaRunCardTool, lint_card
from tools.pythia.settings_db import (
    get_setting,
    load_snapshot,
    parse_xmldoc,
    search_settings,
)
from tools.pythia.settings_topics import ALIASES, TOPICS, topic_for

try:
    import pythia8mc  # noqa: F401

    HAVE_PYTHIA = True
except ImportError:
    HAVE_PYTHIA = False

# Tests that need the installed Pythia are marked; everything else runs
# without it, against the bundled database.
needs_pythia = pytest.mark.skipif(
    not HAVE_PYTHIA, reason="pythia8mc not installed"
)


# --------------------------------------------------------------------- #
# Snapshot integrity
# --------------------------------------------------------------------- #


def test_snapshot_loads_with_expected_shape():
    snapshot = load_snapshot()
    assert snapshot["schema"] == "pythia-settings-1.0"
    assert snapshot["pythia_version"]
    # The 8.3xx series declares on the order of two thousand settings; a
    # snapshot far below that means the parser regressed.
    assert snapshot["n_settings"] > 1800
    assert len(snapshot["settings"]) == snapshot["n_settings"]


def test_core_settings_present_with_types_and_ranges():
    expected = {
        "Beams:eCM": ("float", "14000."),
        "Beams:idA": ("int", "2212"),
        "Beams:frameType": ("int", "1"),
        "HardQCD:all": ("bool", "off"),
        "PhaseSpace:pTHatMin": ("float", "0."),
        "Beams:LHEF": ("string", "void"),
    }
    for name, (kind, default) in expected.items():
        record = get_setting(name)
        assert record is not None, f"{name} missing from snapshot"
        assert record["type"] == kind
        assert record["default"] == default

    assert get_setting("PhaseSpace:pTHatMin")["min"] == "0."
    assert get_setting("Beams:frameType")["max"] == "5"


def test_lookup_is_case_insensitive_like_pythia():
    assert get_setting("beams:ecm")["name"] == "Beams:eCM"
    assert get_setting("BEAMS:ECM")["name"] == "Beams:eCM"


def test_modepick_options_are_captured():
    record = get_setting("Beams:frameType")
    values = {option["value"] for option in record["options"]}
    assert {"1", "2", "3", "4", "5"} <= values
    # Option 4 is the Les Houches path -- the one the showering workflow needs.
    option4 = next(o for o in record["options"] if o["value"] == "4")
    assert "les houches" in option4["description"].lower()


def test_self_closing_settings_do_not_inherit_a_neighbours_prose():
    """VinciaQCD declares `<parm .../>` with no body.

    A previous parser gave these the *next* setting's description, which
    attached the FSR-cutoff prose to a colour-charge factor.
    """
    record = get_setting("Vincia:QGEmitFF:chargeFactor")
    assert record is not None
    assert record["default"] == "2.85"
    assert "cutoff" not in record["description"].lower()


def test_settings_after_a_self_closing_block_are_not_swallowed():
    # SigmaDiffractive:MBRalpha follows a run of self-closing declarations and
    # was previously lost entirely.
    record = get_setting("SigmaDiffractive:MBRalpha")
    assert record is not None
    assert record["min"] == "0.1"


def test_names_containing_angle_brackets_are_parsed(tmp_path):
    # Dire setting names in 8.312 and earlier embed '>' (Dire_fsr_qcd_1->1&21),
    # which breaks a naive attribute scanner.
    from tools.pythia.settings_db import _parse_file

    page = tmp_path / "Dire.xml"
    page.write_text(
        '<chapter name="Dire">\n<parm name="Enhance:Dire_fsr_qcd_1->1&21" '
        'default="1.0" min="0.0">\nAn enhancement.\n</parm>\n</chapter>\n'
    )
    [record] = _parse_file(page)
    assert record["name"] == "Enhance:Dire_fsr_qcd_1->1&21"
    assert record["min"] == "0.0"


def test_only_the_files_pythia_indexes_are_read(tmp_path):
    """Two Pythia builds sharing one share/ prefix leave stray pages behind.

    Pythia reads only what Index.xml links to; a stray page from another
    release must not leak into the database.
    """
    (tmp_path / "Index.xml").write_text('<aidx href="Main">Main</aidx>')
    (tmp_path / "Main.xml").write_text(
        '<flag name="Main:ok" default="on">Kept.</flag>'
    )
    (tmp_path / "Stray.xml").write_text(
        '<flag name="Stray:gone" default="on">Old.</flag>'
    )
    names = {r["name"] for r in parse_xmldoc(tmp_path)["settings"]}
    assert names == {"Main:ok"}


def test_conflicting_declarations_are_refused(tmp_path):
    (tmp_path / "Index.xml").write_text(
        '<aidx href="A">A</aidx><aidx href="B">B</aidx>'
    )
    (tmp_path / "A.xml").write_text(
        '<parm name="X:y" default="1." min="0.">A.</parm>'
    )
    (tmp_path / "B.xml").write_text(
        '<parm name="X:y" default="1." min="-1.">B.</parm>'
    )
    with pytest.raises(ValueError, match="X:y"):
        parse_xmldoc(tmp_path)


def test_shared_description_blocks_are_inherited():
    # LowEnergyQCD:sEffAQM has an empty block; the prose lives with bEffAQM.
    record = get_setting("LowEnergyQCD:sEffAQM")
    assert record is not None
    assert record["description"]
    assert record.get("shared_description") is True


def test_particle_table_is_captured():
    snapshot = load_snapshot()
    particles = snapshot["particles"]
    assert particles["6"] == "t"
    assert particles["25"] == "h0"
    assert len(particles) > 500


def test_every_curated_alias_targets_a_real_setting():
    """A curated alias pointing at a name Pythia no longer has is dead weight.

    This is the check that catches a Pythia upgrade renaming a process group.
    """
    snapshot = load_snapshot()
    assert snapshot["stale_aliases"] == []
    names = {record["name"] for record in snapshot["settings"]}
    missing = sorted(name for name in ALIASES if name not in names)
    assert missing == []


def test_topics_are_assigned_from_the_prefix_table():
    assert topic_for("PhaseSpace:pTHatMin") == "cuts"
    assert topic_for("Beams:LHEF") == "lhe"  # more specific than Beams:
    assert topic_for("Beams:eCM") == "beams"
    assert topic_for("Nonsense:foo") == ""
    assert set(TOPICS) >= {"beams", "cuts", "process", "lhe"}


def test_fvec_settings_are_vectors_of_flags():
    record = get_setting("Charmonium:gg2ccbar(3S1)[3S1(1)]g")
    assert record["type"] == "vector<bool>"


@needs_pythia
def test_validation_uses_the_installed_pythias_database():
    """Whatever the bundled snapshot's release, a card is checked against the
    Pythia that will run it: its names, types, defaults and ranges."""
    from tools.pythia.settings_db import (
        _quiet_pythia,
        installed_settings,
        load_settings,
    )

    live = installed_settings(_quiet_pythia())
    used = load_settings()
    assert used["database"]["source"] == "installed"
    assert {r["name"].lower() for r in used["settings"]} == set(live)
    by_name = {r["name"].lower(): r for r in used["settings"]}
    for key in ("stringz:avgzlund", "beams:ecm", "timeshower:alphasvalue"):
        assert float(by_name[key]["min"]) == float(live[key]["min"])


@needs_pythia
def test_bundled_snapshot_matches_its_release():
    """The shipped snapshot is the fallback; it must be exactly one release."""
    from tools.pythia.settings_db import _quiet_pythia, installed_settings

    shipped = load_snapshot()
    pythia = _quiet_pythia()
    version = f"{float(pythia.settings.parm('Pythia:versionNumber')):.3f}"
    if version != shipped["pythia_version"]:
        pytest.skip(
            f"installed Pythia {version}, snapshot {shipped['pythia_version']}"
        )
    assert {r["name"].lower() for r in shipped["settings"]} == set(
        installed_settings(pythia)
    )


# --------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query,expected",
    [
        ("beam energy", "Beams:eCM"),
        ("les houches file", "Beams:LHEF"),
        ("pt cut", "PhaseSpace:pTHatMin"),
        ("random seed", "Random:seed"),
        ("pdf set", "PDF:pSet"),
        ("ttbar", "Top:all"),
        ("turn off hadronization", "HadronLevel:all"),
        ("isr alphas", "SpaceShower:alphaSvalue"),
    ],
)
def test_physics_phrases_reach_the_intended_setting(query, expected):
    hits = search_settings(query, limit=3)
    assert expected in [hit["name"] for hit in hits], (
        f"{query!r} -> {[h['name'] for h in hits]}"
    )


def test_exact_name_ranks_first():
    hits = search_settings("PhaseSpace:pTHatMin", limit=5)
    assert hits[0]["name"] == "PhaseSpace:pTHatMin"


def test_topic_and_prefix_filters_restrict_results():
    hits = search_settings("mass", topic="cuts", limit=10)
    assert hits and all(hit["topic"] == "cuts" for hit in hits)

    hits = search_settings("", prefix="Beams:", limit=100)
    assert hits and all(hit["name"].startswith("Beams:") for hit in hits)


# --------------------------------------------------------------------- #
# Line parsing -- mirrors Settings::readString
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Beams:eCM = 13000.", ("setting", "beams:ecm", "13000.")),
        ("Beams:eCM=13000.", ("setting", "beams:ecm", "13000.")),
        ("  Beams:eCM   13000.  ", ("setting", "beams:ecm", "13000.")),
        (
            "Beams:eCM = 13000. ! trailing comment",
            ("setting", "beams:ecm", "13000."),
        ),
        ("Beams::eCM = 13000.", ("setting", "beams:ecm", "13000.")),
        ("BEAMS:ECM = 13000.", ("setting", "beams:ecm", "13000.")),
        ("6:m0 = 172.5", ("particle", "6:m0", "172.5")),
        ("! a comment", None),
        ("# also a comment", None),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_card_line_matches_pythia_semantics(line, expected):
    assert parse_card_line(line) == expected


def test_vector_values_keep_their_braces():
    kind, key, value = parse_card_line("Parallelism:seeds = {1, 2, 3}")
    assert kind == "setting"
    assert key == "parallelism:seeds"
    assert value == "{1, 2, 3}"


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #

GOOD_CARD = """! ttbar at the LHC
Beams:idA = 2212
Beams:idB = 2212
Beams:eCM = 13600.
Top:gg2ttbar = on
Top:qqbar2ttbar = on
PhaseSpace:pTHatMin = 20.
6:m0 = 172.5
"""


def test_a_correct_card_validates_clean():
    report = lint_card(GOOD_CARD)
    assert report["valid"] is True
    assert report["counts"]["error"] == 0
    assert report["counts"]["warning"] == 0


def _messages(report, severity="error"):
    return [
        f["message"] for f in report["findings"] if f["severity"] == severity
    ]


def test_unknown_setting_is_an_error_with_a_spelling_suggestion():
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\nPhaseSpace:pTHatMinn = 25.\n"
    )
    assert report["valid"] is False
    finding = next(
        f for f in report["findings"] if "unknown setting" in f["message"]
    )
    assert finding["line"] == 3
    assert "PhaseSpace:pTHatMin" in finding["suggestion"]


def test_out_of_range_value_is_an_error():
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\nTimeShower:alphaSvalue = 5.0\n"
    )
    assert report["valid"] is False
    assert any("above the declared maximum" in m for m in _messages(report))


def test_unrecognised_boolean_is_an_error_because_pythia_reads_it_as_off():
    """`boolString` accepts only true/1/on/yes/ok; anything else is false.

    Pythia neither warns nor fails on `= onn`, so the run silently has
    hadronization disabled. This is the check that catches it.
    """
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\nHadronLevel:all = onn\n"
    )
    assert report["valid"] is False
    assert any("not a boolean" in m for m in _messages(report))


def test_valid_boolean_spellings_are_accepted():
    for value in ("on", "off", "true", "false", "yes", "no", "1", "0"):
        report = lint_card(
            f"Beams:eCM = 13000.\nHardQCD:all = {value}\nSoftQCD:all = on\n"
        )
        assert not any("not a boolean" in m for m in _messages(report)), value


def test_undocumented_mode_option_is_a_warning_not_an_error():
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\nBeams:frameType = 3\n"
    )
    assert not any("frameType" in m for m in _messages(report, "error"))


def test_unknown_particle_id_is_an_error():
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\n9999999:mayDecay = off\n"
    )
    assert report["valid"] is False
    assert any(
        "not in Pythia's particle data table" in m for m in _messages(report)
    )


def test_unknown_particle_property_is_an_error():
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\n15:notAProperty = 3\n"
    )
    assert report["valid"] is False
    assert any("not a particle property" in m for m in _messages(report))


def test_known_particle_lines_pass():
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\n6:m0 = 172.5\n15:mayDecay = "
        "off\n"
        "23:onMode = off\n23:onIfAny = 13 -13\n"
    )
    assert report["valid"] is True


def test_card_with_no_process_is_an_error():
    report = lint_card("Beams:eCM = 13000.\n")
    assert report["valid"] is False
    assert any("no hard process" in m for m in _messages(report))


def test_missing_beam_energy_is_a_warning():
    report = lint_card("HardQCD:all = on\n")
    assert any("no beam energy set" in m for m in _messages(report, "warning"))


def test_duplicate_assignment_is_flagged():
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\nPhaseSpace:pTHatMin = 20.\n"
        "PhaseSpace:pTHatMin = 50.\n"
    )
    assert any("is set again here" in m for m in _messages(report, "warning"))


# ---- integration with PythiaFromRunCardTool -------------------------- #


def test_lhe_card_without_a_lhef_line_is_valid_for_the_tool():
    """PythiaFromRunCardTool sets Beams:frameType and Beams:LHEF from lhe_path,
    so the card need not carry them; a note says it will not read an LHE file
    anywhere else."""
    report = lint_card("PartonLevel:MPI = on\n", lhe_mode=True)
    assert report["valid"] is True
    assert any("lhe_path" in m for m in _messages(report, "note"))


def test_lhe_mode_is_inferred_from_frame_type():
    report = lint_card("Beams:frameType = 4\n")
    assert report["valid"] is True
    assert any("Beams:LHEF" in m for m in _messages(report, "note"))


def test_complete_lhe_card_validates_clean():
    report = lint_card(
        "Beams:frameType = 4\nBeams:LHEF = dummy.lhe\nPartonLevel:MPI = on\n",
        lhe_mode=True,
    )
    assert report["valid"] is True


def test_card_seed_is_noted_as_yielding_to_the_tool_seed():
    """The tool appends its seed after the card, so the tool's value wins."""
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\nRandom:setSeed = "
        "on\nRandom:seed = 42\n"
    )
    assert any(
        "`seed`, when given, overrides" in m for m in _messages(report, "note")
    )
    assert _messages(report, "warning") == []


def test_seed_without_set_seed_is_flagged():
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\nRandom:seed = 42\n"
    )
    assert any("Random:setSeed" in m for m in _messages(report, "warning"))


def test_number_of_events_in_card_is_a_note():
    report = lint_card(
        "Beams:eCM = 13000.\nHardQCD:all = on\nMain:numberOfEvents = 1000\n"
    )
    assert any(
        "ignored by PythiaFromRunCardTool" in m
        for m in _messages(report, "note")
    )


# --------------------------------------------------------------------- #
# Tool wrappers
# --------------------------------------------------------------------- #


def test_search_tool_returns_json(tmp_path):
    tool = PythiaSettingsSearchTool(base_directory=str(tmp_path))
    payload = json.loads(tool.execute(query="beam energy", limit=3))
    assert payload["status"] == "ok"
    assert payload["hits"][0]["name"] == "Beams:eCM"
    assert "default" in payload["hits"][0]


def test_search_tool_expands_an_exact_name():
    tool = PythiaSettingsSearchTool(base_directory=".")
    payload = json.loads(tool.execute(query="Beams:frameType"))
    top = payload["hits"][0]
    assert top["name"] == "Beams:frameType"
    assert "options" in top, (
        "an exact-name lookup should return the full record"
    )


def test_search_tool_rejects_an_unknown_topic():
    tool = PythiaSettingsSearchTool(base_directory=".")
    assert "Error" in tool.execute(query="mass", topic="nonsense")


def test_search_tool_points_at_particle_data_when_nothing_matches():
    tool = PythiaSettingsSearchTool(base_directory=".")
    payload = json.loads(tool.execute(query="zzzzqqqxyzzy"))
    assert payload["n_hits"] == 0
    assert "pdgid" in payload["suggestion"].lower()


def test_validate_tool_reads_a_card_from_disk(tmp_path):
    card = tmp_path / "run.cmnd"
    card.write_text(GOOD_CARD, encoding="utf-8")
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(tool.execute(cmnd_path="run.cmnd"))
    assert payload["status"] == "ok"
    assert payload["valid"] is True
    assert payload["cmnd_path"] == "run.cmnd"


def test_validate_tool_requires_an_input(tmp_path):
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    assert "Error" in tool.execute()


def test_validate_tool_rejects_paths_outside_the_sandbox(tmp_path):
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    assert "Access Denied" in tool.execute(cmnd_path="../escape.cmnd")


def test_validate_tool_reports_a_missing_file(tmp_path):
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    assert "File Not Found" in tool.execute(cmnd_path="nope.cmnd")


# --------------------------------------------------------------------- #
# Dry run against the real Pythia
# --------------------------------------------------------------------- #


@needs_pythia
def test_dry_run_confirms_a_good_card_initialises(tmp_path):
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(
        tool.execute(
            card_text="Beams:eCM = 13000.\nHardQCD:all = "
            "on\nPhaseSpace:pTHatMin = 20.\n",
            dry_run_init=True,
        )
    )
    assert payload["dry_run"]["ran"] is True
    assert payload["dry_run"]["init_ok"] is True
    assert payload["valid"] is True


@needs_pythia
def test_dry_run_surfaces_pythia_own_error_text(tmp_path):
    """A card can pass every static check and still fail to initialise.

    pTHatMin above the beam energy leaves no phase space; nothing about the
    line is individually invalid, so only running Pythia catches it.
    """
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(
        tool.execute(
            card_text="Beams:eCM = 13000.\nHardQCD:all = "
            "on\nPhaseSpace:pTHatMin = 1e9\n",
            dry_run_init=True,
        )
    )
    assert payload["dry_run"]["init_ok"] is False
    assert payload["valid"] is False
    assert any("PYTHIA" in m for m in payload["dry_run"]["messages"])


@needs_pythia
def test_the_validator_agrees_with_pythia_on_a_card_pythia_rejects(tmp_path):
    """Cross-check: what the static checks reject, Pythia rejects too."""
    bad = "Beams:eCM = 13000.\nHardQCD:all = on\nPhaseSpace:pTHatMinn = 25.\n"
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(tool.execute(card_text=bad, dry_run_init=True))
    assert payload["valid"] is False
    assert payload["dry_run"]["init_ok"] is False


# --------------------------------------------------------------------- #
# Effective configuration -- what Pythia actually ended up holding
# --------------------------------------------------------------------- #


@needs_pythia
def test_dry_run_reports_the_effective_configuration(tmp_path):
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(
        tool.execute(
            card_text="Beams:eCM = 13000.\nHardQCD:all = "
            "on\nPhaseSpace:pTHatMin = 20.\n",
            dry_run_init=True,
        )
    )
    changed = {
        entry["name"]: entry
        for entry in payload["dry_run"]["changed_settings"]
    }
    assert "Beams:eCM" in changed
    assert float(changed["Beams:eCM"]["default"]) == 14000.0
    assert "HardQCD:all" in changed and changed["HardQCD:all"]["value"] == "on"


@needs_pythia
def test_a_tune_silently_overriding_an_earlier_setting_is_caught(tmp_path):
    """`Tune:pp` resets the parameters it owns when the line is read.

    A card that sets a tuned parameter *before* the tune line loses that value
    with no warning from Pythia, and the card is entirely valid. Only the
    effective configuration reveals it.
    """
    card = (
        "Beams:eCM = 13000.\nHardQCD:all = on\n"
        "MultipartonInteractions:pT0Ref = 3.14\n"
        "Tune:pp = 14\n"
    )
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(tool.execute(card_text=card, dry_run_init=True))

    overridden = {
        entry["name"]: entry for entry in payload["overridden_settings"]
    }
    assert "MultipartonInteractions:pT0Ref" in overridden
    entry = overridden["MultipartonInteractions:pT0Ref"]
    assert entry["requested"] == "3.14"
    assert entry["effective"] != "3.14"
    assert any("did not survive" in f["message"] for f in payload["findings"])


@needs_pythia
def test_the_correct_tune_ordering_produces_no_warning(tmp_path):
    """Tune first, then the override: the user's value survives."""
    card = (
        "Beams:eCM = 13000.\nHardQCD:all = on\n"
        "Tune:pp = 14\n"
        "MultipartonInteractions:pT0Ref = 3.14\n"
    )
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(tool.execute(card_text=card, dry_run_init=True))
    assert payload.get("overridden_settings", []) == []


@needs_pythia
def test_a_setting_assigned_its_own_default_is_not_reported_as_overridden(
    tmp_path,
):
    """Absent from the changes-only table because it did not change."""
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(
        tool.execute(
            card_text="Beams:eCM = 14000.\nHardQCD:all = "
            "on\nPhaseSpace:pTHatMin = 20.\n",
            dry_run_init=True,
        )
    )
    assert payload.get("overridden_settings", []) == []


@needs_pythia
def test_changed_particle_data_is_reported(tmp_path):
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    with_change = json.loads(
        tool.execute(
            card_text="Beams:eCM = 13000.\nTop:gg2ttbar = on\n6:m0 = 172.5\n",
            dry_run_init=True,
        )
    )
    without = json.loads(
        tool.execute(
            card_text="Beams:eCM = 13000.\nHardQCD:all = "
            "on\nPhaseSpace:pTHatMin = 20.\n",
            dry_run_init=True,
        )
    )
    assert with_change["dry_run"]["particle_data_changed"] is True
    assert without["dry_run"]["particle_data_changed"] is False


# --------------------------------------------------------------------- #
# Audit regressions -- each of these was once misjudged
# --------------------------------------------------------------------- #

BASE = "Beams:eCM = 13600.\nHardQCD:all = on\n"


@pytest.mark.parametrize(
    "ask_pythia", [False, pytest.param(True, marks=needs_pythia)]
)
@pytest.mark.parametrize(
    "card",
    [
        # decay-channel lines: <id>:<channel>:<property>
        BASE + "23:onMode = off\n23:3:onMode = on\n",
        # a braced value continued on the next line
        BASE + "Parallelism:seeds = {1, 2,\n 3, 4}\n",
        # a particle defined in the card, then modified
        BASE
        + "7777777:new = zeta zetabar 1 0 0 100.\n7777777:mayDecay = off\n",
        # an onia process switched on through a vector of flags
        "Beams:eCM = 13600.\nCharmonium:gg2ccbar(3S1)[3S1(1)]g = {on,off}\n",
        # frameType 3 gives the beam momenta, not an energy
        "Beams:frameType = 3\nBeams:pzA = 6800.\nBeams:pzB = "
        "-6800.\nHardQCD:all = on\n",
        # the Pythia spelling of the left-right-symmetric process group
        "Beams:eCM = 13600.\nLeftRightSymmmetry:ffbar2ZR = on\n",
        # settings Angantyr creates in code, absent from the manual
        BASE + "HIMultipartonInteractions:pT0Ref = 2.1\n",
    ],
)
def test_valid_cards_pythia_accepts_are_not_rejected(card, ask_pythia):
    report = lint_card(card, ask_pythia=ask_pythia)
    assert report["valid"] is True, report["findings"]


@pytest.mark.parametrize(
    "ask_pythia", [False, pytest.param(True, marks=needs_pythia)]
)
@pytest.mark.parametrize(
    "card,needle",
    [
        (BASE + "PhaseSpace:pTHatMin += 20.\n", "+="),
        (BASE + "PhaseSpace:pTHatMin = default\n", "DEFAULT"),
        (BASE + "23:m0 = heavy\n", "not a number"),
        (BASE + "23:isResonance = maybe\n", "not a boolean"),
        (BASE + "include no/such/file.cmnd\n", "not found"),
    ],
)
def test_broken_cards_are_rejected(card, needle, ask_pythia):
    report = lint_card(card, ask_pythia=ask_pythia)
    assert report["valid"] is False
    assert any(needle in m for m in _messages(report)), report["findings"]


def test_a_truncated_integer_is_a_warning_naming_what_pythia_reads():
    report = lint_card(BASE + "Tune:pp = 5.5\n", ask_pythia=False)
    assert any("leading '5'" in m for m in _messages(report, "warning"))


def test_repeating_a_setting_in_different_subruns_is_not_a_duplicate():
    card = BASE + (
        "Main:subrun = 1\nPhaseSpace:pTHatMin = 20.\n"
        "Main:subrun = 2\nPhaseSpace:pTHatMin = 50.\n"
    )
    report = lint_card(card, ask_pythia=False)
    assert not any("set again" in m for m in _messages(report, "warning"))


def test_an_include_is_read_in_place(tmp_path):
    (tmp_path / "common.cmnd").write_text("PhaseSpace:pTHatMinn = 20.\n")
    card = BASE + f"include {tmp_path / 'common.cmnd'}\n"
    report = lint_card(card, ask_pythia=False)
    [finding] = [
        f for f in report["findings"] if "unknown setting" in f["message"]
    ]
    assert finding["file"].endswith("common.cmnd") and finding["line"] == 1


@needs_pythia
def test_pythias_verdict_overrides_the_static_one():
    report = lint_card(BASE + "PhaseSpace:pTHatMinn = 20.\n")
    assert report["checked_by"] == "installed Pythia"
    [error] = [f for f in report["findings"] if f["severity"] == "error"]
    assert "not found in settings databases" in error["pythia_says"]


@needs_pythia
def test_the_dry_run_does_not_change_the_settings_it_reports(tmp_path):
    """Forcing Print:quiet = off after the card reset every Init:show* and
    Next:* setting, which were then reported as overridden by a tune."""
    card = BASE + (
        "Print:quiet = on\nInit:showChangedSettings = off\n"
        "Next:numberCount = 0\n"
    )
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(tool.execute(card_text=card, dry_run_init=True))
    assert payload.get("overridden_settings", []) == []


@needs_pythia
def test_a_tune_override_is_caught_under_a_non_default_tune(tmp_path):
    """The value, not mere presence in the changed table, decides: under
    Tune:pp = 5 pT0Ref stays changed from the default, just not to 3.14."""
    card = BASE + "MultipartonInteractions:pT0Ref = 3.14\nTune:pp = 5\n"
    tool = ValidatePythiaRunCardTool(base_directory=str(tmp_path))
    payload = json.loads(tool.execute(card_text=card, dry_run_init=True))
    [entry] = payload["overridden_settings"]
    assert entry["name"] == "MultipartonInteractions:pT0Ref"
    assert float(entry["effective"]) != 3.14
    assert "Tune:pp = 5" in entry["suggestion"]


def test_validator_path_guard_rejects_a_sibling_directory(tmp_path):
    base = tmp_path / "proj"
    base.mkdir()
    (tmp_path / "proj2").mkdir()
    (tmp_path / "proj2" / "x.cmnd").write_text(BASE)
    tool = ValidatePythiaRunCardTool(base_directory=str(base))
    assert "Access Denied" in tool.execute(cmnd_path="../proj2/x.cmnd")


def test_validator_schema_requires_nothing():
    assert (
        ValidatePythiaRunCardTool.get_input_schema().get("required", []) == []
    )


def run_all_tests() -> bool:
    import sys as _sys

    verbose = "-v" in _sys.argv or "--verbose" in _sys.argv
    return pytest.main([__file__, "-v" if verbose else "-q"]) == 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(0 if run_all_tests() else 1)
