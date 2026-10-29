#!/usr/bin/env python3
"""
# test_feynrules.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
FeynRules Tools Test Suite

Tests cover:
- UFO model generation from FeynRules
- Helper functions (_abs_path, _ensure_dir)
- Parameter validation
- Path resolution
- Integration with Mathematica/wolframscript
"""

# Standard library imports
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
FEYNRULES_DIR = SCRIPT_PATH.parent                            # .../heptapod/tools/feynrules
TOOLS_DIR = FEYNRULES_DIR.parent                              # .../heptapod/tools
REPO_ROOT = TOOLS_DIR.parent                                  # .../heptapod

# Add repository root to path for local imports (prompts, tools, etc.)
sys.path.insert(0, str(REPO_ROOT))

# Tool imports
from tools.feynrules import FeynRulesToUFOTool

# Future: When orchestral-ai PyPI package is fixed, use these imports instead:
# from orchestral.tools import FeynRulesToUFOTool

# Try to import config for configuration
try:
    # Paths come from the resolver, not from config.py directly: config.py is
    # gitignored, so on any clone but the one where somebody filled it in it
    # holds config.example.py's placeholders and every live test "fails" for
    # want of configuration. The resolver reads the environment first, then a
    # real config.py, then toolbase -- which already stores these paths for
    # this toolkit and keeps them out of the repository by construction.
    from config_resolve import resolve_path
    feynrules_path = resolve_path("feynrules_path")
    wolframscript_path = resolve_path("wolframscript_path")
    config_is_incomplete = not (feynrules_path and wolframscript_path)
except ImportError:
    feynrules_path = os.environ.get("FEYNRULES_PATH")
    wolframscript_path = os.environ.get("WOLFRAMSCRIPT_PATH")
    config_is_incomplete = not (feynrules_path and wolframscript_path)

# Initialize base directory
base_directory = str(FEYNRULES_DIR / "test_files")

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run FeynRules tools test suite",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
parser.add_argument(
    "--keep-files",
    action="store_true",
    help="Keep test-generated files after tests complete (useful for debugging)"
)
# Under pytest, sys.argv holds pytest's own arguments, which this parser
# rejects (SystemExit: 2 at collection): read them only as a script.
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])


# ============================================================================
# Test Functions
# ============================================================================

def test_helper_functions():
    """Test helper functions (_abs_path, _ensure_dir)."""
    print(">> Testing FeynRules helper functions...\n")

    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path
    )

    # Test _abs_path with relative path
    rel_path = "models/test.fr"
    abs_path = tool._abs_path(rel_path)
    assert abs_path is not None
    assert Path(abs_path).is_absolute()
    assert "models/test.fr" in abs_path or "models\\test.fr" in abs_path
    print("[✓] Test 1 passed: _abs_path converts relative to absolute")

    # Test _abs_path with absolute path
    abs_input = "/absolute/path/test.fr"
    abs_output = tool._abs_path(abs_input)
    assert abs_output is not None
    assert Path(abs_output).is_absolute()
    print("[✓] Test 2 passed: _abs_path handles absolute paths")

    # Test _abs_path with None
    none_output = tool._abs_path(None)
    assert none_output is None
    print("[✓] Test 3 passed: _abs_path handles None input")

    # Test _ensure_dir creates directories
    test_dir = Path(base_directory) / "test_temp_dir"
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)

    tool._ensure_dir(str(test_dir))
    assert test_dir.exists()
    assert test_dir.is_dir()
    print("[✓] Test 4 passed: _ensure_dir creates directories")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)

    print("\nAll helper function tests passed! [✓]\n")


def test_missing_parameters():
    """Test that missing required parameters are handled correctly."""
    print(">> Testing parameter validation...\n")

    # Test 1: Model path that doesn't exist
    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path,
        model_path="nonexistent/model.fr",
        output_dir="test_output"
    )

    result = tool._run()
    # Refused before FeynRules runs, naming the missing file
    assert "Model Not Found" in result and "nonexistent/model.fr" in result
    print("[✓] Test 1 passed: Nonexistent model file is handled")

    # Test 2: Invalid wolframscript path
    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path="/nonexistent/wolframscript",
        model_path="models/test.fr",
        output_dir="test_output"
    )

    result = tool._run()
    # Should fail because wolframscript doesn't exist
    assert "error" in result.lower() or "not found" in result.lower() or "executable" in result.lower()
    print("[✓] Test 2 passed: Invalid wolframscript path is handled")

    print("\nAll parameter validation tests passed! [✓]\n")


def test_path_resolution():
    """Test that paths are correctly resolved."""
    print(">> Testing path resolution...\n")

    # Create a tool instance with relative paths
    model_rel = "models/S1_LQ_RR.fr"
    output_rel = "output/UFO_test"

    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path,
        model_path=model_rel,
        output_dir=output_rel
    )

    # Check that _abs_path resolves correctly
    abs_model = tool._abs_path(model_rel)
    abs_output = tool._abs_path(output_rel)

    assert Path(abs_model).is_absolute()
    assert Path(abs_output).is_absolute()
    assert str(base_directory) in abs_model
    assert str(base_directory) in abs_output

    print("[✓] Test 1 passed: Relative paths resolved to absolute within base_directory")

    print("\nAll path resolution tests passed! [✓]\n")


def test_ufo_generation(verbose=False):
    """Test UFO generation from FeynRules model (if Mathematica is available).

    Returns:
        True if test passed, False if test failed, None if test was skipped
    """
    print(">> Testing UFO generation...\n")

    # Check if test model exists
    model_path = "models/S1_LQ_RR.fr"
    model_abs = Path(base_directory) / model_path

    if not model_abs.exists():
        print(f"[⊘] Skipping UFO generation test: model file not found at {model_abs}\n")
        return None

    # Check if wolframscript is available
    import shutil
    if not shutil.which(wolframscript_path):
        print(f"[⊘] Skipping UFO generation test: wolframscript not found at {wolframscript_path}\n")
        return None

    output_dir = "data/UFO_test_output"

    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path,
        model_path=model_path,
        # Required, and never inferred: this fixture happens to name its BSM
        # Lagrangian LBSM, which is a property of this model and not a
        # convention. Omitting it makes the tool report the model's symbols.
        lagrangian="LSM + LBSM",
        output_dir=output_dir,
        timeout_sec=600  # 10 minutes
    )

    if verbose:
        print(f"  Base directory: {base_directory}")
        print(f"  Model path: {model_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  FeynRules path: {feynrules_path}")
        print(f"  wolframscript: {wolframscript_path}\n")

    output_raw = tool._run()

    try:
        output = json.loads(output_raw)
    except Exception:
        print(f"[✗] Failed to parse tool output")
        print(f"  Raw output: {output_raw}\n")
        return False

    if output.get("ok"):
        print("[✓] UFO generation succeeded!")
        print(f"  Output directory: {output.get('output_dir')}")
        print(f"  Files created: {len(output.get('files_created', []))}")

        if verbose and output.get('files_created'):
            print(f"\n  Generated files:")
            for fname in output['files_created'][:10]:  # Show first 10
                print(f"    - {fname}")
            if len(output['files_created']) > 10:
                print(f"    ... and {len(output['files_created']) - 10} more")

        print(f"  Logs: {output.get('logs', {}).get('stdout', 'N/A')}\n")

        # Verify some expected UFO files exist
        output_path = Path(output.get('output_dir'))
        expected_files = ['__init__.py', 'particles.py', 'vertices.py', 'couplings.py']
        found_expected = []
        for fname in expected_files:
            if (output_path / fname).exists():
                found_expected.append(fname)

        if found_expected:
            print(f"  Verified UFO files: {', '.join(found_expected)}")

        return True
    else:
        print("[✗] UFO generation failed.")
        error_info = output.get('error', output.get('reason', 'Unknown error'))
        print(f"  Error: {error_info}")

        if verbose:
            logs = output.get('logs', {})
            if logs.get('stderr'):
                print(f"  Stderr log: {logs['stderr']}")

        print()
        return False


# ============================================================================
# Cleanup Functions
# ============================================================================

def test_check_verdict_parsing():
    """Test consistency-check verdict parsing (no Mathematica required)."""
    print(">> Testing consistency-check verdict parsing...\n")

    from tools.feynrules.lagrangian_checks import (
        parse_check_verdicts, failed_checks, summarise, FAIL, INCONCLUSIVE,
        DEFAULT_CHECKS, OPTIONAL_CHECKS, CHECK_ORDER,
    )

    # Test 0: the checks that run by default are the ones that pass for the
    # Standard Model alone. The other two fail for the SM itself -- the R_xi
    # gauge-Goldstone term and the massless neutrinos -- so they would fail for
    # every model built on it and are opt-in.
    assert set(DEFAULT_CHECKS) | set(OPTIONAL_CHECKS) == set(CHECK_ORDER)
    assert "quadratic_diagonal" in OPTIONAL_CHECKS
    assert "kinetic_norm" in OPTIONAL_CHECKS
    assert "hermiticity" in DEFAULT_CHECKS
    print("[\u2713] Test 0 passed: SM-failing checks are excluded by default")

    def line(scope, name, verdict, detail=""):
        return f"HEPTAPOD-CHECK|scope={scope}|name={name}|verdict={verdict}|detail={detail}"

    # Test 1: a verdict is three-valued; "not pass" is never reported as passed.
    res = parse_check_verdicts("\n".join([
        line("model", "hermiticity", "pass"),
        line("model", "mass_diagonal", "inconclusive"),
    ]))
    by_name = {r["name"]: r for r in res}
    assert by_name["hermiticity"]["passed"] is True
    assert by_name["mass_diagonal"]["verdict"] == INCONCLUSIVE
    assert by_name["mass_diagonal"]["passed"] is False
    print("[\u2713] Test 1 passed: inconclusive is not reported as a pass")

    # Test 2: results the SM baseline shares are marked inherited rather than
    # blamed on the model. CheckDiagonalQuadraticTerms trips on the SM's own
    # gauge-Goldstone mixing term, so without this every SM extension would
    # report a failure it did nothing to cause.
    both = "\n".join([
        line("baseline", "quadratic_diagonal", "fail", "GP W mixing"),
        line("model", "quadratic_diagonal", "fail", "GP W mixing"),
    ])
    res = parse_check_verdicts(both)
    assert res[0]["pre_existing"] is True
    assert failed_checks(res) == []
    assert failed_checks(res, include_pre_existing=True) != []
    print("[\u2713] Test 2 passed: SM-inherited failures are not model failures")

    # Test 3: a failure the SM baseline does not share still surfaces, with a
    # message that says what to do about it.
    res = parse_check_verdicts("\n".join([
        line("baseline", "hermiticity", "pass"),
        line("model", "hermiticity", "fail", "2 residual terms"),
    ]))
    assert [r["name"] for r in failed_checks(res)] == ["hermiticity"]
    assert "HC[" in res[0]["message"]
    print("[\u2713] Test 3 passed: a model-introduced failure is reported")

    # Test 4: CheckKineticTermNormalisation returns Null on failure rather than
    # False, so an undecided verdict becomes a failure only when the kinetic
    # terms are diagonal and the SM baseline does not share the result.
    res = parse_check_verdicts("\n".join([
        line("model", "kinetic_diagonal", "pass"),
        line("model", "kinetic_norm", "inconclusive"),
    ]))
    assert {r["name"]: r["verdict"] for r in res}["kinetic_norm"] == FAIL
    res = parse_check_verdicts("\n".join([
        line("baseline", "kinetic_diagonal", "pass"),
        line("baseline", "kinetic_norm", "inconclusive"),
        line("model", "kinetic_diagonal", "pass"),
        line("model", "kinetic_norm", "inconclusive"),
    ]))
    assert {r["name"]: r["verdict"] for r in res}["kinetic_norm"] == INCONCLUSIVE
    print("[\u2713] Test 4 passed: kinetic-normalisation Null resolved correctly")

    # Test 5: output from a generator predating the scope field still parses.
    res = parse_check_verdicts(
        "HEPTAPOD-CHECK|name=hermiticity|verdict=pass|detail=")
    assert len(res) == 1 and res[0]["passed"] is True
    print("[\u2713] Test 5 passed: scope-less verdict lines still parse")

    # Test 6: the summary counts inherited results separately.
    assert "inherited" in summarise(parse_check_verdicts(both))
    assert summarise([]) == "no checks run"
    print("[\u2713] Test 6 passed: summary distinguishes inherited results")

    print("\nAll verdict parsing tests passed! [\u2713]\n")


def test_lagrangian_is_not_guessed(verbose=False):
    """Test that a model naming its Lagrangian differently fails loudly.

    Regression test for the failure this replaced: the generator hardcoded
    `LSM + LBSM`, so a model calling its Lagrangian anything else produced a UFO
    directory of boilerplate -- no particles.py, no vertices.py, no couplings.py
    -- and still reported success with exit code 0.
    """
    print(">> Testing that the Lagrangian name is never guessed...\n")

    if config_is_incomplete:
        print("[i] Skipped: feynrules_path/wolframscript_path not configured\n")
        return None

    from tools.feynrules.wl_probe import wolframscript_activated
    if not wolframscript_activated(wolframscript_path):
        print("[i] Skipped: wolframscript is not installed or not activated\n")
        return None

    source = FEYNRULES_DIR / "test_files" / "models" / "S1_LQ_RR.fr"
    if not source.exists():
        print(f"[i] Skipped: fixture not found: {source}\n")
        return None

    # Derived at test time rather than committed: all that matters is a model
    # whose Lagrangian is called something the tool has never heard of.
    work = Path(base_directory) / "test_lagrangian_naming"
    work.mkdir(parents=True, exist_ok=True)
    renamed = work / "S1_LQ_RR_renamed.fr"
    renamed.write_text(re.sub(r"\bLBSM\b", "LNP", source.read_text()))

    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path,
        model_path=str(renamed),
        output_dir=str(work / "UFO_out"),
        run_checks=False,
    )
    raw = tool._run()

    # A failing tool returns BaseTool.format_error(), which is formatted text
    # rather than JSON, so the result has to be read both ways.
    try:
        result = json.loads(raw)
        succeeded = result.get("ok") is True
        reported = json.dumps(result)
    except json.JSONDecodeError:
        succeeded = False
        reported = raw

    assert not succeeded, f"a misnamed Lagrangian must not succeed: {reported}"
    assert "LNP" in reported, (
        f"the error should name the symbols the model defines: {reported}")
    print("[\u2713] Test 1 passed: missing Lagrangian fails and lists candidates")

    if not args.keep_files:
        shutil.rmtree(work, ignore_errors=True)

    print("\nAll Lagrangian naming tests passed! [\u2713]\n")
    return True


def cleanup_test_files():
    """Remove all test-generated files and directories."""
    print("\n>> Cleaning up test files...\n")

    # Directories to clean up
    cleanup_dirs = [
        Path(base_directory) / "data",
        Path(base_directory) / "test_output",
    ]

    cleaned = 0

    # Remove directories
    for dir_path in cleanup_dirs:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"[✓] Removed directory: {dir_path.name}")
                cleaned += 1
            except Exception as e:
                print(f"[⚠] Failed to remove {dir_path.name}: {e}")

    if cleaned == 0:
        print("[i] No test files to clean up")
    else:
        print(f"\n[✓] Cleaned up {cleaned} directory(ies)\n")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    import sys

    print(f"Base directory: {base_directory}\n")
    print("=" * 70)

    # Run all tests and track failures
    all_passed = True

    try:
        test_helper_functions()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_helper_functions failed: {e}\n")
        all_passed = False

    try:
        test_missing_parameters()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_missing_parameters failed: {e}\n")
        all_passed = False

    try:
        test_path_resolution()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_path_resolution failed: {e}\n")
        all_passed = False

    try:
        test_check_verdict_parsing()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_check_verdict_parsing failed: {e}\n")
        all_passed = False

    # Needs a real FeynRules + activated Wolfram; returns None when skipped.
    try:
        if test_lagrangian_is_not_guessed(verbose=args.verbose) is False:
            all_passed = False
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_lagrangian_is_not_guessed failed: {e}\n")
        all_passed = False

    ufo_result = test_ufo_generation(verbose=args.verbose)
    # Note: UFO generation test can be skipped if dependencies not available
    # ufo_result: True = passed, False = failed, None = skipped
    if ufo_result is False:
        all_passed = False
    elif ufo_result is None and config_is_incomplete:
        # SKIP, not fail. The live tests need a FeynRules and a Wolfram install
        # that a clone cannot assume; every test that does not need them has
        # already run and reported. Failing here made a fresh checkout red on
        # arrival, which trains people to ignore the suite -- and this one hid
        # three real breakages behind that red.
        missing = [k for k, v in (("feynrules_path", feynrules_path),
                                  ("wolframscript_path", wolframscript_path))
                   if not v]
        print(f"[i] Live UFO tests SKIPPED: no {' or '.join(missing)}.")
        try:
            from config_resolve import describe_sources
            print("    Resolution:")
            print(describe_sources())
        except Exception:
            pass
        print("    Set one of: the environment variable, config.py, or")
        print("    `tb config set heptapod <key> <path>`.\n")

    # Cleanup test files unless --keep-files flag is set
    if not args.keep_files:
        cleanup_test_files()
    else:
        print("\n[i] Keeping test files (--keep-files flag set)\n")

    print("=" * 70)
    if all_passed:
        print("Test suite completed successfully! [✓]")
    else:
        print("Test suite completed with failures! [✗]")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)
