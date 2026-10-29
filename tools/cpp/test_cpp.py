#!/usr/bin/env python3
"""
# test_cpp.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

"""
The C++ route to Pythia 8 and Sherpa, and the choice of Python interface.

Tests cover:
- toolkit.yaml: pythia8mc carries the platform marker that keeps pip from
  compiling it from source, sherpa-mc is not a dependency at all, and the
  cpp bundles are gated on the install paths and free of generator wheels
- locating an install (configured path or environment variable) and its
  error messages
- build + cache + run against a FAKE install (a stub pythia8-config with
  no flags) -- needs only a C++ compiler
- PythiaCPPFromProgramTool against the fake install: build, run, compiler
  errors, sandbox escapes
- the driver's event stream rebuilt into Pythia.nextBatch's layout, against
  pythia8mc when it is installed
- import_pythia: the module on sys.path, a build's own extension when
  pythia_path names one, and the one-interface-per-process rule
- the real chain, gated on a configured install (PYTHIA_PATH / SHERPA_PATH,
  config.py or toolbase): the CPP run-card tools, every output format, init
  failures, the size guard stopping the driver, and byte-identical datasets
  from the Python and C++ tools when the versions match
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.cpp.build import (
    BuildError,
    build,
    cache_dir,
    configured_path,
    resolve_install,
    run,
)  # noqa: E402

HAVE_CXX = bool(os.environ.get("CXX") or shutil.which("c++"))

try:
    import yaml
except ImportError:
    yaml = None
try:
    from packaging.markers import Marker
except ImportError:
    Marker = None


def _resolved(key):
    try:
        from config_resolve import resolve_path

        return resolve_path(key)
    except Exception:
        return None


PYTHIA_PATH = _resolved("pythia_path")
SHERPA_PATH = _resolved("sherpa_path")


def _fresh(code: str, **env) -> subprocess.CompletedProcess:
    """Run code in a new interpreter: Pythia's Python interface loads once
    per process, so every import_pythia case needs a process of its own."""
    environment = dict(os.environ)
    environment.pop("PYTHIA_PATH", None)
    environment.update(env)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n" + code,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
    )


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="heptapod-cpp-"))
        self._env = dict(os.environ)
        os.environ["HEPTAPOD_CPP_CACHE"] = str(self.tmp / "cache")
        for var in ("PYTHIA_PATH", "SHERPA_PATH"):
            os.environ.pop(var, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_pythia(self) -> Path:
        """An install prefix whose pythia8-config reports no flags at all."""
        prefix = self.tmp / "fake-pythia"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "share" / "xmldoc").mkdir(parents=True)
        script = prefix / "bin" / "pythia8-config"
        script.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            f"  --prefix) echo {prefix} ;;\n"
            f"  --xmldoc) echo {prefix}/share/xmldoc ;;\n"
            "  --version) echo 8.999 ;;\n"
            "  *) echo ;;\n"
            "esac\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        return prefix


class TestToolkitDeps(unittest.TestCase):
    """What tb install may and may not pull in."""

    @unittest.skipUnless(yaml and Marker, "needs pyyaml and packaging")
    def test_pythia_wheel_is_marker_guarded(self):
        manifest = yaml.safe_load((REPO_ROOT / "toolkit.yaml").read_text())
        bundles = manifest["bundles"]
        specs = set()
        for name in ("event_gen", "pythia"):
            for dep in bundles[name]["deps"]:
                pkg = dep.split(">")[0].split("=")[0].split(";")[0].strip()
                if pkg == "pythia8mc":
                    self.assertIn(
                        ";",
                        dep,
                        f"{name}: pythia8mc has no marker; pip would build it "
                        "from source wherever PyPI has no wheel",
                    )
                    specs.add(dep)
        self.assertEqual(
            len(specs),
            1,
            "pythia8mc spelled differently across bundles; toolbase dedups "
            "by exact string",
        )
        pythia = Marker(next(iter(specs)).split(";", 1)[1])

        def env(plat, machine, release, py):
            return {
                "sys_platform": plat,
                "platform_machine": machine,
                "platform_release": release,
                "python_version": py,
            }

        # (environment, pythia8mc has a wheel), per PyPI 2026-10
        cases = [
            (env("darwin", "arm64", "24.1.0", "3.12"), True),
            (env("darwin", "arm64", "24.1.0", "3.13"), True),
            (env("darwin", "arm64", "23.6.0", "3.13"), False),  # macOS 14
            (env("darwin", "x86_64", "24.1.0", "3.13"), False),
            (env("linux", "x86_64", "6.8.0", "3.12"), True),
            (env("linux", "x86_64", "6.8.0", "3.14"), True),
            (env("linux", "aarch64", "6.8.0", "3.12"), False),
        ]
        for e, want in cases:
            self.assertEqual(pythia.evaluate(e), want, e)

    @unittest.skipUnless(yaml, "needs pyyaml")
    def test_sherpa_wheel_is_never_a_dependency(self):
        # Its Linux wheels need glibc 2.34, which no marker can test; on an
        # older glibc pip would compile Sherpa from source.
        manifest = yaml.safe_load((REPO_ROOT / "toolkit.yaml").read_text())
        for name, bundle in manifest["bundles"].items():
            for dep in bundle.get("deps") or []:
                self.assertNotIn("sherpa-mc", dep, name)
                self.assertNotIn("sherpa_mc", dep, name)

    @unittest.skipUnless(yaml, "needs pyyaml")
    def test_cpp_bundles_gated_and_wheel_free(self):
        manifest = yaml.safe_load((REPO_ROOT / "toolkit.yaml").read_text())
        for name, key in (
            ("pythia_cpp", "pythia_path"),
            ("sherpa_cpp", "sherpa_path"),
        ):
            bundle = manifest["bundles"][name]
            self.assertEqual(bundle["requires"], [key])
            self.assertFalse(
                [
                    d
                    for d in bundle["deps"]
                    if "pythia8mc" in d or "sherpa-mc" in d
                ]
            )
        served = {}
        for tool in manifest["tools"]:
            bundles = tool.get("bundle")
            bundles = [bundles] if isinstance(bundles, str) else bundles or []
            served[tool["name"]] = set(bundles)
        self.assertEqual(served["PythiaCPPFromRunCardTool"], {"pythia_cpp"})
        self.assertEqual(served["PythiaCPPFromProgramTool"], {"pythia_cpp"})
        self.assertNotIn("pythia_cpp", served["PythiaFromRunCardTool"])
        self.assertIn("sherpa_cpp", served["SherpaCPPFromRunCardTool"])
        self.assertIn("sherpa_cpp", served["SherpaCPPFromProgramTool"])
        self.assertNotIn("sherpa_cpp", served["SherpaFromRunCardTool"])


class TestLocateInstall(_TempDirCase):
    def test_unconfigured_path_explains(self):
        self.assertEqual(configured_path("pythia", ""), "")
        with self.assertRaisesRegex(
            BuildError, "pythia_path is not configured"
        ):
            resolve_install("pythia", "")

    def test_broken_path_is_an_error(self):
        with self.assertRaisesRegex(BuildError, "no executable Sherpa-config"):
            resolve_install("sherpa", str(self.tmp))

    def test_environment_variable_counts_as_configured(self):
        os.environ["PYTHIA_PATH"] = str(self.fake_pythia())
        self.assertEqual(
            configured_path("pythia", ""), os.environ["PYTHIA_PATH"]
        )
        install = resolve_install("pythia", configured_path("pythia", None))
        self.assertEqual(install.version, "8.999")

    def test_path_may_name_prefix_bin_or_script(self):
        prefix = self.fake_pythia()
        for path in (
            prefix,
            prefix / "bin",
            prefix / "bin" / "pythia8-config",
        ):
            self.assertEqual(
                resolve_install("pythia", str(path)).prefix, str(prefix)
            )


@unittest.skipUnless(HAVE_CXX, "needs a C++ compiler")
class TestBuildAndRun(_TempDirCase):
    def test_build_caches_and_rebuilds_on_change(self):
        install = resolve_install("pythia", str(self.fake_pythia()))
        src = self.tmp / "hello.cc"
        src.write_text(
            "#include <cstdio>\n#include <cstdlib>\nint main(){"
            'std::printf("%s\\n", std::getenv("PYTHIA8DATA")); return 3;}\n'
        )
        out = self.tmp / "work" / "hello"
        first = build(install, [str(src)], str(out))
        self.assertFalse(first.cached)
        self.assertTrue(build(install, [str(src)], str(out)).cached)
        src.write_text(src.read_text().replace("return 3", "return 4"))
        self.assertFalse(build(install, [str(src)], str(out)).cached)

        r = run(install, [str(out)], cwd=str(out.parent), log_prefix=str(out))
        self.assertEqual(r.returncode, 4)
        # The install's runtime environment reaches the program; its output
        # goes to the log files, never this process's stdout.
        self.assertIn("share/xmldoc", r.tail("stdout"))
        self.assertTrue(os.path.isfile(r.stderr_log))

    def test_compiler_error_is_reported(self):
        install = resolve_install("pythia", str(self.fake_pythia()))
        src = self.tmp / "broken.cc"
        src.write_text("int main() { return undeclared_thing; }\n")
        with self.assertRaisesRegex(BuildError, "undeclared_thing"):
            build(install, [str(src)], str(self.tmp / "broken"))
        self.assertFalse((self.tmp / "broken").exists())

    def test_cache_dir_override(self):
        self.assertEqual(cache_dir(), self.tmp / "cache")


@unittest.skipUnless(HAVE_CXX, "needs a C++ compiler")
class TestProgramTool(_TempDirCase):
    def setUp(self):
        super().setUp()
        try:
            from tools.cpp.program import PythiaCPPFromProgramTool
        except ImportError as e:
            self.skipTest(f"orchestral not importable: {e}")
        self.base = self.tmp / "sandbox"
        (self.base / "src").mkdir(parents=True)
        self.tool = PythiaCPPFromProgramTool(
            base_directory=str(self.base), pythia_path=str(self.fake_pythia())
        )

    def test_build_and_run(self):
        (self.base / "src" / "main.cc").write_text(
            "#include <fstream>\n#include <cstdlib>\n"
            "int main(int argc, char** argv){ "
            'std::ofstream("out.txt") << argv[1]; return 0; }\n'
        )
        res = json.loads(
            self.tool.execute(
                source_path="src/main.cc", work_dir="run", args=["hello"]
            )
        )
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(res["run"]["returncode"], 0)
        self.assertIn("run/out.txt", res["run"]["changed_files"])
        self.assertEqual((self.base / "run" / "out.txt").read_text(), "hello")
        again = json.loads(
            self.tool.execute(
                source_path="src/main.cc", work_dir="run", args=["bye"]
            )
        )
        self.assertTrue(again["build"]["cached"])
        self.assertIn("run/out.txt", again["run"]["changed_files"])

    def test_nonzero_exit_is_an_error_status(self):
        (self.base / "src" / "fail.cc").write_text("int main(){ return 7; }\n")
        res = json.loads(
            self.tool.execute(source_path="src/fail.cc", work_dir="run")
        )
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["run"]["returncode"], 7)

    def test_build_failure(self):
        (self.base / "src" / "bad.cc").write_text(
            "int main(){ return nope; }\n"
        )
        out = self.tool.execute(source_path="src/bad.cc", work_dir="run")
        self.assertIn("Build Failed", out)
        self.assertIn("nope", out)

    def test_sandbox(self):
        self.assertIn(
            "Access Denied",
            self.tool.execute(source_path="../outside.cc", work_dir="run"),
        )
        (self.base / "src" / "ok.cc").write_text("int main(){}\n")
        self.assertIn(
            "Access Denied",
            self.tool.execute(source_path="src/ok.cc", work_dir="../escape"),
        )

    def test_no_install_configured(self):
        from tools.cpp.program import PythiaCPPFromProgramTool

        (self.base / "src" / "ok.cc").write_text("int main(){}\n")
        out = PythiaCPPFromProgramTool(base_directory=str(self.base)).execute(
            source_path="src/ok.cc", work_dir="run"
        )
        self.assertIn("Setup Error", out)
        self.assertIn("pythia_path is not configured", out)


class _GeneratorCase(_TempDirCase):
    def setUp(self):
        super().setUp()
        self.base = self.tmp / "sandbox"
        (self.base / "cards").mkdir(parents=True)


# e+e- -> Z -> hadrons at the Z pole: no PDFs or MPI, so Pythia initialises
# in a second, with shower weight variations so the weights path is exercised.
_EE_CARD = """Beams:idA = 11
Beams:idB = -11
Beams:eCM = 91.2
WeakSingleBoson:ffbar2gmZ = on
23:onMode = off
23:onIfAny = 1 2 3 4 5
UncertaintyBands:doVariations = on
UncertaintyBands:List = {fsrup fsr:muRfac=0.5}
"""


class TestDriverBatchLayout(_TempDirCase):
    """batch_from_events rebuilds nextBatch's layout from the driver's rows."""

    def _events(self, full):
        from tools.pythia.driver import FULL_ROW, KINEMATIC_ROW

        layout = FULL_ROW if full else KINEMATIC_ROW

        def row(id_, status, px, py, pz, e, m, **extra):
            values = {
                "id": id_,
                "status": status,
                "px": px,
                "py": py,
                "pz": pz,
                "e": e,
                "m": m,
                "mother1": 0,
                "mother2": 0,
                "daughter1": 0,
                "daughter2": 0,
                "col": 0,
                "acol": 0,
                "scale": 0.0,
                "pol": 9.0,
                "tau": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "vt": 0.0,
            }
            values.update(extra)
            return [values[name] for name in layout]

        info = [
            11,
            -11,
            1.0,
            1.0,
            0.0,
            0.0,
            0.13,
            0.0078,
            8317.4,
            8317.4,
            91.2,
            8317.4,
            0.0,
            0.0,
            0.0,
        ]
        return [
            {
                "p": [
                    row(
                        23,
                        -22,
                        0.0,
                        0.0,
                        0.0,
                        91.2,
                        91.2,
                        daughter1=1,
                        daughter2=2,
                    ),
                    row(1, 1, 0.0, 0.0, 45.6, 45.6, 0.0, mother1=0, vx=0.1),
                    row(-1, 1, 0.0, 0.0, -45.6, 45.6, 0.0, mother1=0),
                ],
                "i": info,
                "w": [1.0, 0.9, 1.1],
            },
            {
                "p": [row(211, 1, 1.0, 2.0, 3.0, 3.75, 0.13957)],
                "i": info,
                "w": [1.0, 0.9, 1.1],
            },
        ]

    def test_full_rows_reduce_like_a_pythia_batch(self):
        import awkward as ak
        from tools import event_io
        from tools.pythia.driver import batch_from_events

        batch = batch_from_events(self._events(full=True))
        self.assertEqual(batch.fields, ["prt", "info"])
        self.assertEqual(
            batch.prt.fields,
            [
                "id",
                "status",
                "mother1",
                "mother2",
                "daughter1",
                "daughter2",
                "col",
                "acol",
                "p",
                "m",
                "scale",
                "pol",
                "vProd",
                "tau",
            ],
        )
        self.assertIn("weights", batch.info.fields)
        self.assertEqual(ak.to_list(ak.num(batch.prt)), [3, 1])
        full = event_io.apply_store(batch, store="full", precision="f64")
        self.assertEqual(ak.to_list(ak.num(full.prt)), [3, 1])
        self.assertIn("vx", full.prt.fields)
        self.assertEqual(ak.to_list(full.prt.vx[0]), [0.0, 0.1, 0.0])
        self.assertEqual(ak.to_list(full.weight), [1.0, 1.0])
        self.assertEqual(ak.to_list(full.weights[0]), [1.0, 0.9, 1.1])
        self.assertEqual(ak.to_list(full.info_id1), [11, 11])
        finals = event_io.apply_store(
            batch, store="finals+kinematics", precision="f32"
        )
        self.assertEqual(ak.to_list(ak.num(finals.prt)), [2, 1])
        self.assertEqual(finals.prt.fields, list(event_io._KINEMATIC_FIELDS))
        self.assertTrue(event_io.has_genealogy(batch))

    def test_kinematic_rows_carry_only_kinematics(self):
        import awkward as ak
        from tools import event_io
        from tools.pythia.driver import batch_from_events

        batch = batch_from_events(self._events(full=False))
        self.assertEqual(batch.prt.fields, ["id", "status", "p", "m"])
        reduced = event_io.apply_store(batch, store="finals+kinematics")
        self.assertEqual(ak.to_list(reduced.prt.id), [[1, -1], [211]])
        self.assertEqual(ak.to_list(reduced.prt.e[1]), [3.75])

    def test_empty_stream_gives_an_empty_batch(self):
        from tools.pythia.driver import batch_from_events

        self.assertEqual(len(batch_from_events([])), 0)

    def test_matches_nextbatch_layout(self):
        """The same events through pythia8mc and through the driver's row
        format reduce to identical arrays."""
        try:
            import pythia8mc as p8
        except Exception:
            self.skipTest("pythia8mc not installed")
        import awkward as ak
        from tools import event_io
        from tools.pythia.driver import (
            FULL_ROW,
            INFO_FIELDS,
            batch_from_events,
        )
        from tools.pythia.runtime import STDERR, STDOUT, FdSilence

        with FdSilence(STDOUT, STDERR):
            py = p8.Pythia("", printBanner=False)
            for line in _EE_CARD.splitlines() + [
                "Print:quiet = on",
                "Random:setSeed = on",
                "Random:seed = 7",
            ]:
                py.readString(line)
            self.assertTrue(py.init())
            batch = py.nextBatch(3, "skip")
        # Write the batch the way the driver writes an event line.
        events = []
        for ev in ak.to_list(batch):
            rows = []
            for prt in ev["prt"]:
                v = prt["vProd"] or {"px": 0.0, "py": 0.0, "pz": 0.0, "e": 0.0}
                flat = {
                    **{
                        k: prt[k]
                        for k in (
                            "id",
                            "status",
                            "mother1",
                            "mother2",
                            "daughter1",
                            "daughter2",
                            "col",
                            "acol",
                            "m",
                            "scale",
                            "pol",
                            "tau",
                        )
                    },
                    **prt["p"],
                    "vx": v["px"],
                    "vy": v["py"],
                    "vz": v["pz"],
                    "vt": v["e"],
                }
                rows.append([flat[name] for name in FULL_ROW])
            events.append(
                {
                    "p": rows,
                    "i": [ev["info"][name] for name in INFO_FIELDS],
                    "w": ev["info"]["weights"],
                }
            )
        rebuilt = batch_from_events(events)
        for store in ("full", "finals", "finals+kinematics"):
            a = event_io.apply_store(batch, store=store, precision="f64")
            b = event_io.apply_store(rebuilt, store=store, precision="f64")
            self.assertEqual(a.fields, b.fields, store)
            self.assertEqual(ak.to_list(a), ak.to_list(b), store)


class TestImportPythia(_TempDirCase):
    """Which Python interface PythiaFromRunCardTool runs."""

    def _extension(self):
        from tools.pythia.runtime import find_extension

        return find_extension(PYTHIA_PATH) if PYTHIA_PATH else None

    def test_install_without_extension_is_left_alone(self):
        from tools.pythia.runtime import find_extension

        fake = str(self.fake_pythia())
        self.assertIsNone(find_extension(fake))
        # A configured install without an extension falls through to the
        # module on sys.path, or to the error that says how to get one.
        proc = _fresh(
            "from tools.pythia.runtime import import_pythia\n"
            "try:\n"
            f"    print(import_pythia({fake!r}).name)\n"
            "except ImportError as e:\n"
            "    print('ImportError:', e)\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout.strip()
        self.assertTrue(
            out in ("pythia8mc", "pythia8")
            or "no Python interface to Pythia 8" in out,
            out,
        )

    def test_module_on_sys_path(self):
        proc = _fresh(
            "from tools.pythia.runtime import import_pythia\n"
            "p = import_pythia()\n"
            "print(p.name, p.describe())\n"
        )
        if proc.returncode != 0 and "no Python interface" in (
            proc.stderr + proc.stdout
        ):
            self.skipTest("no Pythia module installed")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        name = proc.stdout.split()[0]
        self.assertIn(name, ("pythia8mc", "pythia8"))
        self.assertNotIn("prefix", proc.stdout)

    def test_extension_of_a_configured_build(self):
        extension = self._extension()
        if extension is None:
            self.skipTest("pythia_path has no Python extension")
        proc = _fresh(
            "from tools.pythia.runtime import import_pythia\n"
            f"p = import_pythia({PYTHIA_PATH!r})\n"
            "print(p.name, p.module.__file__, p.describe().get('prefix'))\n"
            "print(p.note)\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        name, path, prefix = proc.stdout.splitlines()[0].split()
        if name != "pythia8":
            # Built for another Python: the module on sys.path ran instead,
            # and the note says why.
            self.assertIn("could not be imported", proc.stdout)
            self.assertIn("ran the", proc.stdout)
            self.skipTest("the extension was built for another Python")
        self.assertEqual(path, str(extension))
        self.assertEqual(prefix, PYTHIA_PATH)

    def test_only_one_interface_per_process(self):
        # With the wheel already loaded, the install's extension cannot be;
        # the wheel runs and the note says so.
        extension = self._extension()
        if extension is None:
            self.skipTest("pythia_path has no Python extension")
        proc = _fresh(
            "try:\n"
            "    import pythia8mc\n"
            "except Exception:\n"
            "    raise SystemExit(3)\n"
            "from tools.pythia.runtime import import_pythia\n"
            f"p = import_pythia({PYTHIA_PATH!r})\n"
            "print(p.name)\n"
            "print(p.note)\n"
        )
        if proc.returncode == 3:
            self.skipTest("pythia8mc not installed")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.splitlines()[0], "pythia8mc")
        self.assertIn("loads only once per process", proc.stdout)
        self.assertIn("PythiaCPPFromRunCardTool", proc.stdout)

    def test_unloadable_extension_falls_back_with_a_note(self):
        prefix = self.fake_pythia()
        (prefix / "lib").mkdir()
        (prefix / "lib" / "pythia8.so").write_bytes(b"not a module")
        proc = _fresh(
            "from tools.pythia.runtime import import_pythia\n"
            "try:\n"
            f"    p = import_pythia({str(prefix)!r})\n"
            "except ImportError as e:\n"
            "    print('ImportError:', e)\n"
            "else:\n"
            "    print(p.name)\n"
            "    print(p.note)\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        if proc.stdout.startswith("ImportError:"):
            self.assertIn("could not be imported", proc.stdout)
            self.skipTest("no Pythia module to fall back to")
        self.assertIn(proc.stdout.splitlines()[0], ("pythia8mc", "pythia8"))
        self.assertIn("could not be imported", proc.stdout)
        self.assertIn("ran the", proc.stdout)


@unittest.skipUnless(
    PYTHIA_PATH and HAVE_CXX,
    "set pythia_path (or PYTHIA_PATH) to a Pythia 8 install",
)
class TestPythiaCPP(_GeneratorCase):
    def setUp(self):
        super().setUp()
        (self.base / "cards" / "ee.cmnd").write_text(_EE_CARD)
        from tools.pythia import (
            PythiaCPPFromRunCardTool,
            PythiaFromRunCardTool,
        )

        self.CPP = PythiaCPPFromRunCardTool
        self.Python = PythiaFromRunCardTool

    def generate(self, data_dir, Tool=None, n_events=5, **kw):
        tool = (Tool or self.CPP)(
            base_directory=str(self.base), pythia_path=PYTHIA_PATH
        )
        kw.setdefault("cmnd_path", "cards/ee.cmnd")
        res = json.loads(
            tool.execute(data_dir=data_dir, n_events=n_events, seed=17, **kw)
        )
        return res, self.base / data_dir

    def test_cpp_run(self):
        res, out = self.generate(
            "cpp", output_format="jsonl", select_pids=[211, 321]
        )
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(res["backend"], "cpp")
        self.assertEqual(res["accepted"], 5)
        self.assertGreater(res["xsec"]["sigmaGen_mb"], 0)
        self.assertTrue(res["weighted"] is False or "weight_range" in res)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["inputs"]["generator"]["backend"], "cpp")
        self.assertEqual(
            manifest["inputs"]["generator"]["prefix"],
            resolve_install("pythia", PYTHIA_PATH).prefix,
        )
        self.assertRegex(manifest["pythia_version"] or "", r"^8\.\d{3}$")
        self.assertEqual([b["id"] for b in manifest["beams"]], [11, -11])
        self.assertEqual(manifest["outputs"]["weights"]["names"][0], "Weight")
        rows = [
            json.loads(line)
            for line in (out / "events.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 5)
        self.assertTrue(
            all(
                abs(p["id"]) in (211, 321)
                for r in rows
                for p in r["data"]["particles"]
            )
        )
        self.assertEqual(len(rows[0]["weights"]), 3)
        log = (out / "pythia.log").read_text()
        self.assertIn("PYTHIA Event and Cross Section Statistics", log)

    def test_full_record_to_root_and_hepmc3(self):
        from tools import event_io

        res, out = self.generate("root", store="full")
        self.assertEqual(res["status"], "ok", res)
        self.assertTrue((out / "events.root").is_file())
        self.assertEqual(event_io.count_events(str(out / "events.root")), 5)
        try:
            import pyhepmc  # noqa: F401
        except Exception:
            self.skipTest("pyhepmc not installed")
        res, out = self.generate("hepmc", store="full", output_format="hepmc3")
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(res["hepmc3_record"], "full")
        self.assertEqual(event_io.count_events(str(out / "events.hepmc")), 5)

    def test_init_failure_is_reported_from_the_log(self):
        (self.base / "cards" / "bad.cmnd").write_text(
            "Beams:idA = 11\nBeams:idB = -11\nBeams:eCM = 0.001\n"
            "WeakSingleBoson:ffbar2gmZ = on\n"
        )
        tool = self.CPP(base_directory=str(self.base), pythia_path=PYTHIA_PATH)
        res = tool.execute(
            data_dir="bad", cmnd_path="cards/bad.cmnd", n_events=1
        )
        self.assertTrue(res.startswith("Error: Pythia Init Failed"), res)
        self.assertIn(
            "too low energy", res
        )  # Pythia's own diagnostic, read back from the log
        self.assertTrue((self.base / "bad" / "pythia.log").is_file())

    def test_size_guard_stops_the_driver(self):
        res, out = self.generate(
            "trunc",
            n_events=200,
            output_format="jsonl",
            store="full",
            chunk_size=5,
            max_output_gb=1e-6,
        )
        self.assertEqual(res["status"], "ok", res)
        self.assertTrue(res.get("truncated"), res)
        self.assertLess(res["accepted"], 200)
        self.assertGreater(res["accepted"], 0)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertIn("sigmaGen_mb", manifest["outputs"]["xsec"])
        self.assertEqual(
            sum(
                1
                for line in (out / "events.jsonl").read_text().splitlines()
                if line
            ),
            res["accepted"],
        )

    def test_matches_the_python_interface(self):
        """The C++ tool and the Python tool write identical datasets from
        one card when they run the same Pythia version. The card carries no
        variations: a build's own extension stores the weight groups where
        the wheel and the driver store every weight, so with variations the
        two layouts differ by design (see the test after this one)."""
        (self.base / "cards" / "plain.cmnd").write_text(
            _EE_CARD.split("UncertaintyBands")[0]
        )
        local = resolve_install("pythia", PYTHIA_PATH).version
        # The Pythia the Python tool will run, in a process of its own: with
        # pythia_path set it takes the install's extension when that loads,
        # which cannot happen beside a wheel this process may have imported.
        probe = _fresh(
            "from tools.pythia.runtime import import_pythia, pythia_version\n"
            f"p = import_pythia({PYTHIA_PATH!r})\n"
            "print(p.name, pythia_version(p.module.Pythia('', False)))\n"
        )
        if probe.returncode != 0:
            self.skipTest("no Python interface: " + probe.stderr[-300:])
        name, version = probe.stdout.strip().splitlines()[-1].split()
        if version != local:
            self.skipTest(
                f"the Python tool would run {name} {version}, the C++ tool "
                f"{local}; events would differ"
            )
        for store in ("finals+kinematics", "full"):
            cpp, c_out = self.generate(
                f"c_{store}",
                store=store,
                output_format="jsonl",
                cmnd_path="cards/plain.cmnd",
            )
            proc = _fresh(
                "import json\n"
                "from tools.pythia import PythiaFromRunCardTool\n"
                "tool = PythiaFromRunCardTool("
                f"base_directory={str(self.base)!r},"
                f" pythia_path={PYTHIA_PATH!r})\n"
                "print(tool.execute(data_dir="
                f"{'p_' + store!r}, cmnd_path='cards/plain.cmnd', n_events=5,"
                f" seed=17, store={store!r}, output_format='jsonl'))\n",
                HEPTAPOD_CPP_CACHE=os.environ["HEPTAPOD_CPP_CACHE"],
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            raw = proc.stdout.strip().splitlines()[-1]
            if store == "full" and "wrong particles" in proc.stdout:
                # The extension a Pythia 8.317 build makes misplaces the
                # vertices; the tool refuses a store that keeps them.
                self.assertIn("Generation Error", proc.stdout)
                self.assertIn("PythiaCPPFromRunCardTool", proc.stdout)
                continue
            python = json.loads(raw)
            p_out = self.base / f"p_{store}"
            self.assertEqual(cpp["status"], "ok", cpp)
            self.assertEqual(python["status"], "ok", python)
            self.assertEqual(
                (c_out / "events.jsonl").read_bytes(),
                (p_out / "events.jsonl").read_bytes(),
            )
            self.assertEqual(cpp["summary"], python["summary"])
            self.assertEqual(cpp["xsec"], python["xsec"])

    def test_extension_names_its_weights(self):
        """Through a build's own extension, nextBatch stores the weight
        groups; the names stored beside them must be the groups' names."""
        from tools.pythia.runtime import find_extension

        if find_extension(PYTHIA_PATH) is None:
            self.skipTest("pythia_path has no Python extension")
        proc = _fresh(
            "import json\n"
            "from tools.pythia import PythiaFromRunCardTool\n"
            "tool = PythiaFromRunCardTool("
            f"base_directory={str(self.base)!r},"
            f" pythia_path={PYTHIA_PATH!r})\n"
            "print(tool.execute(data_dir='ext', cmnd_path='cards/ee.cmnd',"
            " n_events=3, seed=17, output_format='jsonl'))\n",
            HEPTAPOD_CPP_CACHE=os.environ["HEPTAPOD_CPP_CACHE"],
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        res = json.loads(proc.stdout.strip().splitlines()[-1])
        if res["backend"] != "pythia8":
            self.skipTest("the extension was built for another Python")
        manifest = json.loads(
            (self.base / "ext" / "manifest.json").read_text()
        )
        names = manifest["outputs"]["weights"]["names"]
        rows = [
            json.loads(line)
            for line in (self.base / "ext" / "events.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(names[0], "Weight")
        self.assertTrue(all(len(r["weights"]) == len(names) for r in rows))
        self.assertIn("fsrup", names)


@unittest.skipUnless(
    SHERPA_PATH and HAVE_CXX,
    "set sherpa_path (or SHERPA_PATH) to a Sherpa install",
)
class TestSherpaCPP(_GeneratorCase):
    def test_cpp_run(self):
        # e+e- -> jets at the Z pole: no PDFs, initializes in seconds.
        (self.base / "cards" / "ee.yaml").write_text(
            "BEAMS: [11, -11]\nBEAM_ENERGIES: 45.6\nPDF_LIBRARY: None\n"
            "PROCESSES:\n- 11 -11 -> 93 93: {}\n"
        )
        from tools.sherpa import SherpaCPPFromRunCardTool

        tool = SherpaCPPFromRunCardTool(
            base_directory=str(self.base), sherpa_path=SHERPA_PATH
        )
        res = json.loads(
            tool.execute(
                data_dir="ee",
                card_path="cards/ee.yaml",
                n_events=20,
                seed=3,
                store="finals",
            )
        )
        self.assertEqual(res["status"], "ok", res)
        out = self.base / "ee"
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["inputs"]["generator"]["backend"], "cpp")
        self.assertEqual(
            manifest["sherpa_version"],
            manifest["inputs"]["generator"]["version"],
        )
        import awkward as ak
        from tools import event_io

        events = event_io.load_events(str(out / "events.root"))
        self.assertEqual(len(events), 20)
        ids = ak.to_numpy(ak.flatten(events.prt.id))
        self.assertTrue((ids < 0).any(), "antiparticles must keep their sign")
        # Final states balance the 91.2 GeV beams.
        for e_sum, pz_sum in zip(
            ak.to_list(ak.sum(events.prt.e, axis=1)),
            ak.to_list(ak.sum(events.prt.pz, axis=1)),
        ):
            self.assertAlmostEqual(e_sum, 91.2, places=2)
            self.assertAlmostEqual(pz_sum, 0.0, places=2)
        # The weights are cross sections in pb: positive, and on average the
        # total cross section to the precision 20 events allow.
        weights = ak.to_numpy(events.weight)
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(
            float(weights.mean()),
            res["xsec"]["sigmaGen_pb"],
            delta=0.3 * res["xsec"]["sigmaGen_pb"],
        )


if __name__ == "__main__":
    # test_runner.py passes --keep-files to every suite; unittest has no such
    # flag.
    sys.argv = [a for a in sys.argv if a != "--keep-files"]
    unittest.main()
