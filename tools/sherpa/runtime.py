"""
# runtime.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Running Sherpa from inside a tool.

Sherpa comes two ways. The ``sherpa-mc`` wheel is a Python module whose
package directory is also a complete install prefix: ``bin/Sherpa``,
``bin/Sherpa-config``, ``include/SHERPA-MC``, ``lib/SHERPA-MC`` and the data
under ``share/SHERPA-MC``. A local build is the same layout under its own
prefix, with a Python module under ``lib/python3.*/site-packages`` when it
was configured with Python support. Everything here locates one or the
other and reads what a run reports: the version, the executable for a dry
initialisation, and the diagnostics in a log. Sherpa writes to the C-level
stdout and stderr, which under the MCP transport is the JSON-RPC channel,
so every call into the wheel runs with those descriptors redirected
(``FdSilence``, shared with the Pythia tools).
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.pythia.runtime import STDERR, STDOUT, FdSilence  # noqa: F401

#: The platforms PyPI has a sherpa-mc wheel for, and the command that
#: installs it into the toolkit environment without ever compiling Sherpa.
WHEEL_PLATFORMS = (
    "Linux x86_64 with glibc 2.34 or newer on Python 3.12 or 3.13, and "
    "macOS 15 arm64 on Python 3.13"
)
WHEEL_INSTALL = (
    "<toolkit environment>/bin/pip install --only-binary=:all: sherpa-mc"
)

#: Where a Sherpa install keeps its Python module.
_MODULE_GLOBS = (
    "lib/python3*/site-packages/Sherpa.py",
    "lib/python3*/site-packages/Sherpa*.so",
    "lib64/python3*/site-packages/Sherpa.py",
    "lib64/python3*/site-packages/Sherpa*.so",
)

#: Sherpa's own messages: a fatal error ends the run with this heading; the
#: run-time warnings carry the WARNING tag.
FATAL = re.compile(r"(Fatal error|Error|Exception)[^\n]*thrown", re.I)
WARNING = re.compile(r"\bWARNING\b")
#: ANSI colour and cursor sequences Sherpa prints even to a file.
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass
class PythonSherpa:
    """Sherpa's Python interface as located: the install it belongs to, the
    directory the worker puts on PYTHONPATH (empty for the wheel), and what
    the manifest calls it."""

    prefix: str
    pythonpath: str = ""
    name: str = "sherpa-mc"

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"backend": self.name}
        if self.pythonpath:
            out["prefix"] = self.prefix
        return out


def find_python_module(prefix: str) -> Optional[Path]:
    """The site-packages directory holding a Sherpa install's Python module,
    or None if the install has none."""
    root = Path(prefix).expanduser()
    for pattern in _MODULE_GLOBS:
        for candidate in sorted(root.glob(pattern)):
            if candidate.is_file():
                return candidate.parent
    return None


def _importable(pythonpath: str) -> Optional[str]:
    """Import Sherpa in a fresh interpreter with pythonpath first; the error
    text if that fails, else None."""
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import Sherpa"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)
    if proc.returncode != 0:
        return (
            clean(proc.stderr).strip().splitlines()[-1:][0]
            if (proc.stderr.strip())
            else f"exit status {proc.returncode}"
        )
    return None


def import_sherpa(sherpa_path: str = "") -> PythonSherpa:
    """
    Locate Sherpa's Python interface for the worker.

    A Sherpa install named by sherpa_path (or SHERPA_PATH) that carries a
    Python module is used, and a module this Python cannot import is an
    error rather than a fallback: whoever set the path meant it. An install
    without a module is the C++ tools' business, and the sherpa-mc wheel is
    used instead.

    Raises:
        ImportError: no interface is available; the message says how to get
            one
    """
    prefix = (sherpa_path or os.environ.get("SHERPA_PATH", "")).strip()
    module_dir = find_python_module(prefix) if prefix else None
    if module_dir is not None:
        problem = _importable(str(module_dir))
        if problem:
            raise ImportError(
                f"the Sherpa module under {module_dir} could not be imported "
                f"by this Python ({sys.version.split()[0]}, {sys.executable}):"
                f" {problem}. It loads only under the Python the build was "
                "configured with. Use SherpaCPPFromRunCardTool for this "
                "install, which needs no Python interface."
            )
        return PythonSherpa(
            prefix=prefix, pythonpath=str(module_dir), name="sherpa"
        )
    wheel = wheel_prefix()
    if wheel:
        return PythonSherpa(prefix=wheel)
    raise ImportError(
        "sherpa-mc is not installed. heptapod does not install it with the "
        "toolkit, since pip compiles Sherpa from source wherever PyPI has no "
        f"wheel ({WHEEL_PLATFORMS}). On those platforms install it into the "
        f"toolkit environment with `{WHEEL_INSTALL}`, which refuses to "
        "compile; otherwise point sherpa_path at a Sherpa install and use "
        "SherpaCPPFromRunCardTool."
    )


def require_sherpa() -> Any:
    """Import the sherpa-mc module, or raise ImportError saying how to get
    one."""
    try:
        import Sherpa

        return Sherpa
    except Exception as e:
        raise ImportError(
            "sherpa-mc is not installed. heptapod does not install it with "
            "the toolkit, since pip compiles Sherpa from source wherever PyPI "
            f"has no wheel ({WHEEL_PLATFORMS}). On those platforms install it "
            f"into the toolkit environment with `{WHEEL_INSTALL}`, which "
            "refuses to compile; otherwise point sherpa_path at a Sherpa "
            "install."
        ) from e


def wheel_prefix() -> Optional[str]:
    """
    The sherpa-mc wheel's package directory, which doubles as an install
    prefix, or None when the wheel is not importable.

    The wheel's ``Sherpa-config`` reports the paths of the machine it was
    built on, so the prefix is taken from the module's location instead.
    """
    try:
        import Sherpa
    except Exception:
        return None
    prefix = Path(Sherpa.__file__).resolve().parent
    if (prefix / "bin" / "Sherpa").is_file() and (
        prefix / "share" / "SHERPA-MC"
    ).is_dir():
        return str(prefix)
    return None


def sherpa_executable(prefix: Optional[str]) -> Optional[str]:
    """``bin/Sherpa`` under the prefix, when it is there and runs."""
    if not prefix:
        return None
    exe = Path(prefix) / "bin" / "Sherpa"
    return str(exe) if exe.is_file() and os.access(exe, os.X_OK) else None


def sherpa_version(prefix: Optional[str] = None) -> Optional[str]:
    """Sherpa's version as "3.0.3", from Sherpa-config or the module."""
    for candidate in (
        Path(prefix) / "bin" / "Sherpa-config" if prefix else None,
    ):
        if candidate is not None and candidate.is_file():
            try:
                out = subprocess.run(
                    [str(candidate), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
    try:
        import Sherpa

        version = getattr(Sherpa, "__version__", None)
        return str(version) if version else None
    except Exception:
        return None


def clean(text: str) -> str:
    """Sherpa's output without its colour and cursor sequences."""
    return ANSI.sub("", text)


def log_tail(path: str, n_lines: int = 15) -> str:
    """
    The lines of a Sherpa log worth showing when a run fails.

    Sherpa announces a failure with a "Fatal error thrown" heading followed
    by the message; those lines come first. Otherwise the last lines of the
    log, with the colour codes removed.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = [clean(line.rstrip("\n")) for line in handle]
    except OSError:
        return ""
    lines = [line for line in lines if line.strip()]
    fatal = [i for i, line in enumerate(lines) if "Fatal error" in line]
    if fatal:
        start = fatal[0]
        return "\n".join(lines[start : start + n_lines])
    return "\n".join(lines[-n_lines:])


def fatal_messages(text: str) -> List[str]:
    """The messages of the fatal errors in a Sherpa log or output."""
    lines = [clean(line) for line in text.splitlines()]
    out: List[str] = []
    for i, line in enumerate(lines):
        if "Fatal error" in line and "thrown" in line:
            message = " ".join(
                part.strip()
                for part in lines[i + 1 : i + 4]
                if part.strip() and "Terminator" not in part
            )
            out.append(message or line.strip())
    return out


def unused_settings(text: str) -> Dict[str, List[str]]:
    """
    The settings Sherpa reports it never read, by source.

    With ``CHECK_SETTINGS: true`` Sherpa ends a run by listing, per source
    (the config file by name, or "command line"), the top-level keys nothing
    asked for -- a misspelt key, or a block whose every entry was misspelt.
    """
    out: Dict[str, List[str]] = {}
    for line in clean(text).splitlines():
        match = re.match(r"\s+-\s+(.+?):\s+(.*)$", line)
        if match and match.group(2).strip():
            keys = match.group(2).split()
            if all(re.match(r"^[A-Za-z0-9_/()@.:-]+$", k) for k in keys):
                out.setdefault(match.group(1).strip(), []).extend(keys)
    return out
