"""
# runtime.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Running Pythia's Python interface from inside a tool.

The interface comes three ways: the pythia8mc wheel, conda's pythia8
package, or the pythia8 extension a Pythia 8 build configured with
--with-python installs under its lib/. import_pythia finds one. Pythia
writes to the C-level stdout and stderr; under the MCP stdio transport
stdout is the JSON-RPC channel, so every call into Pythia has to run with
those file descriptors redirected (FdSilence) -- to /dev/null, or to a log
file whose Pythia diagnostics can be read back (log_tail). The rest is
access to what a run reports: its Info object, its version, and the names
of its event weights.
"""

import ctypes
import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

STDIN, STDOUT, STDERR = 0, 1, 2

# Progress bars go to stderr, on one line, so they never touch the transport.
TQDM_CONFIG = {
    "file": sys.stderr,
    "ncols": 80,
    "leave": True,
    "dynamic_ncols": False,
    "mininterval": 0.1,
    "ascii": False,
}

# Pythia's diagnostics: "Error in <method>: ...", "Abort from ...",
# "Warning in ...".
DIAGNOSTIC = re.compile(r"(Abort from|Error in|Warning in) ")


#: Where a Pythia 8 install keeps its Python extension: lib/ for a source
#: build configured --with-python, site-packages for a conda environment.
_EXTENSION_GLOBS = (
    "lib/pythia8*.so",
    "lib64/pythia8*.so",
    "lib/python3*/site-packages/pythia8*.so",
    "lib64/python3*/site-packages/pythia8*.so",
)

#: The Python modules Pythia comes as, in the order they are tried.
_MODULES = ("pythia8mc", "pythia8")

#: Pythia.nextBatch, which the generation loop needs, arrived in 8.314.
_BATCH_SINCE = "8.314"


@dataclass
class PythonPythia:
    """Pythia's Python interface as imported: the module, its name, the
    install it belongs to when it is a local build's extension, and a note
    when the install's extension was asked for but could not be used."""

    module: Any
    name: str
    prefix: str = ""
    note: str = ""

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"backend": self.name}
        if self.prefix:
            out["prefix"] = self.prefix
        return out


def find_extension(prefix: str) -> Optional[Path]:
    """The Python extension of a Pythia 8 install, or None if it has none
    (a build configured without --with-python)."""
    root = Path(prefix).expanduser()
    for pattern in _EXTENSION_GLOBS:
        for candidate in sorted(root.glob(pattern)):
            if candidate.is_file():
                return candidate
    return None


def _check_batch(module: Any, where: str) -> None:
    if not hasattr(module.Pythia, "nextBatch"):
        raise ImportError(
            f"{where} has no Pythia.nextBatch, the batch interface the "
            f"generation loop uses (Pythia {_BATCH_SINCE} or later, built "
            "with the awkward package available)"
        )


def _import_extension(extension: Path) -> Any:
    """Import a build's pythia8 extension, or raise ImportError saying why
    it cannot be used in this process."""
    # Pythia's bindings register their C++ types with pybind11 once per
    # process, so a second binding cannot load beside the first.
    for name in _MODULES:
        loaded = sys.modules.get(name)
        if loaded is not None and getattr(loaded, "__file__", None) != str(
            extension
        ):
            raise ImportError(
                f"{name} is already loaded in this process from "
                f"{loaded.__file__}, and Pythia's Python interface loads "
                f"only once per process, so {extension} cannot"
            )
    if str(extension.parent) not in sys.path:
        sys.path.insert(0, str(extension.parent))
    try:
        module = importlib.import_module("pythia8")
    except Exception as e:
        raise ImportError(
            f"{extension} could not be imported by this Python "
            f"({sys.version.split()[0]}, {sys.executable}): {e}. The "
            "extension loads only under the Python the build was configured "
            "with (--with-python-include)"
        ) from e
    _check_batch(module, str(extension))
    return module


def import_pythia(pythia_path: str = "") -> PythonPythia:
    """
    Import Pythia's Python interface.

    A Pythia 8 install named by pythia_path (or PYTHIA_PATH) that carries a
    Python extension is used when this process can load it. When it cannot
    -- built for another Python, or another Pythia module is already loaded
    -- the module on sys.path is used instead and the result's ``note``
    says so, since pythia_path is first of all the C++ tools' key and a
    conda Pythia always carries an extension for its own Python. An install
    without an extension is the C++ tools' business. The module on sys.path
    is pythia8mc (the wheel), then pythia8 (conda, or a build whose lib/ is
    on PYTHONPATH). Either way Pythia.nextBatch must exist.

    Raises:
        ImportError: no interface could be imported; the message says how to
            get one
    """
    prefix = (pythia_path or os.environ.get("PYTHIA_PATH", "")).strip()
    extension = find_extension(prefix) if prefix else None
    note = ""
    if extension is not None:
        try:
            module = _import_extension(extension)
        except ImportError as e:
            note = str(e)
        else:
            return PythonPythia(module, "pythia8", prefix=prefix)
    errors = []
    for name in _MODULES:
        try:
            module = importlib.import_module(name)
            _check_batch(module, f"the {name} module at {module.__file__}")
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        if note:
            note = (
                f"{note}; ran the {name} module instead, which may be "
                "another Pythia version. PythiaCPPFromRunCardTool runs the "
                "install itself."
            )
        return PythonPythia(module, name, note=note)
    raise ImportError(
        "no Python interface to Pythia 8 is installed: "
        + "; ".join(([note] if note else []) + errors)
        + ". Install the pythia8mc wheel in the toolkit environment (`pip "
        "install pythia8mc`; wheels for Linux x86_64 and macOS 15 arm64), "
        "or point pythia_path at a Pythia 8 build configured "
        "--with-python for this Python, or use PythiaCPPFromRunCardTool "
        "with any Pythia 8 build."
    )


def require_pythia() -> Any:
    """The Pythia module on sys.path (pythia8mc, else pythia8), or raise
    ImportError."""
    return import_pythia().module


def get_info(pythia: Any) -> Any:
    """Return Pythia's Info object across binding flavours.

    The pythia8mc wheel exposes it only as the method ``infoPython()``;
    ``pythia.info`` (the C++ spelling) raises AttributeError there.
    """
    if hasattr(pythia, "infoPython"):
        return pythia.infoPython()
    info = getattr(pythia, "info", None)
    return info() if callable(info) else info


def weight_names(info: Any) -> List[str]:
    """Names of the event weights, nominal first, in the order nextBatch
    stores them.

    Pythia labels the nominal weight "Weight" and each variation by its
    definition (e.g. "fsr:murfac=0.5"). Empty if the Info object has no names.
    """
    try:
        return [str(name) for name in info.weightNameVector()]
    except Exception:
        return []


def pythia_version(pythia: Any) -> Optional[str]:
    """Pythia's version as "8.317", for the run's provenance; None if
    unknown."""
    try:
        return f"{float(pythia.settings.parm('Pythia:versionNumber')):.3f}"
    except Exception:
        return None


def log_tail(path: str, n_lines: int = 15) -> str:
    """
    Return the lines of a Pythia log worth showing when a run fails.

    Pythia's diagnostics read "Error in <method>: ...", "Abort from ..." or
    "Warning in ...", both as they happen and in the table stat() prints; they
    are preferred over the raw tail, which is usually a statistics banner.
    Empty if the log is missing or blank.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [line.rstrip() for line in fh if line.strip()]
    except OSError:
        return ""
    flagged = []
    for line in lines:
        match = DIAGNOSTIC.search(line)
        if match:
            text = line[match.start() :].rstrip(" |")
            if text not in flagged:
                flagged.append(text)
    return "\n".join((flagged or lines)[-n_lines:])


def flush_std_buffers():
    """Flush Python- and C-level stdout/stderr buffers.

    Native libraries (Pythia8's C++ iostreams, MG5 helpers, ...) buffer their
    writes. Without flushing across an fd swap, that buffered output is emitted
    *after* the fd is restored -- landing on the real stream instead of the
    silenced one, which under toolbase's MCP stdio transport corrupts the
    JSON-RPC channel. ``fflush(NULL)`` flushes every C stdio stream; C++ cout
    is synced with stdio by default, so it is covered too.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


class FdSilence:
    def __init__(self, *fds, target: Optional[str] = None):
        """Context manager to redirect file descriptors (e.g., STDOUT, STDERR).

        Output is discarded by default, or appended to `target` when a path is
        given, so a caller can keep a native library's log without letting it
        reach the real stdout.
        """
        self.fds = fds
        self.target = target
        self.saved = []

    def __enter__(self):
        # Flush pending legitimate output before redirecting so it isn't lost.
        flush_std_buffers()
        # open null once
        if self.target:
            self.null = os.open(
                self.target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
            )
        else:
            self.null = os.open(os.devnull, os.O_WRONLY)
        for fd in self.fds:
            # dup original fd so we can restore later
            saved_fd = os.dup(fd)
            self.saved.append((fd, saved_fd))
            # redirect fd -> /dev/null
            os.dup2(self.null, fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush buffered library output to the still-silenced fd BEFORE
        # restoring, so it is discarded rather than emitted onto the real fd.
        flush_std_buffers()
        # restore each fd
        for fd, saved_fd in self.saved:
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(self.null)
