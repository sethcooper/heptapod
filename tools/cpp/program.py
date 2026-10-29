"""
# program.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Compile and run a C++ main program against Pythia 8 or Sherpa:
PythiaCPPFromProgramTool and SherpaCPPFromProgramTool.

The run-card tools cover what a card can say. Anything past that --
UserHooks, a custom veto, reading Pythia's event record mid-shower, a
Sherpa program calling the ATOOLS API -- needs a main program. These tools
build one from the sandbox with the install's own flags (the same build
path heptapod's drivers use) and run it there.
"""

import json
import os
from typing import ClassVar, Dict, List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField
from pydantic import Field

from tools.cpp.build import (
    BuildError,
    build,
    configured_path,
    resolve_install,
    run,
)

_SOURCE_SUFFIXES = (".cc", ".cpp", ".cxx", ".C")


def _snapshot(root: str) -> Dict[str, int]:
    """mtime of every file under root, to report what a run wrote."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            path = os.path.join(d, f)
            try:
                out[path] = os.stat(path).st_mtime_ns
            except OSError:
                pass
    return out


def _runtime_list(description: str) -> List[str]:
    # Field(default_factory=list) rather than RuntimeField(default=None): a
    # non-None default is what marks a runtime field optional in the schema.
    return Field(
        default_factory=list,
        json_schema_extra={"runtime": True},
        description=description,
    )


class _CPPProgramTool(BaseTool):
    generator: ClassVar[str] = ""  # "pythia" or "sherpa", set by subclasses

    # --------------------------- Runtime fields --------------------------- #
    source_path: str = RuntimeField(
        description="Relative path to the C++ source holding main() "
        "(.cc/.cpp/.cxx/.C)"
    )
    work_dir: str = RuntimeField(
        description="Relative directory to build into and run in; created if "
        "missing. "
        "The program's relative output paths land here."
    )
    extra_sources: List[str] = _runtime_list(
        "Optional further relative .cc sources to compile and link in"
    )
    args: List[str] = _runtime_list(
        "Optional command-line arguments passed to the program"
    )
    extra_cxxflags: List[str] = _runtime_list(
        "Optional extra compiler flags, e.g. ['-O3', '-I/opt/fastjet/include']"
    )
    extra_ldflags: List[str] = _runtime_list(
        "Optional extra linker flags, e.g. ['-L/opt/fastjet/lib', '-lfastjet']"
    )
    build_only: bool = RuntimeField(
        default=False, description="Compile and link but do not run"
    )
    timeout_s: int = RuntimeField(
        default=3600, description="Wall-clock limit for the run, in seconds"
    )
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(
        default=".", description="Base directory for safe path resolution"
    )
    # ---------------------------------------------------------------------- #

    def _install_path(self) -> str:
        """The install to build against: the configured path, or what the
        subclass falls back to."""
        raise NotImplementedError

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(
                f"Base directory does not exist: {self.base_directory}"
            )

    def _safe_path(self, rel: str) -> Optional[str]:
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        if full != self.base_directory and not full.startswith(
            self.base_directory + os.sep
        ):
            return None
        return full

    def _rel(self, path: str) -> str:
        return os.path.relpath(path, self.base_directory)

    def _run(self) -> str:
        for key in ("source_path", "work_dir"):
            if not getattr(self, key, None):
                return self.format_error(
                    error="Missing Parameter", reason=f"{key} is required"
                )
        sources = []
        for rel in [self.source_path, *self.extra_sources]:
            full = self._safe_path(rel)
            if not full:
                return self.format_error(
                    error="Access Denied",
                    reason="Path escapes base_directory",
                    context=f"source={rel}",
                )
            if not os.path.isfile(full):
                return self.format_error(
                    error="File Not Found",
                    reason="Source does not exist",
                    context=f"source={rel}",
                )
            if not full.endswith(_SOURCE_SUFFIXES):
                return self.format_error(
                    error="Invalid Source",
                    reason=f"not a C++ source: {rel}",
                    suggestion=f"Use one of {', '.join(_SOURCE_SUFFIXES)}",
                )
            sources.append(full)
        work = self._safe_path(self.work_dir)
        if not work:
            return self.format_error(
                error="Access Denied",
                reason="work_dir escapes base_directory",
                context=f"work_dir={self.work_dir}",
            )
        os.makedirs(work, exist_ok=True)

        try:
            install = resolve_install(self.generator, self._install_path())
        except BuildError as e:
            return self.format_error(error="Setup Error", reason=str(e))
        exe = os.path.join(
            work, os.path.splitext(os.path.basename(sources[0]))[0]
        )
        try:
            b = build(
                install,
                sources,
                exe,
                extra_cxxflags=self.extra_cxxflags,
                extra_ldflags=self.extra_ldflags,
            )
        except BuildError as e:
            return self.format_error(
                error="Build Failed",
                reason=str(e),
                suggestion="Fix the compiler errors above and call again",
            )
        result: Dict = {
            "status": "ok",
            "generator": install.describe(),
            "exe": self._rel(b.exe),
            "build": {
                "cached": b.cached,
                "command": b.command,
                **({"warnings": b.log[-4000:]} if b.log else {}),
            },
        }
        if self.build_only:
            return json.dumps(result, ensure_ascii=False)

        before = _snapshot(work)
        r = run(
            install,
            [b.exe, *self.args],
            cwd=work,
            log_prefix=os.path.join(work, os.path.basename(exe)),
            timeout=self.timeout_s or None,
        )
        after = _snapshot(work)
        result["run"] = {
            "returncode": r.returncode,
            "elapsed_s": r.elapsed_s,
            "timed_out": r.timed_out,
            "stdout_log": self._rel(r.stdout_log),
            "stderr_log": self._rel(r.stderr_log),
            "stdout_tail": r.tail("stdout", 3000),
            "stderr_tail": r.tail("stderr", 3000),
            "changed_files": sorted(
                self._rel(p) for p, m in after.items() if before.get(p) != m
            ),
        }
        if r.returncode != 0:
            result["status"] = "error"
            result["error"] = (
                f"timed out after {self.timeout_s}s"
                if r.timed_out
                else f"program exited with status {r.returncode}"
            )
        return json.dumps(result, ensure_ascii=False)


class PythiaCPPFromProgramTool(_CPPProgramTool):
    """
    Compile a C++ main program against the Pythia 8 install named by
    pythia_path and run it in the sandbox.

    Use it for what a .cmnd card cannot express: UserHooks, custom vetoes,
    reading the event record during generation, Pythia plugins. The program
    includes "Pythia8/Pythia.h" as usual; the install's pythia8-config
    supplies the flags, and PYTHIA8DATA is set so `Pythia pythia;` finds
    the install's settings database. The pythia8mc wheel ships no headers,
    so this tool always needs a Pythia 8 build of your own.

    Builds are cached: a rerun with unchanged sources and flags skips the
    compile. Output: build command, run status, stdout/stderr tails (full
    logs are written to work_dir), and the files the run wrote.
    """

    generator: ClassVar[str] = "pythia"
    pythia_path: str = StateField(
        default="",
        description="Pythia 8 install prefix (holds bin/pythia8-config); "
        "PYTHIA_PATH when unset",
    )

    def _install_path(self) -> str:
        return configured_path("pythia", self.pythia_path)


class SherpaCPPFromProgramTool(_CPPProgramTool):
    """
    Compile a C++ main program against a Sherpa install and run it in the
    sandbox.

    The program drives Sherpa through its C++ API (SHERPA::Sherpa,
    GenerateOneEvent, GetBlobList, ...), as Sherpa's own Examples/API do.
    It links the same libraries as the install's bin/Sherpa, so it can call
    into any part of Sherpa. Sherpa reads its run card from the working
    directory (Sherpa.yaml there, or -f<card> in args) and writes Process/
    and Results/ there.

    The install is the one sherpa_path names; without a path, the sherpa-mc
    wheel's own libraries and headers, when the wheel is installed.

    Builds are cached: a rerun with unchanged sources and flags skips the
    compile. Output: build command, run status, stdout/stderr tails (full
    logs are written to work_dir), and the files the run wrote.
    """

    generator: ClassVar[str] = "sherpa"
    sherpa_path: str = StateField(
        default="",
        description="Sherpa install prefix (holds bin/Sherpa-config); "
        "SHERPA_PATH when unset, else the sherpa-mc wheel",
    )

    def _install_path(self) -> str:
        path = configured_path("sherpa", self.sherpa_path)
        if path:
            return path
        from tools.sherpa.runtime import wheel_prefix

        return wheel_prefix() or ""
