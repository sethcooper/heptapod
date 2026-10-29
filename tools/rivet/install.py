"""
# install.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Locate a Rivet installation and run its programs.

Rivet is a C++ library with a Python front end (`rivet`, `rivet-mkhtml`, the
`rivet` and `yoda` modules) that is not on PyPI, so heptapod drives whatever
installation the user has -- a conda-forge environment, a `rivet-bootstrap`
build, an LCG release -- through its `rivet-config` script, the way the
Delphes and MadGraph bundles drive theirs. Everything that touches Rivet is a
subprocess run with the install's own Python and environment, so the
interpreter serving heptapod and the one Rivet was built for need not agree.

`rivet_path` (or the RIVET_PATH environment variable) names the install
prefix: the directory holding `bin/rivet-config`. The prefix's `bin/` or
the script itself are accepted too.
"""

import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class RivetError(RuntimeError):
    """A Rivet installation is missing or broken, or one of its programs
    failed."""


_CONFIG_SCRIPT = "rivet-config"

_INSTALL_HINT = (
    "Point rivet_path (`tb config set heptapod rivet_path <prefix>`) or "
    "RIVET_PATH "
    "at a Rivet installation prefix, the directory holding bin/rivet-config. "
    "One way to get one: `conda create -n rivet -c conda-forge rivet` and use "
    "that environment's prefix."
)


@dataclass
class RivetInstall:
    """A Rivet installation, located through `rivet-config`."""

    prefix: str
    version: str
    bin_dir: str
    data_dir: str  # share/Rivet: the .info, .plot and .yoda.gz files
    lib_dir: str
    python: str  # an interpreter that imports rivet and yoda
    yoda_version: str = ""
    env: Dict[str, str] = field(default_factory=dict)  # for every subprocess

    def executable(self, name: str) -> str:
        """The path of one of the install's programs (`rivet`, `rivet-merge`,
        ...)."""
        path = os.path.join(self.bin_dir, name)
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            raise RivetError(
                f"{name} is not part of the Rivet installation at "
                f"{self.prefix}"
            )
        return path

    def describe(self) -> Dict[str, str]:
        return {
            "prefix": self.prefix,
            "version": self.version,
            "yoda_version": self.yoda_version,
        }


@dataclass
class RunResult:
    """What a Rivet program did: exit status, timing and where its output
    went."""

    returncode: int
    elapsed_s: float
    stdout_log: str
    stderr_log: str
    timed_out: bool = False
    command: List[str] = field(default_factory=list)

    def tail(self, which: str = "stderr", n: int = 4000) -> str:
        path = self.stderr_log if which == "stderr" else self.stdout_log
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - n))
                return handle.read().decode("utf-8", errors="replace")
        except OSError:
            return ""


# Resolved installs, keyed by the configured path: resolving one costs several
# subprocess calls, and a tool may be called many times in a session.
_CACHE: Dict[str, RivetInstall] = {}


def _query(
    script: str, *options: str, env: Optional[Dict[str, str]] = None
) -> str:
    try:
        proc = subprocess.run(
            [script, *options],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    except OSError as e:
        raise RivetError(f"cannot run {script}: {e}") from e
    except subprocess.TimeoutExpired:
        raise RivetError(
            f"{script} {' '.join(options)} did not finish in 60 s"
        )
    if proc.returncode != 0:
        raise RivetError(
            f"{os.path.basename(script)} {' '.join(options)} failed: "
            f"{proc.stderr.strip()[-500:] or proc.stdout.strip()[-500:]}"
        )
    return proc.stdout.strip()


def _find_config_script(path: str) -> Optional[Path]:
    """`rivet-config` for a path naming the prefix, its bin/, or the script."""
    candidate = Path(path).expanduser()
    for option in (
        candidate,
        candidate / "bin" / _CONFIG_SCRIPT,
        candidate / _CONFIG_SCRIPT,
    ):
        if (
            option.is_file()
            and option.name == _CONFIG_SCRIPT
            and os.access(option, os.X_OK)
        ):
            return option.resolve()
    return None


def _shebang_interpreter(script: str) -> Optional[str]:
    """The interpreter a script's first line names, resolved; None if there is
    none."""
    try:
        with open(script, "rb") as handle:
            first = handle.readline().decode("utf-8", errors="replace").strip()
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    parts = shlex.split(first[2:])
    if not parts:
        return None
    if os.path.basename(parts[0]) == "env" and len(parts) > 1:
        # "#!/usr/bin/env python3": whatever python3 is, where the script lives
        # wins over PATH, since an install's bin/ normally holds its own.
        for name in parts[1:]:
            if name.startswith("-"):
                continue
            local = os.path.join(os.path.dirname(script), name)
            return local if os.access(local, os.X_OK) else shutil.which(name)
    return parts[0] if os.access(parts[0], os.X_OK) else None


def _prepend(env: Dict[str, str], key: str, *values: str) -> None:
    present = [v for v in env.get(key, "").split(os.pathsep) if v]
    env[key] = os.pathsep.join(
        [v for v in values if v] + [v for v in present if v not in values]
    )


def resolve_rivet(path: Optional[str] = None) -> RivetInstall:
    """
    Locate a Rivet installation and check that it works.

    Args:
        path: The install prefix (or its bin/, or rivet-config itself). When
            empty, the RIVET_PATH environment variable is used.

    Returns:
        The install, with the environment its programs need.

    Raises:
        RivetError: when nothing is configured, when the path holds no
            rivet-config, or when the install's Python cannot import rivet
            and yoda -- which is what a half-built install looks like.
    """
    path = (path or os.environ.get("RIVET_PATH", "")).strip()
    if not path:
        raise RivetError("rivet_path is not configured. " + _INSTALL_HINT)
    cached = _CACHE.get(path)
    if cached is not None and os.path.isdir(cached.prefix):
        return cached

    script = _find_config_script(path)
    if script is None:
        raise RivetError(
            f"no executable {_CONFIG_SCRIPT} under {path}. " + _INSTALL_HINT
        )
    script_path = str(script)
    prefix = _query(script_path, "--prefix")
    version = _query(script_path, "--version")
    data_dir = _query(script_path, "--datadir")
    lib_dir = _query(script_path, "--libdir")
    bin_dir = os.path.dirname(script_path)
    python_paths = _query(script_path, "--pythonpath").split()

    yoda_config = os.path.join(bin_dir, "yoda-config")
    yoda_version = ""
    if os.access(yoda_config, os.X_OK):
        try:
            python_paths += _query(yoda_config, "--pythonpath").split()
            yoda_version = _query(yoda_config, "--version")
        except RivetError:
            pass

    env = dict(os.environ)
    _prepend(env, "PYTHONPATH", *dict.fromkeys(python_paths))
    _prepend(env, "PATH", bin_dir)
    _prepend(
        env,
        "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH",
        lib_dir,
    )
    # Not inherited: a serving process's own buffering and encoding settings
    # have no business in Rivet's Python.
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")

    python = (
        _shebang_interpreter(os.path.join(bin_dir, "rivet"))
        or next(
            (
                p
                for p in (
                    os.path.join(prefix, "bin", "python3"),
                    os.path.join(prefix, "bin", "python"),
                )
                if os.access(p, os.X_OK)
            ),
            None,
        )
        or shutil.which("python3")
    )
    if python is None:
        raise RivetError(
            f"no Python interpreter found for the Rivet installation at "
            f"{prefix}"
        )
    try:
        probe = subprocess.run(
            [
                python,
                "-c",
                "import rivet, yoda; print(rivet.version(), yoda.version())",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RivetError(
            f"cannot run {python} for the Rivet installation at {prefix}: {e}"
        ) from e
    if probe.returncode != 0:
        raise RivetError(
            f"the Rivet installation at {prefix} is not usable: {python} "
            f"cannot import "
            f"rivet and yoda:\n{probe.stderr.strip()[-800:]}"
        )
    reported = probe.stdout.split()
    if len(reported) == 2:
        version, yoda_version = reported

    install = RivetInstall(
        prefix=prefix,
        version=version,
        bin_dir=bin_dir,
        data_dir=data_dir,
        lib_dir=lib_dir,
        python=python,
        yoda_version=yoda_version,
        env=env,
    )
    _CACHE[path] = install
    return install


def run_program(
    install: RivetInstall,
    argv: Sequence[str],
    *,
    cwd: str,
    log_prefix: str,
    timeout: Optional[float] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> RunResult:
    """
    Run one of the install's programs with its environment.

    stdout and stderr go to `<log_prefix>.stdout.log` and `.stderr.log`, never
    to this process's own descriptors: under `tb serve`, stdout is the MCP
    JSON-RPC channel, and a program's chatter on it breaks the transport.
    """
    env = dict(install.env)
    if extra_env:
        env.update(extra_env)
    stdout_log, stderr_log = (
        f"{log_prefix}.stdout.log",
        f"{log_prefix}.stderr.log",
    )
    os.makedirs(os.path.dirname(stdout_log) or ".", exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with open(stdout_log, "wb") as out, open(stderr_log, "wb") as err:
        try:
            proc = subprocess.run(
                list(argv),
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                timeout=timeout,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode, timed_out = -9, True
        except OSError as e:
            raise RivetError(f"cannot run {argv[0]}: {e}") from e
    return RunResult(
        returncode=returncode,
        elapsed_s=round(time.monotonic() - started, 3),
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        timed_out=timed_out,
        command=list(argv),
    )


def run_python(
    install: RivetInstall,
    script: str,
    args: Sequence[str] = (),
    *,
    timeout: float = 600,
) -> Tuple[int, str, str]:
    """
    Run a helper script with the install's Python, which imports rivet and
    yoda.

    Returns:
        (returncode, stdout, stderr). The caller parses stdout; the helpers in
        this package print JSON.
    """
    try:
        proc = subprocess.run(
            [install.python, "-I", script, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=install.env,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise RivetError(
            f"{os.path.basename(script)} did not finish in {timeout:.0f} s"
        )
    except OSError as e:
        raise RivetError(f"cannot run {install.python}: {e}") from e
    return proc.returncode, proc.stdout, proc.stderr


def popen_program(
    install: RivetInstall,
    argv: Sequence[str],
    *,
    cwd: str,
    log_prefix: str,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[Any, str, str]:
    """
    Start one of the install's programs without waiting for it, logging as
    :func:`run_program` does. For a Rivet run that reads events from a pipe
    while a generator writes them.

    Returns:
        (process, stdout_log, stderr_log). The caller owns the process and
        must close its log handles through ``process.stdout``/``stderr``
        attributes being files -- they are closed here when the process ends
        via :func:`finish_program`.
    """
    env = dict(install.env)
    if extra_env:
        env.update(extra_env)
    stdout_log, stderr_log = (
        f"{log_prefix}.stdout.log",
        f"{log_prefix}.stderr.log",
    )
    os.makedirs(os.path.dirname(stdout_log) or ".", exist_ok=True)
    out = open(stdout_log, "wb")
    err = open(stderr_log, "wb")
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
        )
    except OSError as e:
        out.close()
        err.close()
        raise RivetError(f"cannot run {argv[0]}: {e}") from e
    process._heptapod_logs = (out, err)  # closed by finish_program
    return process, stdout_log, stderr_log


def finish_program(
    process: Any,
    *,
    timeout: Optional[float] = None,
    command: Sequence[str] = (),
    started: Optional[float] = None,
    stdout_log: str = "",
    stderr_log: str = "",
) -> RunResult:
    """Wait for a process started by :func:`popen_program` and close its
    logs."""
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        returncode, timed_out = -9, True
    finally:
        for handle in getattr(process, "_heptapod_logs", ()):
            handle.close()
    elapsed = (
        round(time.monotonic() - started, 3) if started is not None else 0.0
    )
    return RunResult(
        returncode=returncode,
        elapsed_s=elapsed,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        timed_out=timed_out,
        command=list(command),
    )
