"""
# build.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Compile and run C++ against a local Pythia 8 or Sherpa install.

A local install is found through its own config script -- `pythia8-config`
or `Sherpa-config` -- which knows the compiler flags the install was built
with. That is what lets heptapod drive a user's own build, at whatever
version and with whatever plugins it has, without the pip wheels and
without caring which Python the install was built against: everything
here is a subprocess.

Programs are compiled once and cached by a hash of their sources, the
install's flags and the compiler, so a rebuild happens exactly when one of
those changes.
"""

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class BuildError(RuntimeError):
    """A generator install is missing or broken, or a build failed."""


#: Each supported generator: its config script, the environment variable
#: that can point at it, and what the install prefix must hold.
_GENERATORS = {
    "pythia": {
        "script": "pythia8-config",
        "key": "pythia_path",
        "env": "PYTHIA_PATH",
        "label": "Pythia 8",
    },
    "sherpa": {
        "script": "Sherpa-config",
        "key": "sherpa_path",
        "env": "SHERPA_PATH",
        "label": "Sherpa",
    },
}

#: Sherpa's config script reports only -lSherpaMain, but Sherpa's libraries
#: are linked with undefined symbols left to load time (flat namespace on
#: macOS), so a program has to load every library they resolve against --
#: the set the stock bin/Sherpa links, which _sherpa_link_flags copies. These
#: are the ones a program calling into ATOOLS needs even if that read fails.
_SHERPA_EXTRA_LIBS = ["-lToolsPhys", "-lToolsMath", "-lToolsOrg"]

#: Standard for heptapod's drivers and agent programs. Config scripts often
#: say -std=c++11 (Pythia's does); the later flag wins with gcc and clang.
_CXX_STD = "-std=c++17"


@dataclass
class Install:
    """A local generator install, located through its config script."""

    generator: str  # "pythia" or "sherpa"
    prefix: str
    config_script: str
    version: str = ""
    cxx: str = "c++"
    cxxflags: List[str] = field(default_factory=list)
    ldflags: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)  # for running programs

    @property
    def label(self) -> str:
        return _GENERATORS[self.generator]["label"]

    def describe(self) -> Dict[str, str]:
        return {
            "generator": self.generator,
            "prefix": self.prefix,
            "version": self.version,
        }


def _query(script: str, *opts: str) -> str:
    proc = subprocess.run(
        [script, *opts], capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        raise BuildError(
            f"{script} {' '.join(opts)} failed: {proc.stderr.strip()[-500:]}"
        )
    return proc.stdout.strip()


def _find_config_script(path: str, script: str) -> Optional[Path]:
    """The config script for a path that names the prefix, its bin/, or the
    script."""
    p = Path(path).expanduser()
    for cand in (p, p / "bin" / script, p / script):
        if cand.is_file() and cand.name == script and os.access(cand, os.X_OK):
            return cand.resolve()
    return None


def _rpaths(ldflags: Sequence[str]) -> List[str]:
    """-Wl,-rpath for every existing -L directory, so a program finds the
    install's shared libraries without DYLD_/LD_LIBRARY_PATH."""
    out = []
    for flag in ldflags:
        if flag.startswith("-L") and os.path.isdir(flag[2:]):
            rp = f"-Wl,-rpath,{flag[2:]}"
            if rp not in ldflags and rp not in out:
                out.append(rp)
    return out


def _sherpa_link_flags(prefix: str) -> List[str]:
    """The libraries bin/Sherpa itself links, as linker flags.

    Read from the executable (otool -L on macOS, the ELF NEEDED entries
    elsewhere): Sherpa libraries become -l flags, and any other non-system
    library (Homebrew's libzip, say) is passed by path. Empty when the
    executable or the inspection tool is missing.
    """
    exe = Path(prefix) / "bin" / "Sherpa"
    libdirs = [
        d
        for d in (
            Path(prefix) / "lib" / "SHERPA-MC",
            Path(prefix) / "lib64" / "SHERPA-MC",
        )
        if d.is_dir()
    ]
    if not exe.is_file() or not libdirs:
        return []
    own = {p.name for d in libdirs for p in d.iterdir()}
    flags: List[str] = []
    try:
        if sys.platform == "darwin":
            lines = _query("otool", "-L", str(exe)).splitlines()[1:]
            deps = [ln.strip().split(" (")[0] for ln in lines]
        else:
            lines = _query("readelf", "-d", str(exe)).splitlines()
            deps = [
                ln.split("[", 1)[1].rstrip("]")
                for ln in lines
                if "(NEEDED)" in ln
            ]
            flags.append("-Wl,--no-as-needed")
    except (BuildError, OSError, subprocess.TimeoutExpired):
        return []
    for dep in deps:
        name = os.path.basename(dep)
        if name in own and name.startswith("lib"):
            flags.append("-l" + name[3:].split(".")[0])
        elif (
            sys.platform == "darwin"
            and dep.startswith("/")
            and not dep.startswith(("/usr/lib/", "/System/"))
        ):
            flags.append(dep)
    return flags


def resolve_install(generator: str, path: Optional[str] = None) -> Install:
    """Locate a Pythia 8 or Sherpa install and read its build flags.

    Args:
        generator: "pythia" or "sherpa".
        path: the install prefix (or its bin/, or the config script). When
            empty, the generator's environment variable (PYTHIA_PATH,
            SHERPA_PATH) is used.
    """
    spec = _GENERATORS[generator]
    path = (path or os.environ.get(spec["env"], "")).strip()
    if not path:
        raise BuildError(
            f"{spec['key']} is not configured. Point it at a {spec['label']} "
            f"install "
            f"prefix containing bin/{spec['script']} (`tb config set heptapod "
            f"{spec['key']} <prefix>`, or set {spec['env']})."
        )
    script = _find_config_script(path, spec["script"])
    if script is None:
        raise BuildError(
            f"no executable {spec['script']} under {path}. {spec['key']} must "
            f"be the "
            f"{spec['label']} install prefix (the directory holding "
            f"bin/{spec['script']})."
        )
    s = str(script)

    if generator == "pythia":
        install = Install(
            generator,
            prefix=_query(s, "--prefix"),
            config_script=s,
            cxxflags=shlex.split(_query(s, "--cxxflags")),
            ldflags=shlex.split(_query(s, "--libs")),
        )
        xmldoc = _query(s, "--xmldoc")
        if xmldoc:
            # Pythia() with no argument reads PYTHIA8DATA; set it so an agent's
            # program finds the same settings database the driver uses.
            install.env["PYTHIA8DATA"] = xmldoc
    else:
        install = Install(
            generator,
            prefix=_query(s, "--prefix"),
            config_script=s,
            cxx=_query(s, "--cxx") or "c++",
            cxxflags=shlex.split(_query(s, "--cppflags")),
            ldflags=shlex.split(_query(s, "--ldflags")),
        )
        # The sherpa-mc wheel ships a Sherpa-config that names the paths of
        # the machine it was built on. The prefix is then the directory the
        # script lives under, and its include and lib directories are found
        # beside it.
        if not os.path.isdir(install.prefix):
            install.prefix = str(script.parent.parent)
            install.cxxflags = [
                "-I" + os.path.join(install.prefix, "include", "SHERPA-MC")
            ]
            libdir = next(
                (
                    d
                    for d in (
                        os.path.join(install.prefix, "lib", "SHERPA-MC"),
                        os.path.join(install.prefix, "lib64", "SHERPA-MC"),
                    )
                    if os.path.isdir(d)
                ),
                os.path.join(install.prefix, "lib", "SHERPA-MC"),
            )
            install.ldflags = ["-L" + libdir, "-lSherpaMain"]
        for flag in _SHERPA_EXTRA_LIBS + _sherpa_link_flags(install.prefix):
            if flag not in install.ldflags:
                install.ldflags.append(flag)
    try:
        install.version = _query(s, "--version")
    except BuildError:
        pass
    # A config script reports the compiler the install was built with, which
    # may be a toolchain-specific name (conda's
    # arm64-apple-darwin20.0.0-clang++) that is not on PATH where heptapod
    # runs. CXX overrides; else fall back to c++.
    install.cxx = os.environ.get("CXX") or (
        install.cxx if shutil.which(install.cxx) else "c++"
    )
    install.ldflags += _rpaths(install.ldflags)
    return install


def cache_dir() -> Path:
    """Where compiled drivers live: HEPTAPOD_CPP_CACHE, else
    $XDG_CACHE_HOME/heptapod/cpp (~/.cache/heptapod/cpp)."""
    root = os.environ.get("HEPTAPOD_CPP_CACHE")
    if root:
        return Path(root).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return Path(xdg) / "heptapod" / "cpp"


@dataclass
class Build:
    exe: str
    command: List[str]
    cached: bool
    log: str = ""


def build_key(
    install: Install,
    sources: Sequence[Path],
    extra_cxxflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
) -> str:
    h = hashlib.sha256()
    for part in (
        install.cxx,
        install.version,
        install.prefix,
        *install.cxxflags,
        *install.ldflags,
        _CXX_STD,
        *extra_cxxflags,
        *extra_ldflags,
    ):
        h.update(part.encode() + b"\0")
    for src in sources:
        h.update(Path(src).read_bytes() + b"\0")
    return h.hexdigest()


def build(
    install: Install,
    sources: Sequence[str],
    output: str,
    *,
    extra_cxxflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
    timeout: float = 600,
) -> Build:
    """Compile and link sources against install into output.

    Skips the compile when output exists and was built from identical
    inputs (recorded in a `<output>.buildhash` file beside it).
    """
    srcs = [Path(s) for s in sources]
    for s in srcs:
        if not s.is_file():
            raise BuildError(f"source not found: {s}")
    out = Path(output)
    stamp = out.with_name(out.name + ".buildhash")
    key = build_key(install, srcs, extra_cxxflags, extra_ldflags)
    cmd = [
        install.cxx,
        *install.cxxflags,
        _CXX_STD,
        *extra_cxxflags,
        *[str(s) for s in srcs],
        "-o",
        str(out),
        *install.ldflags,
        *extra_ldflags,
    ]
    if out.is_file() and stamp.is_file() and stamp.read_text().strip() == key:
        return Build(exe=str(out), command=cmd, cached=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    # Compile to a temporary name and rename: a concurrent run never sees a
    # half-written executable, and a failed build leaves the old one alone.
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp_cmd = (
        cmd[: cmd.index("-o") + 1] + [str(tmp)] + cmd[cmd.index("-o") + 2 :]
    )
    try:
        proc = subprocess.run(
            tmp_cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        raise BuildError(
            f"compiling against {install.label} timed out after {timeout:.0f}s"
        )
    except FileNotFoundError:
        raise BuildError(
            f"C++ compiler not found: {install.cxx} (set CXX to override)"
        )
    log = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise BuildError(
            f"compiling against {install.label} {install.version} at "
            f"{install.prefix} failed:\n{log[-4000:]}\ncommand: "
            f"{shlex.join(cmd)}"
        )
    os.replace(tmp, out)
    stamp.write_text(key + "\n")
    return Build(exe=str(out), command=cmd, cached=False, log=log)


def build_driver(install: Install, source: Path, name: str) -> Build:
    """A heptapod-shipped driver, built into the shared cache."""
    key = build_key(install, [source])[:16]
    return build(
        install, [str(source)], str(cache_dir() / f"{name}-{key}" / name)
    )


@dataclass
class Run:
    returncode: int
    elapsed_s: float
    stdout_log: str
    stderr_log: str
    timed_out: bool = False

    def tail(self, which: str = "stderr", n: int = 4000) -> str:
        path = self.stderr_log if which == "stderr" else self.stdout_log
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - n))
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""


def run(
    install: Install,
    argv: Sequence[str],
    *,
    cwd: str,
    log_prefix: str,
    timeout: Optional[float] = None,
) -> Run:
    """Run a built program in cwd with the install's runtime environment.

    stdout and stderr go to `<log_prefix>.stdout.log` / `.stderr.log`, never
    to this process's own descriptors: under `tb serve`, stdout is the MCP
    JSON-RPC channel, and a generator's banner on it breaks the transport.
    """
    env = dict(os.environ)
    env.update(install.env)
    out_log, err_log = f"{log_prefix}.stdout.log", f"{log_prefix}.stderr.log"
    t0 = time.monotonic()
    timed_out = False
    with open(out_log, "wb") as out, open(err_log, "wb") as err:
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
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
    return Run(
        returncode=rc,
        elapsed_s=round(time.monotonic() - t0, 3),
        stdout_log=out_log,
        stderr_log=err_log,
        timed_out=timed_out,
    )


@dataclass
class Streamed:
    """A running program whose stdout is read by the caller.

    The drivers behind the CPP run-card tools write their events to stdout
    as they generate them; the tool consumes them from here rather than
    from a file. stderr still goes to a log.
    """

    proc: subprocess.Popen
    stderr_log: str
    started: float = field(default_factory=time.monotonic)

    @property
    def stdout(self) -> Any:
        return self.proc.stdout

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        """Ask the program to finish (SIGTERM); the driver ends its run
        cleanly."""
        if self.alive():
            self.proc.terminate()

    def finish(self, timeout: Optional[float] = 300) -> Run:
        """Wait for the program to exit and report the run."""
        timed_out = False
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            rc, timed_out = self.proc.wait(), True
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        return Run(
            returncode=rc,
            elapsed_s=round(time.monotonic() - self.started, 3),
            stdout_log="",
            stderr_log=self.stderr_log,
            timed_out=timed_out,
        )


def stream(
    install: Install, argv: Sequence[str], *, cwd: str, stderr_log: str
) -> Streamed:
    """Start a built program with its stdout piped back to the caller.

    stderr is appended to `stderr_log`, so nothing the program prints reaches
    this process's own descriptors (see `run`).
    """
    env = dict(os.environ)
    env.update(install.env)
    err = open(stderr_log, "ab")
    try:
        proc = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=err,
            bufsize=1 << 20,
        )
    finally:
        err.close()
    return Streamed(proc=proc, stderr_log=stderr_log)


def configured_path(generator: str, path: Optional[str]) -> str:
    """The install path a tool was given, or the generator's environment
    variable (PYTHIA_PATH, SHERPA_PATH), or an empty string."""
    return (path or os.environ.get(_GENERATORS[generator]["env"], "")).strip()
