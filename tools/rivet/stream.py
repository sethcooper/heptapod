"""
# stream.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Run Rivet on events as a generator produces them, through a named pipe.

A complete hadron-level event is over a hundred kilobytes of HepMC3 text,
so a sample large enough for a measurement -- a million events -- is a
hundred gigabytes on disk. Rivet reads from a pipe as readily as from a
file, which is how its own documentation suggests running it next to a
generator. :class:`RivetStream` starts `rivet` reading a FIFO, hands the
FIFO's path to whatever writes HepMC3 (an ``event_io.EventWriter``), and
collects the result when the writer closes the pipe.
"""

import errno
import fcntl
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from tools.rivet.install import (
    RivetError,
    RivetInstall,
    RunResult,
    finish_program,
    popen_program,
)
from tools.rivet.run import parse_rivet_log


class RivetStreamError(RivetError):
    """Rivet stopped, or never started, while events were being written to
    it."""


class RivetStream:
    """
    A `rivet` process reading HepMC3 events from a named pipe.

    Use as a context manager around the generation loop::

        stream = RivetStream(install, analyses, output_yoda=..., work_dir=...)
        with stream:
            with EventWriter(stream.fifo_path, fmt="hepmc3", ...) as writer:
                for batch in batches:
                    writer.write(batch)
                    stream.check()        # raises if rivet has died
            outcome = stream.finish()

    The pipe is created in `work_dir` and removed on exit; Rivet's logs go
    next to the YODA output.
    """

    def __init__(
        self,
        install: RivetInstall,
        analyses: Sequence[str],
        *,
        output_yoda: str,
        work_dir: str,
        cross_section_pb: float = -1.0,
        ignore_beams: bool = False,
        skip_weights: bool = False,
        nominal_weight: str = "",
        custom_dirs: Sequence[str] = (),
        timeout_s: Optional[float] = None,
    ) -> None:
        self.install = install
        self.analyses = list(analyses)
        self.output_yoda = output_yoda
        self.work_dir = work_dir
        self.cross_section_pb = cross_section_pb
        self.ignore_beams = ignore_beams
        self.skip_weights = skip_weights
        self.nominal_weight = nominal_weight
        self.custom_dirs = list(custom_dirs)
        self.timeout_s = timeout_s
        self.fifo_path = os.path.join(work_dir, "events.hepmc.fifo")
        self.stdout_log = ""
        self.stderr_log = ""
        self.command: List[str] = []
        self._process: Any = None
        self._probe_fd: Optional[int] = None
        self._started: Optional[float] = None

    # ------------------------------------------------------------------ #

    def __enter__(self) -> "RivetStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """Create the pipe and start `rivet` reading it; wait until it does."""
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_yoda) or ".", exist_ok=True)
        if os.path.exists(self.fifo_path):
            os.remove(self.fifo_path)
        os.mkfifo(self.fifo_path)
        argv = [self.install.executable("rivet"), "-o", self.output_yoda]
        for spec in self.analyses:
            argv += ["-a", spec]
        if self.cross_section_pb > 0:
            argv += ["-x", repr(float(self.cross_section_pb))]
        if self.ignore_beams:
            argv.append("--ignore-beams")
        if self.skip_weights:
            argv.append("--skip-weights")
        if self.nominal_weight:
            argv += ["--nominal-weight", self.nominal_weight]
        for directory in self.custom_dirs:
            argv += ["--analysis-path-append", directory]
        argv.append(self.fifo_path)
        extra_env = {}
        if self.custom_dirs:
            extra_env["RIVET_DATA_PATH"] = os.pathsep.join(
                self.custom_dirs + [self.install.data_dir]
            )
        self.command = argv
        self._started = time.monotonic()
        log_prefix = os.path.join(os.path.dirname(self.output_yoda), "rivet")
        self._process, self.stdout_log, self.stderr_log = popen_program(
            self.install,
            argv,
            cwd=self.work_dir,
            log_prefix=log_prefix,
            extra_env=extra_env,
        )

        # Opening a FIFO for writing blocks until something reads it. Rivet
        # opens the pipe once it has loaded its analyses, and exits instead
        # if one of them is unknown -- so poll with a non-blocking open rather
        # than wait on a reader that may never come.
        deadline = time.monotonic() + 600
        while True:
            try:
                self._probe_fd = os.open(
                    self.fifo_path, os.O_WRONLY | os.O_NONBLOCK
                )
                break
            except OSError as e:
                if e.errno != errno.ENXIO:
                    self.close()
                    raise RivetStreamError(
                        f"cannot open the event pipe for writing: {e}"
                    ) from e
            if self._process.poll() is not None:
                message = self.failure_message()
                self.close()
                raise RivetStreamError(message)
            if time.monotonic() > deadline:
                self.close()
                raise RivetStreamError(
                    "rivet did not start reading events within 600 s"
                )
            time.sleep(0.05)
        # The writer that follows opens the pipe itself; it must block on a
        # full pipe rather than fail, so the probe descriptor goes back to
        # blocking mode and is kept until the pipe is closed, which keeps the
        # reader's end-of-file from arriving early.
        flags = fcntl.fcntl(self._probe_fd, fcntl.F_GETFL)
        fcntl.fcntl(self._probe_fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)

    def alive(self) -> bool:
        """Whether `rivet` is still running."""
        return self._process is not None and self._process.poll() is None

    def check(self) -> None:
        """Raise if `rivet` has stopped: nothing written now would be read."""
        if not self.alive():
            raise RivetStreamError(self.failure_message())

    def finish(self) -> Dict[str, Any]:
        """
        Close the writer's side of the pipe and wait for Rivet to finalise.

        Call after the event writer has been closed. Returns the run's exit
        status, timing, parsed log and log paths; raises when Rivet failed.
        """
        self._release_probe()
        result = finish_program(
            self._process,
            timeout=self.timeout_s,
            command=self.command,
            started=self._started,
            stdout_log=self.stdout_log,
            stderr_log=self.stderr_log,
        )
        self._process = None
        self._remove_fifo()
        log = parse_rivet_log(
            result.tail("stdout", 200000)
            + "\n"
            + result.tail("stderr", 200000)
        )
        outcome = {
            "returncode": result.returncode,
            "elapsed_s": result.elapsed_s,
            "timed_out": result.timed_out,
            "log": log,
            "logs": {"stdout": self.stdout_log, "stderr": self.stderr_log},
        }
        if result.timed_out:
            raise RivetStreamError(
                f"rivet did not finish within {self.timeout_s} s of the last "
                f"event"
            )
        if result.returncode != 0 or not os.path.exists(self.output_yoda):
            raise RivetStreamError(self.failure_message(result, log))
        return outcome

    def close(self) -> None:
        """Stop everything and remove the pipe; safe to call more than once."""
        self._release_probe()
        if self._process is not None:
            if self._process.poll() is None:
                self._process.kill()
            finish_program(
                self._process,
                command=self.command,
                started=self._started,
                stdout_log=self.stdout_log,
                stderr_log=self.stderr_log,
            )
            self._process = None
        self._remove_fifo()

    # ------------------------------------------------------------------ #

    def _release_probe(self) -> None:
        if self._probe_fd is not None:
            try:
                os.close(self._probe_fd)
            except OSError:
                pass
            self._probe_fd = None

    def _remove_fifo(self) -> None:
        try:
            os.remove(self.fifo_path)
        except OSError:
            pass

    def failure_message(
        self,
        result: Optional[RunResult] = None,
        log: Optional[Dict[str, Any]] = None,
    ) -> str:
        if log is None:
            text = ""
            for path in (self.stdout_log, self.stderr_log):
                try:
                    with open(
                        path, "r", encoding="utf-8", errors="replace"
                    ) as handle:
                        text += handle.read()[-200000:] + "\n"
                except OSError:
                    pass
            log = parse_rivet_log(text)
        if log["unknown"]:
            return f"unknown analysis: {', '.join(log['unknown'])}"
        if log["all_removed"]:
            return (
                "every analysis was incompatible with the events' beams: "
                + ", ".join(log["removed"])
            )
        if log["errors"]:
            return log["errors"][0]
        status = (
            result.returncode
            if result is not None
            else (
                self._process.returncode if self._process is not None else None
            )
        )
        return f"rivet stopped with status {status}; see {self.stderr_log}"
