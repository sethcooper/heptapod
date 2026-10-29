"""
# source.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Events from a heptapod driver, a C++ program compiled against a generator
install that streams the event record to its stdout one JSON line per
event.

Both C++ drivers (``tools/pythia/pythia_driver.cc`` and
``tools/sherpa/sherpa_driver.cc``) and the Python worker that runs the
sherpa-mc wheel (``tools/sherpa/sherpa_worker.py``) speak the same protocol,
so one reader serves them all. Lines are JSON objects with one top-level
key::

    {"header": {...}}        after initialisation; the generator's version
    {"names": [...]}         weight names, nominal first, after the first event
    {"p": [...], "i": [...], "w": [...]}   one event: particle rows, scalars,
                                           weights
    {"f": 1}                 one event the generator failed to produce
    {"xsec": {...}}          the running cross section, after every chunk
    {"end": {"init_ok": true, "accepted": A, "failed": F, ...}}  the last line

What the rows and scalars mean is the business of each generator's
``batch_from_events``, which turns a list of parsed event lines into the
Awkward Array the shared generation loop consumes.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from tools.cpp.build import (
    Install,
    BuildError,
    Streamed,
    build_driver,
    stream,
)


class DriverInitError(RuntimeError):
    """The driver's generator failed to initialise; the log explains why."""


class StreamSource:
    """
    Events from a program speaking the driver protocol, consumed the way the
    shared generation loop consumes any source.

    Lifecycle: ``start()`` launches the program and reads the header;
    ``next_batch(want)`` returns up to ``want`` events; ``finish()`` ends the
    run (asking the program to stop if events are still wanted) and returns
    its exit status. ``failed``, ``lhe_exhausted``, ``weight_names()`` and
    ``cross_section()`` carry what an in-process backend would read from the
    generator directly. A subclass supplies ``_launch()``.
    """

    #: What the messages call the program.
    program_name = "the driver"

    def __init__(
        self,
        *,
        log_path: str,
        work_dir: str,
        batch_builder: Callable[[Sequence[Dict[str, Any]]], Any],
    ) -> None:
        self.log_path = log_path
        self.work_dir = work_dir
        self.batch_builder = batch_builder

        self.header: Dict[str, Any] = {}
        self.version: Optional[str] = None
        self.beams: List[Dict[str, float]] = []
        self.failed = 0
        self.accepted = 0
        self.lhe_exhausted = False
        self.ended = False
        self.end: Dict[str, Any] = {}
        self._names: List[str] = []
        self._xsec: Dict[str, float] = {}
        self._stream: Optional[Streamed] = None
        self._pending: Optional[Dict[str, Any]] = None
        self.build = None

    # ------------------------------------------------------------------ #

    def _launch(self) -> Streamed:
        """Start the program with its stdout piped back here."""
        raise NotImplementedError

    def start(self) -> bool:
        """Launch the program and read its header.

        Raises:
            BuildError: the program could not start, or sent no header
            DriverInitError: the generator's initialisation failed (see the
                log)
        """
        self._stream = self._launch()
        message = self._read()
        if message is None or "end" in message:
            run = self._stream.finish()
            if message is not None and message["end"].get("init_ok") is False:
                raise DriverInitError("Initialization returned false")
            raise BuildError(
                f"{self.program_name} exited with status {run.returncode} "
                "before producing events: "
                f"{run.tail('stderr', 1500).strip() or 'see ' + self.log_path}"
            )
        header = message.get("header")
        if header is None:
            self._stream.stop()
            self._stream.finish()
            raise BuildError(
                f"{self.program_name} sent {str(message)[:200]!r} instead of "
                "a header"
            )
        self.header = header
        self.version = header.get("version")
        self.beams = list(header.get("beams", []))
        self.started(header)
        return True

    def started(self, header: Dict[str, Any]) -> None:
        """Hook for a subclass to read more out of the header."""

    def _read(self) -> Optional[Dict[str, Any]]:
        """The next line of the stream as a dict, or None at its end."""
        assert self._stream is not None
        line = self._stream.stdout.readline()
        if not line:
            return None
        return json.loads(line)

    def _control(self, message: Dict[str, Any]) -> None:
        """Apply a control line: a failure, the running cross section, the
        weight names, or the end of the run."""
        if "f" in message:
            self.failed += int(message["f"])
        elif "xsec" in message:
            self._xsec = message["xsec"]
        elif "names" in message:
            self._names = [str(n) for n in message["names"]]
        elif "end" in message:
            self.end = message["end"]
            self.lhe_exhausted = bool(self.end.get("lhe_exhausted"))
            self.ended = True

    def next_batch(self, want: int) -> Any:
        """Up to ``want`` events as one Awkward Array (empty when the run is
        over).

        After the batch is complete the stream is read on to the next event
        (held back for the following call), so the cross-section line the
        driver writes after a batch has been applied when this returns.
        """
        events: List[Dict[str, Any]] = []
        if self._pending is not None:
            events.append(self._pending)
            self._pending = None
        while not self.ended:
            message = self._read()
            if message is None:
                self.ended = True
                break
            if "p" not in message:
                self._control(message)
            elif len(events) < want:
                events.append(message)
            else:
                self._pending = message
                break
        self.accepted += len(events)
        return self.batch_builder(events)

    def weight_names(self) -> List[str]:
        return list(self._names)

    def cross_section(self) -> Dict[str, float]:
        """The running cross section as the driver last reported it."""
        return dict(self._xsec)

    def cross_section_pb(self) -> Dict[str, float]:
        """The running estimate in pb; a subclass converts its generator's
        units."""
        return {}

    def finish(self):
        """End the run and return the driver's exit (a tools.cpp.build.Run).

        When the tool stops early (size guard, Rivet gone), the driver is
        asked to stop; it then writes its final cross section and end line,
        which are read here so ``cross_section()`` is the run's last word.
        """
        assert self._stream is not None
        if not self.ended:
            self._stream.stop()
            while True:
                message = self._read()
                if message is None:
                    break
                if "p" not in message:
                    self._control(message)
            self.ended = True
        return self._stream.finish()

    def abort(self) -> None:
        """Stop the driver after a failure in the tool; its exit is not
        needed."""
        if self._stream is not None and self._stream.alive():
            self._stream.stop()
        if self._stream is not None:
            self._stream.finish(timeout=60)
        self.ended = True

    def describe(self) -> Dict[str, Any]:
        """What ran, for the manifest."""
        return {"backend": self.program_name}


class DriverSource(StreamSource):
    """A heptapod C++ driver, built against a generator install into the
    shared cache on first use."""

    def __init__(
        self,
        install: Install,
        *,
        driver_source: Path,
        driver_name: str,
        arguments: Sequence[str],
        log_path: str,
        work_dir: str,
        batch_builder: Callable[[Sequence[Dict[str, Any]]], Any],
    ) -> None:
        super().__init__(
            log_path=log_path, work_dir=work_dir, batch_builder=batch_builder
        )
        self.install = install
        self.driver_source = driver_source
        self.driver_name = driver_name
        self.program_name = driver_name
        self.arguments = list(arguments)

    def _launch(self) -> Streamed:
        """Build (if the cache lacks it) and start the driver.

        Raises:
            BuildError: the build failed
        """
        self.build = build_driver(
            self.install, self.driver_source, self.driver_name
        )
        argv = [self.build.exe, *self.arguments]
        return stream(
            self.install, argv, cwd=self.work_dir, stderr_log=self.log_path
        )

    def describe(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"backend": "cpp", **self.install.describe()}
        if self.build is not None:
            out["driver"] = os.path.basename(os.path.dirname(self.build.exe))
        return out


class CommandSource(StreamSource):
    """A program given as a command line, run in this process's environment
    plus ``env``: the Python worker that runs a generator's Python module,
    for one."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        program_name: str,
        log_path: str,
        work_dir: str,
        batch_builder: Callable[[Sequence[Dict[str, Any]]], Any],
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            log_path=log_path, work_dir=work_dir, batch_builder=batch_builder
        )
        self.argv = list(argv)
        self.program_name = program_name
        self.env = dict(env or {})

    def _launch(self) -> Streamed:
        environment = dict(os.environ)
        environment.update(self.env)
        err = open(self.log_path, "ab")
        try:
            proc = subprocess.Popen(
                self.argv,
                cwd=self.work_dir,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=err,
                text=True,
                bufsize=1,
            )
        finally:
            err.close()
        return Streamed(proc=proc, stderr_log=self.log_path)
