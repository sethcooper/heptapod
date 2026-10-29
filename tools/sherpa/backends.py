"""
# backends.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

The two sources of Sherpa events: the Python interface for
SherpaFromRunCardTool, the C++ driver for SherpaCPPFromRunCardTool.

Both run Sherpa in a process of its own and stream the event record as
JSON lines that ``record.batch_from_events`` turns into the dataset batch.
Sherpa keeps its state in statics and cannot be initialised twice in one
process, so neither runs inside the tool server.

``WorkerSource`` runs Sherpa's Python module -- the sherpa-mc wheel, or a
local build's -- through ``sherpa_worker.py`` with this interpreter.
``SherpaDriverSource`` runs heptapod-sherpa (``sherpa_driver.cc``), compiled
once against an install -- a local build named by ``sherpa_path``, or the
wheel's own libraries, which ship with their headers -- and is the one that
carries Sherpa's weight variations, which the Python interface does not
expose.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tools.cpp.build import Install
from tools.cpp.source import (  # noqa: F401
    CommandSource,
    DriverInitError,
    DriverSource,
)
from tools.sherpa.record import batch_from_events
from tools.sherpa.runtime import PythonSherpa

#: The driver source beside this module, and the name it is cached under.
DRIVER_SOURCE = Path(__file__).with_name("sherpa_driver.cc")
DRIVER_NAME = "heptapod-sherpa"
#: The worker that runs the wheel.
WORKER = Path(__file__).with_name("sherpa_worker.py")


def _arguments(
    card: str,
    log_path: str,
    n_events: int,
    chunk: int,
    store: str,
    select_pids: Optional[Sequence[int]],
    seed: Optional[int],
) -> List[str]:
    """The command-line options both programs take."""
    arguments = [
        "--card",
        card,
        "--log",
        log_path,
        "--nevents",
        str(int(n_events)),
        "--chunk",
        str(max(1, int(chunk))),
        "--store",
        store,
    ]
    if select_pids:
        arguments += [
            "--select-pids",
            ",".join(str(abs(int(p))) for p in select_pids),
        ]
    if seed is not None:
        arguments += ["--seed", str(int(seed))]
    return arguments


def _cross_section_pb(xsec: Dict[str, float]) -> Dict[str, float]:
    """The running estimate in pb, as the programs report it."""
    if "sigma_pb" not in xsec:
        return {}
    return {
        "sigma_pb": float(xsec["sigma_pb"]),
        "sigma_err_pb": float(xsec.get("sigma_err_pb", 0.0)),
        "n_accepted": int(xsec.get("n_accepted", -1)),
        "n_tried": int(float(xsec.get("n_tried", -1))),
    }


class WorkerSource(CommandSource):
    """Events from Sherpa's Python interface, run by sherpa_worker.py in a
    process of its own."""

    def __init__(
        self,
        sherpa: PythonSherpa,
        *,
        card: str,
        log_path: str,
        n_events: int,
        chunk: int,
        store: str,
        select_pids: Optional[Sequence[int]],
        seed: Optional[int],
        work_dir: str,
        beams: List[Dict[str, float]],
    ) -> None:
        argv = [sys.executable, str(WORKER)] + _arguments(
            card, log_path, n_events, chunk, store, select_pids, seed
        )
        if sherpa.prefix:
            argv += ["--prefix", sherpa.prefix]
        env = {}
        if sherpa.pythonpath:
            env["PYTHONPATH"] = (
                sherpa.pythonpath
                + os.pathsep
                + os.environ.get("PYTHONPATH", "")
            )
        super().__init__(
            argv,
            program_name=sherpa.name,
            log_path=log_path,
            work_dir=work_dir,
            batch_builder=batch_from_events,
            env=env,
        )
        self.sherpa = sherpa
        self.card_beams = list(beams)

    def started(self, header: Dict[str, Any]) -> None:
        self.beams = list(self.card_beams)

    def cross_section_pb(self) -> Dict[str, float]:
        return _cross_section_pb(self.cross_section())

    def describe(self) -> Dict[str, Any]:
        return {**self.sherpa.describe(), "version": self.version}


class SherpaDriverSource(DriverSource):
    """Events from heptapod-sherpa, for the shared generation loop."""

    def __init__(
        self,
        install: Install,
        *,
        card: str,
        log_path: str,
        n_events: int,
        chunk: int,
        store: str,
        select_pids: Optional[Sequence[int]],
        seed: Optional[int],
        work_dir: str,
        beams: List[Dict[str, float]],
    ) -> None:
        super().__init__(
            install,
            driver_source=DRIVER_SOURCE,
            driver_name=DRIVER_NAME,
            arguments=_arguments(
                card, log_path, n_events, chunk, store, select_pids, seed
            ),
            log_path=log_path,
            work_dir=work_dir,
            batch_builder=batch_from_events,
        )
        self.card_beams = list(beams)

    def started(self, header: Dict[str, Any]) -> None:
        # The driver reports Sherpa's version; the beams come from the card.
        self.beams = list(self.card_beams)

    def cross_section_pb(self) -> Dict[str, float]:
        return _cross_section_pb(self.cross_section())
