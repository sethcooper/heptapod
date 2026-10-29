"""
# sherpa_worker.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Sherpa's Python interface as a heptapod driver.

Sherpa keeps its state in statics and cannot be initialised twice in one
process: a second ``Sherpa.Sherpa(...)`` hangs. A tool server calls a tool
many times, so the module is never run inside the server. This script runs
it in a process of its own and speaks the protocol of the C++ drivers
(tools.cpp.source), so the reader is the same one:

    python sherpa_worker.py --card run.yaml --log sherpa.log --nevents N
                            --chunk C [--seed S] [--store ...]
                            [--select-pids 11,13] [--prefix <install>]

The module is the sherpa-mc wheel, or a local build's when its
site-packages directory is on PYTHONPATH.

Everything Sherpa prints goes to the log; the event lines go to the
process's original stdout. The Python interface exposes the nominal weight
only, so the ``names`` line carries just "Weight"; a card with variations
is the C++ driver's business (tools/sherpa/sherpa_driver.cc).
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.sherpa.record import event_rows  # noqa: E402

STOP = {"requested": False}


def _on_term(signum, frame):  # noqa: ARG001
    STOP["requested"] = True


def _parse():
    parser = argparse.ArgumentParser(prog="sherpa_worker")
    parser.add_argument("--card", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--nevents", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=10000)
    parser.add_argument("--seed", default="")
    parser.add_argument(
        "--store",
        default="finals+kinematics",
        choices=("full", "finals", "finals+kinematics"),
    )
    parser.add_argument("--select-pids", default="")
    parser.add_argument("--prefix", default="")
    return parser.parse_args()


def _xsec(generator, accepted, tried, weight_sum):
    out = {
        "weightSum": weight_sum,
        "n_accepted": accepted,
        "n_tried": tried,
    }
    try:
        out["sigma_pb"] = float(generator.TotalXS())
        out["sigma_err_pb"] = float(generator.TotalErr())
    except Exception:
        pass
    return {"xsec": out}


def main() -> int:
    args = _parse()
    chunk = max(1, args.chunk)
    all_particles = args.store == "full"
    select = {abs(int(p)) for p in args.select_pids.split(",") if p.strip()}

    # The original stdout is the event stream; the C-level stdout and
    # stderr, which Sherpa writes to, go to the log.
    data = os.fdopen(os.dup(1), "w", encoding="utf-8")
    log_fd = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)

    def emit(obj) -> None:
        data.write(json.dumps(obj, separators=(",", ":")) + "\n")

    import Sherpa

    argv = ["Sherpa", "-O0", f"-e{args.nevents}"]
    if args.seed:
        argv.append(f"-R{int(args.seed)}")
    argv.append(args.card)
    try:
        generator = Sherpa.Sherpa(len(argv), argv)
        ok = bool(
            generator.InitializeTheRun()
            and generator.InitializeTheEventHandler()
        )
    except Sherpa.SherpaException:
        ok = False
    if not ok:
        emit({"end": {"init_ok": False}})
        data.flush()
        return 3
    version = None
    try:
        from tools.sherpa.runtime import sherpa_version, wheel_prefix

        version = sherpa_version(args.prefix or wheel_prefix())
    except Exception:
        pass
    emit({"header": {"version": version}})
    emit({"names": ["Weight"]})
    data.flush()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    accepted = 0
    failed = 0
    consecutive = 0
    tried = 0.0
    weight_sum = 0.0
    while accepted < args.nevents and not STOP["requested"]:
        if not generator.GenerateOneEvent():
            failed += 1
            emit({"f": 1})
            consecutive += 1
            if consecutive >= chunk:
                break
            continue
        consecutive = 0
        event = event_rows(generator.GetBlobList())
        if not all_particles:
            event["p"] = [row for row in event["p"] if row[1] > 0]
        if select:
            event["p"] = [row for row in event["p"] if abs(row[0]) in select]
        tried += event["i"][0]
        weight_sum += event["w"][0]
        emit(event)
        accepted += 1
        if accepted % chunk == 0:
            emit(_xsec(generator, accepted, tried, weight_sum))
            data.flush()

    # Sherpa's own summary to the log, then the final figures.
    final = _xsec(generator, accepted, tried, weight_sum)
    try:
        generator.SummarizeRun()
    except Exception:
        pass
    emit(final)
    emit(
        {
            "end": {
                "init_ok": True,
                "accepted": accepted,
                "failed": failed,
                "stopped": STOP["requested"],
            }
        }
    )
    data.flush()
    # A normal exit: the C++ streams Sherpa wrote to are flushed into the
    # log on the way out, and Sherpa's destructor runs once, at the end.
    return 0


if __name__ == "__main__":
    sys.exit(main())
