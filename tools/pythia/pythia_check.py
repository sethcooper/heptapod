"""
# pythia_check.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Checks that ask the installed Pythia rather than a model of it.

The static checks in ``validate`` reimplement Pythia's reading rules, and a
reimplementation can drift from the release that will run the card. Two
questions are therefore put to Pythia itself:

``line_verdicts``
    Does Pythia accept each line of the card? Every line is passed to
    ``Pythia::readString`` in order, on a fresh instance, and its answer and
    any message it prints are recorded. This is exact for the installed
    version -- decay-channel lines, ``:new`` particles and braced vectors
    included -- and needs no ``init()``, so it is cheap.

``dry_run``
    Does the card initialise, and what configuration does Pythia end up
    holding? The effective values are read back from the settings database
    after ``init()``, and compared with what the card asked for: a tune
    applied after a setting silently resets it, and only this shows it.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence

from .settings_db import _LIVE_MAPS, _format_value


def _diagnostics(text: str) -> List[str]:
    """Pythia's own error, abort and warning lines, de-duplicated, in order."""
    from .runtime import DIAGNOSTIC

    out: List[str] = []
    for line in text.splitlines():
        match = DIAGNOSTIC.search(line)
        if match:
            message = line[match.start() :].strip().rstrip(" |").strip()
        elif (
            "PYTHIA Error" in line
            or "PYTHIA Abort" in line
            or "PYTHIA Warning" in line
        ):
            message = line.strip()
        else:
            continue
        if message not in out:
            out.append(message)
    return out


class _Captured:
    """Redirect the C-level stdout/stderr into a file that can be read back
    in pieces, so each call's output can be attributed to that call."""

    def __init__(self) -> None:
        handle = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        handle.close()
        self.path = handle.name
        self._offset = 0

    def __enter__(self) -> "_Captured":
        from .runtime import STDERR, STDOUT, FdSilence

        self._silence = FdSilence(STDOUT, STDERR, target=self.path)
        self._silence.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._silence.__exit__(*exc)
        with contextlib.suppress(OSError):
            os.unlink(self.path)

    def read_new(self) -> str:
        """Everything written since the previous call."""
        from .runtime import flush_std_buffers

        flush_std_buffers()
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self._offset)
            text = fh.read()
            self._offset = fh.tell()
        return text


def _new_pythia() -> Any:
    from .runtime import require_pythia

    return require_pythia().Pythia("", printBanner=False)


def line_verdicts(lines: Sequence[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Feed each line to the installed Pythia's readString, in order.

    Parameters:
        lines: Card statements as Pythia should see them -- a braced value
            spanning several lines already joined, include directives
            already expanded

    Returns:
        {"lines": one {"accepted", "messages"} per line, "has_hard_process":
        Pythia's Settings::hasHardProc() after reading them}, or None when
        pythia8mc is not available
    """
    try:
        pythia = None
        with _Captured() as capture:
            pythia = _new_pythia()
            pythia.readString("Print:quiet = off")  # errors must print
            capture.read_new()  # drop the constructor's output
            verdicts = []
            for line in lines:
                accepted = bool(pythia.readString(line, True))
                verdicts.append(
                    {
                        "accepted": accepted,
                        "messages": _diagnostics(capture.read_new()),
                    }
                )
            has_hard_process = bool(pythia.settings.hasHardProc())
        return {"lines": verdicts, "has_hard_process": has_hard_process}
    except ImportError:
        return None


def effective_values(pythia: Any) -> Dict[str, Dict[str, str]]:
    """Every setting's current and default value, keyed by lowercase name."""
    out: Dict[str, Dict[str, str]] = {}
    for getter, kind in _LIVE_MAPS:
        for key, entry in getattr(pythia.settings, getter)("").items():
            out[key] = {
                "name": entry.name,
                "type": kind,
                "value": _format_value(entry.valNow, kind),
                "default": _format_value(entry.valDefault, kind),
            }
    return out


def dry_run(card_text: str) -> Dict[str, Any]:
    """
    Initialise Pythia on a card and report what it holds afterwards.

    The card is read exactly as written: nothing is forced after it. (An
    earlier version set Print:quiet = off to see Pythia's report, but setting
    that flag resets every Init:show* and Next:* setting, so the dry run
    altered the configuration it was reporting.) Pythia's messages come from
    stat(), which lists every error and warning even under Print:quiet.

    Returns:
        {"ran", "init_ok", "messages", "effective"} -- "effective" maps each
        setting to its value after init -- or {"ran": False, "reason"} when
        pythia8mc is unavailable
    """
    try:
        from .runtime import require_pythia

        require_pythia()
    except ImportError as exc:
        return {
            "ran": False,
            "reason": str(exc),
            "suggestion": "Install pythia8mc to enable dry-run initialisation",
        }

    handle = tempfile.NamedTemporaryFile("w", suffix=".cmnd", delete=False)
    try:
        handle.write(card_text)
        handle.close()
        init_ok: Optional[bool] = None
        exception: Optional[str] = None
        with _Captured() as capture:
            pythia = _new_pythia()
            try:
                pythia.readFile(handle.name)
                init_ok = bool(pythia.init())
            except Exception as exc:  # pragma: no cover - depends on the build
                exception = f"{type(exc).__name__}: {exc}"
            pythia.stat()
            log = capture.read_new()
        out: Dict[str, Any] = {
            "ran": True,
            "init_ok": init_ok,
            "messages": _diagnostics(log),
            "effective": effective_values(pythia),
        }
        if exception:
            out["exception"] = exception
        return out
    finally:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
