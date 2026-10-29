"""
# config_resolve.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Where an external tool's path comes from.

`config.py` is per-clone and gitignored, so a fresh checkout has only
`config.example.py` and its placeholders -- and a test suite that reads
`config.py` directly reports "not configured" on every machine but the one
where somebody filled it in. Meanwhile toolbase already stores exactly these
paths for the same toolkit, set once with `tb config set heptapod
feynrules_path ...`, and keeps them out of the repository by construction.

So paths resolve through a chain rather than a file:

    1. environment    FEYNRULES_PATH, WOLFRAMSCRIPT_PATH, MG5_PATH.
                      CI sets these; they override everything.
    2. config.py      a real value, i.e. anything that is not the
                      placeholder `config.example.py` ships.
    3. toolbase       the effective config for this toolkit, project scope
                      over user scope, exactly as `tb serve` resolves it.
    4. None           unresolvable. A caller decides whether that is a skip
                      or an error; this module does not.

Nothing here raises on a missing toolbase: the toolkit has to work for
someone who cloned it and never installed one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

#: Paths this module knows how to resolve, and the environment variable for
#: each. The names match both `config.py`'s and toolbase's field names, which
#: is what lets one key address all three sources.
PATH_KEYS: Dict[str, str] = {
    "feynrules_path": "FEYNRULES_PATH",
    "wolframscript_path": "WOLFRAMSCRIPT_PATH",
    "mg5_path": "MG5_PATH",
    "delphes_path": "DELPHES_PATH",
    "pythia_path": "PYTHIA_PATH",
    "sherpa_path": "SHERPA_PATH",
    "rivet_path": "RIVET_PATH",
}

#: What `config.example.py` ships. A value still equal to one of these means
#: the field was never filled in, not that the path is genuinely "/path/to".
_PLACEHOLDERS = {
    "/path/to/FeynRules", "/path/to/wolframscript", "/path/to/MG5_aMC",
    "/path/to/Delphes", "/path/to/pythia", "/path/to/Sherpa", "/path/to/rivet",
    "", None,
}


def _from_config_py(key: str) -> Optional[str]:
    """The value in a local `config.py`, if it holds a real one."""
    try:
        import config  # noqa: PLC0415 - optional, per-clone, may not exist
    except Exception:
        return None
    val = getattr(config, key, None)
    if not isinstance(val, str) or val in _PLACEHOLDERS:
        return None
    return val


def _from_toolbase(key: str) -> Optional[str]:
    """The value toolbase resolves for this toolkit, project over user scope.

    Best-effort by design: toolbase is how heptapod is usually served, but a
    clone without it must still be able to run its own tests.
    """
    try:
        from toolbase.envs.config import resolve_toolkit_config  # noqa: PLC0415
        from toolbase.envs.discovery import (  # noqa: PLC0415
            project_root_or_default,
        )
    except Exception:
        return None
    try:
        # `project_root_or_default` falls back to toolbase's default project
        # when the cwd is not inside one -- which is the usual case here, since
        # the toolkit checkout is not itself a toolbase project. That default
        # is where user-scope config lives, so the lookup still works.
        root = project_root_or_default()
    except Exception:
        return None
    try:
        cfg = resolve_toolkit_config("heptapod", root) or {}
    except Exception:
        return None
    val = cfg.get(key)
    return val if isinstance(val, str) and val not in _PLACEHOLDERS else None


def resolve_path(key: str) -> Optional[str]:
    """Resolve one external-tool path, or None.

    Args:
        key: One of :data:`PATH_KEYS` (``feynrules_path`` etc.).

    Returns:
        The first value found in the chain described in the module docstring,
        or None when no source supplies one.
    """
    env = PATH_KEYS.get(key)
    if env:
        val = os.environ.get(env)
        if val:
            return val
    return _from_config_py(key) or _from_toolbase(key)


def resolve_paths(*keys: str) -> Dict[str, Optional[str]]:
    """Resolve several paths at once. Convenience over :func:`resolve_path`."""
    return {k: resolve_path(k) for k in (keys or PATH_KEYS)}


def describe_sources() -> str:
    """One line per key saying where its value came from, for a test banner.

    A suite that skips should say WHY, and "unset" is a different problem from
    "set to a path that does not exist".
    """
    lines = []
    for key in PATH_KEYS:
        env = PATH_KEYS[key]
        if os.environ.get(env):
            src, val = f"${env}", os.environ[env]
        elif _from_config_py(key):
            src, val = "config.py", _from_config_py(key)
        elif _from_toolbase(key):
            src, val = "toolbase", _from_toolbase(key)
        else:
            src, val = "unset", None
        exists = "" if val is None else ("" if Path(val).exists() else "  [MISSING ON DISK]")
        lines.append(f"  {key:20s} {src:10s} {val or '-'}{exists}")
    return "\n".join(lines)
