"""
# lorentz.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Static reading of a UFO's ``lorentz.py``.

Each ``Lorentz(...)`` declaration names a spin structure, the spins of the
legs it applies to, and the structure itself as a string in the UFO's
tensor language (``Gamma(3,2,-1)*ProjM(-1,1)``). The string is returned
as written; interpreting it is the consumer's business.

Parsed with ``ast`` and never imported, like the rest of this package:
``lorentz.py`` imports the model's object library.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict, List


def parse_lorentz(ufo_dir: str) -> Dict[str, Dict[str, Any]]:
    """Every Lorentz structure declared in a UFO.

    Args:
        ufo_dir: The UFO directory.

    Returns:
        ``{variable_name: {"name", "spins", "structure"}}``, keyed by the
        Python variable a vertex cites (``L.FFV1`` -> ``"FFV1"``). ``spins``
        is the list of 2s+1 values, ``structure`` the tensor string. Empty
        when the file is missing or unparsable.
    """
    path = os.path.join(ufo_dir, "lorentz.py")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for stmt in tree.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call)):
            continue
        func = stmt.value.func
        fname = getattr(func, "id", None) or getattr(func, "attr", None)
        if fname != "Lorentz":
            continue
        kws = {kw.arg: kw.value for kw in stmt.value.keywords if kw.arg}

        spins: List[int] = []
        if isinstance(kws.get("spins"), ast.List):
            for e in kws["spins"].elts:
                try:
                    spins.append(int(ast.literal_eval(e)))
                except (ValueError, TypeError):
                    spins = []
                    break

        structure = kws.get("structure")
        try:
            structure = ast.literal_eval(structure) if structure is not None else None
        except ValueError:
            structure = None
        if not isinstance(structure, str):
            continue

        var = stmt.targets[0].id
        name = kws.get("name")
        try:
            name = ast.literal_eval(name) if name is not None else var
        except ValueError:
            name = var
        out[var] = {"name": name, "spins": spins, "structure": structure}
    return out
