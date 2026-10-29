"""
# cards.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

SLHA parameter cards, as overrides for a UFO's external parameters.

A MadGraph ``param_card.dat`` and a model's ``restrict_*.dat`` files are the
same format: ``Block NAME`` sections of ``code value`` lines, plus
``DECAY pdg width`` lines. Every external UFO parameter declares the
(lhablock, lhacode) it is read from, so a card maps onto parameters by that
pair -- never by the comment after the value, which is free text.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .evaluate import parse_parameters

SLHAKey = Tuple[str, Tuple[int, ...]]


def read_slha(path: str) -> Dict[SLHAKey, float]:
    """Read an SLHA card into ``{(BLOCK, codes): value}``.

    Block names are upper-cased, as SLHA blocks are case-insensitive.
    ``DECAY pdg width`` lines become ``("DECAY", (pdg,))``. Decay-table
    lines after a DECAY header (branching ratios) are skipped: they are
    not parameters. ``QNUMBERS`` blocks are skipped for the same reason.

    Raises:
        OSError: the card cannot be read.
        ValueError: a line that should hold a number does not, naming it.
    """
    values: Dict[SLHAKey, float] = {}
    block = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            head = fields[0].upper()
            if head == "BLOCK":
                block = fields[1].upper() if len(fields) > 1 else None
                if block == "QNUMBERS":
                    block = None
                continue
            if head == "DECAY":
                if len(fields) < 3:
                    raise ValueError(f"{path}:{n}: DECAY line needs a PDG code and a width")
                values[("DECAY", (int(fields[1]),))] = float(fields[2])
                block = None
                continue
            if block is None:
                continue
            try:
                *codes, value = fields
                values[(block, tuple(int(c) for c in codes))] = float(value)
            except ValueError as exc:
                raise ValueError(f"{path}:{n}: cannot read {raw.strip()!r} in block {block}") from exc
    return values


def card_overrides(ufo_dir: str, card_path: str) -> Tuple[Dict[str, float], List[str]]:
    """External-parameter overrides for a UFO, read from an SLHA card.

    Args:
        ufo_dir: The UFO directory.
        card_path: A param_card or restriction card.

    Returns:
        ``(overrides, missing)`` -- ``{parameter_var: value}`` for every
        external parameter the card sets, and the external parameters the
        card does not mention (they keep their UFO defaults).
    """
    card = read_slha(card_path)
    overrides: Dict[str, float] = {}
    missing: List[str] = []
    for p in parse_parameters(ufo_dir):
        if p.get("nature") != "external" or "lhablock" not in p:
            continue
        key = (str(p["lhablock"]).upper(), tuple(int(c) for c in p.get("lhacode", [])))
        if key in card:
            overrides[p["var"]] = card[key]
        else:
            missing.append(p["var"])
    return overrides, missing
