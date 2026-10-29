"""
# card.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Reading and preparing a Sherpa run card.

A Sherpa 3 run card is YAML: top-level settings such as ``BEAMS`` and
``BEAM_ENERGIES``, blocks such as ``SHOWER`` and ``HARD_DECAYS``, and the
``PROCESSES`` list whose entries are process strings (``93 93 -> 11 -11``)
mapping to their options. This module parses a card with line numbers for
the validator, reads the beams out of it, and prepares the copy a run uses:
the card as given, with the model lines a UFO model needs appended. Seeds
and event counts are not written into the card; Sherpa takes them on the
command line (``-R``, ``-e``), which both backends pass.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

#: The process string: incoming particles, an arrow, outgoing particles, each
#: a particle id (PDG code or Sherpa container such as 93) with an optional
#: multiplicity suffix ``{n}`` for "up to n more".
PROCESS = re.compile(
    r"^\s*(?P<in>(?:-?\d+\s+)*-?\d+)\s*->\s*"
    r"(?P<out>(?:-?\d+(?:\{\d+\})?\s*)+)$"
)
#: Sherpa's particle containers (``share/SHERPA-MC`` lists them at start-up).
CONTAINERS = {
    90: "lepton",
    91: "neutrino",
    92: "fermion",
    93: "jet (light quarks and gluon)",
    94: "light quark",
    95: "jet plus photon",
    98: "ewj",
}


class CardError(ValueError):
    """The card cannot be read as YAML, with the line that failed."""

    def __init__(self, message: str, line: Optional[int] = None) -> None:
        super().__init__(message)
        self.line = line


def _require_yaml() -> Any:
    try:
        import yaml

        return yaml
    except Exception as e:
        raise ImportError(
            "pyyaml is needed to read Sherpa run cards (pip install pyyaml)"
        ) from e


def parse_card(text: str) -> Dict[str, Any]:
    """
    The card as a mapping of top-level settings.

    Raises:
        CardError: when the text is not valid YAML or not a mapping
    """
    yaml = _require_yaml()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        line = mark.line + 1 if mark is not None else None
        problem = getattr(e, "problem", None) or str(e)
        raise CardError(f"YAML syntax: {problem}", line) from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise CardError(
            "a Sherpa card is a YAML mapping of settings, not a "
            f"{type(data).__name__}"
        )
    return data


def key_lines(text: str) -> Dict[str, int]:
    """
    The line on which each top-level key starts, for reporting.

    YAML's own parser does not keep positions, so this is a scan for lines
    that begin at column zero with ``KEY:``.
    """
    out: Dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^([A-Za-z0-9_/()@.-]+)\s*:", line)
        if match and match.group(1) not in out:
            out[match.group(1)] = number
    return out


def parse_process(spec: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Split a process string into its incoming ids and its outgoing
    ``(id, extra)`` pairs, where ``extra`` is the ``{n}`` multiplicity.

    Raises:
        ValueError: when the string is not of the form ``a b -> c d{n}``
    """
    match = PROCESS.match(spec)
    if not match:
        raise ValueError(
            f"'{spec}' is not a process: expected 'a b -> c d', with PDG "
            "ids or containers such as 93 and an optional {n} after an "
            "outgoing particle"
        )
    incoming = [int(tok) for tok in match.group("in").split()]
    outgoing: List[Tuple[int, int]] = []
    for tok in re.findall(r"-?\d+(?:\{\d+\})?", match.group("out")):
        if "{" in tok:
            pid, extra = tok.rstrip("}").split("{")
            outgoing.append((int(pid), int(extra)))
        else:
            outgoing.append((int(tok), 0))
    return incoming, outgoing


def prepare_card(text: str, *, ufo_model: Optional[str] = None) -> str:
    """
    The card text a run uses.

    The card is kept as written. A UFO model adds the ``MODEL`` and
    ``UFO_PARAM_CARD`` lines Sherpa needs to load it, under a comment that
    says the tool put them there. The seed and the event count are passed on
    the command line, so the card never says anything the card author did
    not write.
    """
    out = text.rstrip("\n") + "\n"
    if ufo_model:
        name = os.path.basename(ufo_model.rstrip("/"))
        out += (
            "\n# ---- Set by SherpaFromRunCardTool for the UFO model ----\n"
            f"MODEL: {name}\n"
            f"UFO_PARAM_CARD: param_{name}.dat\n"
        )
    return out


def has_variations(card: Dict[str, Any]) -> bool:
    """Whether the card asks for weight variations, which only the C++
    driver can store."""
    return any(
        key in card and card[key]
        for key in (
            "SCALE_VARIATIONS",
            "PDF_VARIATIONS",
            "QCUT_VARIATIONS",
            "ASSOCIATED_CONTRIBUTIONS_VARIATIONS",
        )
    )
