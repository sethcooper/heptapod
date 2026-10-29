"""
# search.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Searching Pythia 8's settings database: PythiaSettingsSearchTool.
"""

import json
from typing import Any, Dict, List

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .settings_db import brief, get_setting, load_settings, search_settings
from .settings_topics import TOPICS


class PythiaSettingsSearchTool(BaseTool):
    """
    Search Pythia 8's settings database by name, physics term, or topic.

    Pythia declares roughly two thousand run-card settings across its XML
    manual. This tool searches that database -- names, descriptions, option
    lists, and a curated set of physics synonyms -- and returns each hit with
    the type, default, and allowed range needed to write a correct card line.

    Inputs (runtime):
      - query: free text ("beam energy", "pt cut") or an exact setting name
        ("PhaseSpace:pTHatMin"). An exact name returns the full record.
      - topic: optional filter, one of run, beams, lhe, process, cuts, pdf,
        shower, mpi, hadronization, decays, tune, matching, bsm
      - prefix: optional name-prefix filter, e.g. "PhaseSpace:" or "Beams:"
      - limit: maximum hits to return (default 15)
      - detail: "brief" (default) for a hit list, "full" for complete
        descriptions and every option of each hit

    Output (JSON):
      {
        "status": "ok",
        "query": "<query>",
        "pythia_version": "8.317",
        "n_hits": <int>,
        "hits": [
          {"name": "PhaseSpace:pTHatMin", "type": "float", "default": "0.",
           "min": "0.", "summary": "The minimum invariant pT."}
        ]
      }

    Notes:
      - Particle properties (masses, widths, lifetimes, decay switches) are NOT
        settings. They are set as `<pdgid>:<property> = <value>`, e.g.
        `6:m0 = 172.5`, and are not returned by this search.
      - Types, defaults and ranges are those of the installed pythia8mc when
        it is available, and of the bundled Pythia 8.317 database otherwise.
    """

    query: str = RuntimeField(
        description="Setting name or physics phrase to search for"
    )
    topic: str = RuntimeField(
        default="",
        description="Optional topic filter: run, beams, lhe, process, cuts, "
        "pdf, shower, mpi, hadronization, decays, tune, matching, bsm",
    )
    prefix: str = RuntimeField(
        default="",
        description="Optional setting-name prefix filter, e.g. 'Beams:'",
    )
    limit: int = RuntimeField(default=15, description="Maximum number of hits")
    detail: str = RuntimeField(
        default="brief",
        description="'brief' for a hit list, 'full' for complete records",
    )

    base_directory: str = StateField(
        description="Base directory (unused; kept for tool uniformity)"
    )

    def _run(self) -> str:
        if not self.query and not (self.topic or self.prefix):
            return self.format_error(
                error="Missing Parameter",
                reason="query is required (or a topic/prefix filter)",
                suggestion="Provide a setting name or a physics phrase",
            )

        try:
            snapshot = load_settings()
        except Exception as exc:
            return self.format_error(
                error="Database Unavailable",
                reason=str(exc),
                suggestion="Regenerate with `python -m "
                "tools.pythia.settings_db`",
            )

        if self.topic and self.topic.lower() not in TOPICS:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"unknown topic '{self.topic}'",
                context=f"valid topics: {', '.join(TOPICS)}",
                suggestion="Use one of the listed topics, or omit the filter",
            )

        query = (self.query or "").strip()
        exact = get_setting(query, snapshot) if query else None
        full = str(self.detail).lower() == "full"

        hits = search_settings(
            query,
            snapshot=snapshot,
            topic=self.topic,
            prefix=self.prefix,
            limit=max(1, int(self.limit)),
        )

        # An exact name is a lookup, not a search: answer it in full and keep
        # the remaining hits as a brief "related settings" list rather than
        # returning fifteen complete records.
        rendered: List[Dict[str, Any]] = []
        for hit in hits:
            if full or (exact is not None and hit["name"] == exact["name"]):
                rendered.append(_full_record(hit))
            else:
                rendered.append(brief(hit))

        result: Dict[str, Any] = {
            "status": "ok",
            "query": query,
            "pythia_version": snapshot.get("pythia_version"),
            "n_hits": len(hits),
            "hits": rendered,
        }
        if not hits:
            result["suggestion"] = (
                "No settings matched. If you are after a particle property "
                "(mass, width, lifetime, decay on/off), that is not a "
                "setting: "
                "write `<pdgid>:<property> = <value>` in the card instead, "
                "e.g. `6:m0 = 172.5` or `15:mayDecay = off`."
            )
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)


def _full_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        key: value for key, value in record.items() if not key.startswith("_")
    }
    return out
