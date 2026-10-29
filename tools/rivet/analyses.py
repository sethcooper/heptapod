"""
# analyses.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Find the Rivet analyses that fit a measurement, a process or a planned
run.
"""

import json
from typing import Optional

from orchestral.tools.base.field_utils import RuntimeField

from tools.rivet.base import RivetToolBase
from tools.rivet.catalogue import (
    caution,
    load_catalogue,
    plot_style,
    plot_styles,
    reference_paths,
)
from tools.rivet.install import RivetError


class RivetFindAnalysesTool(RivetToolBase):
    """
    Search the analyses the Rivet installation ships, or describe one.

    Rivet ships about two thousand analyses, each a published measurement
    (or a Monte Carlo validation study, the MC_* ones) with the reference
    data to compare against. Searching by what was measured and filtering by
    the run you can produce -- beams and sqrt(s) -- is how to find the ones
    worth running: an analysis measured at 7 TeV run on 13 TeV events fills
    its histograms with the wrong physics, and Rivet does not stop it.

    Inputs (runtime), all optional:
      - query: words that must all occur in the name, summary, description,
        keywords, experiment, collider or INSPIRE id ("Z transverse momentum",
        "jet shapes", "1204784"). LaTeX markup in the metadata is ignored.
      - name: one analysis to describe in full, with the process to generate
        (run_info), its options, references, histograms and plot titles.
        Overrides the search.
      - beams, energy_gev: keep analyses compatible with such a run. Beams as
        two names ("p+ p+", "e+ e-", "p+ Pb") or PDG ids; energy is sqrt(s).
      - experiment, collider, status, keyword, year_min, year_max,
        with_reference_data: further filters. status is VALIDATED,
        UNVALIDATED, PRELIMINARY or OBSOLETE.
      - max_results: at most this many (default 25).

    Output (JSON): the matching analyses, best first, each with name,
    summary, experiment, collider, year, status, beams, energies, whether
    reference data ships, whether a cross section is needed, and options.
    With `name`, the full record plus "histograms" (path, type, title) and
    any "cautions" about using it.
    """

    query: str = RuntimeField(
        default="",
        description="Words that must all occur in the analysis metadata, e.g. "
        "'Z transverse momentum'",
    )
    name: str = RuntimeField(
        default="",
        description="Describe this one analysis in full (e.g. "
        "'ATLAS_2012_I1204784'); overrides the search",
    )
    beams: str = RuntimeField(
        default="",
        description="Keep analyses for these beams: two names or PDG ids, "
        "e.g. 'p+ p+', 'e+ e-', '2212 2212'",
    )
    energy_gev: float = RuntimeField(
        default=-1.0,
        description="Keep analyses measured at this sqrt(s) in GeV "
        "(within 1%)",
    )
    experiment: str = RuntimeField(
        default="",
        description="Experiment, e.g. ATLAS, CMS, LHCb, ALICE, CDF, ALEPH",
    )
    collider: str = RuntimeField(
        default="", description="Collider, e.g. LHC, Tevatron, LEP, HERA, RHIC"
    )
    status: str = RuntimeField(
        default="",
        description="VALIDATED, UNVALIDATED, PRELIMINARY or OBSOLETE",
    )
    keyword: str = RuntimeField(
        default="", description="A Rivet keyword, e.g. 'Jets', 'Z'"
    )
    with_reference_data: bool = RuntimeField(
        default=False,
        description="Only analyses that ship the measured data to compare "
        "against",
    )
    year_min: int = RuntimeField(
        default=0, description="Earliest publication year"
    )
    year_max: int = RuntimeField(
        default=0, description="Latest publication year"
    )
    max_results: int = RuntimeField(
        default=25, description="At most this many results"
    )

    def _run(self) -> str:
        try:
            install = self._install()
            catalogue = load_catalogue(install.data_dir)
        except (RivetError, FileNotFoundError) as e:
            return self.format_error(
                error="Rivet Not Available", reason=str(e)
            )

        if self.name:
            analysis = catalogue.get(self.name)
            if analysis is None:
                return self.format_error(
                    error="Unknown Analysis",
                    reason=f"'{self.name}' is not an analysis of Rivet "
                    f"{install.version}",
                    suggestion=(
                        "Did you mean: "
                        + ", ".join(catalogue.suggest(self.name))
                    )
                    if catalogue.suggest(self.name)
                    else "Search with `query` instead",
                )
            record = analysis.to_dict(detail=True)
            styles = plot_styles(install.data_dir, analysis.name)
            record["histograms"] = [
                {
                    "path": path.replace("/REF/", "/", 1),
                    "type": kind.lower(),
                    "title": plot_style(
                        styles, path.replace("/REF/", "/", 1)
                    ).get("Title", ""),
                }
                for path, kind in reference_paths(
                    install.data_dir, analysis.name
                )
            ]
            record["cautions"] = caution(analysis)
            record["rivet_version"] = install.version
            return json.dumps(record, ensure_ascii=False)

        energy: Optional[float] = (
            float(self.energy_gev) if self.energy_gev > 0 else None
        )
        try:
            matches = catalogue.search(
                self.query,
                experiment=self.experiment,
                collider=self.collider,
                beams=self.beams or None,
                energy_gev=energy,
                status=self.status,
                keyword=self.keyword,
                with_reference_data=True if self.with_reference_data else None,
                year_min=self.year_min or None,
                year_max=self.year_max or None,
            )
        except ValueError as e:
            return self.format_error(
                error="Invalid Parameter",
                reason=str(e),
                suggestion="Give beams as two particles, e.g. 'p+ p+'",
            )
        limit = max(1, int(self.max_results))
        result = {
            "rivet_version": install.version,
            "n_analyses": len(catalogue),
            "n_matches": len(matches),
            "analyses": [a.to_dict() for a in matches[:limit]],
        }
        if len(matches) > limit:
            result["note"] = (
                f"{len(matches) - limit} more match; narrow the query or "
                f"raise max_results"
            )
        if not matches:
            result["note"] = (
                "Nothing matches every criterion. Loosen the query, or drop a "
                "filter: "
                "many analyses list no beams or energies in their metadata "
                "and "
                "only match when neither is given."
            )
        return json.dumps(result, ensure_ascii=False)
