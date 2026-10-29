"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Rivet: generator predictions against published measurements.

    RivetFindAnalysesTool   which analyses fit a measurement, a process, a run
    RunRivetTool            run them on HepMC3 events -> YODA histograms
    RivetCompareTool        chi2, pulls and p-values against the data
    RivetPlotTool           the plots: data, predictions, variations, ratio
    RivetMergeTool          combine the outputs of independent runs

Everything runs through a local Rivet installation (`rivet_path`), as the
Delphes bundle runs through a local Delphes build.
"""

from .analyses import RivetFindAnalysesTool
from .compare import RivetCompareTool
from .merge import RivetMergeTool
from .plot import RivetPlotTool
from .run import RunRivetTool

__all__ = [
    "RivetFindAnalysesTool",
    "RunRivetTool",
    "RivetCompareTool",
    "RivetPlotTool",
    "RivetMergeTool",
]
