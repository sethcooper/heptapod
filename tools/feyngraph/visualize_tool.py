"""
# visualize_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

VisualizeDiagrams tool for generating Feynman diagram visualizations.

This tool uses FeynGraph's native SVG drawing to visualize diagrams.
"""

import json
import os
from typing import List, Optional
from pathlib import Path

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .feyngraph_interface import FeynGraphInterface


class VisualizeDiagramsTool(BaseTool):
    """
    Generate Feynman diagram visualizations using FeynGraph's native SVG drawing.

    This tool enumerates diagrams for a specified process and saves them as SVG
    images to the sandbox directory. Uses FeynGraph's native drawing capabilities
    for accurate physics diagram representation.

    Inputs (runtime):
        initial: List of initial state particle labels
                 Examples: ["H"], ["mu-"], ["e+", "e-"]

        final: List of final state particle labels
               Examples: ["b", "bbar"], ["gamma", "gamma"]

        max_loops: Maximum loop order (default 0 = tree level)
                   Set to 1 for 1-loop diagrams

        model: Physics model (default "SM" for Standard Model)

        output_prefix: Prefix for output files (default "diagram")
                      Files will be named: <prefix>_0.svg, <prefix>_1.svg, etc.

        max_diagrams: Maximum number of diagrams to save (default 10)
                     Set to None to save all diagrams

    Output (JSON):
        {
            "status": "ok",
            "n_diagrams": 3,
            "n_saved": 3,
            "process": "H -> gamma gamma",
            "files": ["diagram_0.svg", ...],
            "output_directory": "/path/to/sandbox"
        }

    Examples:
        VisualizeDiagramsTool(initial=["H"], final=["b", "bbar"])
        VisualizeDiagramsTool(initial=["H"], final=["gamma", "gamma"], max_loops=1)
    """

    # ======================== Runtime fields ======================== #
    initial: List[str] = RuntimeField(
        description='Initial state particles, e.g., ["H"]'
    )
    final: List[str] = RuntimeField(
        description='Final state particles, e.g., ["gamma", "gamma"]'
    )
    max_loops: int = RuntimeField(
        default=0,
        description="Maximum loop order (0=tree, 1=1-loop)"
    )
    model: str = RuntimeField(
        default="SM",
        description="SM uses FeynGraph's built-in local model; otherwise an explicit UFO directory (relative to workspace)"
    )
    output_prefix: str = RuntimeField(
        default="diagram",
        description="Prefix for output SVG files"
    )
    max_diagrams: Optional[int] = RuntimeField(
        default=10,
        description="Maximum diagrams to save (None=all)"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Sandbox directory for saving files"
    )
    # ================================================================ #

    def _run(self) -> str:
        """Main execution method."""
        if not self.initial:
            return self.format_error(
                error="Empty Initial State",
                reason="initial list must contain at least one particle",
                suggestion='Provide initial state, e.g., initial=["H"]'
            )

        if not self.final:
            return self.format_error(
                error="Empty Final State",
                reason="final list must contain at least one particle",
                suggestion='Provide final state, e.g., final=["gamma", "gamma"]'
            )

        output_dir = Path(self.base_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            fg_interface = FeynGraphInterface(model=self.model, base_directory=self.base_directory)
        except ImportError as e:
            return self.format_error(
                error="FeynGraph Not Available",
                reason=str(e),
                suggestion="Install FeynGraph for diagram visualization. pip install feyngraph"
            )
        except ValueError as e:
            return self.format_error(
                error="Model Error",
                reason=str(e),
                suggestion='Use model="SM" for Standard Model'
            )

        try:
            fg_diagrams = fg_interface.enumerate_diagrams(
                initial_particles=self.initial,
                final_particles=self.final,
                max_loop_order=self.max_loops
            )
        except ValueError as e:
            return self.format_error(
                error="Invalid Particles",
                reason=str(e),
                suggestion="Check particle labels (e.g., 'b', 'bbar', 'gamma')"
            )
        except RuntimeError as e:
            return self.format_error(
                error="FeynGraph Error",
                reason=str(e),
                suggestion="Check that the process is allowed in the model"
            )

        if not fg_diagrams:
            return self.format_error(
                error="No Diagrams Found",
                reason=f"No diagrams for {self.initial} -> {self.final}",
                suggestion="For loop processes, increase max_loops"
            )

        n_total = len(fg_diagrams)
        if self.max_diagrams is not None:
            fg_diagrams = fg_diagrams[:self.max_diagrams]

        saved_files = []
        errors = []

        for i, fg_diagram in enumerate(fg_diagrams):
            filename = f"{self.output_prefix}_{i}.svg"
            filepath = output_dir / filename

            try:
                if hasattr(fg_diagram, 'draw_svg'):
                    fg_diagram.draw_svg(str(filepath))
                    saved_files.append(filename)
                else:
                    errors.append(f"Diagram {i}: No draw_svg method")
            except Exception as e:
                errors.append(f"Diagram {i}: {str(e)}")

        result = {
            "status": "ok" if saved_files else "error",
            "n_diagrams": n_total,
            "n_saved": len(saved_files),
            "process": f"{' '.join(self.initial)} -> {' '.join(self.final)}",
            "max_loops": self.max_loops,
            "files": saved_files,
            "output_directory": str(output_dir),
        }

        if errors:
            result["errors"] = errors

        if saved_files:
            result["message"] = (
                f"Saved {len(saved_files)} diagram(s) to {output_dir}. "
                f"Files: {', '.join(saved_files)}"
            )
        else:
            result["message"] = "No diagrams could be saved."

        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
