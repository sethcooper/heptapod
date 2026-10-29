"""
# enumerate_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

EnumerateDiagrams tool for automatic Feynman diagram enumeration.

This tool wraps FeynGraph to enumerate all Feynman diagrams for a given process,
rank them by physics importance, and provide diagram specifications for NDA analysis.
"""

import json
import re
from typing import List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .feyngraph_interface import FeynGraphInterface
from .diagram_converter import DiagramConverter
from .ranking import DiagramRanker
from tools.logging.findings import append_finding


class EnumerateDiagramsTool(BaseTool):
    """
    Enumerate all Feynman diagrams for a process using FeynGraph.

    This tool uses FeynGraph to automatically enumerate all Feynman diagrams
    contributing to a given process, then ranks them by physics importance
    and provides diagram specifications for each diagram.

    The output includes for each diagram:
    - rank: Importance ranking (1 = most important)
    - diagram: Full diagram dict for EstimateDecayWidthNDATool
    - topology: Diagram topology (tree_2body, loop_2body_triangle, etc.)
    - explanation: Physics reasoning for the ranking
    - couplings: Coupling constants involved

    Inputs (runtime):
        initial: List of initial state particle labels (HEPTAPOD notation)
                 Examples: ["H"], ["mu-"], ["e+", "e-"]

        final: List of final state particle labels
               Examples: ["b", "bbar"], ["e-", "nu_ebar", "nu_mu"]

        max_loops: Maximum loop order (default 0 = tree level)
                   Set to 1 for 1-loop diagrams, 2 for 2-loop, etc.

        model: Physics model (default "SM" for Standard Model)
               Can also be a path to UFO model directory

        save_diagrams: Save all diagrams in group folders with JSON+SVG pairs (default True)
                       Structure: heavy_N/diagram_XXX_rankYY.{json,svg} + summary.md

        visualize: Generate diagram visualizations alongside each JSON (default True)

        output_format: Visualization format — 'svg' (default), 'tikz' (LaTeX-native), or 'both'

        max_visualize: Max diagrams to visualize (default 10, 0=all). All JSONs always saved.

    Output (JSON):
        {
            "status": "ok",
            "n_diagrams": 3,
            "process": "H -> b bbar",
            "diagrams": [...],
            "summary": "Found 3 diagrams. Tree-level Yukawa dominates.",
            "saved_files": {...}
        }

    Each diagram in the output can be passed directly to `EstimateDecayWidthNDA`
    or `ComputeSymbolicAmplitude` via the `diagrams[i].diagram` field. For
    branching ratio calculations across multiple diagram classes, group diagrams
    by topology or propagator count and pass to `EstimateBranchingRatioNDA`.

    Examples:
        # Higgs to bottom quarks
        EnumerateDiagramsTool(initial=["H"], final=["b", "bbar"])

        # Muon decay
        EnumerateDiagramsTool(initial=["mu-"], final=["e-", "nu_ebar", "nu_mu"])

        # Higgs to photons (with loops)
        EnumerateDiagramsTool(initial=["H"], final=["gamma", "gamma"], max_loops=1)
    """

    # ======================== Runtime fields ======================== #
    initial: List[str] = RuntimeField(
        description='Initial state particles, e.g., ["H"] or ["mu-"]'
    )
    final: List[str] = RuntimeField(
        description='Final state particles, e.g., ["b", "bbar"]'
    )
    max_loops: int = RuntimeField(
        default=0,
        description="Maximum loop order (0=tree, 1=1-loop, etc.)"
    )
    model: str = RuntimeField(
        default="SM",
        description="SM uses FeynGraph's built-in local model; otherwise an explicit UFO directory (relative to workspace)"
    )
    save_diagrams: bool = RuntimeField(
        default=True,
        description="Save all diagram JSON files (individual + grouped by n_heavy_propagators) and summary"
    )
    visualize: bool = RuntimeField(
        default=True,
        description="Save Feynman diagram visualizations to sandbox directory"
    )
    output_format: str = RuntimeField(
        default="svg",
        description=(
            "Visualization format: 'svg' (default), 'tikz' (LaTeX-native via \\input{}), "
            "or 'both'. TikZ is recommended for LaTeX documents."
        )
    )
    max_visualize: int = RuntimeField(
        default=10,
        description="Maximum number of diagrams to visualize (0=all). Only applies if visualize=True"
    )
    metadata_only: bool = RuntimeField(
        default=False,
        description=(
            "Return only diagram counts and classification (no files written). "
            "Useful for large enumerations where you only need the class structure."
        )
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for saving diagrams"
    )
    # ================================================================ #

    def _run(self) -> str:
        """Main execution method."""
        # Validate inputs
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
                suggestion='Provide final state, e.g., final=["b", "bbar"]'
            )

        if self.max_loops < 0:
            return self.format_error(
                error="Invalid Loop Order",
                reason=f"max_loops must be non-negative, got {self.max_loops}",
                suggestion="Use max_loops=0 for tree-level, 1 for 1-loop, etc."
            )

        # Initialize FeynGraph interface
        try:
            fg_interface = FeynGraphInterface(model=self.model, base_directory=self.base_directory)
        except ImportError as e:
            return self.format_error(
                error="FeynGraph Not Available",
                reason=str(e),
                suggestion="Install FeynGraph to use automatic diagram enumeration."
            )
        except ValueError as e:
            return self.format_error(
                error="Model Error",
                reason=str(e),
                suggestion='Use model="SM" for Standard Model or provide valid UFO path'
            )

        # Enumerate diagrams
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
                suggestion="Check particle labels. Use HEPTAPOD notation (e.g., 'b', 'bbar', 'gamma')"
            )
        except RuntimeError as e:
            return self.format_error(
                error="FeynGraph Error",
                reason=str(e),
                suggestion="Check that the process is allowed in the specified model"
            )

        if not fg_diagrams:
            return self.format_error(
                error="No Diagrams Found",
                reason=f"FeynGraph found no diagrams for {self.initial} -> {self.final}",
                suggestion=(
                    "Check that the process is allowed in the model. "
                    "For loop diagrams, increase max_loops."
                )
            )

        # ------------------------------------------------------------------
        # Lightweight path: metadata_only
        # Classify at the FeynGraph object level — no full Diagram conversion.
        # Only converts one representative per class.
        # ------------------------------------------------------------------
        if self.metadata_only:
            return self._run_lightweight(fg_diagrams, fg_interface)

        # ------------------------------------------------------------------
        # Full path: convert all diagrams, rank, optionally save to disk
        # ------------------------------------------------------------------

        # Convert to HEPTAPOD diagrams
        converter = DiagramConverter(strict_mode=False, auto_infer_qn=True)
        heptapod_diagrams = []
        fg_diagram_map = []

        for fg_diagram in fg_diagrams:
            try:
                diagram = converter.convert(fg_diagram)
                heptapod_diagrams.append(diagram)
                fg_diagram_map.append(fg_diagram)
            except Exception:
                continue

        if not heptapod_diagrams:
            return self.format_error(
                error="Conversion Failed",
                reason="Failed to convert FeynGraph diagrams to HEPTAPOD format",
                suggestion="This may indicate an unsupported diagram topology"
            )

        # Rank diagrams
        ranker = DiagramRanker(calculate_nda=False, explain=True)
        ranked = ranker.rank(heptapod_diagrams)

        # Build output
        result = {
            "status": "ok",
            "n_diagrams": len(ranked),
            "process": f"{' '.join(self.initial)} -> {' '.join(self.final)}",
            "max_loops": self.max_loops,
        }

        # Build diagram list for saving
        diagrams_for_saving = []
        for rd in ranked:
            diagram_info = rd.to_dict()
            diagram_info["process_string"] = self._diagram_to_process_string(rd.diagram)
            diagrams_for_saving.append(diagram_info)

        # Add summary
        result["summary"] = self._generate_summary(ranked)

        if self.save_diagrams and self.base_directory:
            # Save diagrams to disk
            saved_files = self._save_diagrams_by_group(
                diagrams_for_saving, ranked, fg_diagram_map, heptapod_diagrams
            )
            result["saved_files"] = saved_files
            if "summary_file" in saved_files:
                result["summary_file"] = saved_files["summary_file"]

        # Append findings (best-effort)
        try:
            process_str = f"{' '.join(self.initial)} → {' '.join(self.final)}"
            n_tree = sum(1 for r in ranked if r.ranking_info.loop_order == 0)
            n_loop = len(ranked) - n_tree
            entries = [f"Found {len(ranked)} diagram(s)"]
            if n_tree > 0:
                entries.append(f"Tree-level: {n_tree}")
            if n_loop > 0:
                entries.append(f"Loop: {n_loop}")
            # List unique topologies
            topos = set()
            for r in ranked:
                if hasattr(r.ranking_info, 'topology') and r.ranking_info.topology:
                    topos.add(r.ranking_info.topology)
            if topos:
                entries.append(f"Topologies: {', '.join(sorted(topos))}")
            append_finding(self.base_directory, f"Diagrams: {process_str}", entries)
        except Exception:
            pass

        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

    # ------------------------------------------------------------------
    # Lightweight classification helpers
    # ------------------------------------------------------------------

    # FeynGraph particle names for heavy propagators (mass > 10 GeV)
    _HEAVY_PROPAGATOR_PARTICLES = {"W+", "W-", "Z", "H", "t", "t~"}

    def _count_heavy_propagators_fg(self, fg_diagram) -> int:
        """Count heavy propagators directly from a FeynGraph diagram object.

        Uses particle names instead of masses — avoids full Diagram conversion.
        """
        n_heavy = 0
        for prop in fg_diagram.propagators():
            name = prop.particle().name()
            if name in self._HEAVY_PROPAGATOR_PARTICLES:
                n_heavy += 1
        return n_heavy

    def _run_lightweight(self, fg_diagrams: list, fg_interface: FeynGraphInterface) -> str:
        """Lightweight path: classify FeynGraph diagrams without full conversion.

        Groups all diagrams by heavy propagator count, converts only one
        representative per class through DiagramConverter, and returns class
        counts + representative diagrams.  No files written to disk.
        """
        # Classify all FeynGraph diagrams by heavy propagator count
        groups: dict[int, list] = {}  # n_heavy -> [fg_diagram, ...]
        for fg_d in fg_diagrams:
            n_heavy = self._count_heavy_propagators_fg(fg_d)
            if n_heavy not in groups:
                groups[n_heavy] = []
            groups[n_heavy].append(fg_d)

        # Convert one representative per class (the first diagram in each group)
        converter = DiagramConverter(strict_mode=False, auto_infer_qn=True)
        classes = []
        # Keep Diagram objects for compact serialization in br_classes
        representative_objects: dict[int, object] = {}  # n_heavy -> Diagram
        for n_heavy in sorted(groups.keys()):
            members = groups[n_heavy]
            # Convert only the representative
            representative_diagram = None
            for fg_d in members:
                try:
                    hep_diagram = converter.convert(fg_d)
                    ranker = DiagramRanker(calculate_nda=False, explain=False)
                    ranked = ranker.rank([hep_diagram])
                    if ranked:
                        representative_diagram = ranked[0].to_dict().get("diagram")
                        representative_objects[n_heavy] = ranked[0].diagram
                    break
                except Exception:
                    continue

            class_info = {
                "n_heavy_propagators": n_heavy,
                "n_diagrams": len(members),
            }
            if representative_diagram is not None:
                class_info["representative_diagram"] = representative_diagram
            classes.append(class_info)

        total_diagrams = sum(c["n_diagrams"] for c in classes)

        # Build BR-ready classes array for direct use with EstimateBranchingRatioNDA.
        # Uses compact=True to omit quantum numbers, cutting ~60% of token cost.
        br_classes = []
        for c in classes:
            n_h = c["n_heavy_propagators"]
            diag_obj = representative_objects.get(n_h)
            if diag_obj is not None:
                br_classes.append({
                    "diagram": diag_obj.to_dict(compact=True),
                    "n_diagrams": c["n_diagrams"],
                    "n_heavy": n_h,
                })

        result = {
            "status": "ok",
            "n_diagrams": total_diagrams,
            "process": f"{' '.join(self.initial)} -> {' '.join(self.final)}",
            "max_loops": self.max_loops,
            "metadata_only": True,
            "summary": f"Found {total_diagrams} diagram(s) in {len(classes)} class(es).",
            "classes": classes,
        }
        if br_classes:
            result["br_classes"] = br_classes

        # Append findings (best-effort)
        try:
            process_str = f"{' '.join(self.initial)} → {' '.join(self.final)}"
            entries = [f"Found {total_diagrams} diagram(s) (lightweight)"]
            for c in classes:
                entries.append(
                    f"heavy_{c['n_heavy_propagators']}: {c['n_diagrams']} diagrams"
                )
            append_finding(self.base_directory, f"Diagrams: {process_str}", entries)
        except Exception:
            pass

        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

    def _save_diagrams_by_group(self, diagrams: list, ranked, fg_diagram_map, heptapod_diagrams) -> dict:
        """Save ALL diagrams organized by ranking category with JSON + SVG pairs."""
        import os
        from collections import Counter

        process_name = f"{'_'.join(self.initial)}_to_{'_'.join(self.final)}"
        output_dir = os.path.join(self.base_directory, f"diagrams_{process_name}")
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}
        viz_errors = []
        total_viz = 0
        do_svg = self.output_format in ("svg", "both")
        do_tikz = self.output_format in ("tikz", "both")

        diagram_to_fg = {id(hd): fg for hd, fg in zip(heptapod_diagrams, fg_diagram_map)}

        # Group diagrams by n_heavy_propagators
        groups = {}
        for i, d in enumerate(diagrams):
            n_heavy = d.get("ranking_info", {}).get("n_heavy_propagators", 0)
            key = f"heavy_{n_heavy}"
            if key not in groups:
                groups[key] = []
            rd = ranked[i] if i < len(ranked) else None
            groups[key].append((i, d, rd))

        for group_name in sorted(groups.keys()):
            group_dir = os.path.join(output_dir, group_name)
            os.makedirs(group_dir, exist_ok=True)

            group_items = groups[group_name]
            n_viz_saved = 0

            for idx, d, rd in group_items:
                rank = d.get("rank", idx + 1)
                base_name = f"diagram_{idx:03d}_rank{rank:02d}"

                json_filepath = os.path.join(group_dir, f"{base_name}.json")
                with open(json_filepath, 'w') as f:
                    json.dump(d, f, indent=2)

                # Respect max_visualize: 0 means unlimited
                viz_limit_reached = (
                    self.max_visualize > 0 and total_viz >= self.max_visualize
                )

                if rd is not None and self.visualize and not viz_limit_reached:
                    fg_diagram = diagram_to_fg.get(id(rd.diagram))
                    if fg_diagram is not None:
                        try:
                            # SVG output
                            if do_svg:
                                svg_filepath = os.path.join(group_dir, f"{base_name}.svg")
                                svg_content = None
                                if hasattr(fg_diagram, '_repr_svg_'):
                                    svg_content = fg_diagram._repr_svg_()
                                elif hasattr(fg_diagram, 'draw_svg'):
                                    fg_diagram.draw_svg(svg_filepath)
                                    with open(svg_filepath, 'r') as f:
                                        svg_content = f.read()

                                if svg_content:
                                    svg_content = self._add_white_background(svg_content)
                                    with open(svg_filepath, 'w') as f:
                                        f.write(svg_content)

                            # TikZ output
                            if do_tikz and hasattr(fg_diagram, 'draw_tikz'):
                                tikz_filepath = os.path.join(group_dir, f"{base_name}.tikz")
                                fg_diagram.draw_tikz(tikz_filepath)

                            n_viz_saved += 1
                            total_viz += 1
                        except Exception as e:
                            viz_errors.append(f"Diagram {idx}: {str(e)}")

            # Create representative.json — symlink to the top-ranked diagram
            if group_items:
                top_ranked = min(group_items, key=lambda x: x[1].get("rank", 999))
                top_rank = top_ranked[1].get("rank", top_ranked[0] + 1)
                top_name = f"diagram_{top_ranked[0]:03d}_rank{top_rank:02d}.json"
                rep_path = os.path.join(group_dir, "representative.json")
                try:
                    if os.path.exists(rep_path):
                        os.remove(rep_path)
                    os.symlink(top_name, rep_path)
                except OSError:
                    # Fallback: copy instead of symlink (e.g. on some filesystems)
                    import shutil
                    src = os.path.join(group_dir, top_name)
                    if os.path.exists(src):
                        shutil.copy2(src, rep_path)

            saved_files[group_name] = {
                "directory": group_dir,
                "n_diagrams": len(group_items),
                "n_visualized": n_viz_saved,
                "format": self.output_format,
            }

        # Generate and save summary markdown
        groups_for_summary = {k: [d for _, d, _ in v] for k, v in groups.items()}
        summary_md = self._generate_diagram_summary_markdown(diagrams, groups_for_summary, output_dir)
        summary_filepath = os.path.join(output_dir, "summary.md")
        with open(summary_filepath, 'w') as f:
            f.write(summary_md)
        saved_files["summary_file"] = summary_filepath
        saved_files["output_directory"] = output_dir

        if viz_errors:
            saved_files["viz_errors"] = viz_errors

        return saved_files

    def _generate_diagram_summary_markdown(self, diagrams: list, groups: dict, output_dir: str) -> str:
        """Generate a minimal markdown summary table of enumerated diagrams."""
        lines = []
        process_str = f"{' '.join(self.initial)} \u2192 {' '.join(self.final)}"

        lines.append(f"**Diagram Enumeration:** {process_str}")
        lines.append("")

        lines.append("| Heavy W | Diagrams | Couplings |")
        lines.append("|:-------:|:--------:|:----------|")

        total_diagrams = 0
        for group_name in sorted(groups.keys()):
            group_diagrams = groups[group_name]
            n_heavy = int(group_name.replace("heavy_", ""))
            n_diag = len(group_diagrams)
            total_diagrams += n_diag

            couplings = self._extract_couplings(group_diagrams[0] if group_diagrams else None)

            lines.append(f"| {n_heavy} | {n_diag} | {couplings} |")

        lines.append(f"| **Total** | **{total_diagrams}** | |")
        lines.append("")

        return "\n".join(lines)

    def _extract_couplings(self, diagram_dict: dict) -> str:
        """Extract coupling structure from a diagram dict with explicit labels."""
        if not diagram_dict:
            return "-"

        diagram = diagram_dict.get("diagram", {})
        vertices = diagram.get("vertices", [])

        if not vertices:
            return "-"

        # Count by explicit coupling name from each vertex
        counts: dict[str, int] = {}
        for v in vertices:
            coupling = v.get("coupling", "")
            vtype = v.get("type", "").lower()

            # Map to physics labels
            if isinstance(coupling, str) and coupling:
                label = coupling
            elif isinstance(coupling, dict):
                # Chiral etc. — use first value
                label = list(coupling.values())[0] if coupling else "g"
            elif vtype in ("em", "qed", "electromagnetic"):
                label = "e"
            elif vtype in ("weak", "gauge", "gauge-axial", "gauge-vector"):
                label = "g_W"
            else:
                label = "g"

            counts[label] = counts.get(label, 0) + 1

        parts = []
        for label, n in counts.items():
            parts.append(f"{label}^{n}" if n > 1 else label)

        return " ".join(parts) if parts else "-"

    def _add_white_background(self, svg_content: str) -> str:
        """Add a white background rectangle to an SVG for better visibility."""
        svg_match = re.search(r'<svg[^>]*>', svg_content)
        if not svg_match:
            return svg_content

        svg_tag = svg_match.group(0)

        width = "100%"
        height = "100%"

        w_match = re.search(r'width=["\']([^"\']+)["\']', svg_tag)
        h_match = re.search(r'height=["\']([^"\']+)["\']', svg_tag)
        if w_match and h_match:
            width = w_match.group(1)
            height = h_match.group(1)
        else:
            vb_match = re.search(r'viewBox=["\'][\d.\s]+\s+([\d.]+)\s+([\d.]+)["\']', svg_tag)
            if vb_match:
                width = vb_match.group(1)
                height = vb_match.group(2)

        bg_rect = f'<rect width="{width}" height="{height}" fill="white"/>'
        modified_svg = svg_content.replace(svg_tag, svg_tag + '\n' + bg_rect, 1)

        return modified_svg

    def _diagram_to_process_string(self, diagram) -> str:
        """Convert a Diagram object to process string notation."""
        SPIN_TO_LETTER = {0: "S", 0.5: "F", 1: "V", 2: "T"}

        try:
            if hasattr(diagram, 'to_process_string'):
                return diagram.to_process_string()

            parts = []

            if diagram.initial:
                init = diagram.initial[0]
                spin_letter = SPIN_TO_LETTER.get(init.spin, "?")
                mass = init.mass if init.mass is not None else 0
                parts.append(f"{spin_letter}({mass})")

            parts.append("->")

            if diagram.propagators:
                for prop in diagram.propagators:
                    spin_letter = SPIN_TO_LETTER.get(prop.spin, "V")
                    mass = prop.mass if hasattr(prop, 'mass') and prop.mass is not None else 0
                    is_loop = getattr(prop, 'is_loop_propagator', False)
                    if is_loop:
                        parts.append(f"{{{spin_letter}({mass})}}")
                    else:
                        parts.append(f"[{spin_letter}({mass})]")
                parts.append("->")

            final_parts = []
            for p in diagram.final:
                spin_letter = SPIN_TO_LETTER.get(p.spin, "?")
                mass = p.mass if p.mass is not None else 0
                final_parts.append(f"{spin_letter}({mass})")
            parts.append(" ".join(final_parts))

            return " ".join(parts)

        except Exception:
            return f"? -> {' '.join(['?'] * len(diagram.final))}"

    def _generate_summary(self, ranked) -> str:
        """Generate a summary of the enumeration results."""
        n_total = len(ranked)
        if n_total == 0:
            return "No diagrams found."

        n_tree = sum(1 for r in ranked if r.ranking_info.loop_order == 0)
        n_loop = n_total - n_tree

        parts = [f"Found {n_total} diagram(s)."]

        if n_tree > 0 and n_loop > 0:
            parts.append(f"{n_tree} tree-level, {n_loop} loop.")
        elif n_tree > 0:
            parts.append("All tree-level.")
        else:
            parts.append("All loop-induced.")

        if ranked:
            top = ranked[0]
            if "Tree-level" in top.ranking_info.explanation:
                parts.append("Tree-level diagram dominates.")
            elif "1-loop" in top.ranking_info.explanation:
                parts.append("Loop-induced process (no tree-level).")

        return " ".join(parts)
