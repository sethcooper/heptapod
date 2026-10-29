"""
# feyngraph_interface.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

FeynGraph API wrapper for HEPTAPOD integration.

This module provides a clean interface to the FeynGraph library for
automatic Feynman diagram enumeration.
"""

from typing import List, Optional, Any, Dict
import os
from pathlib import Path
import hashlib
import json
import importlib.metadata

try:
    from .model_mapping import nda_to_feyngraph_label, feyngraph_to_nda_label
except ImportError:
    from model_mapping import nda_to_feyngraph_label, feyngraph_to_nda_label


def _require_feyngraph() -> Any:
    """
    Ensure feyngraph is available, or raise ImportError.

    Returns:
        feyngraph module

    Raises:
        ImportError: If feyngraph is not installed with helpful message
    """
    try:
        import feyngraph

        return feyngraph
    except ImportError as e:
        raise ImportError(
            "FeynGraph is not installed. This module requires FeynGraph for "
            "automatic diagram enumeration.\n\n"
            "Install with:\n"
            "  pip install feyngraph\n\n"
            "Or add to requirements.txt:\n"
            "  feyngraph>=0.1.0\n\n"
            "Note: Manual Diagram specification does not require FeynGraph."
        ) from e


class FeynGraphInterface:
    """
    Wrapper for FeynGraph library to integrate with HEPTAPOD.

    This class provides a clean interface to FeynGraph for automatic
    Feynman diagram enumeration, handling particle label mappings and
    model loading.

    Attributes:
        model: Model name ("SM") or path to UFO model directory
        feyngraph: FeynGraph module (lazily loaded)
    """

    def __init__(self, model: str = "SM", base_directory: Optional[str] = None):
        """
        Initialize FeynGraph interface.

        Args:
            model: Model to use. Options:
                  - "SM" (default): Standard Model (built-in)
                  - "/path/to/model_UFO": Path to UFO model directory

        Raises:
            ImportError: If FeynGraph is not installed
            ValueError: If model file not found
        """
        self.model = model
        self.base_directory = Path(base_directory or os.getcwd()).resolve()
        self.feyngraph = None  # Lazy loading
        self._model_obj = None

    def _ensure_loaded(self):
        """Lazy load FeynGraph module."""
        if self.feyngraph is None:
            self.feyngraph = _require_feyngraph()
        if self._model_obj is None:
            self._load_model()

    def _load_model(self):
        """
        Load the specified model.

        Raises:
            ValueError: If model file not found or invalid
        """
        if self.model == "SM":
            # Standard Model is built-in to FeynGraph
            # No explicit loading needed
            self._model_obj = self.feyngraph.Model()
        else:
            path = Path(self.model)
            if not path.is_absolute():
                path = self.base_directory / path
            path = path.resolve()
            if not path.is_dir() or not (path / "particles.py").is_file():
                raise ValueError(f"Not a UFO directory: {path}")
            # Explicitly bind the supplied UFO. Never fall back to the built-in SM.
            self.model = str(path)
            self._model_obj = self.feyngraph.Model.from_ufo(str(path))

    def resolve_label(self, label: str) -> str:
        """Prefer native labels (including BSM), then model-aware SM aliases."""
        self._ensure_loaded()
        names = {p.name() for p in self._model_obj.particles()}
        if label in names:
            return label
        aliases = {"tau": "ta-", "tau-": "ta-", "tau+": "ta+", "taubar": "ta+"}
        candidate = aliases.get(label)
        if candidate in names:
            return candidate
        try:
            candidate = nda_to_feyngraph_label(label)
        except ValueError:
            candidate = None
        if candidate in names:
            return candidate
        raise ValueError(
            f"Unknown particle {label!r} in the selected model; inspect its particle labels"
        )

    def model_info(self) -> Dict[str, Any]:
        self._ensure_loaded()
        particles = sorted(
            [
                {
                    "name": p.name(),
                    "antiname": p.anti_name(),
                    "pdg": p.pdg(),
                    "spin": p.spin(),
                    "color": p.color(),
                }
                for p in self._model_obj.particles()
            ],
            key=lambda p: p["pdg"],
        )
        vertices = sorted(
            [
                {"name": v.name(), "orders": v.coupling_orders()}
                for v in self._model_obj.vertices()
            ],
            key=lambda v: v["name"],
        )
        info = {
            "source": "builtin" if self.model == "SM" else "ufo",
            "model": self.model,
            "feyngraph_version": importlib.metadata.version("feyngraph"),
            "particles": particles,
            "vertices": vertices,
        }
        if self.model == "SM":
            info["restrictions"] = (
                "FeynGraph built-in SM; not assumed equivalent to an unrestricted SM UFO"
            )
            info["fingerprint_scope"] = (
                "FeynGraph version and exposed structural inventory; not numerical parameters"
            )
            identity = {k: v for k, v in info.items() if k != "model"}
        else:
            root = Path(self.model)
            if any(p.is_symlink() for p in root.rglob("*")):
                raise ValueError(
                    "UFO symlinks are not supported; supply a self-contained model"
                )
            info["files"] = [
                {
                    "path": p.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in sorted(root.rglob("*.py"))
                if p.is_file()
            ]
            identity = info["files"]
        info["model_sha256"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        return info

    def enumerate_diagrams(
        self,
        initial_particles: List[str],
        final_particles: List[str],
        max_loop_order: int = 0,
        **kwargs,
    ) -> List[Any]:
        """
        Generate all Feynman diagrams for the specified process.

        Args:
            initial_particles: Initial state particle labels (HEPTAPOD notation)
                             e.g., ["H"], ["mu-"], ["e+", "e-"]
            final_particles: Final state particle labels (HEPTAPOD notation)
                           e.g., ["b", "bbar"], ["e-", "gamma"]
            max_loop_order: Maximum loop order (0=tree, 1=1-loop, 2=2-loop)
            **kwargs: Additional arguments passed to FeynGraph

        Returns:
            List of FeynGraph diagram objects

        Raises:
            ImportError: If FeynGraph not installed
            ValueError: If particles invalid or model not loaded

        Examples:
            >>> interface = FeynGraphInterface("SM")
            >>> diagrams = interface.enumerate_diagrams(
            ...     initial_particles=["H"],
            ...     final_particles=["b", "bbar"],
            ...     max_loop_order=0
            ... )
            >>> len(diagrams)
            1  # One tree-level diagram for H → bb̄
        """
        self._ensure_loaded()

        # Map HEPTAPOD labels to FeynGraph notation
        try:
            fg_initial = [self.resolve_label(p) for p in initial_particles]
            fg_final = [self.resolve_label(p) for p in final_particles]
        except ValueError as e:
            raise ValueError(
                f"Invalid particle label in diagram specification: {e}\n"
                f"Initial: {initial_particles}\n"
                f"Final: {final_particles}"
            ) from e

        # Call FeynGraph API
        try:
            # Generate diagrams using FeynGraph
            diagram_container = self.feyngraph.generate_diagrams(
                fg_initial, fg_final, max_loop_order, model=self._model_obj, **kwargs
            )

            # Convert DiagramContainer to list
            diagrams = [diagram_container[i] for i in range(len(diagram_container))]

            return diagrams

        except Exception as e:
            raise RuntimeError(
                f"FeynGraph diagram generation failed: {e}\n"
                f"Initial: {initial_particles} ({fg_initial})\n"
                f"Final: {final_particles} ({fg_final})\n"
                f"Loop order: {max_loop_order}"
            ) from e

    def get_diagram_info(self, diagram: Any) -> Dict[str, Any]:
        """
        Extract information from a FeynGraph diagram object.

        Args:
            diagram: FeynGraph diagram object

        Returns:
            Dictionary with diagram information:
              {
                "n_vertices": int,
                "n_propagators": int,
                "n_initial": int,
                "n_final": int,
                "symmetry_factor": float,
                "sign": int,
                "particles": {
                  "initial": List[str],  # HEPTAPOD labels
                  "final": List[str],
                  "propagators": List[str]
                }
              }

        Raises:
            ValueError: If diagram is invalid
        """
        self._ensure_loaded()

        try:
            # Extract basic properties
            vertices = diagram.vertices()
            propagators = diagram.propagators()
            incoming = diagram.incoming()
            outgoing = diagram.outgoing()

            # Get particle names (convert to HEPTAPOD notation)
            initial_particles = [
                feyngraph_to_nda_label(leg.particle().name()) for leg in incoming
            ]
            final_particles = [
                feyngraph_to_nda_label(leg.particle().name()) for leg in outgoing
            ]
            propagator_particles = [
                feyngraph_to_nda_label(prop.particle().name()) for prop in propagators
            ]

            return {
                "n_vertices": len(vertices),
                "n_propagators": len(propagators),
                "n_initial": len(incoming),
                "n_final": len(outgoing),
                "symmetry_factor": diagram.symmetry_factor(),
                "sign": diagram.sign(),
                "particles": {
                    "initial": initial_particles,
                    "final": final_particles,
                    "propagators": propagator_particles,
                },
            }

        except Exception as e:
            raise ValueError(f"Failed to extract diagram info: {e}") from e


# Convenience function for quick enumeration
def enumerate_sm_diagrams(
    initial: List[str], final: List[str], max_loops: int = 0
) -> List[Any]:
    """
    Quick enumeration of Standard Model diagrams.

    Args:
        initial: Initial state particles (HEPTAPOD labels)
        final: Final state particles (HEPTAPOD labels)
        max_loops: Maximum loop order

    Returns:
        List of FeynGraph diagram objects

    Examples:
        >>> diagrams = enumerate_sm_diagrams(["H"], ["b", "bbar"])
        >>> len(diagrams)
        1
    """
    interface = FeynGraphInterface("SM")
    return interface.enumerate_diagrams(initial, final, max_loops)
