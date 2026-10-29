"""Agent-facing, bounded catalogue operations with persistent execution receipts."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField
from .catalogue import dump, sha


class CatalogueTool(BaseTool):
    model: str = RuntimeField(
        default="SM",
        description="SM uses FeynGraph's locally built-in model; otherwise give an explicit trusted UFO directory. No fallback.",
    )
    param_card: Optional[str] = RuntimeField(
        default=None,
        description="SLHA card for the explicit UFO. Omit to evaluate UFO defaults; not accepted with built-in SM.",
    )
    masses_json: Optional[str] = RuntimeField(
        default=None,
        description="Optional complete particle-name to mass map for built-in SM only. Without it, enumeration is structural, without kinematic cuts.",
    )
    output_dir: str = RuntimeField(
        description="New directory relative to base_directory for receipts and artifacts. Must not already exist."
    )
    timeout_seconds: int = RuntimeField(
        default=300, description="Worker timeout in seconds, 1 to 3600."
    )
    base_directory: str = StateField(
        description="Workspace root for input and output files"
    )

    def _execute_catalogue(self, action, **options):
        # Not named execute(): BaseTool.execute(**runtime_args) is how the
        # harness calls every tool, and overriding it with a positional
        # `action` made all catalogue tools fail with "missing 1 required
        # positional argument: 'action'".

        output = None
        try:
            base = Path(self.base_directory).resolve()

            def local(value):
                p = (base / value).resolve()
                if not p.is_relative_to(base):
                    raise ValueError("Path must be within base_directory")
                return str(p)

            if not 1 <= self.timeout_seconds <= 3600:
                raise ValueError("Timeout must be between 1 and 3600 seconds")
            request = dict(
                action=action,
                base_directory=str(base),
                model="SM" if self.model == "SM" else local(self.model),
                param_card=local(self.param_card) if self.param_card else None,
                masses=json.loads(self.masses_json) if self.masses_json else None,
                **options
            )
            if "catalogue_path" in request:
                request["catalogue_path"] = local(request["catalogue_path"])
            output = Path(local(self.output_dir))
            output.mkdir(parents=True, exist_ok=False)
            dump(output / "request.json", request)
            root = Path(__file__).resolve().parents[2]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
            with (output / "stdout.log").open("w") as stdout, (
                output / "stderr.log"
            ).open("w") as stderr:
                try:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.feyngraph.catalogue_worker",
                            str(output / "request.json"),
                        ],
                        cwd=output,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=self.timeout_seconds,
                    )
                    if not (output / "response.json").exists():
                        dump(
                            output / "response.json",
                            {
                                "status": "error",
                                "error": "Worker exited without response",
                                "returncode": proc.returncode,
                            },
                        )
                except OSError as exc:
                    dump(
                        output / "response.json",
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                except subprocess.TimeoutExpired:
                    dump(
                        output / "response.json",
                        {
                            "status": "error",
                            "error": "Worker timeout; partial artifacts are not a complete catalogue",
                        },
                    )
            manifest = {
                p.relative_to(output).as_posix(): sha(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            }
            dump(output / "manifest.json", manifest)
            result = json.loads((output / "response.json").read_text())
            result["artifacts_directory"] = str(output)
            return json.dumps(result)
        except Exception as exc:
            return json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
            )


class InspectProcessModelTool(CatalogueTool):
    """Inspect the locally available built-in SM or explicit UFO, including provenance and physical particle inventory."""

    def _run(self):
        return self._execute_catalogue("inspect")


class EnumerateProcessesTool(CatalogueTool):
    """Enumerate tree-level 1->N decays (any N >= 2) with resolved flavors and separate conjugates: FeynTable graphs (unitary gauge with a UFO), coupling classes (orders and coupling monomials) and exclusion accounting. No widths or scattering."""

    parents: Optional[List[str]] = RuntimeField(
        default=None,
        description="Parent particle names; omit for all physical particles.",
    )
    orders_json: Optional[str] = RuntimeField(
        default=None,
        description='JSON map of exact coupling powers, e.g. {"QED":1}. Unspecified orders unconstrained.',
    )
    final_multiplicity: int = RuntimeField(
        default=2,
        description="Number of daughters N >= 2; the candidate budget grows quickly with N.",
    )
    max_candidates: int = RuntimeField(
        default=200000,
        description="Explicit cap on candidate parent/final-state combinations (before charge and kinematic filtering).",
    )
    render: bool = RuntimeField(
        default=True,
        description="Save native FeynGraph SVG and TikZ companions for each graph.",
    )

    def _run(self):
        try:
            options = dict(
                parents=self.parents,
                orders=json.loads(self.orders_json) if self.orders_json else None,
                final_multiplicity=self.final_multiplicity,
                max_candidates=self.max_candidates,
                render=self.render,
            )
            return self._execute_catalogue("enumerate", enumeration=options)
        except Exception as exc:
            return json.dumps({"status": "error", "error": str(exc)})


class ValidateProcessCatalogueTool(CatalogueTool):
    """Independently regenerate every 1->N channel in the catalogue's scope with MG5 from the same explicit UFO (unitary gauge) and compare graphs, signed lines, vertex terms and coupling classes. This is not a numerical rate check."""

    catalogue_path: str = RuntimeField(
        description="Path to catalogue.json within workspace."
    )
    expected_scope_json: str = RuntimeField(
        description="Independently requested full scope JSON; must equal the catalogue scope. Prevents silently narrowed coverage."
    )
    mg5_path: str = StateField(
        description="Configured MadGraph installation directory."
    )

    def _run(self):
        try:
            return self._execute_catalogue(
                "validate",
                catalogue_path=self.catalogue_path,
                expected_scope=json.loads(self.expected_scope_json),
                mg5_path=self.mg5_path,
            )
        except Exception as exc:
            return json.dumps({"status": "error", "error": str(exc)})
