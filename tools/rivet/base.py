"""
# base.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Sandbox and installation plumbing shared by the Rivet tools."""

import os
from typing import List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import StateField
from pydantic import Field

from tools.rivet.install import RivetInstall, resolve_rivet


def runtime_list(description: str) -> List[str]:
    """An optional list-valued runtime field.

    `RuntimeField(default=None)` is what marks a field *required* in the
    schema, so an optional list takes an empty default instead.
    """
    return Field(
        default_factory=list,
        json_schema_extra={"runtime": True},
        description=description,
    )


class RivetToolBase(BaseTool):
    """Path resolution inside the sandbox and access to the Rivet install."""

    base_directory: str = StateField(
        default=".", description="Base directory for safe path resolution"
    )
    rivet_path: str = StateField(
        default="",
        description="Rivet installation prefix (the directory holding "
        "bin/rivet-config); RIVET_PATH when unset",
    )

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(
                f"Base directory does not exist: {self.base_directory}"
            )

    def _safe_path(self, rel: str) -> Optional[str]:
        """The absolute path of a sandbox-relative one; None if it escapes."""
        if not rel:
            return None
        full = os.path.abspath(
            os.path.join(self.base_directory, os.path.expanduser(rel))
        )
        if full != self.base_directory and not full.startswith(
            self.base_directory + os.sep
        ):
            return None
        return full

    def _rel(self, path: str) -> str:
        return os.path.relpath(path, self.base_directory)

    def _install(self) -> RivetInstall:
        return resolve_rivet(self.rivet_path)
