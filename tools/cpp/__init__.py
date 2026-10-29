"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

The C++ route to Pythia 8 and Sherpa: build and run programs against an
install."""

from .build import (
    BuildError,
    Install,
    build,
    build_driver,
    cache_dir,
    configured_path,
    resolve_install,
    run,
)
