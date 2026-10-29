"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
FeynRules tools for UFO model generation and model validation.

Tools:
    FeynRulesToUFOTool          — compile a .fr model to a UFO directory
    ValidateLagrangianTool  — run FeynRules' consistency checks only
    ValidateUFOModelTool    — compile to a UFO, then check what was produced
    WriteFeynArtsModel      — write a FeynArts model, with a manifest and a linter
    WriteToyModel           — a .fr for a toy theory from a compact spec

Library:
    parse_check_verdicts        — read check verdicts out of generator stdout
    failed_checks               — the failures a model actually introduced
    summarise                   — one-line summary of a check run
"""
from .feynrules import FeynRulesToUFOTool, ValidateLagrangianTool
from .validate_tool import ValidateUFOModelTool
from .feynarts_model import WriteFeynArtsModel
from .toy_model import WriteToyModel
from .lagrangian_checks import parse_check_verdicts, failed_checks, summarise

__all__ = [
    "FeynRulesToUFOTool",
    "ValidateLagrangianTool",
    "ValidateUFOModelTool",
    "WriteFeynArtsModel",
    "WriteToyModel",
    "parse_check_verdicts",
    "failed_checks",
    "summarise",
]
