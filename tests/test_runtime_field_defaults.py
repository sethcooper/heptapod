"""
A RuntimeField that defaults to None must accept None.

PDFToTeX declares ``pages: str = RuntimeField(default=None)``; called
without ``pages`` ("omit to extract the whole document") it must not fail
with ``Input should be a valid string ... input_value=None``. Every field
with that pattern is checked.
"""
import importlib
import pkgutil
import sys
import types
import typing
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orchestral.tools.base.tool import BaseTool  # noqa: E402


def _tool_classes():
    import tools
    seen = {}
    for mod in pkgutil.walk_packages(tools.__path__, "tools."):
        if ".test" in mod.name or "test_files" in mod.name:
            continue
        try:
            m = importlib.import_module(mod.name)
        except Exception:
            continue  # optional dependency missing; not this test's concern
        for obj in vars(m).values():
            if isinstance(obj, type) and issubclass(obj, BaseTool) and obj is not BaseTool \
                    and obj.__module__.startswith("tools."):
                seen[f"{obj.__module__}.{obj.__name__}"] = obj
    return seen


def _accepts_none(ann) -> bool:
    if ann is None or ann is typing.Any or ann is type(None):
        return True
    origin = typing.get_origin(ann)
    if origin in (typing.Union, types.UnionType):
        return any(_accepts_none(a) for a in typing.get_args(ann))
    return False


def test_none_default_fields_accept_none():
    bad = []
    for name, cls in sorted(_tool_classes().items()):
        for fname, field in cls.model_fields.items():
            extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
            # Required RuntimeFields get an injected default=None (marked
            # optional=False); only author-declared optional ones count.
            if not (extra.get("runtime") and extra.get("optional")):
                continue
            if field.default is None and not _accepts_none(field.annotation):
                bad.append(f"{name}.{fname}: {field.annotation}")
    assert not bad, "fields default to None but reject it:\n" + "\n".join(bad)


def test_pdftotex_without_pages_is_not_a_validation_error(tmp_path):
    from tools.literature.pdf_to_tex_tool import PDFToTeXTool
    out = PDFToTeXTool(base_directory=str(tmp_path)).execute(pdf_path="paper.pdf")
    assert "Validation Error" not in out


@pytest.mark.parametrize("modname,clsname", [
    ("tools.units.natural_units_converter", "NaturalUnitsConverter"),
    ("tools.units.metric_prefix_converter", "MetricPrefixConverter"),
])
def test_units_missing_request_gets_the_tool_message(tmp_path, modname, clsname):
    cls = getattr(importlib.import_module(modname), clsname)
    out = cls(base_directory=str(tmp_path)).execute()
    assert "Validation Error" not in out
    assert "to <target_unit>" in out
