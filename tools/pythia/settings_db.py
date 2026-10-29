"""
# settings_db.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Pythia 8 settings database.

Pythia's user manual is authored as a set of XML files
(``share/Pythia8/xmldoc``) in which every run-card setting is declared inline
with its type, default, and allowed range. That is the same source Pythia
itself reads at initialisation, so it is authoritative rather than a
transcription: if a setting is not in here, ``readString``/``readFile`` will
reject it as unknown.

This module turns that XML into a searchable snapshot so an agent writing a
``.cmnd`` run card can find the right setting name and its default without
loading a megabyte of manual into context.

Regenerate the snapshot against a specific Pythia install with::

    python -m tools.pythia.settings_db --xmldoc /path/to/share/Pythia8/xmldoc

With no ``--xmldoc``, the xmldoc directory of the installed ``pythia8mc`` wheel
is used -- which is the copy ``PythiaFromRunCardTool`` actually runs against.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .settings_topics import ALIASES, TOPICS, topic_for

SNAPSHOT_VERSION = "pythia-settings-1.0"

# Tags in the xmldoc that declare a run-card setting, mapped to the value type
# Pythia parses them as. The "...more" variants declare a further setting that
# shares the prose of its neighbours; the "...fix" variants are settings the
# user is not meant to change (e.g. Pythia:versionNumber).
_SETTING_TAGS: Dict[str, str] = {
    "flag": "bool",
    "flagfix": "bool",
    "flagmore": "bool",
    "parm": "float",
    "parmfix": "float",
    "parmmore": "float",
    "mode": "int",
    "modefix": "int",
    "modemore": "int",
    "modepick": "int",
    "modeopen": "int",
    "word": "string",
    "wordfix": "string",
    "wordmore": "string",
    # fvec is a vector of flags (e.g. the onia process switches), not floats.
    "fvec": "vector<bool>",
    "mvec": "vector<int>",
    "pvec": "vector<float>",
    "wvec": "vector<string>",
}

# The opening tag of a setting block. Two details of the real xmldoc make a
# single open-to-close regex wrong:
#   * Some settings are self-closing with no prose at all (VinciaQCD.xml:
#     `<parm name="Vincia:QGEmitFF:chargeFactor" default="2.85"/>`).
#     Requiring a closing tag makes such a block run on to the *next*
#     `</parm>`, swallowing every setting in between.
#   * Dire setting names contain '>' (`Enhance:Dire_fsr_qcd_1->1&21`), so the
#     attribute scanner has to consume quoted strings whole rather than stop at
#     the first '>'.
# Bodies are therefore found by scanning forward from each opening tag.
_OPEN_RE = re.compile(
    r"<(?P<tag>" + "|".join(_SETTING_TAGS) + r")\s+"
    r"(?P<attrs>(?:\"[^\"]*\"|[^>\"])*?)(?P<selfclose>/?)>",
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_OPTION_RE = re.compile(
    r'<option\s+value\s*=\s*"([^"]*)"\s*>(.*?)</option>', re.DOTALL
)
_CHAPTER_RE = re.compile(r'<chapter\s+name\s*=\s*"([^"]*)"')
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Entities the Pythia manual uses that html.unescape does not resolve to
# something readable in a terminal.
_ENTITY_FIXUPS = (
    ("&rarr;", "->"),
    ("&larr;", "<-"),
    ("&harr;", "<->"),
    ("&rArr;", "=>"),
    ("&ge;", ">="),
    ("&le;", "<="),
    ("&ne;", "!="),
    ("&approx;", "~"),
    ("&asymp;", "~"),
    ("&times;", "*"),
    ("&minus;", "-"),
    ("&plusmn;", "+-"),
    ("&nbsp;", " "),
)


def _clean_text(raw: str) -> str:
    """Strip the manual's inline markup down to plain prose."""
    text = raw
    for entity, replacement in _ENTITY_FIXUPS:
        text = text.replace(entity, replacement)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _first_sentence(text: str, limit: int = 240) -> str:
    """One-line summary of a description, for brief search output."""
    if not text:
        return ""
    # Split on a period followed by whitespace, avoiding numeric defaults like
    # "0. " that appear mid-sentence in the Pythia prose.
    match = re.search(r"(?<![0-9])\.\s", text)
    summary = text[: match.start() + 1] if match else text
    if len(summary) > limit:
        summary = summary[: limit - 1].rstrip() + "…"
    return summary


def _extract_body(
    raw: str, match: "re.Match[str]", next_match: Optional["re.Match[str]"]
) -> str:
    """Return the prose between a setting's opening and closing tag.

    Bounded by the next setting's opening tag so that an unclosed or
    self-closing block cannot absorb its successors.
    """
    if match.group("selfclose"):
        return ""
    start = match.end()
    stop = next_match.start() if next_match is not None else len(raw)
    close = raw.find(f"</{match.group('tag')}>", start, stop)
    return raw[start:close] if close != -1 else raw[start:stop]


def _parse_file(path: Path) -> List[Dict[str, Any]]:
    """Extract every setting declared in one xmldoc file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    chapter_match = _CHAPTER_RE.search(raw)
    chapter = (
        _clean_text(chapter_match.group(1)) if chapter_match else path.stem
    )

    opens = list(_OPEN_RE.finditer(raw))
    settings: List[Dict[str, Any]] = []
    for index, match in enumerate(opens):
        tag = match.group("tag")
        attrs = dict(_ATTR_RE.findall(match.group("attrs")))
        name = attrs.get("name")
        if not name:
            continue

        body = _extract_body(
            raw, match, opens[index + 1] if index + 1 < len(opens) else None
        )
        options = [
            {"value": value.strip(), "description": _clean_text(text)}
            for value, text in _OPTION_RE.findall(body)
        ]
        description = _clean_text(_OPTION_RE.sub(" ", body))

        record: Dict[str, Any] = {
            "name": name,
            "type": _SETTING_TAGS[tag],
            "tag": tag,
            "chapter": chapter,
            "page": path.stem,
            "default": attrs.get("default"),
            "description": description,
        }
        if "min" in attrs:
            record["min"] = attrs["min"]
        if "max" in attrs:
            record["max"] = attrs["max"]
        if options:
            record["options"] = options
        if tag.endswith("fix"):
            record["read_only"] = True
        if match.group("selfclose"):
            record["_selfclose"] = True
        settings.append(record)

    _inherit_shared_prose(settings)
    for record in settings:
        record.pop("_selfclose", None)
    return settings


def _inherit_shared_prose(settings: List[Dict[str, Any]]) -> None:
    """Give settings that share a description block a copy of that prose.

    The manual documents groups such as ``LowEnergyQCD:sEffAQM`` /
    ``cEffAQM`` / ``bEffAQM`` by leaving the first blocks empty and writing the
    prose once, in the last block of the run. Without this the first members of
    such a group would be undiscoverable by description search.

    Self-closing declarations are excluded: they carry no prose by design (the
    Vincia charge factors are declared this way) and are not part of a shared
    block, so inheriting the next description would attach a description that
    belongs to an unrelated setting.
    """
    pending: List[Dict[str, Any]] = []
    for record in settings:
        if record.get("_selfclose"):
            continue
        if record["description"]:
            for earlier in pending:
                earlier["description"] = record["description"]
                earlier["shared_description"] = True
            pending = []
        else:
            pending.append(record)


def default_xmldoc_dir() -> Optional[Path]:
    """Locate the xmldoc directory of the Pythia the tools actually run
    against.

    ``pythia8mc`` installs its share tree next to the interpreter prefix rather
    than inside the package, so resolve from ``sys.prefix`` first and fall back
    to the ``PYTHIA8DATA`` convention.
    """
    candidates = []
    env = os.environ.get("PYTHIA8DATA")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(sys.prefix) / "share" / "Pythia8" / "xmldoc")
    try:
        import pythia8mc  # noqa: F401

        pkg_prefix = Path(pythia8mc.__file__).resolve().parent
        candidates.append(pkg_prefix / "share" / "Pythia8" / "xmldoc")
        candidates.append(
            pkg_prefix.parents[2] / "share" / "Pythia8" / "xmldoc"
        )
    except Exception:
        pass

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.xml")):
            return candidate
    return None


_INDEX_HREF_RE = re.compile(r'<aidx\s+href\s*=\s*"([^"]+)"')


def _indexed_files(xmldoc_dir: Path) -> List[Path]:
    """The xmldoc files Pythia itself reads, in the order Index.xml lists them.

    Settings::init reads Index.xml and follows its links; any other .xml file
    in the directory is ignored. Globbing instead picks up whatever else is
    there -- in an environment where two Pythia builds share one share/
    prefix, that includes files from the other version (e.g. the Dire pages
    8.317 no longer ships), and the snapshot then mixes two releases.
    """
    index = xmldoc_dir / "Index.xml"
    if not index.is_file():
        raise FileNotFoundError(
            f"{index} not found; a Pythia xmldoc directory always has one"
        )
    hrefs = _INDEX_HREF_RE.findall(
        index.read_text(encoding="utf-8", errors="replace")
    )
    files: List[Path] = []
    for href in dict.fromkeys(hrefs):
        path = xmldoc_dir / (href if href.endswith(".xml") else href + ".xml")
        if path.is_file():
            files.append(path)
    return files


def parse_xmldoc(xmldoc_dir: Path) -> Dict[str, Any]:
    """Parse a whole xmldoc directory into a snapshot dictionary.

    Raises:
        FileNotFoundError: if the directory or its Index.xml is missing
        ValueError: if two indexed files declare the same setting, which
            means the directory does not hold a single consistent release
    """
    xmldoc_dir = Path(xmldoc_dir)
    if not xmldoc_dir.is_dir():
        raise FileNotFoundError(
            f"xmldoc directory does not exist: {xmldoc_dir}"
        )

    settings: List[Dict[str, Any]] = []
    for path in _indexed_files(xmldoc_dir):
        settings.extend(_parse_file(path))

    # The manual occasionally documents one setting on two pages, which is
    # harmless when the declarations agree. When they disagree on type,
    # default or range there is no right one to keep, and picking silently is
    # how a snapshot ends up half one release and half another.
    first: Dict[str, Dict[str, Any]] = {}
    conflicts: List[str] = []
    for record in settings:
        key = record["name"].lower()
        if key not in first:
            first[key] = record
            continue
        earlier = first[key]
        if any(
            earlier.get(f) != record.get(f)
            for f in ("type", "default", "min", "max")
        ):
            conflicts.append(
                f"{record['name']} ({earlier['page']}, {record['page']})"
            )
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} setting(s) declared inconsistently in "
            f"{xmldoc_dir}: " + "; ".join(conflicts[:5])
        )
    unique = sorted(first.values(), key=lambda r: r["name"].lower())

    stale_aliases = _attach_curated_layer(unique)

    return {
        "schema": SNAPSHOT_VERSION,
        "pythia_version": _read_version(xmldoc_dir),
        "n_settings": len(unique),
        "topics": TOPICS,
        "stale_aliases": stale_aliases,
        "particles": _parse_particles(xmldoc_dir),
        "settings": unique,
    }


_PARTICLE_RE = re.compile(
    r'<particle\s+id\s*=\s*"(-?\d+)"\s+name\s*=\s*"([^"]*)"'
    r'(?:\s+antiName\s*=\s*"([^"]*)")?',
    re.DOTALL,
)


def _parse_particles(xmldoc_dir: Path) -> Dict[str, str]:
    """Map PDG id -> name from ParticleData.xml.

    Run cards address particles directly (``6:m0 = 172.5``,
    ``4900101:mayDecay = off``). Those lines are handled by ParticleData rather
    than Settings, and an id that is not in the table is an error at read time,
    so the validator needs the id list to catch it.

    Antiparticles are keyed under their negative id using the declared
    ``antiName``, so an event record full of negative ids can be summarised in
    names a physicist reads rather than raw PDG codes.
    """
    path = xmldoc_dir / "ParticleData.xml"
    if not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8", errors="replace")
    particles: Dict[str, str] = {}
    for pid, name, anti_name in _PARTICLE_RE.findall(raw):
        particles[pid] = name
        if anti_name and not pid.startswith("-"):
            particles[f"-{pid}"] = anti_name
    return particles


def _attach_curated_layer(settings: List[Dict[str, Any]]) -> List[str]:
    """Add topic and alias fields from ``settings_topics``.

    Returns the curated names that are absent from this Pythia's xmldoc, so the
    regeneration CLI (and the test suite) can flag aliases that have gone stale
    against a newer release rather than failing silently.
    """
    by_name = {record["name"]: record for record in settings}
    for record in settings:
        topic = topic_for(record["name"])
        if topic:
            record["topic"] = topic

    unknown: List[str] = []
    for name, aliases in ALIASES.items():
        record = by_name.get(name)
        if record is None:
            unknown.append(name)
            continue
        record["aliases"] = list(aliases)
        record["curated"] = True
    return unknown


def _read_version(xmldoc_dir: Path) -> Optional[str]:
    """Read the Pythia version the xmldoc belongs to, from Version.xml."""
    version_file = xmldoc_dir / "Version.xml"
    if not version_file.is_file():
        return None
    match = re.search(
        r'name\s*=\s*"Pythia:versionNumber"\s+default\s*=\s*"([^"]*)"',
        version_file.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1) if match else None


# ====================================================================== #
# ===================== The installed Pythia's database ================ #
# ====================================================================== #

# Settings map getters on pythia8mc's Settings, and the value type each holds.
_LIVE_MAPS = (
    ("getFlagMap", "bool"),
    ("getModeMap", "int"),
    ("getParmMap", "float"),
    ("getWordMap", "string"),
    ("getFVecMap", "vector<bool>"),
    ("getMVecMap", "vector<int>"),
    ("getPVecMap", "vector<float>"),
    ("getWVecMap", "vector<string>"),
)


def _format_value(value: Any, kind: str) -> str:
    """Render a live setting value the way the xmldoc spells it."""
    if kind.startswith("vector"):
        return (
            "{" + ",".join(_format_value(v, kind[7:-1]) for v in value) + "}"
        )
    if kind == "bool":
        return "on" if value else "off"
    if kind == "float":
        return format(float(value), "g")
    return str(value)


def installed_settings(pythia: Any) -> Dict[str, Dict[str, Any]]:
    """
    Read the settings database of a constructed Pythia instance.

    This is what Pythia will actually accept: every name, with its type,
    default and range, for the version that is installed. The xmldoc snapshot
    can only describe the version it was built from.

    Parameters:
        pythia: A pythia8mc.Pythia instance (init() not required)

    Returns:
        Map of lowercase name -> {"name", "type", "default", "min"?, "max"?}
    """
    settings = pythia.settings
    out: Dict[str, Dict[str, Any]] = {}
    for getter, kind in _LIVE_MAPS:
        # A vector's bounds apply to each element.
        element = (
            kind[len("vector<") : -1] if kind.startswith("vector") else kind
        )
        for key, entry in getattr(settings, getter)("").items():
            record: Dict[str, Any] = {
                "name": entry.name,
                "type": kind,
                "default": _format_value(entry.valDefault, kind),
            }
            if getattr(entry, "hasMin", False):
                record["min"] = _format_value(entry.valMin, element)
            if getattr(entry, "hasMax", False):
                record["max"] = _format_value(entry.valMax, element)
            out[key] = record
    return out


def _same_setting_value(kind: str, a: str, b: str) -> bool:
    """Whether two spellings of a setting value mean the same thing."""
    if kind == "bool":
        truthy = ("on", "true", "yes", "1", "ok")
        return (a.strip().lower() in truthy) == (b.strip().lower() in truthy)
    try:
        if kind.startswith("vector"):
            left = [v.strip() for v in a.strip("{} ").split(",") if v.strip()]
            right = [v.strip() for v in b.strip("{} ").split(",") if v.strip()]
            element = kind[len("vector<") : -1]
            return len(left) == len(right) and all(
                _same_setting_value(element, x, y) for x, y in zip(left, right)
            )
        if kind in ("int", "float"):
            return math.isclose(
                float(a), float(b), rel_tol=1e-9, abs_tol=1e-300
            )
    except ValueError:
        return False
    return a == b


def merge_installed(
    snapshot: Dict[str, Any],
    live: Dict[str, Dict[str, Any]],
    version: Optional[str],
) -> Dict[str, Any]:
    """
    Combine the installed database with the snapshot's documentation.

    Names, types, defaults and ranges come from `live`, so validation matches
    the Pythia that will run the card. Descriptions, option lists, topics and
    aliases come from the snapshot where it documents the same name; a setting
    the snapshot does not know is kept, undocumented, and one the installed
    Pythia does not have is dropped.
    """
    documented = {
        record["name"].lower(): record for record in snapshot["settings"]
    }
    merged: List[Dict[str, Any]] = []
    for key, authoritative in live.items():
        record = dict(
            documented.get(key, {"page": "", "chapter": "", "description": ""})
        )
        if (
            not record.get("description")
            and key.startswith("hi")
            and key[2:] in documented
        ):
            # Angantyr creates an "HI" copy of many settings in code, to set
            # them separately for heavy-ion sub-collisions; the manual
            # documents only the original.
            base = documented[key[2:]]
            record.update(
                {k: base[k] for k in ("page", "chapter", "topic") if k in base}
            )
            record["description"] = (
                f"Heavy-ion (Angantyr) copy of {base['name']}, applied to the "
                f"sub-collisions. {base.get('description', '')}"
            ).strip()
        spelled = {
            field: record.pop(field, None)
            for field in ("default", "min", "max")
        }
        record.update(authoritative)
        # Keep the manual's spelling ("14000.", "0.") where it is the same
        # value.
        for field, text in spelled.items():
            if (
                field in record
                and text is not None
                and _same_setting_value(record["type"], text, record[field])
            ):
                record[field] = text
        merged.append(record)
    merged.sort(key=lambda r: r["name"].lower())

    out = {k: v for k, v in snapshot.items() if k != "settings"}
    out.update(
        {
            "pythia_version": version,
            "n_settings": len(merged),
            "settings": merged,
            "database": {
                "source": "installed",
                "pythia_version": version,
                "documentation_version": snapshot.get("pythia_version"),
            },
        }
    )
    return out


_SETTINGS_CACHE: Optional[Dict[str, Any]] = None


def load_settings(use_installed: bool = True) -> Dict[str, Any]:
    """
    The settings database to validate against, cached for the process.

    When pythia8mc is importable its own database is used, merged with the
    snapshot's documentation, so a card is checked against the release that
    will run it. Otherwise the bundled snapshot is returned as it is. Either
    way `database.source` says which ("installed" or "bundled").
    """
    global _SETTINGS_CACHE
    if use_installed and _SETTINGS_CACHE is not None:
        return _SETTINGS_CACHE

    snapshot = load_snapshot()
    if use_installed:
        try:
            pythia = _quiet_pythia()
            version = (
                f"{float(pythia.settings.parm('Pythia:versionNumber')):.3f}"
            )
            merged = merge_installed(
                snapshot, installed_settings(pythia), version
            )
            _SETTINGS_CACHE = merged
            return merged
        except Exception:
            pass
    bundled = dict(snapshot)
    bundled["database"] = {
        "source": "bundled",
        "pythia_version": snapshot.get("pythia_version"),
    }
    return bundled


def _quiet_pythia() -> Any:
    """A Pythia instance for reading its databases, with its banner suppressed.

    Constructing Pythia reads the whole xmldoc and prints to the C-level
    stdout, which under the MCP stdio transport is the JSON-RPC channel.
    """
    from .runtime import STDERR, STDOUT, FdSilence, require_pythia

    pythia8 = require_pythia()
    with FdSilence(STDOUT, STDERR):
        return pythia8.Pythia("", printBanner=False)


# ====================================================================== #
# ============================ Snapshot I/O ============================ #
# ====================================================================== #

DATA_DIR = Path(__file__).parent / "data"
SNAPSHOT_PATH = DATA_DIR / "pythia_settings.json.gz"


def write_snapshot(
    snapshot: Dict[str, Any], path: Path = SNAPSHOT_PATH
) -> Path:
    """Write a snapshot to disk, gzip-compressed, with stable key order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        snapshot, indent=1, sort_keys=False, ensure_ascii=False
    )
    # mtime=0 keeps the bytes reproducible so regenerating an unchanged
    # snapshot produces no git diff.
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=open(path, "wb"), mtime=0
    ) as fp:
        fp.write(payload.encode("utf-8"))
    return path


_CACHE: Optional[Dict[str, Any]] = None


def load_snapshot(
    path: Path = SNAPSHOT_PATH, use_cache: bool = True
) -> Dict[str, Any]:
    """Load the shipped snapshot, caching it for the life of the process."""
    global _CACHE
    if use_cache and _CACHE is not None and Path(path) == SNAPSHOT_PATH:
        return _CACHE
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"settings snapshot not found: {path}. Regenerate it with "
            "`python -m tools.pythia.settings_db`."
        )
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        snapshot = json.load(fp)
    if use_cache and path == SNAPSHOT_PATH:
        _CACHE = snapshot
    return snapshot


# ====================================================================== #
# ============================== Searching ============================= #
# ====================================================================== #

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize_name(name: str) -> List[str]:
    """Split ``PhaseSpace:pTHatMin`` into searchable lowercase tokens.

    Pythia setting names are ``Group:camelCaseName``; splitting on the colon
    and on camel-case boundaries is what lets a query like "pT hat min" or
    "phase space" reach the right setting.
    """
    tokens: List[str] = []
    for chunk in _WORD_RE.findall(name):
        tokens.append(chunk.lower())
        for piece in re.findall(
            r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|^[a-z0-9]+", chunk
        ):
            if piece:
                tokens.append(piece.lower())
    return tokens


def _build_index(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Precompute per-setting search fields plus a corpus-wide IDF table.

    Term rarity is what makes ranking work here. A query like "pTHat cut"
    contains one term that appears in a handful of settings ("pthat") and one
    that appears in scores of them ("cut"); without inverse-document-frequency
    weighting the common term dominates and the intended hit
    (``PhaseSpace:pTHatMin``) is buried under every setting whose name happens
    to end in "Cut".
    """
    entries: List[Dict[str, Any]] = []
    document_freq: Dict[str, int] = {}

    for record in snapshot["settings"]:
        name_lower = record["name"].lower()
        name_tokens = set(_tokenize_name(record["name"]))
        description = record.get("description", "").lower()
        description_tokens = set(_WORD_RE.findall(description))
        option_text = " ".join(
            option.get("description", "")
            for option in record.get("options", [])
        ).lower()

        aliases = [alias.lower() for alias in record.get("aliases", [])]
        alias_text = " ".join(aliases)
        alias_tokens = set(_WORD_RE.findall(alias_text))
        # Alias phrases are compared as token sequences, not as substrings: the
        # alias "zz" is a substring of any query containing a double z, which
        # made unrelated queries match WeakDoubleBoson.
        alias_phrases = [tuple(_WORD_RE.findall(alias)) for alias in aliases]

        entries.append(
            {
                "record": record,
                "name_lower": name_lower,
                "name_tokens": name_tokens,
                "description": description,
                "description_tokens": description_tokens,
                "option_text": option_text,
                "aliases": aliases,
                "alias_tokens": alias_tokens,
                "alias_phrases": alias_phrases,
            }
        )
        for token in name_tokens | description_tokens:
            document_freq[token] = document_freq.get(token, 0) + 1

    total = max(len(entries), 1)
    idf = {
        token: max(0.35, math.log(total / (1 + freq)))
        for token, freq in document_freq.items()
    }
    # A term absent from the corpus is maximally rare rather than unweighted --
    # it will only ever match as a substring, and that match is informative.
    snapshot["_index"] = entries
    snapshot["_idf"] = idf
    snapshot["_idf_default"] = math.log(total)
    return entries


def _get_index(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "_index" not in snapshot:
        _build_index(snapshot)
    return snapshot["_index"]


# Relative weight of where a term matched. Name hits dominate description hits
# because a user who half-remembers a setting name is the common case;
# description matching serves the user who knows the physics but not the
# spelling.
_W_NAME_TOKEN = 30.0
_W_NAME_SUBSTRING = 24.0
_W_TOKEN_PREFIX = 14.0
_W_DESCRIPTION_TOKEN = 5.0
_W_DESCRIPTION_SUBSTRING = 2.0
_W_OPTION = 1.5


def _contains_phrase(
    haystack: Tuple[str, ...], needle: Tuple[str, ...]
) -> bool:
    """True when ``needle`` appears as a contiguous run of tokens in
    ``haystack``."""
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[start : start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )


def _score(
    entry: Dict[str, Any],
    terms: List[str],
    raw_query: str,
    snapshot: Dict[str, Any],
) -> float:
    """Rank one setting against a query."""
    name_lower = entry["name_lower"]
    query_lower = raw_query.lower().strip()

    if name_lower == query_lower:
        return 1e6

    idf = snapshot["_idf"]
    idf_default = snapshot["_idf_default"]

    score = 0.0
    if query_lower and query_lower in name_lower:
        score += 150.0
        if name_lower.startswith(query_lower):
            score += 50.0

    # A curated alias matched as a whole phrase is the strongest signal short
    # of the exact name: it means someone wrote down that this wording means
    # this setting. "beam energy" -> Beams:eCM only works through this branch.
    query_phrase = tuple(terms)
    for alias in entry["alias_phrases"]:
        if alias == query_phrase:
            score += 400.0
            break
        if _contains_phrase(query_phrase, alias) or _contains_phrase(
            alias, query_phrase
        ):
            score += 180.0
            break

    matched = 0
    weight_sum = 0.0
    for term in terms:
        term_idf = idf.get(term, idf_default)
        weight_sum += term_idf

        if term in entry["name_tokens"] or term in entry["alias_tokens"]:
            best = _W_NAME_TOKEN
        elif term in name_lower:
            best = _W_NAME_SUBSTRING
        elif any(token.startswith(term) for token in entry["name_tokens"]):
            best = _W_TOKEN_PREFIX
        else:
            best = 0.0

        if term in entry["description_tokens"]:
            best += _W_DESCRIPTION_TOKEN
        elif term in entry["description"]:
            best += _W_DESCRIPTION_SUBSTRING
        if term in entry["option_text"]:
            best += _W_OPTION

        if best > 0.0:
            matched += 1
            score += best * term_idf

    # Reward covering the whole query: matching every term of a two-term query
    # is a far stronger signal than matching one of them twice as well.
    if terms and matched:
        coverage = matched / len(terms)
        score *= 0.55 + 0.45 * coverage
        if matched == len(terms):
            score += 0.5 * weight_sum * _W_NAME_TOKEN / max(len(terms), 1)

    if entry["record"].get("read_only"):
        score -= 20.0
    return score


def search_settings(
    query: str,
    *,
    snapshot: Optional[Dict[str, Any]] = None,
    chapter: Optional[str] = None,
    prefix: Optional[str] = None,
    topic: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return settings matching ``query``, best match first."""
    data = snapshot if snapshot is not None else load_snapshot()
    entries: Iterable[Dict[str, Any]] = _get_index(data)

    if topic:
        needle = topic.lower()
        entries = [e for e in entries if e["record"].get("topic") == needle]
    if chapter:
        needle = chapter.lower()
        entries = [
            e
            for e in entries
            if needle in e["record"]["chapter"].lower()
            or needle in e["record"]["page"].lower()
        ]
    if prefix:
        needle = prefix.lower().rstrip(":")
        entries = [e for e in entries if e["name_lower"].startswith(needle)]

    terms = [t.lower() for t in _WORD_RE.findall(query)]
    if not terms:
        # No query text: the filters alone define the result set.
        return sorted((e["record"] for e in entries), key=lambda r: r["name"])[
            :limit
        ]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in entries:
        value = _score(entry, terms, query, data)
        if value > 0:
            scored.append((value, entry["record"]))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
    return [record for _, record in scored[:limit]]


def get_setting(
    name: str, snapshot: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Exact (case-insensitive) lookup of one setting by name."""
    data = snapshot if snapshot is not None else load_snapshot()
    needle = name.strip().lower()
    for record in data["settings"]:
        if record["name"].lower() == needle:
            return record
    return None


def brief(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compact form of a record, for search hit lists."""
    out = {
        "name": record["name"],
        "type": record["type"],
        "default": record.get("default"),
        "summary": _first_sentence(record.get("description", "")),
        "page": record["page"],
    }
    for key in ("min", "max"):
        if key in record:
            out[key] = record[key]
    if record.get("options"):
        out["n_options"] = len(record["options"])
    if record.get("read_only"):
        out["read_only"] = True
    return out


# ====================================================================== #
# ========================= Regeneration CLI =========================== #
# ====================================================================== #


def _xmldoc_digest(xmldoc_dir: Path) -> str:
    """SHA-256 over the indexed xmldoc files, so a snapshot names its
    source."""
    import hashlib

    digest = hashlib.sha256()
    for path in _indexed_files(Path(xmldoc_dir)):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the Pythia settings snapshot. Names, types, "
        "defaults "
        "and ranges are read from the installed pythia8mc; prose from the "
        "xmldoc of the same release."
    )
    parser.add_argument(
        "--xmldoc",
        type=Path,
        default=None,
        help="Path to share/Pythia8/xmldoc. Defaults to the installed "
        "pythia8mc's copy.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SNAPSHOT_PATH,
        help="Output snapshot path (.json.gz).",
    )
    args = parser.parse_args(argv)

    xmldoc = args.xmldoc or default_xmldoc_dir()
    if xmldoc is None:
        parser.error(
            "no xmldoc directory found; pass --xmldoc "
            "/path/to/share/Pythia8/xmldoc"
        )

    documented = parse_xmldoc(xmldoc)
    pythia = _quiet_pythia()
    version = f"{float(pythia.settings.parm('Pythia:versionNumber')):.3f}"
    if documented.get("pythia_version") and float(
        documented["pythia_version"]
    ) != float(version):
        parser.error(
            f"xmldoc {xmldoc} documents Pythia {documented['pythia_version']} "
            f"but the "
            f"installed pythia8mc is {version}; pass the xmldoc of the "
            f"installed release"
        )
    snapshot = merge_installed(documented, installed_settings(pythia), version)
    snapshot.pop("database", None)
    snapshot["provenance"] = {
        "pythia_version": version,
        "xmldoc_sha256": _xmldoc_digest(xmldoc),
        "generated_by": "python -m tools.pythia.settings_db",
    }
    out = write_snapshot(snapshot, args.out)
    print(
        f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB): "
        f"{snapshot['n_settings']} settings from Pythia "
        f"{snapshot['pythia_version']} "
        f"({xmldoc})"
    )
    stale = snapshot.get("stale_aliases") or []
    if stale:
        print(
            f"warning: {len(stale)} curated alias target(s) absent from this "
            f"Pythia release -- update settings_topics.ALIASES: "
            f"{', '.join(stale)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
