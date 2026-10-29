"""
# catalogue.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

The analyses a Rivet installation ships, read from their metadata files.

Every Rivet analysis comes with an `.info` file (YAML: the experiment, the
beams and energies it was measured at, what to generate for it, its status,
its references), usually a `.plot` file (axis labels and titles for its
histograms) and, for a measurement, a `.yoda.gz` of the published data. They
all live in `rivet-config --datadir`. Reading them directly, rather than
through `rivet --show-analysis`, makes a search over two thousand analyses
take well under a second and need nothing but the data directory.

Beam compatibility is also decided here, from the same metadata, because
Rivet itself does not check it fully: an analysis whose beam *particles* do
not match the events is removed with a warning, but one measured at a
different *energy* runs without a word and fills its histograms with the
wrong physics.
"""

import difflib
import gzip
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Beam names as Rivet's .info files spell them, with the spellings and PDG ids
# that mean the same particle. Nuclei keep the symbol the file uses.
_BEAM_NAMES: Dict[Any, str] = {
    "p": "p+",
    "p+": "p+",
    "proton": "p+",
    2212: "p+",
    "p-": "p-",
    "pbar": "p-",
    "pbar-": "p-",
    "antiproton": "p-",
    -2212: "p-",
    "e-": "e-",
    "electron": "e-",
    11: "e-",
    "e+": "e+",
    "positron": "e+",
    -11: "e+",
    "mu-": "mu-",
    13: "mu-",
    "mu+": "mu+",
    -13: "mu+",
    "gamma": "gamma",
    "photon": "gamma",
    22: "gamma",
    "n": "n0",
    "n0": "n0",
    "neutron": "n0",
    2112: "n0",
    "pi+": "pi+",
    211: "pi+",
    "pi-": "pi-",
    -211: "pi-",
    "k+": "K+",
    321: "K+",
    "k-": "K-",
    -321: "K-",
}

# Nuclear PDG ids (10LZZZAAAI) to the element symbols the .info files use.
_ELEMENTS = {
    1: "d",
    2: "He",
    8: "O",
    10: "Ne",
    18: "Ar",
    20: "Ca",
    29: "Cu",
    36: "Kr",
    54: "Xe",
    79: "Au",
    82: "Pb",
    92: "U",
}

# Tolerance on the centre-of-mass energy when matching an analysis to a run.
ENERGY_TOLERANCE = 0.01

# Status words that mean "do not trust this for physics".
_CAUTION_STATUS = (
    "UNVALIDATED",
    "PRELIMINARY",
    "OBSOLETE",
    "UNPHYSICAL",
    "EXAMPLE",
    "NEW",
)


def beam_name(beam: Any) -> str:
    """
    The canonical name of a beam particle, from a .info spelling or a PDG id.

    Unknown names are returned as given, so a nucleus written "Pb" or an
    exotic beam still compares equal to itself.
    """
    if isinstance(beam, bool):
        raise ValueError(f"not a beam: {beam!r}")
    if isinstance(beam, int) or (
        isinstance(beam, str) and re.fullmatch(r"-?\d+", beam.strip())
    ):
        pid = int(beam)
        if pid in _BEAM_NAMES:
            return _BEAM_NAMES[pid]
        if abs(pid) > 1000000000:  # a nucleus: 10LZZZAAAI
            z = abs(pid) // 10000 % 1000
            return _ELEMENTS.get(z, f"ion(Z={z})")
        return str(pid)
    text = str(beam).strip()
    return _BEAM_NAMES.get(text, _BEAM_NAMES.get(text.lower(), text))


def normalise_beams(beams: Any) -> List[Tuple[str, str]]:
    """
    The beam pairs a .info `Beams` entry allows, each sorted, as canonical
    names.

    Accepts one pair (`[p+, p+]`), a list of pairs, or the numbered mapping
    some files use (`{1: [p+, p+], 2: [p+, Pb]}`); a run's beams may be given
    the same way or as PDG ids.
    """
    if beams is None:
        return []
    if isinstance(beams, dict):
        beams = list(beams.values())
    if isinstance(beams, str):
        groups = re.findall(r"\[([^\[\]]*)\]", beams)
        if len(groups) > 1:
            beams = [
                [b for b in re.split(r"[,\s]+", g.strip()) if b]
                for g in groups
            ]
        else:
            beams = [b for b in re.split(r"[,\s]+", beams.strip("[] ")) if b]
    beams = list(beams)
    if not beams:
        return []
    if all(isinstance(b, (list, tuple)) for b in beams):
        pairs = beams
    else:
        pairs = [beams]
    out = []
    for pair in pairs:
        names = tuple(sorted(beam_name(b) for b in pair))
        if len(names) != 2:
            raise ValueError(
                f"a beam specification needs two particles, got {pair!r}"
            )
        if names not in out:
            out.append(names)
    return out


def normalise_energies(energies: Any) -> List[float]:
    """
    The centre-of-mass energies in GeV a .info `Energies` entry allows.

    An entry is either sqrt(s) or a pair of beam energies, for which sqrt(s)
    is taken as 2 sqrt(E1 E2), exact for massless beams and within a part in
    a million for anything at a collider. The numbered mapping some files use
    (`{1: [5020, 7000], 2: [...]}`, one list per beam option) holds sqrt(s)
    values, not pairs. An empty list means any energy.
    """
    if energies is None:
        return []
    if isinstance(energies, (int, float)):
        return [float(energies)]
    if isinstance(energies, str):
        groups = re.findall(r"\[([^\[\]]*)\]", energies)
        if len(groups) > 1:
            energies = {
                i: [e for e in re.split(r"[,\s]+", g.strip()) if e]
                for i, g in enumerate(groups)
            }
        else:
            energies = [
                e for e in re.split(r"[,\s]+", energies.strip("[] ")) if e
            ]
    if isinstance(energies, dict):
        energies = [
            e
            for value in energies.values()
            for e in (value if isinstance(value, (list, tuple)) else [value])
        ]
    out = []
    for entry in energies:
        if isinstance(entry, (list, tuple)):
            if len(entry) == 1:
                value = float(entry[0])
            elif len(entry) == 2:
                value = 2.0 * math.sqrt(float(entry[0]) * float(entry[1]))
            else:
                raise ValueError(
                    f"an energy entry has one or two numbers, got {entry!r}"
                )
        else:
            value = float(entry)
        out.append(value)
    return out


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("yes", "true", "1", "on"):
        return True
    if text in ("no", "false", "0", "off"):
        return False
    return None


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [
            str(v).strip() for v in value if v is not None and str(v).strip()
        ]
    return [
        s
        for s in (
            part.strip() for part in str(value).replace(";", ",").split(",")
        )
        if s
    ]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return " ".join(str(value).split())


@dataclass
class Analysis:
    """One Rivet analysis, as its metadata files describe it."""

    name: str
    summary: str = ""
    description: str = ""
    experiment: str = ""
    collider: str = ""
    year: Optional[int] = None
    inspire_id: str = ""
    status: str = ""  # VALIDATED, UNVALIDATED, PRELIMINARY, OBSOLETE, ...
    status_flags: List[str] = field(
        default_factory=list
    )  # NOHEPDATA, SINGLEWEIGHT, ...
    beams: List[Tuple[str, str]] = field(default_factory=list)  # [] means any
    energies_gev: List[float] = field(default_factory=list)  # [] means any
    run_info: str = ""
    options: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    needs_cross_section: Optional[bool] = None
    luminosity_fb: Any = None
    num_events: Optional[int] = None
    references: List[str] = field(default_factory=list)
    bibkey: str = ""
    reentrant: Optional[bool] = None
    has_reference_data: bool = False
    has_plot_info: bool = False
    info_path: str = ""
    parse_error: str = ""

    @property
    def validated(self) -> bool:
        return self.status == "VALIDATED"

    def compatible(
        self,
        beams: Any,
        energy_gev: Optional[float],
        tolerance: float = ENERGY_TOLERANCE,
    ) -> Tuple[bool, str]:
        """
        Whether a run with these beams at this sqrt(s) is one the analysis
        was made for, and why not when it is not.

        An analysis that lists no beams or no energies accepts any.
        """
        run = normalise_beams(beams) if beams else []
        if self.beams and run:
            if not any(pair in self.beams for pair in run):
                return False, (
                    f"beams {' '.join(run[0])} are not among the analysis's "
                    f"{' | '.join(' '.join(p) for p in self.beams)}"
                )
        if self.energies_gev and energy_gev is not None:
            if not any(
                abs(energy_gev - e) <= tolerance * e for e in self.energies_gev
            ):
                return False, (
                    f"sqrt(s) = {energy_gev:g} GeV is not among the "
                    f"analysis's "
                    f"{', '.join(f'{e:g}' for e in self.energies_gev)} GeV"
                )
        return True, ""

    def to_dict(self, detail: bool = False) -> Dict[str, Any]:
        """A JSON-ready record; `detail` adds the long fields."""
        record: Dict[str, Any] = {
            "name": self.name,
            "summary": self.summary,
            "experiment": self.experiment,
            "collider": self.collider,
            "year": self.year,
            "status": self.status,
            "beams": [" ".join(pair) for pair in self.beams] or ["any"],
            "energies_gev": self.energies_gev or ["any"],
            "has_reference_data": self.has_reference_data,
            "needs_cross_section": self.needs_cross_section,
            "inspire_id": self.inspire_id,
        }
        if self.status_flags:
            record["status_flags"] = self.status_flags
        if self.options:
            record["options"] = self.options
        if self.keywords:
            record["keywords"] = self.keywords
        if detail:
            record.update(
                {
                    "run_info": self.run_info,
                    "description": self.description,
                    "references": self.references,
                    "bibkey": self.bibkey,
                    "luminosity_fb": self.luminosity_fb,
                    "num_events": self.num_events,
                    "reentrant": self.reentrant,
                    "has_plot_info": self.has_plot_info,
                    "info_path": self.info_path,
                }
            )
            if self.parse_error:
                record["parse_error"] = self.parse_error
        return record


def _fallback_parse(text: str) -> Dict[str, Any]:
    """
    Key: value pairs from a .info file that is not valid YAML.

    Top-level scalars are recovered with their continuation lines, block
    lists (`- item`) as lists, and flow lists as the raw bracketed text, which
    the normalisers accept.
    """
    out: Dict[str, Any] = {}
    key = None
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if match:
            key, value = match.group(1), match.group(2)
            out[key] = value
        elif key and re.match(r"^\s*-\s+", line):
            if not isinstance(out.get(key), list):
                out[key] = [] if not out.get(key) else [out[key]]
            out[key].append(re.sub(r"^\s*-\s+", "", line).strip())
        elif (
            key
            and line.startswith((" ", "\t"))
            and isinstance(out.get(key), str)
        ):
            out[key] = (out[key] + " " + line.strip()).strip()
    return out


_BIBTEX_BLOCK = re.compile(
    r"^BibTeX:.*?(?=^[A-Za-z_]+:|\Z)", re.MULTILINE | re.DOTALL
)


def _load_yaml(text: str) -> Tuple[Dict[str, Any], str]:
    """
    A .info file's mapping, and the parse error if it was not clean YAML.

    The files are hand-written and read by yaml-cpp, which is more forgiving
    than PyYAML; the usual offender is the quoted BibTeX entry, so a file that
    fails is retried without it before the line-based fallback.
    """
    import yaml  # noqa: PLC0415 - a bundle dependency, imported where used

    def attempt(source: str) -> Dict[str, Any]:
        data = yaml.safe_load(source)
        if not isinstance(data, dict):
            raise ValueError("not a mapping")
        return data

    try:
        return attempt(text), ""
    except Exception as first:
        try:
            return attempt(_BIBTEX_BLOCK.sub("", text)), ""
        except Exception:
            message = str(first).splitlines()[0] if str(first) else ""
            return _fallback_parse(
                text
            ), f"{type(first).__name__}: {message}".strip()


def parse_info(path: str) -> Analysis:
    """Read one analysis's .info file into an :class:`Analysis`."""
    name = os.path.basename(path)[: -len(".info")]
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    data, error = _load_yaml(text)

    # The same field under its common misspellings.
    def pick(*keys: str) -> Any:
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return None

    status_words = [
        w for w in re.split(r"[\s,]+", _as_text(pick("Status"))) if w
    ]
    try:
        beams = normalise_beams(pick("Beams"))
    except (ValueError, TypeError):
        beams, error = [], error or f"unreadable Beams: {pick('Beams')!r}"
    try:
        energies = normalise_energies(
            pick("Energies", "Energies_GeV", "Enegies")
        )
    except (ValueError, TypeError):
        energies, error = (
            [],
            error or f"unreadable Energies: {pick('Energies')!r}",
        )
    year = pick("Year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    num_events = pick("NumEvents")
    try:
        num_events = int(float(num_events)) if num_events is not None else None
    except (TypeError, ValueError):
        num_events = None
    base = path[: -len(".info")]
    return Analysis(
        name=str(pick("Name") or name),
        summary=_as_text(pick("Summary")),
        description=_as_text(pick("Description")),
        experiment=_as_text(pick("Experiment")),
        collider=_as_text(pick("Collider")),
        year=year,
        inspire_id=_as_text(pick("InspireID", "SpiresID", "SpireID")),
        status=status_words[0].upper() if status_words else "",
        status_flags=[w.upper() for w in status_words[1:]],
        beams=beams,
        energies_gev=energies,
        run_info=_as_text(pick("RunInfo")),
        options=_as_list(pick("Options")),
        keywords=_as_list(pick("Keywords")),
        needs_cross_section=_as_bool(
            pick("NeedCrossSection", "NeedsCrossSection", "NeedCrosssection")
        ),
        luminosity_fb=pick("Luminosity_fb"),
        num_events=num_events,
        references=_as_list(pick("References")),
        bibkey=_as_text(pick("BibKey")),
        reentrant=_as_bool(pick("Reentrant")),
        has_reference_data=os.path.exists(base + ".yoda.gz")
        or os.path.exists(base + ".yoda"),
        has_plot_info=os.path.exists(base + ".plot"),
        info_path=path,
        parse_error=error,
    )


class Catalogue:
    """Every analysis of one installation, searchable."""

    def __init__(
        self, analyses: Iterable[Analysis], data_dir: str = ""
    ) -> None:
        self.data_dir = data_dir
        self._by_name: Dict[str, Analysis] = {a.name: a for a in analyses}

    def __len__(self) -> int:
        return len(self._by_name)

    def __iter__(self):
        return iter(sorted(self._by_name.values(), key=lambda a: a.name))

    def names(self) -> List[str]:
        return sorted(self._by_name)

    def get(self, name: str) -> Optional[Analysis]:
        """The analysis of that name; an attached option string (`NAME:OPT=V`)
        is ignored."""
        return self._by_name.get(base_name(name))

    def suggest(self, name: str, limit: int = 5) -> List[str]:
        """Names close to a misspelt one."""
        return difflib.get_close_matches(
            base_name(name), self.names(), n=limit, cutoff=0.6
        )

    def search(
        self,
        query: str = "",
        *,
        experiment: str = "",
        collider: str = "",
        beams: Any = None,
        energy_gev: Optional[float] = None,
        status: str = "",
        keyword: str = "",
        with_reference_data: Optional[bool] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> List[Analysis]:
        """
        The analyses matching every given criterion, best matches first.

        `query` is split into words that must all occur in the name, summary,
        description, keywords, experiment, collider or INSPIRE id. `beams`
        and `energy_gev` keep the analyses compatible with such a run, in the
        sense of :meth:`Analysis.compatible`.
        """
        words = [w for w in _searchable(query).split() if w]
        matches: List[Tuple[int, Analysis]] = []
        for analysis in self._by_name.values():
            if (
                experiment
                and analysis.experiment.lower() != experiment.lower()
            ):
                continue
            if collider and analysis.collider.lower() != collider.lower():
                continue
            if status and analysis.status != status.upper():
                continue
            if keyword and keyword.lower() not in [
                k.lower() for k in analysis.keywords
            ]:
                continue
            if (
                with_reference_data is not None
                and analysis.has_reference_data != with_reference_data
            ):
                continue
            if year_min is not None and (
                analysis.year is None or analysis.year < year_min
            ):
                continue
            if year_max is not None and (
                analysis.year is None or analysis.year > year_max
            ):
                continue
            if (beams or energy_gev is not None) and not analysis.compatible(
                beams, energy_gev
            )[0]:
                continue
            score = 0
            if words:
                haystack = _searchable(
                    " ".join(
                        [
                            analysis.name,
                            analysis.summary,
                            analysis.description,
                            " ".join(analysis.keywords),
                            analysis.experiment,
                            analysis.collider,
                            analysis.inspire_id,
                        ]
                    )
                )
                found = haystack.split()
                if not all(w in found for w in words):
                    continue
                score += sum(
                    3 for w in words if w in _searchable(analysis.name).split()
                )
                score += sum(
                    2
                    for w in words
                    if w in _searchable(analysis.summary).split()
                )
            # Measurements with data and a VALIDATED status rank above tools
            # and unvalidated code when the text matches equally.
            score += 1 if analysis.has_reference_data else 0
            score += 1 if analysis.validated else 0
            matches.append((score, analysis))
        matches.sort(key=lambda item: (-item[0], item[1].name))
        return [analysis for _, analysis in matches]


_MARKUP = re.compile(
    r"[\\${}^*_~]|\b(?:mathrm|text|rm|frac|sqrt|left|right)\b"
)
_PUNCTUATION = re.compile(r"[^\w+\-.]+")


def _searchable(text: str) -> str:
    """Text lowered and stripped of LaTeX markup, so `phi*` meets
    `$\\phi^*$`."""
    text = _MARKUP.sub(" ", text.lower())
    return " ".join(_PUNCTUATION.sub(" ", text).split())


def base_name(analysis: str) -> str:
    """`NAME` of a `NAME:OPT1=V1:OPT2=V2` specification, as Rivet reads it."""
    return analysis.split(":", 1)[0].strip()


_CATALOGUES: Dict[str, Tuple[float, Catalogue]] = {}


def load_catalogue(data_dir: str) -> Catalogue:
    """
    The catalogue of a Rivet data directory, cached while the directory's
    modification time is unchanged.
    """
    try:
        stamp = os.stat(data_dir).st_mtime
    except OSError as e:
        raise FileNotFoundError(
            f"Rivet data directory not found: {data_dir}"
        ) from e
    cached = _CATALOGUES.get(data_dir)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    analyses = [
        parse_info(os.path.join(data_dir, name))
        for name in sorted(os.listdir(data_dir))
        if name.endswith(".info")
    ]
    catalogue = Catalogue(analyses, data_dir)
    _CATALOGUES[data_dir] = (stamp, catalogue)
    return catalogue


# ---------------------------------------------------------------------- #
# Reference data and plot styles
# ---------------------------------------------------------------------- #

_YODA_BEGIN = re.compile(r"^BEGIN YODA_(\w+?)(?:_V\d+)? (/\S+)")


def reference_paths(data_dir: str, analysis: str) -> List[Tuple[str, str]]:
    """
    The (path, type) of every object in an analysis's reference YODA file,
    read from the file's own BEGIN lines, so no YODA library is needed.
    """
    for suffix, opener in ((".yoda.gz", gzip.open), (".yoda", open)):
        path = os.path.join(data_dir, base_name(analysis) + suffix)
        if os.path.exists(path):
            out = []
            with opener(
                path, "rt", encoding="utf-8", errors="replace"
            ) as handle:
                for line in handle:
                    match = _YODA_BEGIN.match(line)
                    if match:
                        out.append((match.group(2), match.group(1)))
            return out
    return []


def parse_plot_info(path: str) -> List[Tuple[re.Pattern, Dict[str, str]]]:
    """
    The `# BEGIN PLOT <pattern> ... # END PLOT` blocks of a .plot file.

    Each block's key is a path pattern (a regular expression such as
    `/ANA/d..-x..-y..`) and its body `Key=Value` lines; later blocks override
    earlier ones for the paths they match, which is how Rivet's own plotting
    applies them.
    """
    blocks: List[Tuple[re.Pattern, Dict[str, str]]] = []
    pattern: Optional[re.Pattern] = None
    body: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("# BEGIN PLOT"):
                spec = line[len("# BEGIN PLOT") :].strip()
                try:
                    pattern = re.compile(spec)
                except re.error:
                    pattern = re.compile(re.escape(spec))
                body = {}
            elif line.startswith("# END PLOT"):
                if pattern is not None:
                    blocks.append((pattern, body))
                pattern = None
            elif (
                pattern is not None
                and "=" in line
                and not line.startswith("#")
            ):
                key, value = line.split("=", 1)
                body[key.strip()] = value.strip()
    return blocks


def plot_style(
    blocks: Sequence[Tuple[re.Pattern, Dict[str, str]]], path: str
) -> Dict[str, str]:
    """The merged plot settings for one histogram path."""
    style: Dict[str, str] = {}
    for pattern, body in blocks:
        if pattern.match(path):
            style.update(body)
    return style


def plot_styles(
    data_dir: str, analysis: str
) -> List[Tuple[re.Pattern, Dict[str, str]]]:
    """The .plot blocks of an analysis, or none when it has no .plot file."""
    path = os.path.join(data_dir, base_name(analysis) + ".plot")
    return parse_plot_info(path) if os.path.exists(path) else []


def caution(analysis: Analysis) -> List[str]:
    """Why a result from this analysis should be read with care, if at all."""
    notes = []
    if analysis.status in _CAUTION_STATUS:
        notes.append(
            f"status {analysis.status}: Rivet's own validation of this "
            f"analysis "
            "is incomplete or it is superseded"
        )
    if "SINGLEWEIGHT" in analysis.status_flags:
        notes.append(
            "SINGLEWEIGHT: the analysis handles only the nominal weight; "
            "variation weights are ignored"
        )
    if not analysis.has_reference_data and not analysis.name.startswith("MC_"):
        notes.append(
            "no reference data ships with this analysis; nothing to compare to"
        )
    return notes
