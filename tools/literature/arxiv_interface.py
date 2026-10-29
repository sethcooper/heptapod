"""
# arxiv_interface.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

arXiv Atom API interface: search, PDF download, and LaTeX e-print retrieval.

Complements the INSPIRE tools, which have no arXiv access or PDF retrieval,
so an agent can go from a paper hit to the source that downstream Lagrangian
extraction needs. Uses ``requests`` only (a HEPTAPOD base dependency).

Turning a PDF into text is deliberately NOT done here: ``PDFToTeXTool`` in
this bundle reconstructs TeX macros for symbols, sub/superscripts, fractions
and radicals, which plain text extraction destroys. Prefer the e-print source
where arXiv has it, and that tool where it does not.
"""

import hashlib
import http.client
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# arXiv API usage guidance: no more than one request roughly every 3 seconds.
_MIN_REQUEST_INTERVAL = 3.0

# Cap on downloaded PDF size (defensive; arXiv PDFs are rarely > a few tens of MB).
_MAX_PDF_BYTES = 100 * 1024 * 1024

# arXiv answers bursts with 429 (or 503) and sometimes cuts a download short
# (IncompleteRead). E-print downloads are retried after these pauses, or after
# the server's Retry-After when it sends one, as long as the total wait stays
# within _MAX_RETRY_WAIT seconds; then the error is raised.
_RETRY_DELAYS = (5.0, 15.0, 45.0)
_MAX_RETRY_WAIT = 120.0
_RETRY_STATUS = {429, 503}

# Only arXiv hosts, over HTTPS, may be fetched (avoids SSRF via arbitrary URLs).
_ALLOWED_PDF_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}

# arXiv identifier forms (version suffix optional):
#   new style: 2103.02708, 2103.02708v2   (YYMM.NNNNN)
#   old style: hep-ph/9905221, math.AG/0309136v1
_ARXIV_NEW_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_OLD_RE = re.compile(r"^[a-z][a-z-]*(\.[A-Za-z]{2})?/\d{7}(v\d+)?$")


def _strip_arxiv_prefix(arxiv_id: str) -> str:
    """Drop an optional case-insensitive ``arXiv:`` prefix and surrounding space."""
    return re.sub(r"(?i)^arxiv:", "", (arxiv_id or "").strip())


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Canonical id: strip an ``arXiv:`` prefix and any ``vN`` version suffix.

    e.g. ``arXiv:2101.00001v2`` -> ``2101.00001``. Used for dedup/display, not
    for fetching (fetching preserves an explicit version).
    """
    return re.sub(r"v\d+$", "", _strip_arxiv_prefix(arxiv_id))


def split_version(arxiv_id: str) -> Tuple[str, Optional[str]]:
    """``2103.02708v2`` -> ``("2103.02708", "v2")``; no version -> ``(id, None)``."""
    s = _strip_arxiv_prefix(arxiv_id)
    m = re.match(r"^(.*?)(v\d+)$", s)
    return (m.group(1), m.group(2)) if m else (s, None)


def version_from_response(base_id: str, *texts: Optional[str]) -> Optional[str]:
    """Find ``<base_id>vN`` in a Content-Disposition header or a URL.

    arXiv names the file it serves after the version it chose
    (``filename="2103.02708v2.pdf"``, ``"arXiv-2103.02708v2.tar.gz"``), so an
    unversioned request still reveals which version came back.
    """
    flat = re.escape(base_id.replace("/", "_"))
    raw = re.escape(base_id)
    for t in texts:
        if not t or not isinstance(t, str):
            continue
        m = re.search(rf"(?:{raw}|{flat})(v\d+)", t)
        if m:
            return m.group(1)
    return None


def arxiv_id_from_url(url: Optional[str]) -> Optional[str]:
    """The arXiv id in an ``https://arxiv.org/{pdf,abs}/<id>[.pdf]`` URL, if any."""
    if not url:
        return None
    m = re.search(r"/(?:pdf|abs|e-print|src)/(.+?)(?:\.pdf)?/?$", urlparse(url).path or "")
    return validate_arxiv_id(m.group(1)) if m else None


def validate_arxiv_id(arxiv_id: str) -> Optional[str]:
    """Return the well-formed id (``arXiv:`` prefix stripped, version preserved),
    or ``None`` if ``arxiv_id`` is not a recognizable arXiv identifier."""
    if not arxiv_id:
        return None
    s = _strip_arxiv_prefix(arxiv_id)
    if _ARXIV_NEW_RE.match(s) or _ARXIV_OLD_RE.match(s):
        return s
    return None


def is_allowed_arxiv_url(url: str) -> bool:
    """True only for HTTPS URLs on a known arXiv host."""
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return False
    return parsed.scheme == "https" and (parsed.hostname or "") in _ALLOWED_PDF_HOSTS


# Back-compat alias (original, PDF-specific name).
is_allowed_pdf_url = is_allowed_arxiv_url


def https_arxiv_url(url: Optional[str]) -> Optional[str]:
    """Upgrade an ``http://`` arXiv URL to ``https://`` (arXiv feeds emit http).

    Leaves non-arXiv or already-https URLs unchanged.
    """
    if not url or not url.startswith("http://"):
        return url
    host = (urlparse(url).hostname or "")
    if host in _ALLOWED_PDF_HOSTS or host.endswith(".arxiv.org") or host == "arxiv.org":
        return "https://" + url[len("http://"):]
    return url


def _response_meta(resp: Any, fallback_url: str) -> Tuple[str, Optional[str]]:
    """(final URL after redirects, Content-Disposition) of a response."""
    url = getattr(resp, "url", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        disp = headers.get("Content-Disposition") or headers.get("content-disposition")
    except AttributeError:
        disp = None
    return (url if isinstance(url, str) and url else fallback_url,
            disp if isinstance(disp, str) else None)


class _MinIntervalLimiter:
    """Serialize requests so consecutive calls are >= ``interval`` seconds apart."""

    def __init__(self, interval: float = _MIN_REQUEST_INTERVAL) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> float:
        """Block until at least ``interval`` seconds have passed since the last call."""
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            waited = 0.0
            if self._last and delta < self.interval:
                waited = self.interval - delta
                time.sleep(waited)
            self._last = time.monotonic()
            return waited


def _sleep(seconds: float) -> None:
    """The pause between retries (tests replace it)."""
    time.sleep(seconds)


def _retryable(exc: BaseException) -> bool:
    """A rate-limit or server-busy status, or a download cut short."""
    if isinstance(exc, requests.HTTPError):
        return getattr(exc.response, "status_code", None) in _RETRY_STATUS
    if isinstance(exc, (requests.exceptions.ChunkedEncodingError, http.client.IncompleteRead)):
        return True
    return "IncompleteRead" in str(exc)


def _retry_after(exc: BaseException) -> Optional[float]:
    """Seconds from a numeric Retry-After header on the error's response, if any."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        value = headers.get("Retry-After")
        return max(0.0, float(value)) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def with_retries(fn, delays=_RETRY_DELAYS, max_wait: float = _MAX_RETRY_WAIT):
    """``fn()``, retried after each of ``delays`` while the error is retryable.

    The last error is raised once the delays are used up, the error is not
    retryable, or the next pause would take the total wait past ``max_wait``.
    """
    waited = 0.0
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified below
            if attempt == len(delays) or not _retryable(e):
                raise
            pause = _retry_after(e)
            pause = delays[attempt] if pause is None else pause
            if waited + pause > max_wait:
                raise
            _sleep(pause)
            waited += pause


# Module-level limiter shared across tool invocations in a process so repeated
# arXiv calls from an agent stay polite even though each tool builds its own
# interface instance.
_LIMITER = _MinIntervalLimiter()


class ArxivInterface:
    """Thin client over the arXiv Atom export API plus PDF retrieval/extraction."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.last_raw: Optional[str] = None  # body of the last search response
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "HEPTAPOD/1.0 (Physics research tool)"}
        )

    # ----------------------------- Search ----------------------------- #

    @staticmethod
    def build_query(
        model_name: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        category: Optional[str] = "hep-ph",
    ) -> str:
        """Build an arXiv ``search_query`` from a model name and optional keywords.

        Example: ``cat:hep-ph AND (ti:"scalar leptoquark" OR abs:"scalar leptoquark")
        AND (abs:"BSM" OR abs:"leptoquark")``.

        ``model_name`` is optional. It reflects the BSM-model framing this
        bundle started from, where a search is anchored on a named model.
        Other categories have no such anchor — an astro-ph.EP search for
        transmission spectroscopy is keywords and nothing else — so keywords
        alone build a valid query.
        """
        clauses: List[str] = []
        if category:
            clauses.append(f"cat:{category}")
        if model_name and model_name.strip():
            m = model_name.strip()
            clauses.append(f'(ti:"{m}" OR abs:"{m}")')
        if keywords:
            kw = " OR ".join(f'abs:"{k.strip()}"' for k in keywords if k and k.strip())
            if kw:
                clauses.append(f"({kw})")
        return " AND ".join(clauses)

    def search(
        self,
        query: str,
        max_results: int = 25,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Search arXiv. Returns ``(papers, request_url)``."""
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        _LIMITER.wait()
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        self.last_raw = resp.text
        return self._parse_feed(resp.text), url

    def resolve_version(self, arxiv_id: str) -> Dict[str, Any]:
        """Pin an arXiv id to a version: ``{"versioned_id", "request_url"}``.

        An explicit ``vN`` is returned as is (``request_url`` None). Otherwise
        the Atom metadata for the id is fetched and the version is read from
        its ``<id>`` element, which names the current version. Raises
        ValueError when no version can be determined.
        """
        valid = validate_arxiv_id(arxiv_id)
        if not valid:
            raise ValueError(f"Invalid arXiv id: {arxiv_id!r}")
        base, version = split_version(valid)
        if version:
            return {"versioned_id": valid, "request_url": None}
        url = f"{ARXIV_API_URL}?{urlencode({'id_list': base, 'max_results': 1})}"
        _LIMITER.wait()
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        for entry in ET.fromstring(resp.text).findall("atom:entry", ATOM_NS):
            raw = self._extract_arxiv_id(entry) or ""
            b, v = split_version(raw)
            if v and b == base:
                return {"versioned_id": f"{base}{v}", "request_url": url}
        raise ValueError(f"arXiv metadata for {base!r} names no version")

    def _parse_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        root = ET.fromstring(xml_text)
        papers: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            parsed = self._parse_entry(entry)
            if parsed is not None:
                papers.append(parsed)
        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[Dict[str, Any]]:
        title_el = entry.find("atom:title", ATOM_NS)
        if title_el is None or not title_el.text:
            return None
        title = " ".join(title_el.text.split())

        summary_el = entry.find("atom:summary", ATOM_NS)
        abstract = (
            " ".join(summary_el.text.split())
            if summary_el is not None and summary_el.text
            else None
        )

        authors = [
            name_el.text.strip()
            for author_el in entry.findall("atom:author", ATOM_NS)
            if (name_el := author_el.find("atom:name", ATOM_NS)) is not None
            and name_el.text
        ]

        raw_id = self._extract_arxiv_id(entry)
        if not raw_id:
            return None
        arxiv_id = normalize_arxiv_id(raw_id) or raw_id
        _, version = split_version(raw_id)

        categories: List[str] = []
        primary = entry.find("arxiv:primary_category", ATOM_NS)
        if primary is not None and primary.get("term"):
            categories.append(primary.get("term", ""))
        for cat in entry.findall("atom:category", ATOM_NS):
            term = cat.get("term")
            if term and term not in categories:
                categories.append(term)

        doi_el = entry.find("arxiv:doi", ATOM_NS)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        pdf_url: Optional[str] = None
        abs_url: Optional[str] = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("rel") == "alternate":
                abs_url = link.get("href")
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")

        return {
            "arxiv_id": arxiv_id,
            # The Atom <id> names the version the metadata describes (the
            # latest at query time); evidence must cite that, not the bare id.
            "version": version,
            "versioned_id": f"{arxiv_id}{version}" if version else None,
            "source_id": f"arXiv:{arxiv_id}{version}" if version else None,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "categories": categories,
            "doi": doi,
            "published": self._parse_date(entry.find("atom:published", ATOM_NS)),
            "updated": self._parse_date(entry.find("atom:updated", ATOM_NS)),
            "pdf_url": https_arxiv_url(pdf_url) or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "abs_url": https_arxiv_url(abs_url) or f"https://arxiv.org/abs/{arxiv_id}",
        }

    @staticmethod
    def _parse_date(element: Optional[ET.Element]) -> Optional[str]:
        if element is None or not element.text:
            return None
        return element.text[:10]

    @staticmethod
    def _extract_arxiv_id(entry: ET.Element) -> Optional[str]:
        id_el = entry.find("atom:id", ATOM_NS)
        if id_el is None or not id_el.text:
            return None
        match = re.search(r"arxiv\.org/abs/(.+)$", id_el.text.strip())
        return match.group(1) if match else None

    # -------------------------- PDF retrieval ------------------------- #

    def download_pdf(
        self,
        dest_path: str,
        arxiv_id: Optional[str] = None,
        pdf_url: Optional[str] = None,
        max_bytes: int = _MAX_PDF_BYTES,
    ) -> Dict[str, Any]:
        """Download a PDF to ``dest_path``, streaming with a size cap.

        Provide either ``arxiv_id`` (the canonical arXiv PDF URL is derived,
        preserving an explicit ``vN`` version) or an explicit ``pdf_url`` (which
        must be an HTTPS arXiv URL). Validates the ``%PDF-`` header, writes to a
        temp file, and atomically renames on success. Returns ``{"bytes",
        "sha256"}``.
        """
        if not pdf_url:
            if not arxiv_id:
                raise ValueError("Either arxiv_id or pdf_url is required")
            valid = validate_arxiv_id(arxiv_id)
            if not valid:
                raise ValueError(f"Invalid arXiv id: {arxiv_id!r}")
            pdf_url = f"https://arxiv.org/pdf/{valid}.pdf"
        if not is_allowed_pdf_url(pdf_url):
            raise ValueError(f"Refusing non-arXiv or non-HTTPS PDF URL: {pdf_url!r}")

        _LIMITER.wait()
        tmp = dest_path + ".part"
        final_url, disposition = pdf_url, None
        digest = hashlib.sha256()
        total = 0
        header = b""
        validated = False
        with self._session.get(pdf_url, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            final_url, disposition = _response_meta(resp, pdf_url)
            try:
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        if not validated:
                            # Buffer until we have >= 5 bytes: iter_content does
                            # not guarantee the first chunk contains the full
                            # "%PDF-" magic.
                            header += chunk
                            if len(header) < 5:
                                continue
                            if header[:5] != b"%PDF-":
                                raise ValueError("Downloaded file is not a valid PDF")
                            validated = True
                            chunk = header  # flush the buffered header bytes
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(
                                f"PDF exceeds max size of {max_bytes} bytes"
                            )
                        digest.update(chunk)
                        fh.write(chunk)
                if not validated:
                    raise ValueError("Downloaded file is not a valid PDF")
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise

        try:
            os.replace(tmp, dest_path)
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return {"bytes": total, "sha256": digest.hexdigest(), "url": pdf_url,
                "final_url": final_url, "content_disposition": disposition}

    # ------------------------ E-print retrieval ----------------------- #

    def download_eprint(
        self,
        arxiv_id: str,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> Dict[str, Any]:
        """Download the arXiv e-print source payload for ``arxiv_id``.

        Fetches ``https://export.arxiv.org/e-print/<id>`` (gzipped tar of LaTeX
        sources, gzipped single ``.tex``, or a bare PDF when no source exists),
        streaming with a size cap. Returns ``{"data": bytes, "sha256": str,
        "bytes": int}``; the caller classifies via source_archive.detect_payload.
        Rate limiting (429/503) and truncated downloads are retried with
        backoff (see ``with_retries``) before the error is raised.
        """
        valid = validate_arxiv_id(arxiv_id)
        if not valid:
            raise ValueError(f"Invalid arXiv id: {arxiv_id!r}")
        url = f"https://export.arxiv.org/e-print/{valid}"
        if not is_allowed_arxiv_url(url):  # defensive; host is ours by construction
            raise ValueError(f"Refusing non-arXiv URL: {url!r}")
        return with_retries(lambda: self._fetch_eprint(url, max_bytes))

    def _fetch_eprint(self, url: str, max_bytes: int) -> Dict[str, Any]:
        """One attempt at downloading the e-print at ``url``."""
        _LIMITER.wait()
        digest = hashlib.sha256()
        buf = bytearray()
        with self._session.get(url, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            final_url, disposition = _response_meta(resp, url)
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise ValueError(f"e-print exceeds max size of {max_bytes} bytes")
                digest.update(chunk)
        if not buf:
            raise ValueError("e-print payload is empty")
        return {"data": bytes(buf), "sha256": digest.hexdigest(), "bytes": len(buf),
                "url": url, "final_url": final_url, "content_disposition": disposition}
