import asyncio
from pathlib import Path

import httpx
import os

from cern_mkdocs_mcp.config import (
    load_sources,
    resolve_auth_headers,
    validate_source_id,
)
from cern_mkdocs_mcp.tools._index import DocsIndex

from .http_client import make_docs_http_client
from .fetch_helpers import (
    candidate_source_paths,
    project_markdown,
    rendered_url,
)

from urllib.parse import quote

from concurrent.futures import ThreadPoolExecutor


DEFAULT_CONFIG = Path.home() / ".config/heptapod/cern-docs.json"


class CernDocsService:
    def __init__(self, config_path=DEFAULT_CONFIG):
        self.sources = load_sources(config_path)
        self.indices = {
            sid: DocsIndex(search_index_url=source.search_index_url)
            for sid, source in self.sources.items()
            if source.source_type == "mkdocs"
        }

    @staticmethod
    def _run_async(coro):
        """Run a coroutine safely, even if the caller has an active event loop."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    async def _refresh(self, source_id):
        async with make_docs_http_client() as http:
            await self.indices[source_id].refresh(http)

    def initialize(self):
        """Explicitly refresh all configured MkDocs indexes."""
        for source_id in self.indices:
            self._run_async(self._refresh(source_id))

    async def _search(self, query, source_id, limit):
        validate_source_id(source_id, self.sources)

        if source_id not in self.indices:
            raise ValueError(
                f"No MkDocs index for source {source_id!r}"
            )

        async with make_docs_http_client() as http:
            await self.indices[source_id].ensure_fresh(http)

        return self.indices[source_id].search(
            query,
            limit=max(1, min(int(limit), 25)),
            section=None,
        )

    def search(self, query, source_id="blah", limit=10):
        return self._run_async(self._search(query, source_id, limit))


    def _fetch(
        self,
        url: str,
        source_id: str,
        mode: str,
    ) -> dict:
        validate_source_id(source_id, self.sources)

        source = self.sources[source_id]

        if source.source_type != "mkdocs":
            raise ValueError(
                f"Source {source_id!r} is not an MkDocs source."
            )

        token = os.environ.get("CERN_DOCS_GITLAB_TOKEN")
        if not token:
            raise RuntimeError(
                "Set CERN_DOCS_GITLAB_TOKEN to a read-access GitLab token."
            )

        if mode != "markdown" and mode != "outline":
            if not mode.startswith("sections:") or not mode.split(":", 1)[1].strip():
                raise ValueError(
                    "mode must be 'markdown', 'outline', "
                    "or 'sections:Heading'."
                )

        candidates = candidate_source_paths(
            url,
            source.docs_site_url,
        )

        if not candidates:
            raise ValueError(
                f"Could not derive a Markdown path from {url!r}."
            )

        project_path = quote(source.gitlab_project_path, safe="")
        api_base = "https://gitlab.cern.ch/api/v4"
        headers = {"PRIVATE-TOKEN": token}
        last_error = None

        with httpx.Client(timeout=30.0, follow_redirects=True) as http:
            for path in candidates:
                url = (
                    f"{api_base}/projects/{project_path}"
                    f"/repository/files/{quote(path, safe='')}/raw"
                )
                try:
                    response = http.get(
                        url,
                        params={"ref": source.default_branch},
                        headers=headers,
                    )
                except httpx.HTTPError as exc:
                    last_error = exc
                    continue
        
                if response.status_code == 404:
                    continue
        
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if response.status_code in (401, 403):
                        raise RuntimeError(
                            f"GitLab returned HTTP {response.status_code}. "
                            "Check the token's access to this repository."
                        ) from exc
                    last_error = exc
                    continue
        
                projection = project_markdown(response.text, mode)
                return {
                    "status": "ok",
                    "source": source_id,
                    "source_path": path,
                    "url": rendered_url(source.docs_site_url, path),
                    **projection,
                }

        if last_error:
            raise RuntimeError(
                f"Could not fetch a candidate Markdown file: {last_error}"
            ) from last_error

        raise FileNotFoundError(
            f"No matching Markdown file found for {url!r}. "
            f"Tried: {candidates}"
        )

    def fetch(
        self,
        url: str,
        source_id: str = "blah",
        mode: str = "markdown",
    ) -> dict:
        """Fetch one documentation URL from the configured GitLab repo."""

        return self._fetch(
            url=url,
            source_id=source_id,
            mode=mode,
        )


# One instance per Python process.
_service = None


def get_service():
    global _service
    if _service is None:
        _service = CernDocsService()
    return _service
