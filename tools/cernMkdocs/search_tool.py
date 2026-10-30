import json
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField

from .service import CernDocsService


class CernDocsSearchTool(BaseTool):
    """Search CERN MkDocs documentation.

    Inputs:
      - query: Search terms or a natural-language phrase.
      - source_id: Configured documentation source ID (default: blah).
      - limit: Maximum number of results to return (default: 5, max: 10).

    Returns JSON containing matching document titles, URLs, and snippets.
    """

    query: str = RuntimeField(
        description="Terms or a phrase to search for in CERN documentation."
    )
    source_id: Optional[str] = RuntimeField(
        default="blah",
        description="Configured documentation source ID, e.g. blah."
    )
    limit: Optional[int] = RuntimeField(
        default=5,
        description="Maximum results to return, from 1 to 10."
    )

    def _run(self) -> str:
        try:
            query = self.query.strip()
            if not query:
                return self.format_error(
                    error="Invalid Input",
                    reason="The search query must not be empty.",
                )

            limit = max(1, min(int(self.limit or 5), 10))

            # Keep the service alive between calls so the in-memory
            # index does not need to be rebuilt for every search.
            service = _get_service()

            results = service.search(
                query=query,
                source_id=self.source_id or "blah",
                limit=limit,
            )

            return json.dumps(
                {
                    "status": "ok",
                    "query": query,
                    "source_id": self.source_id or "blah",
                    "count": len(results),
                    "results": results,
                },
                ensure_ascii=False,
            )

        except Exception as exc:
            return self.format_error(
                error="Documentation Search Failed",
                reason=str(exc),
                suggestion=(
                    "Check the source configuration and CERN SSO cookie "
                    "file, and refresh the documentation index if needed."
                ),
            )


_service = None


def _get_service() -> CernDocsService:
    global _service

    if _service is None:
        _service = CernDocsService()
        _service.initialize()

    return _service

