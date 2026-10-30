from typing import Any

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField

from tools.cernMkdocs.service import CernDocsService


_service = None


def _get_service() -> CernDocsService:
    global _service
    if _service is None:
        _service = CernDocsService()
    return _service


class CernDocsAnswerTool(BaseTool):
    query: str = RuntimeField(
        description="Question or search terms for CERN documentation."
    )
    source_id: str = RuntimeField(
        description="Configured CERN documentation source ID, e.g. cern-mkdocs."
    )
    limit: int = RuntimeField(
        description="Maximum number of search results to consider (1-25)."
    )

    def _run(self) -> Any:
        try:
            service = _get_service()

            # Always search first.
            results = service.search(
                query=self.query,
                source_id=self.source_id,
                limit=self.limit,
            )

            if not results:
                return {
                    "status": "not_found",
                    "message": "No relevant documentation search results were found.",
                    "query": self.query,
                }

            # The search results are dictionaries with url and score fields.
            best = max(
                results,
                key=lambda result: float(result.get("score", 0)),
            )
            url = best.get("url")

            if not url:
                return {
                    "status": "error",
                    "message": "The best search result did not contain a page URL.",
                    "result": best,
                }

            # Fetch the exact URL returned by search; don't construct a URL.
            content = service.fetch(
                url=url,
                source_id=self.source_id,
                mode="markdown",
            )

            if not results:
                return {
                    "status": "not_found",
                    "message": "No relevant documentation results found.",
                }
            
            best = max(results, key=lambda r: float(r.get("score", 0)))
            url = best.get("url")
            
            if not url:
                return {
                    "status": "error",
                    "message": "Search result has no URL.",
                }
            
            fetched = service.fetch(
                url=url,
                source_id=self.source_id,
                mode="markdown",
            )
            
            markdown = (
                fetched.get("content", "")
                if isinstance(fetched, dict)
                else str(fetched)
            )
            markdown = str(markdown)
            
            # Keep the first diagnostic response deliberately small.
            max_chars = 4000
            was_truncated = len(markdown) > max_chars
            
            return {
                "status": "ok",
                "title": best.get("title", ""),
                "url": url,
                "content": markdown[:max_chars],
                "content_note": "except from the page. Use the URL to get the full content.",
            }

        except Exception as exc:
            return self.format_error(
                error="Documentation Search and Fetch Failed",
                reason=str(exc),
                suggestion=(
                    "Check the query, configured source ID, documentation "
                    "site URL, GitLab token, and repository access."
                ),
            )
