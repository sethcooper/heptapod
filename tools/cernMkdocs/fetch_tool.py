import json

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField

from .service import CernDocsService

class CernDocsFetchTool(BaseTool):
    """
    Fetch Markdown content from a configured CERN MkDocs documentation source.
       Use this tool when the user asks to read, inspect, or retrieve documentation content.
       The source ID defaults to cern-mkdocs. For a specific page, pass its URL.
       For general documentation or a homepage request, leave page empty to fetch docs/index.md. Do not use the source ID as the page path.
    """

    url: str = RuntimeField(
        description=(
            "Full URL of a page on the configured CERN MkDocs documentation "
            "site. Normally use a URL returned by CernDocsSearchTool. "
            "Do not pass a repository path or a page slug."
        )
    )
    source_id: str = RuntimeField(
        default="blah",
        description="Configured documentation source ID."
    )
    mode: str = RuntimeField(
        default="markdown",
        description=(
            "Output mode: markdown, outline, or sections:Heading."
        )
    )

    def _run(self) -> str:
        try:
            url = self.url.strip()
            if not url:
                return self.format_error(
                    error="Invalid Input",
                    reason="The page URL or path must not be empty.",
                )

            mode = (self.mode or "markdown").strip()
            result = _get_service().fetch(
                    url=url,
                    source_id=self.source_id or "cern-mkdocs",
                    mode=self.mode or "markdown",
            )

            return json.dumps(result, ensure_ascii=False)

        except Exception as exc:
            return self.format_error(
                error="Documentation Fetch Failed",
                reason=str(exc),
                suggestion=(
                    "Check the documentation URL, source configuration, "
                    "CERN GitLab token, and repository access permissions."
                ),
            )


_service = None


def _get_service() -> CernDocsService:
    global _service

    if _service is None:
        # Fetching doesn't require refreshing the search index.
        _service = CernDocsService()

    return _service

