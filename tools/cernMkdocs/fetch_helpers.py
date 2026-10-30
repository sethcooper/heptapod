import re
from typing import Any
from urllib.parse import urlparse, unquote

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def candidate_source_paths(
    url_or_path: str,
    docs_site_url: str = "",
) -> list[str]:
    """Map a rendered MkDocs URL to candidate GitLab Markdown paths."""
    raw = url_or_path.strip()
    if not raw:
        return []

    parsed = urlparse(raw)

    if parsed.scheme or parsed.netloc:
        # A full URL must belong to the configured documentation site.
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid documentation URL: {raw}")

        site = urlparse(docs_site_url)
        if not site.scheme or not site.netloc:
            raise ValueError("The configured documentation site URL is invalid")

        if (
            parsed.scheme.lower() != site.scheme.lower()
            or parsed.netloc.lower() != site.netloc.lower()
        ):
            raise ValueError(
                f"URL is outside the configured documentation site: {raw}"
            )

        path = unquote(parsed.path)
        site_prefix = unquote(site.path).rstrip("/")

        # Remove a configured deployment prefix, e.g. /manual.
        if site_prefix:
            if path == site_prefix:
                path = ""
            elif path.startswith(site_prefix + "/"):
                path = path[len(site_prefix):]
            else:
                raise ValueError(
                    f"URL is outside the configured documentation path: {raw}"
                )
    else:
        # Relative paths are accepted for internal callers, but the
        # agent-facing tool should supply a full URL.
        path = unquote(raw.split("#", 1)[0].split("?", 1)[0])

    path = path.strip("/")

    if not path or path in ("index", "index.html"):
        return ["docs/index.md"]

    # Support an explicit Markdown filename as a convenience.
    if path.endswith(".md"):
        if path.startswith("docs/"):
            return [path]
        return [f"docs/{path}"]

    # MkDocs commonly renders foo/index.md as /foo/ and foo.md as /foo/.
    return [
        #f"docs/{path}/index.md",
        f"docs/{path}.md",
    ]


def project_markdown(markdown: str, mode: str) -> dict[str, Any]:
    if mode == "outline":
        outline = [
            {"level": len(match.group(1)), "heading": match.group(2)}
            for match in _HEADING_RE.finditer(markdown)
            if len(match.group(1)) <= 3
        ]
        return {"mode": "outline", "outline": outline}

    if mode.startswith("sections:"):
        heading = mode.split(":", 1)[1].strip()
        target = heading.lower()
        matches = list(_HEADING_RE.finditer(markdown))
        for i, match in enumerate(matches):
            if match.group(2).strip().lower() != target:
                continue

            end = len(markdown)
            level = len(match.group(1))
            for next_match in matches[i + 1:]:
                if len(next_match.group(1)) <= level:
                    end = next_match.start()
                    break

            content = markdown[match.start():end].rstrip() + "\n"
            return {
                "mode": mode,
                "heading": heading,
                "found": True,
                "content": content,
            }

        return {
            "mode": mode,
            "heading": heading,
            "found": False,
            "content": "",
        }

    return {"mode": "markdown", "content": markdown}


def rendered_url(docs_base: str, source_path: str) -> str:
    inner = (
        source_path[len("docs/"):]
        if source_path.startswith("docs/")
        else source_path
    )
    inner = inner.removesuffix(".md")

    if inner == "index":
        inner = ""
    elif inner.endswith("/index"):
        inner = inner[:-len("/index")]

    base = docs_base.rstrip("/")
    return f"{base}/{inner}/" if inner else f"{base}/"
