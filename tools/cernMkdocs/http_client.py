from http.cookiejar import MozillaCookieJar
from pathlib import Path

import httpx


DEFAULT_COOKIE_FILE = (
    Path.home()
    # / ".config"
    # / "heptapod"
    / "cern-docs-cookies.txt"
)


def make_docs_http_client(
    cookie_file: Path = DEFAULT_COOKIE_FILE,
) -> httpx.AsyncClient:
    """Create an HTTP client authenticated with a saved SSO cookie jar."""
    cookie_file = Path(cookie_file).expanduser()

    if not cookie_file.is_file():
        raise FileNotFoundError(
            f"SSO cookie file not found: {cookie_file}"
        )

    jar = MozillaCookieJar(str(cookie_file))
    jar.load(ignore_discard=True, ignore_expires=False)

    if not list(jar):
        raise ValueError(f"No usable cookies found in {cookie_file}")

    return httpx.AsyncClient(
        cookies=httpx.Cookies(jar),
        follow_redirects=True,
        timeout=30.0,
    )
