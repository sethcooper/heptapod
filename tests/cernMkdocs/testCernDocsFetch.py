import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.cernMkdocs.fetch_tool import CernDocsFetchTool


def main():
    page_url = "https://cmshcal.docs.cern.ch/doc/summary2023/"

    tool = CernDocsFetchTool(
        url=page_url,
        source_id="cmshcal",
        mode="markdown",
    )

    result = tool._run()
    result = json.loads(result)

    print("Status:", result.get("status", "error"))
    print("Source:", result.get("source"))
    print("Repository path:", result.get("source_path"))
    print("Rendered URL:", result.get("url"))

    content = result.get("content", "")
    print("\nMarkdown preview:\n")
    print(content[:2000])

    if result.get("status") != "ok":
        print("\nFull error response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

