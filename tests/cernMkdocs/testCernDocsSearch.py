import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools.cernMkdocs.search_tool import CernDocsSearchTool


def main():
    tool = CernDocsSearchTool(
        query="bad data rate",
        source_id="cmshcal",
        limit=5,
    )

    result = json.loads(tool._run())

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

