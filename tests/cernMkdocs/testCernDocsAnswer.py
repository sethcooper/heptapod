import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.cernMkdocs.answer_tool import CernDocsAnswerTool


def main():
    tool = CernDocsAnswerTool(
        query="channels",
        source_id="cmshcal",
        limit=5,
    )
    result = tool._run()
    print(result)


if __name__ == "__main__":
    main()
