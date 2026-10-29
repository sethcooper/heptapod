"""Isolated catalogue worker; request and response are durable JSON receipts."""

import json
import sys
import traceback
from pathlib import Path
from .catalogue import ProcessModel, enumerate_processes, dump


def main():
    request_path = Path(sys.argv[1]).resolve()
    request = json.loads(request_path.read_text())
    output = request_path.parent
    try:
        model = ProcessModel(
            request["model"],
            request["base_directory"],
            request.get("param_card"),
            request.get("masses"),
        )
        action = request["action"]
        if action == "inspect":
            dump(output / "model.json", model.info)
            result = {"status": "ok", "model": model.info}
        elif action == "enumerate":
            data = enumerate_processes(model, output, **request["enumeration"])
            result = {
                "status": "ok",
                "processes": len(data["processes"]),
                "coverage": data["coverage"],
                "scope": data["scope"],
            }
        elif action == "validate":
            from .catalogue_validation import validate_catalogue

            data = json.loads(Path(request["catalogue_path"]).read_text())
            if data["scope"] != request["expected_scope"]:
                raise ValueError(
                    "Catalogue scope differs from the independently requested scope"
                )
            result = validate_catalogue(
                model, request["catalogue_path"], request["mg5_path"]
            )
            dump(output / "validation.json", result)
        else:
            raise ValueError("Unknown action")
    except Exception as exc:
        traceback.print_exc()
        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    dump(output / "response.json", result)
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
