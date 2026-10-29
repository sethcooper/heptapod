"""Unit tests for the streaming EventJSONLToNumpy rewrite (toolgrad loop 18).

The tool used to load the whole JSONL into RAM (a list of dicts + a list-of-lists
+ the dense array, ~3x the dataset -> OOM on a 50k-event shower). It now streams
two passes (find n/max, then fill a pre-allocated array) and supports a
`max_particles` cap. These tests check correctness of the new path.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.analysis.conversions import EventJSONLToNumpyTool


def _make(d, multiplicities):
    p = os.path.join(d, "ev.jsonl")
    with open(p, "w") as f:
        for n in multiplicities:
            parts = [{"px": i, "py": i, "pz": i, "E": i, "id": 11} for i in range(n)]
            f.write(json.dumps({"data": {"particles": parts}}) + "\n")
    return p


def test_padding_to_global_max():
    d = tempfile.mkdtemp()
    _make(d, [2, 5, 3])
    r = json.loads(EventJSONLToNumpyTool(jsonl_path="ev.jsonl", output_path="o.npy",
                                         base_directory=d)._run())
    assert r["status"] == "ok" and r["output_shape"] == [3, 5, 5]
    arr = np.load(os.path.join(d, "o.npy"))
    assert arr.shape == (3, 5, 5)
    # event 0 had 2 particles -> 2 nonzero rows, rest zero-padded
    assert int((arr[0].any(axis=1)).sum()) == 2
    assert int((arr[1].any(axis=1)).sum()) == 5


def test_max_particles_cap():
    d = tempfile.mkdtemp()
    _make(d, [2, 50, 3])
    r = json.loads(EventJSONLToNumpyTool(jsonl_path="ev.jsonl", output_path="o.npy",
                                         max_particles=10, base_directory=d)._run())
    # capped at 10 even though one event has 50 particles
    assert r["output_shape"] == [3, 10, 5]


def test_all_empty_errors_with_zero_particles_message():
    d = tempfile.mkdtemp()
    _make(d, [0, 0, 0])
    out = EventJSONLToNumpyTool(jsonl_path="ev.jsonl", output_path="o.npy",
                                base_directory=d)._run()
    assert "Empty Dataset" in out and "zero particles" in out.lower()


def test_values_preserved():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "ev.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"data": {"particles": [
            {"px": 1.5, "py": 2.5, "pz": 3.5, "E": 4.5, "id": 22}]}}) + "\n")
    EventJSONLToNumpyTool(jsonl_path="ev.jsonl", output_path="o.npy", base_directory=d)._run()
    arr = np.load(os.path.join(d, "o.npy"))
    assert list(arr[0, 0]) == [1.5, 2.5, 3.5, 4.5, 22.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"[ok] {name}")
    print("all conversions streaming tests passed")
