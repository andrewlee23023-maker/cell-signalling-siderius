"""Print the key fields of a SIDERIUS proposal JSON (model name, hypothesis, architecture summary)."""
import json
import sys
from pathlib import Path

p = next(Path(sys.argv[1]).rglob("proposal_*.json"))
d = json.loads(p.read_text())


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o[:20]):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, o


for path, v in walk(d):
    if isinstance(v, str) and len(v) > 40 and any(k in path.lower() for k in ("hypothesis", "architecture", "description", "summary", "rationale", "model_type", "mechanism")):
        print(f"{path}: {v[:600]}\n")
