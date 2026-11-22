"""List keys of a SIDERIUS training record and any file paths / per-dataset metrics it mentions."""
import json
import sys


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o[:50]):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, o


d = json.load(open(sys.argv[1]))
print("top-level keys:", list(d)[:60])
for p, v in walk(d):
    if isinstance(v, str) and (v.endswith((".pt", ".pth", ".json", ".py", ".safetensors", ".ckpt")) or "/" in v and "workspace" in v):
        print("  path-like:", p, "=", v[:200])
    if any(t in p.lower() for t in ("per_dataset", "seed", "checkpoint", "weights", "deliverable")):
        print("  ", p, "=", str(v)[:160])
