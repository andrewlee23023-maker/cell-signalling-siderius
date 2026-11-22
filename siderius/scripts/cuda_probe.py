"""Tiny probe run inside the project sandbox: CUDA visible? read-only mounts read-only?"""
import os
from pathlib import Path

import torch

print("cuda_available", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for p in (Path.home(), Path(os.environ.get("PWD", "."))):
    pass
base = Path("/home/andre/work/cell-signalling")
for d in (base / "data", base / "package"):
    try:
        (d / ".write_probe").write_text("x")
        print(d, "WRITABLE (unexpected)")
        (d / ".write_probe").unlink()
    except OSError as exc:
        print(d, "read-only as declared:", type(exc).__name__)
print("keys present:", sorted(k for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY") if os.environ.get(k)))
