"""Refuse to publish if the staged repo contains secrets or data files. Exit 1 on any finding."""
import re
import sys
from pathlib import Path

R = Path("/mnt/c/Users/andre/workspace/cell-signalling-siderius")
pats = [re.compile(p) for p in (r"sk-ant-[A-Za-z0-9_\-]{10,}", r"sk-(proj-)?[A-Za-z0-9]{20,}", r"AIza[0-9A-Za-z_\-]{30,}",
                                r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"gh[op]_[A-Za-z0-9]{20,}")]
bad = []
for f in R.rglob("*"):
    if not f.is_file():
        continue
    if f.suffix in (".npz", ".npy", ".pkl", ".h5") or f.name in ("credentials.json", ".env", "console-token"):
        bad.append(f"data/credential file: {f}")
        continue
    if f.suffix in (".pt", ".zip", ".pdf", ".png"):
        continue
    t = f.read_text(errors="ignore")
    for p in pats:
        if p.search(t):
            bad.append(f"secret pattern {p.pattern} in {f}")
print("\n".join(bad) if bad else "scan clean")
sys.exit(1 if bad else 0)
