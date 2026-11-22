"""Summarize a finished SIDERIUS run: termination, scores per training record, Health, model, token use.
    python3 run_summary.py RUN_ENVELOPE_DIR
"""
import collections
import json
import sys
from pathlib import Path

W = Path(sys.argv[1]) / "workspace"
for p in sorted(W.rglob("run_output_*.json")):
    d = json.loads(p.read_text())
    print("run output:", p.relative_to(W))
    for k in ("termination_reason", "best_valid_formal_denoising_score", "best_formal_denoising_score", "best_denoising_score"):
        print(f"  {k}: {d.get(k)}")
    for r in d.get("all_records", []):
        sa = r.get("scientific_authority") or {}
        print(f"  record: status={r.get('status')} score={r.get('denoising_score')} "
              f"authoritative={sa.get('authoritative')} blocking={sa.get('blocking_reasons')}")
models = sorted({p.stem for p in W.rglob("*.py") if "plugin" in str(p) or "model" in p.stem.lower()})[:8]
print("model files:", models)
tot = collections.defaultdict(lambda: [0, 0, 0])
for line in (W / "token_usage.jsonl").read_text().splitlines() if (W / "token_usage.jsonl").exists() else []:
    r = json.loads(line)
    t = r.get("tokens") or {}
    m = tot[r.get("model")]
    m[0] += 1
    m[1] += t.get("prompt", 0)
    m[2] += t.get("completion", 0)
print("tokens:", {k: {"calls": v[0], "in": v[1], "out": v[2]} for k, v in tot.items()})
