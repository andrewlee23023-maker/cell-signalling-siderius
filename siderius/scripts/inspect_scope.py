"""Show what data a SIDERIUS training record used for fitting, checkpoint selection and scoring."""
import json
import sys

d = json.load(open(sys.argv[1]))
for k in ("resolved_data_scope", "selected_checkpoint", "is_trial", "attempt_role"):
    print(f"== {k}:\n{json.dumps(d.get(k), indent=1)[:2500]}\n")
tc = (d.get("params") or {}).get("train_config") or {}
print("== train_config:", json.dumps(tc, indent=1)[:2000])
st = d.get("score_table")
print("\n== score_table type:", type(st).__name__, (list(st)[:10] if isinstance(st, dict) else (st[:3] if isinstance(st, list) else st)))
