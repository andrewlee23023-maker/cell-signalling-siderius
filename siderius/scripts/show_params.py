"""Print the full params / model_params / loss settings of a SIDERIUS training record."""
import json
import sys

d = json.load(open(sys.argv[1]))
for k in ("params", "model_params", "training_diagnosis", "timing", "final_loss"):
    print(f"== {k}:\n{json.dumps(d.get(k), indent=1)[:3000]}\n")
hist = d.get("loss_history") or d.get("training_history")
print("== history (first/last):", json.dumps(hist, indent=None)[:600])
