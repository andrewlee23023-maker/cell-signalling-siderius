"""Confirm a staged plugin reproduces a reference statistic exactly on validation.
    python tools/same_scores.py submissions/m1-c4/final_model.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import load, score  # noqa: E402
from m1_stats import two_part  # noqa: E402

spec = importlib.util.spec_from_file_location("staged", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
va = load("validation")
a = mod.predict_batch(va["X"])
b = np.array([two_part(x, True) for x in va["X"]])
print("max |staged - compared C4| =", float(np.max(np.abs(a - b))))
print("staged official:", {k: round(v, 4) for k, v in score(va, a)["aggregate"].items()})
