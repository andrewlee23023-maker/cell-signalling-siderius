"""Task-specific decode + shortcut/loophole diagnostics for cell-signalling.

    python tools/cs_decode.py        # writes decode_report.md and decode_report.json

Descriptive evidence for the team, not a model choice. Every AP is the OFFICIAL
lr_auprc (per-dataset AP averaged over the 21 validation datasets), random level
0.114 ± 0.014 per the task card. Nothing here is fitted on validation; the one
learned diagnostic (§5) is fitted on train only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_data import load, official_baseline_scores, score  # noqa: E402

RANDOM = "0.114 ± 0.014"


def valid(X):
    return X[:, 4] > 0


def masked(X, row, fn):
    m = valid(X)
    out = np.zeros(len(X))
    for i in range(len(X)):
        v = X[i, row, m[i]]
        out[i] = fn(v) if v.size else 0.0
    return out


def marginal_features(X) -> dict:
    """Per-sample statistics that ignore L-R alignment (no joint row 0 x row 1 terms)."""
    f = {"n_contacts": valid(X).sum(axis=1).astype(float)}
    for row, name in ((0, "lig"), (1, "rec"), (2, "loglib"), (3, "nA")):
        f[f"{name}_mean"] = masked(X, row, np.mean)
        f[f"{name}_std"] = masked(X, row, np.std)
        f[f"{name}_frac_nonzero"] = masked(X, row, lambda v: float(np.mean(v != 0)))
    return f


def agg(d, s):
    return score(d, s)["aggregate"]


def main() -> int:
    tr, va = load("train"), load("validation")
    R: dict = {"random_level": RANDOM}
    L = ["# cell-signalling decode report (validation, official metric)", "",
         f"Random ranking: lr_auprc {RANDOM}, direction_accuracy 0.50 ± 0.04 (task card).", ""]

    # 1. channels over contact cells
    m = valid(va["X"])
    L += ["## 1 · Channels over contact cells (validation)", "",
          "| row | meaning (task card) | zero frac | p50 | p99 | max |", "| --- | --- | --- | --- | --- | --- |"]
    names = ["mean log-norm L over type-A neighbours", "log-norm R of the B cell", "log library − tile mean", "# type-A neighbours / 6"]
    R["channels"] = []
    for r in range(4):
        v = va["X"][:, r][m]
        row = {"row": r, "zero_frac": float(np.mean(v == 0)), "p50": float(np.median(v)), "p99": float(np.percentile(v, 99)), "max": float(v.max())}
        R["channels"].append(row)
        L.append(f"| {r} | {names[r]} | {row['zero_frac']:.3f} | {row['p50']:.3g} | {row['p99']:.3g} | {row['max']:.3g} |")

    # 2. baseline breakdown
    b = official_baseline_scores(va)
    ab = agg(va, b)
    R["baseline"] = ab
    L += ["", "## 2 · Official baseline (contact Spearman)", "",
          f"lr_auprc {ab['lr_auprc']:.4f}, direction_accuracy {ab['direction_accuracy']:.4f}, lr_auroc {ab['lr_auroc']:.4f}", ""]
    n = valid(va["X"]).sum(axis=1)
    pos = va["y"]
    thr_rank = []
    for d in np.unique(va["dataset"]):
        idx = np.flatnonzero(va["dataset"] == d)
        k = int(pos[idx].sum())
        top = idx[np.argsort(-b[idx], kind="mergesort")[:k]]
        thr_rank += [(i, i in set(top)) for i in idx[pos[idx]]]
    hit = {i: h for i, h in thr_rank}
    L += ["Positives found in their dataset's top-k (k = number of positives), by effect size and contact count:", "",
          "| β | positives | found in top-k | median contacts (found / missed) |", "| --- | --- | --- | --- |"]
    R["baseline_by_beta"] = {}
    for beta in (0.5, 1.0, 2.0):
        ids = [i for i in np.flatnonzero(pos & np.isclose(va["beta"], beta))]
        f = [i for i in ids if hit[i]]
        mi = [i for i in ids if not hit[i]]
        R["baseline_by_beta"][str(beta)] = {"n": len(ids), "found": len(f)}
        L.append(f"| {beta} | {len(ids)} | {len(f)} ({len(f) / max(len(ids), 1):.0%}) | "
                 f"{int(np.median(n[f])) if f else '–'} / {int(np.median(n[mi])) if mi else '–'} |")

    # 3. marginal-only shortcuts (reward-hacking leads #2, #5)
    feats = marginal_features(va["X"])
    L += ["", "## 3 · Shortcuts that ignore L–R alignment (loophole leads #2 padding/contacts, #5 confounders)", "",
          "Each scored as is and negated; anything clearly above random would be a confounder the metric rewards.", "",
          "| feature | lr_auprc as is | negated |", "| --- | --- | --- |"]
    R["marginal_ap"] = {}
    for name, v in feats.items():
        a1, a2 = agg(va, v)["lr_auprc"], agg(va, -v)["lr_auprc"]
        R["marginal_ap"][name] = [a1, a2]
    for name, (a1, a2) in sorted(R["marginal_ap"].items(), key=lambda kv: -max(kv[1])):
        L.append(f"| {name} | {a1:.3f} | {a2:.3f} |")

    # 4. order leak (#8)
    order = np.zeros(len(va["y"]))
    for d in np.unique(va["dataset"]):
        idx = np.flatnonzero(va["dataset"] == d)
        order[idx] = np.arange(len(idx))
    o1, o2 = agg(va, order)["lr_auprc"], agg(va, -order)["lr_auprc"]
    R["order_ap"] = [o1, o2]
    L += ["", "## 4 · File-order leak (lead #8)", "",
          f"Position of the triple within its dataset as the score: lr_auprc {o1:.3f} (as is) / {o2:.3f} (negated)."]

    # 5. simulator artifact via marginal features (#9): fit on TRAIN only
    from sklearn.ensemble import GradientBoostingClassifier

    ftr = marginal_features(tr["X"])
    keys = list(ftr)
    clf = GradientBoostingClassifier(max_depth=2, n_estimators=200, random_state=0)
    clf.fit(np.column_stack([ftr[k] for k in keys]), tr["y"])
    p = clf.predict_proba(np.column_stack([feats[k] for k in keys]))[:, 1]
    gb = agg(va, p)
    R["marginal_gbm"] = gb
    L += ["", "## 5 · Simulator-artifact probe (lead #9)", "",
          f"Gradient boosting (depth 2, 200 trees) on the {len(keys)} alignment-free features above, fitted on train only: "
          f"lr_auprc {gb['lr_auprc']:.3f}, direction_accuracy {gb['direction_accuracy']:.3f}. "
          "Near random means no exploitable artifact in the marginals."]

    # 6. reverse-pair (transductive) score (#1/#7): uses ANOTHER sample's input
    sid = va["sample_id"].tolist()
    pos_of = {s: i for i, s in enumerate(sid)}
    rel = np.zeros(len(sid))
    has_rev = np.zeros(len(sid), bool)
    for i, s in enumerate(sid):
        d, t = s.split(":", 1)
        l, r_, a, bb = t.split("|")
        j = pos_of.get(f"{d}:{l}|{r_}|{bb}|{a}")
        if j is not None:
            has_rev[i] = True
            rel[i] = b[i] - b[j]
    rev_pos = sum(1 for i in np.flatnonzero(pos) if has_rev[i] and pos[pos_of[f"{sid[i].split(':')[0]}:{'|'.join(np.array(sid[i].split(':')[1].split('|'))[[0, 1, 3, 2]])}"]])
    ar = agg(va, rel)
    R["reverse_relative"] = {"has_reverse_fraction": float(has_rev.mean()), "both_directions_positive": int(rev_pos // 2), **ar}
    L += ["", "## 6 · Reverse-pair structure (leads #1 batch-relative, #3 mirror pairs, #7 direction)", "",
          f"- Reverse triple present for {has_rev.mean():.1%} of triples; pairs where both directions are injected: {rev_pos // 2}.",
          f"- Pair-relative score s(A→B) − s(B→A) from the baseline: lr_auprc {ar['lr_auprc']:.3f}, direction_accuracy "
          f"{ar['direction_accuracy']:.3f} (baseline {ab['lr_auprc']:.3f} / {ab['direction_accuracy']:.3f}).",
          "- This needs the reverse sample's tensor, which a per-sample `predict(x)` never sees; only `predict_batch` over a",
          "  batch containing both could use it. Whether that is legitimate cross-sample pooling or a batch-composition",
          "  loophole is a team judgement (brief §8.1); the evaluator's batching is not documented in the package."]

    # 7. evaluator behaviour (#4, #10) from reading _lr_metrics.py
    L += ["", "## 7 · Evaluator behaviour (read from task/plugins/_lr_metrics.py; leads #4, #10)", "",
          "- Ties: average precision groups tied scores (scikit-learn convention), so ties are not credited optimistically.",
          "- Non-finite scores raise; a missing triple ranks below every prediction of its dataset.",
          "- Health (blocking): at least 5 occupied within-dataset score deciles, so constant/near-binary scores are refused.",
          "- Metrics are per dataset then averaged with equal weight: small datasets count as much as large ones."]
    out = Path.home() / "work/cell-signalling/decode_report.md"
    out.write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(R, indent=2, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
