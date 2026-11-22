"""Decode diagnostics for cell-signalling (brief §4 Q1-Q3 and §12).

    python tools/decode.py DATA_DIR [--split train] [--out decode_report.md]

Writes a Markdown report and a JSON file next to it. Everything here is
descriptive: it measures the tensor, it does not decide the model.

AP numbers use scikit-learn's average_precision_score as a PROVISIONAL stand-in
for the official `lr_auprc`; replace with the task package's metric code once
it is released (look in task/plugins/ for the metric implementation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import inventory, load_split  # noqa: E402

GROUP_HINTS = ("section", "tissue", "slide", "sample", "sample_id", "fov", "replicate", "donor", "image", "batch")
ROLE_HINTS = {"ligand": ("ligand", "lig"), "receptor": ("receptor", "rec"),
              "sender": ("sender", "source", "src"), "receiver": ("receiver", "target_cell", "dst", "tgt")}


def ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    ok = np.isfinite(s)
    return float(average_precision_score(y[ok], s[ok])) if ok.any() and y[ok].min() != y[ok].max() else float("nan")


def channel_stats(X: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """Per-channel value statistics over NON-PADDED slots only."""
    out = []
    idx = rng.choice(len(X), size=min(len(X), 2000), replace=False)
    sub, el = X[idx], effective_length(X[idx])
    valid = np.arange(X.shape[2])[None, :] < el[:, None]  # [n, L]
    for c in range(X.shape[1]):
        v = sub[:, c, :][valid]
        nz = v[v != 0]
        out.append({
            "channel": c,
            "integer_fraction": float(np.mean(np.isclose(v, np.round(v)))),
            "zero_fraction": float(np.mean(v == 0)),
            "in_0_1_fraction": float(np.mean((v >= 0) & (v <= 1))),
            "negative_fraction": float(np.mean(v < 0)),
            "binary": bool(np.isin(np.unique(v[:200000]), [0, 1]).all()),
            "unique_values_sampled": int(len(np.unique(v[:200000]))),
            "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "std": float(v.std()),
            "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)),
            "nonzero_p99_over_p50": float(np.percentile(nz, 99) / max(np.percentile(nz, 50), 1e-12)) if nz.size else float("nan"),
        })
    return out


def guess_channel(s: dict) -> str:
    if s["binary"]:
        return "binary mask (exposure/validity/contact?)"
    if s["integer_fraction"] > 0.999 and s["min"] >= 0:
        return "integer counts" + (" (sparse, zero-inflated)" if s["zero_fraction"] > 0.5 else "")
    if s["in_0_1_fraction"] > 0.999:
        return "fraction/probability/normalized distance in [0,1]"
    if s["min"] >= 0 and s["nonzero_p99_over_p50"] > 10:
        return "heavy-tailed positive (library size / exposure?)"
    if s["negative_fraction"] > 0.05:
        return "signed continuous (standardized/residual/embedding?)"
    return "continuous positive"


def effective_length(X: np.ndarray) -> np.ndarray:
    """Index after the last slot where ANY channel is nonzero (padding detector)."""
    any_nz = (X != 0).any(axis=1)  # [N, L]
    last = X.shape[2] - np.argmax(any_nz[:, ::-1], axis=1)
    last[~any_nz.any(axis=1)] = 0
    return last


def order_stats(X: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """Lag-1 autocorrelation along the 1024 axis vs a within-sample shuffle.

    Computed per sample over its non-padded prefix only, so trailing padding
    cannot fake an ordering signal.
    """
    sub = X[rng.choice(len(X), size=min(len(X), 1000), replace=False)].astype(np.float64)
    el = effective_length(sub)

    def lag1(v: np.ndarray) -> float:
        v = v - v.mean()
        den = float((v * v).sum())
        return float((v[1:] * v[:-1]).sum() / den) if den > 0 else float("nan")

    res = []
    for c in range(X.shape[1]):
        real, shuf, up, down = [], [], [], []
        for row, m in zip(sub[:, c, :], el):
            if m < 3:
                continue
            v = row[:m]
            real.append(lag1(v))
            shuf.append(lag1(rng.permutation(v)))
            d = np.diff(v)
            up.append(bool((d >= 0).all()))
            down.append(bool((d <= 0).all()))
        res.append({
            "channel": c,
            "lag1_autocorr": float(np.nanmean(real)) if real else float("nan"),
            "lag1_autocorr_shuffled": float(np.nanmean(shuf)) if shuf else float("nan"),
            "monotone_nondecreasing_fraction": float(np.mean(up)) if up else float("nan"),
            "monotone_nonincreasing_fraction": float(np.mean(down)) if down else float("nan"),
        })
    return res


def nuisance_ap(X: np.ndarray, y: np.ndarray) -> list[dict]:
    feats = {"effective_length": effective_length(X).astype(float)}
    for c in range(X.shape[1]):
        ch = X[:, c, :]
        feats[f"ch{c}_mean"] = ch.mean(axis=1)
        feats[f"ch{c}_sum"] = ch.sum(axis=1)
        feats[f"ch{c}_max"] = ch.max(axis=1)
        feats[f"ch{c}_frac_nonzero"] = (ch != 0).mean(axis=1)
        feats[f"ch{c}_std"] = ch.std(axis=1)
    rows = []
    for name, v in feats.items():
        a_pos, a_neg = ap(y, v), ap(y, -v)
        rows.append({"feature": name, "ap_as_is": a_pos, "ap_negated": a_neg, "best": max(a_pos, a_neg)})
    return sorted(rows, key=lambda r: -np.nan_to_num(r["best"]))


def find_col(meta, hints) -> str | None:
    cols = {str(c).lower(): str(c) for c in meta.columns}
    for h in hints:
        for low, orig in cols.items():
            if low == h or low.startswith(h + "_") or low.endswith("_" + h) or h in low.split("_"):
                return orig
    return None


def metadata_checks(meta, y) -> dict:
    out: dict = {"columns": list(map(str, meta.columns))}
    g = find_col(meta, GROUP_HINTS)
    out["group_column"] = g
    if g:
        vc = meta[g].value_counts()
        out["n_groups"] = int(vc.size)
        out["group_sizes_min_median_max"] = [int(vc.min()), int(vc.median()), int(vc.max())]
        if y is not None:
            pr = meta.assign(_y=y).groupby(g)["_y"].mean()
            out["group_positive_rate_min_max"] = [float(pr.min()), float(pr.max())]
    roles = {k: find_col(meta, v) for k, v in ROLE_HINTS.items()}
    out["role_columns"] = roles
    if all(roles.values()):
        lig, rec, snd, rcv = (meta[roles[k]].astype(str) for k in ("ligand", "receptor", "sender", "receiver"))
        fwd = list(zip(lig, rec, snd, rcv))
        rev = set(zip(lig, rec, rcv, snd))
        has_rev = np.array([t in rev for t in fwd])
        out["reverse_direction_present_fraction"] = float(has_rev.mean())
        if y is not None and has_rev.any():
            lab = dict(zip(fwd, y))
            pairs = [(lab[t], lab[(t[0], t[1], t[3], t[2])]) for t, h in zip(fwd, has_rev) if h]
            pa = np.array(pairs)
            out["reverse_pairs_both_positive"] = int(((pa[:, 0] == 1) & (pa[:, 1] == 1)).sum() // 2)
            out["reverse_pairs_exactly_one_positive"] = int(((pa[:, 0] + pa[:, 1]) == 1).sum() // 2)
        if y is not None:
            prior = {}
            for k in ("ligand", "receptor", "sender", "receiver"):
                rate = meta.assign(_y=y).groupby(roles[k])["_y"].transform("mean").to_numpy()
                prior[f"ap_of_{k}_positive_rate (in-sample, leaky)"] = ap(y, rate)
            out["group_identity_ap"] = prior
        out["n_unique"] = {k: int(meta[roles[k]].nunique()) for k in roles}
    return out


def fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="decode_report.md")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    s = load_split(a.data_dir, a.split)
    X, y = s.X, s.y
    report: dict = {"split": a.split, "sources": s.sources, "n": int(len(X)), "shape": list(X.shape)}
    if y is not None:
        report["positive_rate"] = float(y.mean())
        report["label_values"] = sorted(map(float, np.unique(y)))[:10]
    report["channels"] = channel_stats(X, rng)
    for ch in report["channels"]:
        ch["guess"] = guess_channel(ch)
    el = effective_length(X)
    report["effective_length"] = {"min": int(el.min()), "median": float(np.median(el)), "max": int(el.max()),
                                  "fraction_full_1024": float(np.mean(el == X.shape[2]))}
    report["order"] = order_stats(X, rng)
    if y is not None:
        report["nuisance_ap"] = nuisance_ap(X, y)
    if s.meta is not None:
        report["metadata"] = metadata_checks(s.meta, y)
    report["other_vectors"] = list(s.vectors)
    report["inventory"] = [f"{f.path.name}[{f.key}] {f.shape} {f.kind} split={f.split}" for f in inventory(a.data_dir)]

    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float))
    L = [f"# Decode report — split `{a.split}`", "", f"Sources: `{s.sources}`", "",
         f"N = {report['n']}, shape = {report['shape']}"
         + (f", positive rate = {report['positive_rate']:.4f} (AP of a random scorer ≈ this)" if y is not None else ", no labels found"), "",
         "## Q1 · What are the five channels?", "",
         "| ch | guess | int frac | zero frac | in [0,1] | min | p50 | p99 | max | mean | std | uniq |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in report["channels"]:
        L.append(f"| {c['channel']} | {c['guess']} | {fmt(c['integer_fraction'])} | {fmt(c['zero_fraction'])} | {fmt(c['in_0_1_fraction'])} | "
                 f"{fmt(c['min'])} | {fmt(c['p50'])} | {fmt(c['p99'])} | {fmt(c['max'])} | {fmt(c['mean'])} | {fmt(c['std'])} | {c['unique_values_sampled']} |")
    e = report["effective_length"]
    L += ["", f"Effective (non-padded) length: min {e['min']}, median {e['median']}, max {e['max']}, "
          f"full-1024 fraction {e['fraction_full_1024']:.3f}", "",
          "## Q2 · Ordered sequence or exchangeable set?", "",
          "Near-equal real vs shuffled autocorrelation ⇒ exchangeable set. Clearly higher real ⇒ ordering carries information.", "",
          "| ch | lag-1 autocorr | shuffled | monotone ↑ | monotone ↓ |", "| --- | --- | --- | --- | --- |"]
    for o in report["order"]:
        L.append(f"| {o['channel']} | {fmt(o['lag1_autocorr'])} | {fmt(o['lag1_autocorr_shuffled'])} | "
                 f"{fmt(o['monotone_nondecreasing_fraction'])} | {fmt(o['monotone_nonincreasing_fraction'])} |")
    if y is not None:
        L += ["", "## Q3 · How much AP is reachable with no signalling inference?", "",
              f"Provisional AP (sklearn), prior = {report['positive_rate']:.4f}. Anything well above the prior is a confounder to control "
              "and a possible Reward-Hacking lead (brief §7–8).", "",
              "| nuisance feature | AP as is | AP negated |", "| --- | --- | --- |"]
        for r in report["nuisance_ap"][:15]:
            L.append(f"| {r['feature']} | {fmt(r['ap_as_is'])} | {fmt(r['ap_negated'])} |")
    if "metadata" in report:
        m = report["metadata"]
        L += ["", "## Metadata", "", f"Columns: {m['columns']}", "",
              f"- Group/section column guess: **{m.get('group_column')}**"
              + (f" — {m['n_groups']} groups, sizes min/median/max {m['group_sizes_min_median_max']}" if m.get("group_column") else " — none found: ask how the hidden test sections differ"),
              f"- Role columns guess: {m['role_columns']}"]
        for k in ("reverse_direction_present_fraction", "reverse_pairs_both_positive", "reverse_pairs_exactly_one_positive", "n_unique", "group_identity_ap", "group_positive_rate_min_max"):
            if k in m:
                L.append(f"- {k}: {m[k]}")
    else:
        L += ["", "## Metadata", "", "No N-row metadata table found. Check the task card for section IDs and candidate identities."]
    L += ["", "## Inventory", ""] + [f"- {i}" for i in report["inventory"]]
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
