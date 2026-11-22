"""Report figures from saved result JSONs (no recomputation). Print-oriented, light surface.

fig_scores.pdf : validation lr_auprc per predictor, mean ± SD over training seeds (deterministic predictors
                 have no seed variance), baseline as a dashed reference line. Single series -> one color.
fig_beta.pdf   : AP by injected effect size beta (0.5 / 1 / 2) for the four ablation predictors.
                 Categorical palette slots 1-4 (validated: all checks pass; contrast WARN relieved by
                 legend + value labels).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = Path.home() / "work/cell-signalling"
OUT = W / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)
L = W / "runs/loopholes"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42})

abl = json.loads((L / "iter1_ablation.json").read_text())
bce = json.loads((L / "model_eval_iter1-noskip-bce.json").read_text())["rows"]
m1 = json.loads((W / "runs/m1/compare.json").read_text())
nb = json.loads((W / "runs/nb/compare.json").read_text())
import numpy as np  # noqa: E402

rows = [
    ("NB Δdeviance (team idea)", nb[1]["val"]["lr_auprc"], 0.0),
    ("M1 two-part slope t (team)", next(r for r in m1 if r["candidate"].startswith("C4 "))["val"]["lr_auprc"], 0.0),
    ("(a) partial-corr z-score alone", abl["(a) partial-corr z-score alone"]["lr_auprc_mean"], 0.0),
    ("(c) iter1 as is", abl["(c) iter1 as is"]["lr_auprc_mean"], abl["(c) iter1 as is"]["lr_auprc_sd"]),
    ("(b) iter1 without skip + BCE", float(np.mean([r["lr_auprc"] for r in bce])), float(np.std([r["lr_auprc"] for r in bce]))),
    ("(b) iter1 without skip", abl["(b) iter1 without skip"]["lr_auprc_mean"], abl["(b) iter1 without skip"]["lr_auprc_sd"]),
]
for label, name in (("(b) + contact subsampling", "iter1-noskip-aug"), ("(b) + generator data (m1)", "iter1-noskip-gen"),
                    ("R8 ensemble (m2 + gen. data)", "ensemble-aug-both"),
                    ("R7 ensemble (final)", "ensemble-noskip-explore2-v2")):
    rr = json.loads((L / f"model_eval_{name}.json").read_text())["rows"]
    rows.append((label, float(np.mean([r["lr_auprc"] for r in rr])), float(np.std([r["lr_auprc"] for r in rr]))))
base = abl["baseline (contact Spearman)"]["lr_auprc_mean"]

fig, ax = plt.subplots(figsize=(3.3, 2.8))
y = np.arange(len(rows))
ax.axvline(base, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(base - 0.004, -0.75, f"baseline {base:.3f}", color=INK2, fontsize=6.5, va="center", ha="right")
for i, (name, m, sd) in enumerate(rows):
    ax.errorbar(m, i, xerr=sd if sd > 0 else None, fmt="o", ms=4.5, color=SERIES[0], ecolor=SERIES[0],
                elinewidth=1.5, capsize=2.5, zorder=3)
    if base - 0.06 < m < base - 0.01:  # just left of the baseline line: label on the left
        ax.text(m - 0.008, i, f"{m:.3f}", va="center", ha="right", fontsize=7, color=INK)
    else:
        ax.text(m + max(sd, 0) + 0.006, i, f"{m:.3f}", va="center", fontsize=7, color=INK)
ax.set_yticks(y, [r[0] for r in rows])
ax.set_xlabel("validation lr_auprc")
ax.set_xlim(0.38, 0.65)
ax.set_ylim(-1.2, len(rows) - 0.5)
ax.grid(axis="x", color="#e6e5e1", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_scores.pdf")
fig.savefig(OUT / "fig_scores.png", dpi=200)

ens = json.loads((L / "ensemble_report_stats.json").read_text())["beta"]
abl["final R7 ensemble"] = {"ap_by_beta": ens["v2 ensemble (submitted)"]}
preds = ["baseline (contact Spearman)", "(a) partial-corr z-score alone", "(b) iter1 without skip", "final R7 ensemble"]
short = ["baseline", "(a) z-score", "(b) no skip", "final (R7)"]
betas = ["0.5", "1.0", "2.0"]
fig, ax = plt.subplots(figsize=(3.3, 2.0))
w = 0.2
for j, (p, s) in enumerate(zip(preds, short)):
    vals = [abl[p]["ap_by_beta"][b] for b in betas]
    xs = np.arange(3) + (j - 1.5) * w
    ax.bar(xs, vals, width=w - 0.02, color=SERIES[j], label=s, edgecolor="white", linewidth=1.0, zorder=3)
ax.set_xticks(np.arange(3), [f"β = {b.rstrip('0').rstrip('.')}" for b in betas])
ax.set_ylabel("AP (positives of one β\nvs all negatives)")
ax.set_ylim(0, 0.9)
ax.grid(axis="y", color="#e6e5e1", lw=0.6)
ax.set_axisbelow(True)
ax.legend(ncol=4, fontsize=6, frameon=False, loc="upper center", handlelength=1.0, columnspacing=0.8)
fig.tight_layout()
fig.savefig(OUT / "fig_beta.pdf")
fig.savefig(OUT / "fig_beta.png", dpi=200)
print("wrote", sorted(p.name for p in OUT.iterdir()))
