"""Figure for the real-data validation (results/validation_auc.json)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summary_auc import ppv_from_auc

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
res = {r["population"]: r for r in json.load(open(OUT / "validation_auc.json"))}
SE_BIAS = res["East Asian (se)"]["auc"] - res["East Asian"]["auc"]   # SE-derived N runs high

rows = []
for name, key in [("African American", "African American (se)"),
                  ("East Asian", "East Asian"), ("Latino", "Latino (se)")]:
    r = res[key]
    adj = SE_BIAS if key.endswith("(se)") else 0.0
    rows.append((name, r["auc"] - adj, [c - adj for c in r["auc_ci95"]], key.endswith("(se)")))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9893"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})

ax = axes[0]
for i, (name, a, (lo, hi), est) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(a, i, "o", ms=8, color=BLUE, mec=SURF, mew=2)
    ax.text(hi + 0.01, i, f"{a:.2f}" + ("*" if est else ""), va="center", color=INK, fontsize=9)
ax.axvline(0.5, color=GRID, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_xlim(0.45, 0.8); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("AUC of European-trained PRS-CS score (95% CI)")
ax.set_title("A. Real out-of-sample AUC, independent PGC3 cohorts", loc="left", fontweight="bold")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.45, -1.15, "* sample size derived from standard errors; shifted down by the 0.01 bias "
        "measured on the East Asian cohort", fontsize=7.5, color=INK2, transform=ax.transData)

ax = axes[1]
K = 0.0025
best = max(rows, key=lambda r: r[1])
eas = [r for r in rows if r[0] == "East Asian"][0]
qs = [0.01, 0.05, 0.10]
x = np.arange(len(qs))
for j, (row, col) in enumerate([(eas, BLUE), (best, GRAY)]):
    v = [100 * ppv_from_auc(row[1], K, q)["ppv"] for q in qs]
    ax.bar(x + (j - 0.5) * 0.36, v, width=0.34, color=col, label=f"{row[0]} (AUC {row[1]:.2f})")
    for xi, vi in zip(x + (j - 0.5) * 0.36, v):
        ax.text(xi, vi + 0.03, f"{vi:.1f}%", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x, [f"Top {int(q*100)}%" for q in qs])
ax.set_ylabel("Flagged people who become cases (PPV)")
ax.set_xlabel("Share of population flagged as high risk")
ax.set_title(f"B. What it means at {K*100:.2f}% prevalence (illustrative)", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.5)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "real_validation.png", dpi=160)
print("saved", OUT / "real_validation.png")
print([(r[0], round(r[1], 3), [round(c, 3) for c in r[2]]) for r in rows])
