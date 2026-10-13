"""Summary figure from results/results.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
r = json.load(open(OUT / "results.json"))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9893"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})

# A: AUC with 95% CI
ax = axes[0]
order = ["Covariates only", "C+T PRS only (non-Bayesian baseline)", "PRS only (PRS-CS)",
         "PRS + covariates (main model)",
         "Ceiling: true common-SNP liability", "Ceiling: all genetics + environment known"]
order = [k for k in order if k in r["auc_test"]][::-1]
for i, k in enumerate(order):
    a, (lo, hi) = r["auc_test"][k]["auc"], r["auc_test"][k]["ci95"]
    c = ORANGE if k.startswith("LEAKY") else GRAY if k.startswith("Ceiling") else BLUE
    ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
    ax.plot(a, i, "o", ms=8, color=c, mec=SURF, mew=2)
    ax.text(hi + 0.008, i, f"{a:.2f}", va="center", color=INK, fontsize=9)
ax.set_yticks(range(len(order)), order)
ax.axvline(0.5, color=GRID, lw=1)
ax.set_xlim(0.45, 1.04)
ax.set_xlabel("Test-set AUC (95% bootstrap CI)")
ax.set_title("A. Discrimination by model", loc="left", fontweight="bold")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)

# B: calibration in the held-out population
ax = axes[1]
pc = r["population_check"]
pred = np.array([b[0] for b in pc["calibration_bins"]])
obs = np.array([b[1] for b in pc["calibration_bins"]])
lim = [1e-4, 3e-2]
ax.plot(lim, lim, color=GRAY, lw=1, ls="--")
ax.text(1.2e-2, 4.5e-3, "perfect\ncalibration", color=INK2, fontsize=8, ha="left")
ax.plot(pred, np.maximum(obs, 1.2e-4), "o-", color=BLUE, lw=2, ms=8, mec=SURF, mew=2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Predicted risk (decile mean)")
ax.set_ylabel("Observed rate")
ax.set_title(f"B. Calibration, untouched population\n(n={pc['n']:,}, {pc['cases']} cases)",
             loc="left", fontweight="bold")
ax.grid(color=GRID, lw=0.6, which="major"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)

# C: what flagging means at real prevalence
ax = axes[2]
emp = pc["ppv_empirical"]
labels = [f"Top {int(e['flag_top']*100)}%" for e in emp]
ppv = [100 * e["ppv"] for e in emp]
sens = [100 * e["sensitivity"] for e in emp]
ax.bar(range(len(emp)), ppv, color=BLUE, width=0.55)
for i, (p, s) in enumerate(zip(ppv, sens)):
    ax.text(i, p + 0.12, f"{p:.1f} true cases\nper 100 flagged\n(catches {s:.0f}% of cases)",
            ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(range(len(emp)), labels)
ax.set_ylim(0, max(ppv) * 2.1)
ax.set_ylabel("PPV (%)")
ax.set_xlabel("Share of population flagged as high risk")
ax.set_title(f"C. Main model at real prevalence ({r['config']['K_eos']*100:.2f}%)",
             loc="left", fontweight="bold")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)

fig.suptitle("Bayesian early-onset schizophrenia risk model — SIMULATED data",
             x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "summary_figure.png", dpi=160)
print("saved", OUT / "summary_figure.png")
