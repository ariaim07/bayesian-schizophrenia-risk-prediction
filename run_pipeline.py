"""
End-to-end pipeline (simulated data):

  1. simulate genome, LD, true effects, PGC-style GWAS summary statistics
  2. Bayesian PRS (PRS-CS) from summary stats + LD reference; C+T baseline
  3. simulate a population, sample a case-control target cohort
  4. person-level train/test split
  5. Bayesian risk models (PyMC), incl. a deliberately leaky one
  6. evaluate: AUC + CI, calibration, PPV at real prevalence, population check
"""
import json, time, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from simulate import (Config, Genome, simulate_effects, simulate_sumstats,
                      simulate_population, sample_case_control)
from prs_cs import prs_cs, clump_threshold
from risk_model import BayesRisk
from evaluate import (auc_ci, calibration_bins, calibration_slope_intercept,
                      ppv_at_prevalence, empirical_ppv)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT.mkdir(exist_ok=True, parents=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)

cfg = Config()
rng = np.random.default_rng(cfg.seed)
results = {"config": {k: v for k, v in cfg.__dict__.items()}}

# 1 -------------------------------------------------------------------------
genome = Genome(cfg, rng)
R_true = genome.block_ld(cfg.n_ld_truth, rng)      # population LD (truth)
R_ref = genome.block_ld(cfg.n_ref, rng)            # small reference panel (what you'd have)
beta_true = simulate_effects(cfg, genome, R_true, rng)
ss = simulate_sumstats(cfg, genome, beta_true, R_true, rng)
log(f"GWAS: {genome.M} SNPs, N={cfg.n_gwas}, "
    f"{(np.abs(ss['z']) > 5.45).sum()} SNPs pass genome-wide significance (5e-8)")

# 2 -------------------------------------------------------------------------
fit = prs_cs(ss["beta"], R_ref, genome.blocks, ss["n"], seed=cfg.seed)
w_ct = clump_threshold(ss["beta"], ss["se"], genome.blocks)
log(f"PRS-CS done, learned global shrinkage phi={fit['phi']:.2e}")

# 3 -------------------------------------------------------------------------
pop = simulate_population(cfg, genome, beta_true,
                          {"prscs": fit["beta"], "ct": w_ct}, rng)
pop["prs"] = pop["score_prscs"]
# half the population is the recruitment pool for the study; the other half is
# never seen by any model and serves as a ground-truth "deployment" population
perm = rng.permutation(cfg.n_pop)
study_pool, holdout_pop = perm[: cfg.n_pop // 2], perm[cfg.n_pop // 2:]
cohort = sample_case_control(cfg, pop, rng, study_pool)
n_case = int(cohort["y"].sum())
log(f"population {cfg.n_pop:,}: {int(pop['y'].sum())} early-onset cases; "
    f"cohort {len(cohort['y'])} ({n_case} cases); "
    f"held-out population {len(holdout_pop):,} ({int(pop['y'][holdout_pop].sum())} cases)")

prs_check = {
    "corr_beta_prscs_vs_true": float(np.corrcoef(fit["beta"], beta_true)[0, 1]),
    "corr_beta_marginal_vs_true": float(np.corrcoef(ss["beta"], beta_true)[0, 1]),
    "liability_r2_prscs": float(np.corrcoef(pop["score_prscs"], pop["liability"])[0, 1] ** 2),
    "liability_r2_ct": float(np.corrcoef(pop["score_ct"], pop["liability"])[0, 1] ** 2),
    "liability_r2_true_snp_score": float(np.corrcoef(pop["G_snp"], pop["liability"])[0, 1] ** 2),
}
results["prs_check"] = prs_check
log("PRS check:", {k: round(v, 3) for k, v in prs_check.items()})

# 4 -------------------------------------------------------------------------
idx_tr, idx_te = train_test_split(np.arange(len(cohort["y"])), test_size=0.3,
                                  stratify=cohort["y"], random_state=cfg.seed)
assert not set(cohort["person_id"][idx_tr]) & set(cohort["person_id"][idx_te]), "person overlap!"
tr = {k: v[idx_tr] for k, v in cohort.items()}
te = {k: v[idx_te] for k, v in cohort.items()}
ytr, yte = tr["y"].astype(int), te["y"].astype(int)

# 5 -------------------------------------------------------------------------
COV = ["family_history", "male", "heavy_cannabis", "urban_upbringing", "obstetric_complic"]
specs = {
    "Covariates only": COV,
    "PRS only (PRS-CS)": ["prs"],
    "PRS + covariates (main model)": ["prs"] + COV,
    "LEAKY: + antipsychotic Rx": ["prs"] + COV + ["antipsychotic_rx_LEAKY"],
}
models, scores = {}, {}
for name, feats in specs.items():
    models[name] = BayesRisk(feats, seed=cfg.seed).fit(tr, ytr)
    scores[name] = models[name].predict(te)[0]
    log(f"fitted: {name}")

# 6 -------------------------------------------------------------------------
perf = {}
for name, s in scores.items():
    a, lo, hi = auc_ci(yte, s)
    perf[name] = {"auc": a, "ci95": [lo, hi]}
for name, s in {"Ceiling: true common-SNP liability": te["G_snp"],
                "Ceiling: all genetics + environment known": te["L_known"],
                "C+T PRS only (non-Bayesian baseline)": te["score_ct"]}.items():
    a, lo, hi = auc_ci(yte, s)
    perf[name] = {"auc": a, "ci95": [lo, hi]}
results["auc_test"] = perf
for k, v in perf.items():
    log(f"AUC {v['auc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]  {k}")

main = models["PRS + covariates (main model)"]
p_te, lo_te, hi_te = main.predict(te)
slope, intercept = calibration_slope_intercept(yte, p_te)
results["calibration_test"] = {"slope": slope, "intercept": intercept,
                               "bins": calibration_bins(yte, p_te)}
results["posterior_summary"] = main.summary().round(3).to_dict(orient="index")
results["mean_90pct_interval_width_sample_scale"] = float(np.mean(hi_te - lo_te))

K = cfg.K_eos
results["ppv_analytic"] = ppv_at_prevalence(yte, p_te, K)

# ground truth: score the held-out half of the population (nobody from the study)
assert not set(holdout_pop) & set(cohort["person_id"])
popx = {k: v[holdout_pop] for k, v in pop.items()}
p_pop, lo_pop, hi_pop = main.predict(popx, prevalence=K)
y_pop = popx["y"].astype(int)
results["population_check"] = {
    "n": len(holdout_pop), "cases": int(y_pop.sum()),
    "auc": float(roc_auc_score(y_pop, p_pop)),
    "mean_predicted_risk": float(p_pop.mean()), "observed_rate": float(y_pop.mean()),
    "ppv_empirical": empirical_ppv(y_pop, p_pop),
    "calibration_bins": calibration_bins(y_pop, p_pop, 10),
    "top_1pct_mean_risk": float(p_pop[p_pop >= np.quantile(p_pop, 0.99)].mean()),
}
log("population check:", {k: v for k, v in results["population_check"].items()
                           if k in ("auc", "mean_predicted_risk", "observed_rate")})

# arrays for plotting
curves = {}
for name, s in list(scores.items()) + [("Ceiling: true common-SNP liability", te["G_snp"])]:
    fpr, tpr, _ = roc_curve(yte, s)
    curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
results["roc_curves"] = curves
results["beta_scatter"] = {"true": beta_true.tolist(), "prscs": fit["beta"].tolist(),
                           "marginal": ss["beta"].tolist()}
results["runtime_sec"] = time.time() - T0

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, default=float)
main.summary().round(3).to_csv(OUT / "posterior_summary_main_model.csv")
log(f"saved to {OUT}/")
