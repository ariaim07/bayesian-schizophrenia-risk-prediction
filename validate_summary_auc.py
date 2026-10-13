"""
Does the summary-statistics AUC agree with the true individual-level AUC?
Simulated validation cohorts with known individuals, so the answer is checkable.
"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score

from simulate import Config, Genome, simulate_effects, simulate_sumstats
from prs_cs import prs_cs
from summary_auc import summary_auc

cfg = Config()
rng = np.random.default_rng(cfg.seed)
genome = Genome(cfg, rng)
R_true = genome.block_ld(cfg.n_ld_truth, rng)
R_ref = genome.block_ld(cfg.n_ref, rng)
beta = simulate_effects(cfg, genome, R_true, rng)


def case_control_cohort(n_pop, K, ctrl_per_case, rng):
    """Liability-threshold population; keep genotypes of all cases + sampled controls."""
    keep_X, keep_y, chunk = [], [], 20_000
    # threshold from the liability distribution (G_snp variance = h2_snp)
    t = np.sqrt(1.0) * __import__("scipy").stats.norm.ppf(1 - K)
    for s in range(0, n_pop, chunk):
        G = genome.genotypes(min(chunk, n_pop - s), rng)
        L = genome.standardize(G) @ beta + rng.normal(0, np.sqrt(1 - cfg.h2_snp), len(G))
        y = L > t
        ctrl = (~y) & (rng.random(len(G)) < ctrl_per_case * K * 1.05)
        sel = y | ctrl
        keep_X.append(G[sel]); keep_y.append(y[sel])
    X, y = np.concatenate(keep_X), np.concatenate(keep_y).astype(int)
    return X, y


rows = []
for rep, (n_gwas_disc, n_pop, K) in enumerate([(3000, 400_000, 0.01), (6000, 400_000, 0.01),
                                                (3000, 250_000, 0.02), (6000, 600_000, 0.005)]):
    cfg.n_gwas = n_gwas_disc
    ss = simulate_sumstats(cfg, genome, beta, R_true, np.random.default_rng(100 + rep))
    w = prs_cs(ss["beta"], R_ref, genome.blocks, ss["n"], seed=rep)["beta"]

    X, y = case_control_cohort(n_pop, K, 4, np.random.default_rng(200 + rep))
    p_ctrl = X[y == 0].mean(0) / 2
    sd = np.sqrt(2 * p_ctrl * (1 - p_ctrl))
    Xs = (X - 2 * p_ctrl) / sd
    prs = Xs @ w
    auc_true = roc_auc_score(y, prs)

    # validation-cohort GWAS: per-SNP marginal z from the case-control sample
    yc = (y - y.mean()) / y.std()
    Xc = (Xs - Xs.mean(0)) / Xs.std(0)
    z = (Xc.T @ yc) / np.sqrt(len(y))

    # drop 15% of SNPs at random, as happens when real files don't fully overlap
    keep = np.random.default_rng(300 + rep).random(genome.M) > 0.15
    blocks = []
    for idx, R in zip(genome.blocks, R_ref):
        k = keep[idx]
        blocks.append((w[idx][k], z[idx][k], R[np.ix_(k, k)]))
    est = summary_auc(blocks, len(y), y.mean())
    auc_true_sub = roc_auc_score(y, Xs[:, keep] @ w[keep])
    rows.append({"disc_n": n_gwas_disc, "val_cases": int(y.sum()), "val_n": len(y),
                 "auc_true_individual": round(auc_true_sub, 4),
                 "auc_summary_estimate": round(est["auc"], 4),
                 "ci": [round(c, 4) for c in est["auc_ci"]]})
    print(rows[-1], flush=True)

out = sys.argv[1] if len(sys.argv) > 1 else "results/summary_auc_validation.json"
json.dump(rows, open(out, "w"), indent=1)
