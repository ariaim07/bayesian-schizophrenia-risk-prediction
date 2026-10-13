"""
Simulation of genotypes, GWAS summary statistics and a target cohort for
early-onset schizophrenia (EOS) risk modelling.

Everything here is SYNTHETIC. Parameters are chosen to sit in a realistic
ballpark (see CONFIG and README), not to reproduce any specific study.

Generative model (liability-threshold):
    L_child = G_snp + G_other + env + E          Var(L_child) = 1
    G_snp   = X_std @ beta     (common-SNP liability, h2_snp)
    G_other ~ N(0, h2_twin - h2_snp)   (rare variants, untagged genetics)
    EOS     = 1[L_child > t],  P(EOS) = K_eos

Parents' liabilities are drawn conditional on the child's total genetic value,
which makes "family history" informative beyond the SNPs (as in real life).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    seed: int = 7
    # genome
    n_blocks: int = 150
    snps_per_block: int = 20
    maf_range: tuple = (0.05, 0.5)
    ld_rho_range: tuple = (0.5, 0.95)      # AR(1) correlation of latent haplotype
    # genetic architecture
    h2_snp: float = 0.24                    # liability-scale SNP heritability (PGC3 ballpark)
    h2_twin: float = 0.80                   # twin heritability (ballpark)
    frac_causal: float = 0.10
    # discovery GWAS (stands in for PGC). N is tuned so the resulting PRS explains
    # roughly the share of liability variance real schizophrenia PRS achieve.
    n_gwas: int = 3000
    # LD reference panel (stands in for 1000 Genomes)
    n_ref: int = 1000
    n_ld_truth: int = 30000                 # large sample to approximate population LD
    # population / target cohort
    n_pop: int = 300_000
    chunk: int = 25_000
    K_eos: float = 0.0025                   # illustrative early-onset prevalence
    K_adult: float = 0.007                  # parent lifetime prevalence
    controls_per_case: int = 4
    # environment: (prevalence, liability-scale effect) -- illustrative values
    env: dict = field(default_factory=lambda: {
        "male":               (0.50, 0.15),
        "heavy_cannabis":     (0.08, 0.35),
        "urban_upbringing":   (0.50, 0.15),
        "obstetric_complic":  (0.10, 0.25),
    })
    # post-diagnosis feature used ONLY for the leakage demonstration
    leak_rate_cases: float = 0.85
    leak_rate_controls: float = 0.01


class Genome:
    """LD-block genome; haplotypes are thresholded AR(1) Gaussians."""

    def __init__(self, cfg: Config, rng):
        self.cfg = cfg
        B, S = cfg.n_blocks, cfg.snps_per_block
        self.M = B * S
        self.maf = rng.uniform(*cfg.maf_range, size=self.M)
        self.rho = rng.uniform(*cfg.ld_rho_range, size=B)
        from scipy.stats import norm
        self.cut = norm.ppf(1 - self.maf)          # allele = 1 if latent > cut
        self.mu = 2 * self.maf
        self.sd = np.sqrt(2 * self.maf * (1 - self.maf))
        self.blocks = [np.arange(b * S, (b + 1) * S) for b in range(B)]

    def _haplotypes(self, n, rng):
        S = self.cfg.snps_per_block
        H = np.empty((n, self.M), dtype=np.int8)
        for b, idx in enumerate(self.blocks):
            r = self.rho[b]
            e = rng.standard_normal((n, S)).astype(np.float32)
            z = e[:, 0].copy()
            H[:, idx[0]] = z > self.cut[idx[0]]
            s = np.float32(np.sqrt(1 - r * r))
            for j in range(1, S):
                z = r * z + s * e[:, j]
                H[:, idx[j]] = z > self.cut[idx[j]]
        return H

    def genotypes(self, n, rng):
        return self._haplotypes(n, rng) + self._haplotypes(n, rng)   # 0/1/2

    def standardize(self, G):
        return ((G - self.mu) / self.sd).astype(np.float32)

    def block_ld(self, n, rng):
        X = self.standardize(self.genotypes(n, rng))
        return [np.corrcoef(X[:, idx], rowvar=False) for idx in self.blocks]


def simulate_effects(cfg, genome, R_true, rng):
    beta = np.zeros(genome.M)
    causal = rng.random(genome.M) < cfg.frac_causal
    beta[causal] = rng.standard_normal(causal.sum())
    var = sum(beta[idx] @ R @ beta[idx] for idx, R in zip(genome.blocks, R_true))
    beta *= np.sqrt(cfg.h2_snp / var)
    return beta


def simulate_sumstats(cfg, genome, beta, R_true, rng):
    """Marginal standardized effects: beta_hat = R beta + N(0, R/N)  (per block)."""
    bhat = np.zeros(genome.M)
    for idx, R in zip(genome.blocks, R_true):
        Lc = np.linalg.cholesky(R + 1e-8 * np.eye(len(idx)))
        bhat[idx] = R @ beta[idx] + Lc @ rng.standard_normal(len(idx)) / np.sqrt(cfg.n_gwas)
    se = np.full(genome.M, 1 / np.sqrt(cfg.n_gwas))
    return {"beta": bhat, "se": se, "z": bhat / se, "n": cfg.n_gwas}


def simulate_population(cfg, genome, beta, prs_weights, rng):
    """Stream the population in chunks; keep only what the study needs.

    prs_weights: dict name -> per-standardized-SNP weights, scored for everyone.
    Genotype matrices are not kept (memory); scores are computed on the fly.
    """
    n, out = cfg.n_pop, {}
    h2_other = cfg.h2_twin - cfg.h2_snp
    G_snp = np.empty(n, np.float32)
    scores = {k: np.empty(n, np.float32) for k in prs_weights}
    for s in range(0, n, cfg.chunk):
        e = min(s + cfg.chunk, n)
        X = genome.standardize(genome.genotypes(e - s, rng))
        G_snp[s:e] = X @ beta.astype(np.float32)
        for k, w in prs_weights.items():
            scores[k][s:e] = X @ w.astype(np.float32)

    G_other = rng.normal(0, np.sqrt(h2_other), n)
    A = G_snp + G_other
    env_cols, env_liab = {}, np.zeros(n)
    for name, (p, eff) in cfg.env.items():
        x = (rng.random(n) < p).astype(np.int8)
        env_cols[name] = x
        env_liab += eff * (x - p)
    var_env = sum(eff**2 * p * (1 - p) for p, eff in cfg.env.values())
    var_E = 1 - cfg.h2_twin - var_env
    L = A + env_liab + rng.normal(0, np.sqrt(var_E), n)
    y = (L > np.quantile(L, 1 - cfg.K_eos)).astype(np.int8)

    # parents | child's total genetic value (A_m, A_f iid var h2; A_c = mean + segregation)
    h = cfg.h2_twin
    cov_pc, var_c = h / 2, h
    mean_p = (cov_pc / var_c) * A
    S = np.array([[h - cov_pc**2 / var_c, -cov_pc**2 / var_c],
                  [-cov_pc**2 / var_c, h - cov_pc**2 / var_c]])
    Ap = mean_p[:, None] + rng.standard_normal((n, 2)) @ np.linalg.cholesky(S).T
    Lp = Ap + rng.normal(0, np.sqrt(1 - h), (n, 2))
    from scipy.stats import norm
    fam = (Lp > norm.ppf(1 - cfg.K_adult)).any(axis=1).astype(np.int8)

    out.update(y=y, G_snp=G_snp, liability=L, L_known=A + env_liab,
               family_history=fam, **env_cols,
               **{f"score_{k}": v for k, v in scores.items()})
    return out


def sample_case_control(cfg, pop, rng, eligible):
    """Case-control study drawn only from `eligible` people (the rest of the
    population is kept untouched for a ground-truth validation)."""
    y = pop["y"][eligible]
    cases = eligible[y == 1]
    ctrls = rng.choice(eligible[y == 0],
                       size=cfg.controls_per_case * len(cases), replace=False)
    idx = np.concatenate([cases, ctrls])
    rng.shuffle(idx)
    cohort = {k: v[idx] for k, v in pop.items()}
    cohort["person_id"] = idx                      # unique per person -> no duplicates
    y = cohort["y"]
    leak_p = np.where(y == 1, cfg.leak_rate_cases, cfg.leak_rate_controls)
    cohort["antipsychotic_rx_LEAKY"] = (rng.random(len(y)) < leak_p).astype(np.int8)
    return cohort
