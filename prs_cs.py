"""
Bayesian polygenic risk score with a continuous-shrinkage prior (PRS-CS).

Re-implementation of the Gibbs sampler from
    Ge T, Chen C-Y, Ni Y, Feng Y-CA, Smoller JW. Polygenic prediction via
    Bayesian regression and continuous shrinkage priors. Nat Commun 2019.

Prior on standardized SNP effect j:
    beta_j ~ N(0, sigma2/N * phi * psi_j),  psi_j ~ G(a, delta_j),  delta_j ~ G(b, 1)
a=1, b=0.5 gives the Strawderman-Berger prior: lots of mass near zero (most SNPs
do nothing) with heavy tails (a few do a lot). phi is the global scale; here it
is learned from the data ("auto" mode).

Inputs are only summary statistics + an LD reference panel, exactly like the
real PRS-CS with PGC summary stats and a 1000 Genomes / UKB LD panel.
"""
import numpy as np
from scipy.stats import geninvgauss


def _gig(p, a, b, rng):
    """Sample GIG(p, a, b) with density ∝ x^(p-1) exp(-(a x + b / x) / 2)."""
    b = np.maximum(b, 1e-12)
    if p == 0.5:
        # Fast exact path: if X ~ GIG(1/2, a, b) then 1/X ~ InvGauss(mean=sqrt(a/b), shape=a)
        return 1.0 / rng.wald(np.sqrt(a / b), a)
    return np.sqrt(b / a) * geninvgauss.rvs(p, np.sqrt(a * b), random_state=rng)


def prs_cs(beta_mrg, ld_blocks, blocks, n, a=1.0, b=0.5, phi=None,
           n_iter=1000, n_burnin=500, thin=5, seed=1, verbose=False):
    """Return posterior mean of standardized SNP effects.

    beta_mrg  : marginal standardized effects (z / sqrt(n))
    ld_blocks : list of LD (correlation) matrices from the reference panel
    blocks    : list of index arrays aligning SNPs to ld_blocks
    phi       : global shrinkage; None = learn it (PRS-CS-auto)
    """
    rng = np.random.default_rng(seed)
    p = len(beta_mrg)
    beta = np.zeros(p)
    psi = np.ones(p)
    sigma = 1.0
    phi_auto = phi is None
    phi = 1.0 if phi_auto else phi
    delta = np.ones(p)
    beta_est = np.zeros(p)
    psi_est = np.zeros(p)
    phi_trace, n_kept = [], 0

    for it in range(1, n_iter + 1):
        quad = 0.0
        for idx, D in zip(blocks, ld_blocks):
            dinvt = D + np.diag(1.0 / psi[idx])
            L = np.linalg.cholesky(dinvt)
            mean = np.linalg.solve(L.T, np.linalg.solve(L, beta_mrg[idx]))
            draw = np.linalg.solve(L.T, rng.standard_normal(len(idx)))
            beta[idx] = mean + np.sqrt(sigma / n) * draw
            quad += beta[idx] @ dinvt @ beta[idx]

        err = max(n / 2 * (1 - 2 * beta @ beta_mrg + quad),
                  n / 2 * np.sum(beta**2 / psi))
        sigma = 1.0 / rng.gamma((n + p) / 2, 1.0 / err)

        delta = rng.gamma(a + b, 1.0 / (psi + phi))
        psi = _gig(a - 0.5, 2 * delta, n * beta**2 / sigma, rng)
        psi = np.minimum(psi, 1.0)

        if phi_auto:
            w = rng.gamma(1.0, 1.0 / (phi + 1))
            phi = rng.gamma(p * b + 0.5, 1.0 / (delta.sum() + w))

        if it > n_burnin and it % thin == 0:
            beta_est += beta
            psi_est += psi
            n_kept += 1
            phi_trace.append(phi)
        if verbose and it % 100 == 0:
            print(f"  PRS-CS iter {it}/{n_iter}  phi={phi:.2e}  sigma={sigma:.3f}")

    return {"beta": beta_est / n_kept, "psi": psi_est / n_kept,
            "phi": float(np.mean(phi_trace)), "phi_trace": np.array(phi_trace)}


def clump_threshold(beta_mrg, se, blocks, p_thresh=0.05):
    """Naive baseline: best SNP per LD block if p < threshold, marginal effect as weight."""
    from scipy.stats import norm
    pval = 2 * norm.sf(np.abs(beta_mrg / se))
    w = np.zeros_like(beta_mrg)
    for idx in blocks:
        j = idx[np.argmin(pval[idx])]
        if pval[j] < p_thresh:
            w[j] = beta_mrg[j]
    return w
