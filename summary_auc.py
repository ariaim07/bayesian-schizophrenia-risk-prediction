"""
AUC of a polygenic score in a validation cohort, from that cohort's GWAS summary
statistics only (no individual-level data).

For standardized genotypes x_j, standardized weights w_j and a case/control
indicator y in the validation sample:

    corr(PRS, y) = sum_j w_j * corr(x_j, y) / sqrt(w' R w)
    corr(x_j, y) ≈ z_j / sqrt(N)                     (marginal GWAS z-score)

R is the LD matrix of the validation population (taken from a reference panel).
If PRS is normal with a shift d (in within-group SD units) between cases and
controls, a sample with case fraction P has
    r = d*sqrt(P(1-P)) / sqrt(1 + d^2 P(1-P))   ->   d = r / sqrt(P(1-P)(1-r^2))
    AUC = Phi(d / sqrt(2)).
This is the same idea as SummaryAUC (Song et al., Bioinformatics 2019).
"""
import numpy as np
from scipy.stats import norm


def r_to_auc(r, case_frac):
    P = case_frac
    d = r / np.sqrt(P * (1 - P) * (1 - r ** 2))
    return norm.cdf(d / np.sqrt(2))


def summary_auc(blocks, n, case_frac, level=0.95):
    """blocks: iterable of (w, z, R) per LD block, already allele-aligned.
    w = weights per standardized genotype; z = validation GWAS z-scores;
    R = validation-population LD among those SNPs.
    Returns dict with r, AUC and a CI from the sampling variance of r."""
    num = den = 0.0
    m = 0
    for w, z, R in blocks:
        num += w @ z
        den += w @ R @ w
        m += len(w)
    r = num / np.sqrt(n) / np.sqrt(den)
    se_r = (1 - r ** 2) / np.sqrt(n)             # Fisher-style SE for a correlation
    q = norm.ppf(0.5 + level / 2)
    lo, hi = r - q * se_r, r + q * se_r
    return {"n_snps": m, "r": float(r), "r2_observed": float(r ** 2),
            "auc": float(r_to_auc(r, case_frac)),
            "auc_ci": [float(r_to_auc(lo, case_frac)), float(r_to_auc(hi, case_frac))]}


def ppv_from_auc(auc, prevalence, top_frac):
    """Binormal equal-variance model: PPV and sensitivity when flagging the top
    `top_frac` of a population with the given prevalence."""
    from scipy.optimize import brentq
    d = np.sqrt(2) * norm.ppf(auc)                # case mean shift; controls N(0,1)
    flagged = lambda t: prevalence * norm.sf(t - d) + (1 - prevalence) * norm.sf(t)
    t = brentq(lambda t: flagged(t) - top_frac, -10, 10)
    sens = norm.sf(t - d)
    ppv = prevalence * sens / top_frac
    return {"flag_top": top_frac, "sensitivity": float(sens), "ppv": float(ppv)}
