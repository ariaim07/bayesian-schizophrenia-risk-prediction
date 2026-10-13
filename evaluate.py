"""Evaluation that matters for a rare-outcome screening model."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc_ci(y, s, n_boot=2000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    n, aucs = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.quantile(aucs, [(1 - level) / 2, (1 + level) / 2])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def calibration_bins(y, p, n_bins=10):
    """Observed vs predicted event rate by predicted-risk decile."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    return [(float(p[b == k].mean()), float(y[b == k].mean()), int((b == k).sum()))
            for k in range(n_bins) if (b == k).any()]


def calibration_slope_intercept(y, p):
    """Logistic recalibration: ideal slope 1, intercept 0."""
    from sklearn.linear_model import LogisticRegression
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6).fit(lp, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def ppv_at_prevalence(y, s, prevalence, top_fracs=(0.01, 0.05, 0.10)):
    """Sens/spec measured in the case-control test set, PPV computed via Bayes'
    rule at the real population prevalence (a case-control set's own PPV is
    meaningless because cases were oversampled)."""
    fpr, tpr, thr = roc_curve(y, s)
    out = []
    for q in top_fracs:
        # threshold that flags fraction q of the *population*:
        # q = tpr*K + fpr*(1-K)  -> pick the closest ROC point
        flagged = tpr * prevalence + fpr * (1 - prevalence)
        k = int(np.argmin(np.abs(flagged - q)))
        sens, f = tpr[k], fpr[k]
        ppv = sens * prevalence / (sens * prevalence + f * (1 - prevalence))
        out.append({"flag_top": q, "sensitivity": float(sens),
                    "false_pos_rate": float(f), "ppv": float(ppv),
                    "false_pos_per_true_pos": float((1 - ppv) / ppv) if ppv > 0 else np.inf})
    return out


def empirical_ppv(y_pop, s_pop, top_fracs=(0.01, 0.05, 0.10)):
    """Ground-truth PPV/sensitivity in the full simulated population."""
    out = []
    for q in top_fracs:
        t = np.quantile(s_pop, 1 - q)
        flag = s_pop >= t
        out.append({"flag_top": q, "ppv": float(y_pop[flag].mean()),
                    "sensitivity": float(flag[y_pop == 1].mean())})
    return out
