"""
Bayesian logistic regression for early-onset schizophrenia risk (PyMC).

    logit P(EOS) = alpha + X @ b
    alpha ~ Normal(logit(case fraction), 1.5)
    b_k   ~ Normal(0, 1)      on standardized predictors (weakly informative)

Why Bayesian here: the output is a full posterior per person, so every
predicted risk comes with an uncertainty interval, and the case-control
intercept can be shifted to the real population prevalence explicitly.
"""
import numpy as np
import pymc as pm


class BayesRisk:
    def __init__(self, features, prior_sd=1.0, draws=1000, tune=1000, chains=4, seed=0):
        self.features, self.prior_sd = features, prior_sd
        self.draws, self.tune, self.chains, self.seed = draws, tune, chains, seed

    def _X(self, data):
        X = np.column_stack([np.asarray(data[f], float) for f in self.features])
        return (X - self.mu_) / self.sd_

    def fit(self, data, y):
        X = np.column_stack([np.asarray(data[f], float) for f in self.features])
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12      # train-set scaling only
        X = (X - self.mu_) / self.sd_
        f = y.mean()
        self.train_case_frac_ = f
        with pm.Model() as m:
            alpha = pm.Normal("alpha", np.log(f / (1 - f)), 1.5)
            b = pm.Normal("b", 0, self.prior_sd, shape=X.shape[1])
            pm.Bernoulli("y", logit_p=alpha + pm.math.dot(X, b), observed=y)
            self.idata_ = pm.sample(self.draws, tune=self.tune, chains=self.chains,
                                    cores=min(self.chains, 2), random_seed=self.seed,
                                    progressbar=False, target_accept=0.9)
        post = self.idata_.posterior
        self.alpha_ = post["alpha"].values.reshape(-1)
        self.b_ = post["b"].values.reshape(-1, X.shape[1])
        return self

    def logit_draws(self, data, prevalence=None):
        """Posterior draws of the logit; optionally re-anchored to a population prevalence."""
        eta = self.alpha_[:, None] + self.b_ @ self._X(data).T
        if prevalence is not None:
            f = self.train_case_frac_
            eta = eta + np.log(prevalence / (1 - prevalence)) - np.log(f / (1 - f))
        return eta

    def predict(self, data, prevalence=None, interval=0.9, chunk=10_000):
        """Posterior mean risk and credible interval per person (chunked for memory)."""
        n = len(np.asarray(data[self.features[0]]))
        mean, lo, hi = (np.empty(n) for _ in range(3))
        for s in range(0, n, chunk):
            sl = slice(s, min(s + chunk, n))
            part = {f: np.asarray(data[f])[sl] for f in self.features}
            p = 1 / (1 + np.exp(-self.logit_draws(part, prevalence)))
            mean[sl] = p.mean(0)
            lo[sl], hi[sl] = np.quantile(p, [(1 - interval) / 2, (1 + interval) / 2], axis=0)
        return mean, lo, hi

    def summary(self):
        import arviz as az
        s = az.summary(self.idata_, var_names=["alpha", "b"])
        s.index = ["intercept"] + [f"b[{f}]" for f in self.features]
        return s
