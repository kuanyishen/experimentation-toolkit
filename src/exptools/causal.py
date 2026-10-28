"""Covariate-adjusted effect estimation for experiments whose assignment depends on covariates.

In a clean randomized test ``difference in means`` is unbiased. If assignment probability varies with covariates
(e.g. pooled sub-tests with different splits) AND those covariates predict the outcome, it is not. Three
adjustments that agree with each other are strong evidence the adjustment is doing its job:

* IPW   - re-weight each user by 1/P(assigned arm | x)
* AIPW  - doubly robust: outcome-model contrast plus a weighted residual correction (consistent if either the
          propensity or the outcome model is right; lowest variance when the outcome model is good)
* weighted SMD - check that weighting actually restores covariate balance
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _result(psi: np.ndarray, alpha: float) -> dict:
    est, se = float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"estimate": est, "se": se, "ci_low": est - z * se, "ci_high": est + z * se}


def ipw_ate(y, t, propensity, alpha: float = 0.05) -> dict:
    """Horvitz-Thompson IPW estimate of the ATE with an influence-function standard error."""
    y, t, e = (np.asarray(a, float) for a in (y, t, propensity))
    return _result(t * y / e - (1 - t) * y / (1 - e), alpha)


def aipw_ate(y, t, propensity, mu1, mu0, alpha: float = 0.05) -> dict:
    """Augmented IPW (doubly robust). ``mu1``/``mu0`` are outcome-model predictions that must come from a model
    that did not see these rows (cross-fitting / a separate training split)."""
    y, t, e, mu1, mu0 = (np.asarray(a, float) for a in (y, t, propensity, mu1, mu0))
    return _result(mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e), alpha)


def weighted_smd(x, t, propensity) -> float:
    """Standardized mean difference after ATE weighting (1/e for treated, 1/(1-e) for control)."""
    x, t, e = (np.asarray(a, float) for a in (x, t, propensity))
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    sd = x.std(ddof=1)
    if sd == 0:
        return 0.0
    return float((np.average(x[t == 1], weights=w[t == 1]) - np.average(x[t == 0], weights=w[t == 0])) / sd)
