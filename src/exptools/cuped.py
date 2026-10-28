"""CUPED (Controlled-experiment Using Pre-Experiment Data), Deng et al. 2013.

Y_adj = Y - theta' (X - mean(X)),  theta = Cov(X)^-1 Cov(X, Y)

X must be measured *before* randomization, otherwise the adjustment is biased.
The adjusted metric has the same expected treatment effect as Y and variance
Var(Y) * (1 - R^2), where R^2 is the share of Y's variance explained by X.
"""

from __future__ import annotations

import numpy as np

from .ab import Effect, diff_in_means


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def cuped_theta(y, x) -> np.ndarray:
    """Optimal coefficients, estimated on pooled data from all arms."""
    y, x = np.asarray(y, float), _as_2d(x)
    xc = x - x.mean(axis=0)
    cov_xx = xc.T @ xc / (len(y) - 1)
    cov_xy = xc.T @ (y - y.mean()) / (len(y) - 1)
    return np.linalg.solve(np.atleast_2d(cov_xx), cov_xy)


def cuped_adjust(y, x, theta=None) -> np.ndarray:
    """Return the variance-reduced metric."""
    y, x = np.asarray(y, float), _as_2d(x)
    theta = cuped_theta(y, x) if theta is None else np.atleast_1d(theta)
    return y - (x - x.mean(axis=0)) @ theta


def cuped_effect(y, x, treated, alpha: float = 0.05) -> tuple[Effect, dict]:
    """CUPED-adjusted ATE plus diagnostics.

    ``treated`` is a boolean mask; units not in {treated, control} should be
    filtered out beforehand. Theta is fit on pooled data and then treated as
    fixed, which is the standard approach and negligible-error at large n.
    """
    y, x, treated = np.asarray(y, float), _as_2d(x), np.asarray(treated, bool)
    y_adj = cuped_adjust(y, x)
    eff = diff_in_means(y_adj[treated], y_adj[~treated], alpha)
    raw = diff_in_means(y[treated], y[~treated], alpha)
    info = {
        "theta": cuped_theta(y, x).tolist(),
        "variance_reduction": float(1 - y_adj.var(ddof=1) / y.var(ddof=1)),
        "se_ratio": float(eff.se / raw.se),
        "raw": raw,
    }
    return eff, info
