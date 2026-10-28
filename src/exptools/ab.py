"""Basic randomized-experiment analysis and validity checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Effect:
    """Difference in means (treatment - control) with a Welch confidence interval."""

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    mean_treat: float
    mean_ctrl: float
    n_treat: int
    n_ctrl: int

    @property
    def relative_lift(self) -> float:
        return self.estimate / self.mean_ctrl if self.mean_ctrl else float("nan")

    def as_dict(self) -> dict:
        return {**self.__dict__, "relative_lift": self.relative_lift}


def diff_in_means(y_treat, y_ctrl, alpha: float = 0.05) -> Effect:
    """Welch two-sample estimate of the average treatment effect.

    Works for binary outcomes too (then it is a Welch-style test of proportions,
    which is indistinguishable from the z-test at experiment-sized samples).
    """
    y_treat = np.asarray(y_treat, dtype=float)
    y_ctrl = np.asarray(y_ctrl, dtype=float)
    nt, nc = len(y_treat), len(y_ctrl)
    diff = y_treat.mean() - y_ctrl.mean()
    vt, vc = y_treat.var(ddof=1) / nt, y_ctrl.var(ddof=1) / nc
    se = np.sqrt(vt + vc)
    # Welch-Satterthwaite degrees of freedom
    dof = (vt + vc) ** 2 / (vt**2 / (nt - 1) + vc**2 / (nc - 1))
    crit = stats.t.ppf(1 - alpha / 2, dof)
    p = 2 * stats.t.sf(abs(diff / se), dof)
    return Effect(diff, se, diff - crit * se, diff + crit * se, p,
                  y_treat.mean(), y_ctrl.mean(), nt, nc)


def srm_test(counts, expected_share=None) -> dict:
    """Sample-ratio-mismatch check: chi-square of arm counts vs the intended split.

    A tiny p-value (conventionally < 0.001) means assignment is broken and the
    experiment's results should not be trusted.
    """
    counts = np.asarray(counts, dtype=float)
    share = np.full(len(counts), 1 / len(counts)) if expected_share is None else np.asarray(expected_share, float)
    chi2, p = stats.chisquare(counts, counts.sum() * share)
    return {"chi2": float(chi2), "p_value": float(p), "counts": counts.astype(int).tolist()}


def standardized_mean_diff(x_treat, x_ctrl) -> float:
    """Covariate balance metric; |SMD| < 0.1 is the usual 'balanced' rule of thumb."""
    x_treat, x_ctrl = np.asarray(x_treat, float), np.asarray(x_ctrl, float)
    pooled = np.sqrt((x_treat.var(ddof=1) + x_ctrl.var(ddof=1)) / 2)
    return float((x_treat.mean() - x_ctrl.mean()) / pooled) if pooled > 0 else 0.0
