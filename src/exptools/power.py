"""Power analysis for two-sample comparisons of means and proportions.

All functions use the normal approximation. ``ratio`` is n_treatment / n_control
and ``variance_reduction`` (e.g. 0.3 for 30%) scales the outcome variance, which
is how CUPED enters a power calculation.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _z(alpha: float, power: float, two_sided: bool) -> float:
    return norm.ppf(1 - alpha / (2 if two_sided else 1)) + norm.ppf(power)


def _n_control(var_c: float, var_t: float, delta: float, ratio: float, z: float) -> int:
    return math.ceil((var_c + var_t / ratio) * z**2 / delta**2)


def sample_size_proportion(p_base: float, mde: float, alpha: float = 0.05, power: float = 0.8,
                           two_sided: bool = True, ratio: float = 1.0, relative: bool = False,
                           variance_reduction: float = 0.0) -> dict:
    """Per-arm sample size to detect an absolute (or relative) lift in a conversion rate."""
    delta = p_base * mde if relative else mde
    p_new = p_base + delta
    if not (0 < p_base < 1 and 0 < p_new < 1):
        raise ValueError("baseline and treated rates must lie strictly between 0 and 1")
    k = 1 - variance_reduction
    n_c = _n_control(k * p_base * (1 - p_base), k * p_new * (1 - p_new), delta, ratio,
                     _z(alpha, power, two_sided))
    n_t = math.ceil(n_c * ratio)
    return {"n_control": n_c, "n_treatment": n_t, "n_total": n_c + n_t, "delta": delta, "p_treatment": p_new}


def sample_size_mean(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8,
                     two_sided: bool = True, ratio: float = 1.0, mean_base: float | None = None,
                     relative: bool = False, variance_reduction: float = 0.0) -> dict:
    """Per-arm sample size to detect a lift in a continuous metric with std. dev. ``sd``."""
    delta = mean_base * mde if relative else mde
    if relative and mean_base is None:
        raise ValueError("relative=True needs mean_base")
    var = (1 - variance_reduction) * sd**2
    n_c = _n_control(var, var, delta, ratio, _z(alpha, power, two_sided))
    n_t = math.ceil(n_c * ratio)
    return {"n_control": n_c, "n_treatment": n_t, "n_total": n_c + n_t, "delta": delta}


def power_proportion(p_base: float, delta: float, n_control: int, alpha: float = 0.05,
                     two_sided: bool = True, ratio: float = 1.0, variance_reduction: float = 0.0) -> float:
    p_new = p_base + delta
    k = 1 - variance_reduction
    se = math.sqrt(k * (p_base * (1 - p_base) / n_control + p_new * (1 - p_new) / (n_control * ratio)))
    return _power_from_se(delta, se, alpha, two_sided)


def power_mean(sd: float, delta: float, n_control: int, alpha: float = 0.05, two_sided: bool = True,
               ratio: float = 1.0, variance_reduction: float = 0.0) -> float:
    se = math.sqrt((1 - variance_reduction) * sd**2 * (1 / n_control + 1 / (n_control * ratio)))
    return _power_from_se(delta, se, alpha, two_sided)


def _power_from_se(delta: float, se: float, alpha: float, two_sided: bool) -> float:
    za = norm.ppf(1 - alpha / (2 if two_sided else 1))
    d = abs(delta) / se
    return float(norm.cdf(d - za) + (norm.cdf(-d - za) if two_sided else 0.0))


def mde_proportion(p_base: float, n_control: int, alpha: float = 0.05, power: float = 0.8,
                   two_sided: bool = True, ratio: float = 1.0, variance_reduction: float = 0.0) -> float:
    """Smallest absolute lift detectable with the given sample (solved by bisection)."""
    lo, hi = 1e-9, 1 - p_base - 1e-9
    for _ in range(100):
        mid = (lo + hi) / 2
        if power_proportion(p_base, mid, n_control, alpha, two_sided, ratio, variance_reduction) < power:
            lo = mid
        else:
            hi = mid
    return hi


def mde_mean(sd: float, n_control: int, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True,
             ratio: float = 1.0, variance_reduction: float = 0.0) -> float:
    """Closed form: MDE = (z_a + z_b) * se."""
    se = math.sqrt((1 - variance_reduction) * sd**2 * (1 / n_control + 1 / (n_control * ratio)))
    return _z(alpha, power, two_sided) * se
