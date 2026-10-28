"""Uplift (heterogeneous treatment effect) modeling and evaluation for randomized data.

Learners estimate tau(x) = E[Y | T=1, x] - E[Y | T=0, x]. Because the data come
from a randomized test, scores can be *evaluated* without any modelling
assumptions: rank customers by predicted uplift and compare treated vs control
outcomes inside the top-k slices (Qini / uplift curves, IPW policy value).
Every score used for evaluation is cross-fitted, i.e. produced by a model that
never saw that customer.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------------- learners


def _predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


def s_learner(model, X_tr, y_tr, t_tr, X_te) -> np.ndarray:
    """One model with treatment as a feature; tau = f(x, 1) - f(x, 0)."""
    m = clone(model).fit(np.column_stack([X_tr, t_tr]), y_tr)
    n = len(X_te)
    return _predict(m, np.column_stack([X_te, np.ones(n)])) - _predict(m, np.column_stack([X_te, np.zeros(n)]))


def t_learner(model, X_tr, y_tr, t_tr, X_te) -> np.ndarray:
    """Separate outcome models for treated and control."""
    m1 = clone(model).fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
    m0 = clone(model).fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
    return _predict(m1, X_te) - _predict(m0, X_te)


def x_learner(model, X_tr, y_tr, t_tr, X_te, reg_model=None) -> np.ndarray:
    """Kunzel et al. (2019): impute individual effects, regress them, blend by propensity."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    reg = reg_model or HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=100,
                                                     min_samples_leaf=200)
    X1, y1, X0, y0 = X_tr[t_tr == 1], y_tr[t_tr == 1], X_tr[t_tr == 0], y_tr[t_tr == 0]
    m1 = clone(model).fit(X1, y1)
    m0 = clone(model).fit(X0, y0)
    d1 = y1 - _predict(m0, X1)  # treated: observed minus imputed control outcome
    d0 = _predict(m1, X0) - y0  # control: imputed treated outcome minus observed
    tau1 = clone(reg).fit(X1, d1)
    tau0 = clone(reg).fit(X0, d0)
    g = t_tr.mean()  # constant propensity: the test was randomized
    return g * tau0.predict(X_te) + (1 - g) * tau1.predict(X_te)


LEARNERS: dict[str, Callable] = {"S-learner": s_learner, "T-learner": t_learner, "X-learner": x_learner}


def crossfit_uplift(X, y, t, learner: Callable, model, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold uplift score for every row."""
    X, y, t = np.asarray(X, float), np.asarray(y, float), np.asarray(t, int)
    tau = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, t):
        tau[te] = learner(model, X[tr], y[tr], t[tr], X[te])
    return tau


def crossfit_arm_outcomes(X, y, arm, model, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predicted outcome under every arm (columns follow sorted arm labels)."""
    X, y, arm = np.asarray(X, float), np.asarray(y, float), np.asarray(arm, int)
    arms = np.unique(arm)
    out = np.zeros((len(y), len(arms)))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, arm):
        for j, a in enumerate(arms):
            m = clone(model).fit(X[tr][arm[tr] == a], y[tr][arm[tr] == a])
            out[te, j] = _predict(m, X[te])
    return out

# ----------------------------------------------------------------------------- evaluation


def qini_curve(tau, y, t, n_points: int = 100) -> pd.DataFrame:
    """Cumulative incremental outcome when treating the top fraction ranked by ``tau``.

    qini(k) = k * (mean y of treated in top k - mean y of control in top k)

    i.e. the expected extra outcome from treating those k customers. (The textbook
    form sum_t y - sum_c y * n_t/n_c counts only the treated units, which understates
    the gain when treatment is not 50/50; this version is always in whole-customer units.)
    ``random`` is the straight line to the same end point, i.e. the gain from treating
    a random subset of the same size. Divide by n for per-customer-in-population values.
    """
    tau, y, t = np.asarray(tau), np.asarray(y, float), np.asarray(t, int)
    order = np.argsort(-tau, kind="stable")
    y, t = y[order], t[order]
    ct, cc = np.cumsum(t), np.cumsum(1 - t)
    cyt, cyc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where((ct > 0) & (cc > 0), (cyt / ct - cyc / cc) * (ct + cc), 0.0)
    n = len(y)
    idx = np.unique(np.linspace(0, n - 1, n_points).astype(int))
    frac = (idx + 1) / n
    return pd.DataFrame({"fraction": np.r_[0, frac], "qini": np.r_[0, q[idx]],
                         "random": np.r_[0, frac * q[-1]]})


def qini_coefficient(tau, y, t) -> float:
    """Area between the Qini curve and the random line, per customer (outcome units)."""
    c = qini_curve(tau, y, t, n_points=200)
    gap = c["qini"] - c["random"]
    area = np.sum((gap.values[1:] + gap.values[:-1]) / 2 * np.diff(c["fraction"].values))
    return float(area / len(y))


def uplift_by_group(tau, y, t, n_groups: int = 10) -> pd.DataFrame:
    """Observed treated-vs-control difference inside equal-sized bins of predicted uplift."""
    df = pd.DataFrame({"tau": tau, "y": y, "t": t})
    df["group"] = pd.qcut(df["tau"].rank(method="first", ascending=False), n_groups, labels=range(1, n_groups + 1))
    rows = []
    for g, d in df.groupby("group", observed=True):
        a, b = d[d.t == 1].y, d[d.t == 0].y
        rows.append({"group": int(g), "n": len(d), "mean_pred_uplift": d.tau.mean(),
                     "observed_uplift": a.mean() - b.mean(),
                     "se": np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))})
    return pd.DataFrame(rows)


def ipw_gain(y, t, p_treat, mask=None) -> float:
    """Unbiased estimate of sum_i tau_i / N over ``mask`` using inverse propensity weights."""
    y, t = np.asarray(y, float), np.asarray(t, float)
    contrib = (t / p_treat - (1 - t) / (1 - p_treat)) * y
    if mask is not None:
        contrib = contrib * np.asarray(mask)
    return float(contrib.mean())


def targeting_curve(tau, y, t, p_treat, costs=(0.0,), fractions=None) -> pd.DataFrame:
    """Expected incremental profit per customer in the *whole* population vs targeting depth."""
    tau = np.asarray(tau)
    fractions = np.linspace(0, 1, 41) if fractions is None else fractions
    rank = np.argsort(np.argsort(-tau, kind="stable"), kind="stable") / len(tau)
    rows = []
    for f in fractions:
        gain = ipw_gain(y, t, p_treat, mask=rank < f)
        for c in costs:
            rows.append({"fraction": f, "cost": c, "profit": gain - c * f})
    return pd.DataFrame(rows)


def bootstrap(stat_fn: Callable[[np.ndarray], float], n: int, n_boot: int = 300, seed: int = 0,
              alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of ``stat_fn(row_indices)`` under resampling of rows."""
    rng = np.random.default_rng(seed)
    vals = [stat_fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
