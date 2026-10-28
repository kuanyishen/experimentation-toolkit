import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from exptools.uplift import (LEARNERS, crossfit_uplift, ipw_gain, qini_coefficient, qini_curve,
                             targeting_curve, uplift_by_group)


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    n = 15000
    X = rng.normal(size=(n, 3))
    t = (rng.random(n) < 0.5).astype(int)
    tau = 1.0 * (X[:, 0] > 0) - 0.5            # heterogeneous: helps half, hurts half
    y = X[:, 1] + tau * t + rng.normal(size=n)
    return X, y, t, tau


def test_qini_ranks_oracle_above_random(synthetic):
    X, y, t, tau = synthetic
    oracle = qini_coefficient(tau, y, t)
    rand = qini_coefficient(np.random.default_rng(1).normal(size=len(y)), y, t)
    assert oracle > 0.1 and abs(rand) < 0.03


def test_curve_endpoints(synthetic):
    _, y, t, _ = synthetic
    c = qini_curve(np.arange(len(y)), y, t)
    assert c.fraction.iloc[0] == 0 and c.fraction.iloc[-1] == 1
    assert c.qini.iloc[-1] == pytest.approx(c.random.iloc[-1])


@pytest.mark.parametrize("name", list(LEARNERS))
def test_learners_recover_heterogeneity(synthetic, name):
    X, y, t, tau = synthetic
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=100, random_state=0)
    scores = crossfit_uplift(X, y, t, LEARNERS[name], model, seed=0)
    assert np.corrcoef(scores, tau)[0, 1] > 0.6


def test_ipw_gain_is_unbiased_for_ate(synthetic):
    _, y, t, tau = synthetic
    assert ipw_gain(y, t, 0.5) == pytest.approx(tau.mean(), abs=0.05)


def test_targeting_curve_peaks_at_positive_uplift_share(synthetic):
    _, y, t, tau = synthetic
    c = targeting_curve(tau, y, t, 0.5, fractions=np.linspace(0, 1, 21))
    best = c.loc[c.profit.idxmax(), "fraction"]
    assert 0.4 <= best <= 0.6  # tau>0 for ~half the customers


def test_decile_table_is_monotone_for_oracle(synthetic):
    _, y, t, tau = synthetic
    d = uplift_by_group(tau + np.random.default_rng(2).normal(0, 1e-6, len(tau)), y, t, 4)
    assert d.observed_uplift.iloc[0] > d.observed_uplift.iloc[-1] + 0.5


def test_propensity_weighted_qini_is_unbiased_under_confounded_assignment():
    """Assignment depends on x0, which also drives the outcome: plain Qini is biased, the IPW one is not."""
    rng = np.random.default_rng(5)
    n = 200000
    x0 = rng.normal(size=n)
    e = 1 / (1 + np.exp(-1.5 * x0))                      # strong selection on x0
    t = (rng.random(n) < e).astype(int)
    tau = np.full(n, 0.3)                                # constant true effect
    y = 2.0 * x0 + tau * t + rng.normal(size=n)
    score = np.random.default_rng(6).normal(size=n)      # uninformative ranking
    end_plain = qini_curve(score, y, t).qini.iloc[-1] / n
    end_ipw = qini_curve(score, y, t, propensity=e).qini.iloc[-1] / n
    assert abs(end_plain - 0.3) > 0.5                    # naive difference in means is badly confounded
    assert end_ipw == pytest.approx(0.3, abs=0.06)
    g = uplift_by_group(score, y, t, 4, propensity=e)
    assert g.observed_uplift.mean() == pytest.approx(0.3, abs=0.06)
    assert ipw_gain(y, t, e) == pytest.approx(0.3, abs=0.06)  # ipw_gain already accepts per-row propensities
