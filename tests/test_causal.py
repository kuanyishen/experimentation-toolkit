import numpy as np
import pytest

from exptools import aipw_ate, diff_in_means, ipw_ate, standardized_mean_diff, weighted_smd


@pytest.fixture(scope="module")
def confounded():
    rng = np.random.default_rng(0)
    n = 200000
    x = rng.normal(size=n)
    e = 1 / (1 + np.exp(-1.0 * x))
    t = (rng.random(n) < e).astype(float)
    y = 3.0 * x + 0.5 * t + rng.normal(size=n)  # true ATE = 0.5
    return x, t, e, y


def test_naive_is_biased_but_adjusted_estimators_recover_effect(confounded):
    x, t, e, y = confounded
    naive = diff_in_means(y[t == 1], y[t == 0])
    assert naive.estimate > 1.5                       # confounding inflates the naive estimate
    assert ipw_ate(y, t, e)["estimate"] == pytest.approx(0.5, abs=0.1)
    mu1, mu0 = 3.0 * x + 0.5, 3.0 * x                 # oracle outcome model
    a = aipw_ate(y, t, e, mu1, mu0)
    assert a["estimate"] == pytest.approx(0.5, abs=0.02)
    assert a["se"] < ipw_ate(y, t, e)["se"]           # outcome model reduces variance


def test_aipw_is_doubly_robust(confounded):
    x, t, e, y = confounded
    bad_outcome = np.zeros_like(y)                    # useless outcome model, true propensity
    assert aipw_ate(y, t, e, bad_outcome, bad_outcome)["estimate"] == pytest.approx(0.5, abs=0.1)
    good_mu1, good_mu0 = 3.0 * x + 0.5, 3.0 * x       # good outcome model, wrong (constant) propensity
    assert aipw_ate(y, t, np.full_like(e, t.mean()), good_mu1, good_mu0)["estimate"] == pytest.approx(0.5, abs=0.02)


def test_weighting_restores_balance(confounded):
    x, t, e, _ = confounded
    assert abs(standardized_mean_diff(x[t == 1], x[t == 0])) > 0.5
    assert abs(weighted_smd(x, t, e)) < 0.02


def test_randomized_case_matches_difference_in_means():
    rng = np.random.default_rng(1)
    t = (rng.random(50000) < 0.5).astype(float)
    y = rng.normal(size=50000) + 0.2 * t
    e = np.full(50000, 0.5)
    assert ipw_ate(y, t, e)["estimate"] == pytest.approx(diff_in_means(y[t == 1], y[t == 0]).estimate, abs=0.03)
