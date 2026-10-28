import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from exptools import cuped_adjust, cuped_effect, cuped_theta, diff_in_means, srm_test, standardized_mean_diff


def test_diff_in_means_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 2, 500), rng.normal(0.8, 3, 700)
    e = diff_in_means(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert e.p_value == pytest.approx(ref.pvalue)
    assert e.ci_low < e.estimate < e.ci_high


def test_srm_detects_broken_split():
    assert srm_test([5000, 5010])["p_value"] > 0.5
    assert srm_test([5000, 5400])["p_value"] < 0.001


def test_smd_zero_for_identical_groups():
    x = np.arange(100.0)
    assert standardized_mean_diff(x, x) == 0.0


def _simulate(rho, n=20000, effect=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(5, 2, n)
    t = rng.random(n) < 0.5
    y = 3 + rho * (x - 5) / 2 * 4 + np.sqrt(1 - rho**2) * 4 * rng.normal(size=n) + effect * t
    return y, x, t


def test_theta_matches_ols_slope():
    y, x, _ = _simulate(0.6)
    slope = np.polyfit(x, y, 1)[0]
    assert cuped_theta(y, x)[0] == pytest.approx(slope)


def test_adjustment_preserves_mean():
    y, x, _ = _simulate(0.6)
    assert cuped_adjust(y, x).mean() == pytest.approx(y.mean())


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_variance_reduction_is_rho_squared(rho):
    y, x, t = _simulate(rho)
    _, info = cuped_effect(y, x, t)
    assert info["variance_reduction"] == pytest.approx(rho**2, abs=0.03)


def test_cuped_matches_ancova_and_has_smaller_se():
    y, x, t = _simulate(0.7)
    eff, info = cuped_effect(y, x, t)
    ols = sm.OLS(y, sm.add_constant(np.column_stack([t.astype(float), x]))).fit(cov_type="HC1")
    assert eff.estimate == pytest.approx(ols.params[1], abs=0.02)
    assert eff.se == pytest.approx(ols.bse[1], rel=0.05)
    assert eff.se < info["raw"].se


def test_multiple_covariates():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30000, 3))
    t = rng.random(30000) < 0.5
    y = X @ [1.0, 2.0, -1.0] + rng.normal(size=30000) + 0.3 * t
    eff, info = cuped_effect(y, X, t)
    assert info["variance_reduction"] > 0.8
    assert eff.estimate == pytest.approx(0.3, abs=0.03)


def test_unbiased_across_replications():
    est = [cuped_effect(*_simulate(0.8, n=4000, seed=s))[0].estimate for s in range(200)]
    assert np.mean(est) == pytest.approx(0.5, abs=0.02)
