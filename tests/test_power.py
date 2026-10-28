import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower

from exptools import (mde_mean, mde_proportion, power_mean, power_proportion, sample_size_mean,
                      sample_size_proportion)


def test_mean_matches_statsmodels():
    sd, delta = 10.0, 1.5
    ours = sample_size_mean(sd, delta)["n_control"]
    ref = NormalIndPower().solve_power(effect_size=delta / sd, alpha=0.05, power=0.8, ratio=1.0)
    assert abs(ours - ref) <= 1


def test_mean_unequal_allocation_matches_statsmodels():
    ours = sample_size_mean(5.0, 1.0, ratio=2.0)["n_control"]
    ref = NormalIndPower().solve_power(effect_size=0.2, alpha=0.05, power=0.8, ratio=2.0)
    assert abs(ours - ref) <= 1


def test_proportion_textbook_value():
    # 10% -> 12%: ~3,800 per arm with unpooled variance
    n = sample_size_proportion(0.10, 0.02)["n_control"]
    assert 3700 < n < 3900


def test_roundtrip_is_tight():
    for p, d in [(0.05, 0.01), (0.3, 0.02), (0.009, 0.003)]:
        n = sample_size_proportion(p, d)["n_control"]
        assert power_proportion(p, d, n) >= 0.8 - 1e-9
        assert power_proportion(p, d, n - 1) < 0.8


def test_mde_inverts_sample_size():
    n = 20000
    d = mde_proportion(0.1, n)
    assert abs(power_proportion(0.1, d, n) - 0.8) < 1e-6
    d2 = mde_mean(4.0, n)
    assert abs(power_mean(4.0, d2, n) - 0.8) < 1e-3  # closed form ignores the far tail


def test_variance_reduction_cuts_sample_size_proportionally():
    base = sample_size_mean(3.0, 0.2)["n_control"]
    reduced = sample_size_mean(3.0, 0.2, variance_reduction=0.4)["n_control"]
    assert reduced == pytest.approx(0.6 * base, rel=0.01)


def test_one_sided_needs_fewer_samples():
    assert sample_size_mean(1, 0.1, two_sided=False)["n_control"] < sample_size_mean(1, 0.1)["n_control"]


def test_relative_mde_and_validation():
    assert sample_size_proportion(0.1, 0.2, relative=True)["delta"] == pytest.approx(0.02)
    with pytest.raises(ValueError):
        sample_size_proportion(0.95, 0.1)


def test_monte_carlo_power_matches_formula():
    rng = np.random.default_rng(1)
    p, delta, n = 0.2, 0.03, 2000
    rej = 0
    reps = 1500
    for _ in range(reps):
        a, b = rng.binomial(n, p), rng.binomial(n, p + delta)
        pa, pb = a / n, b / n
        se = np.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
        rej += abs(pb - pa) / se > 1.96
    assert rej / reps == pytest.approx(power_proportion(p, delta, n), abs=0.04)
