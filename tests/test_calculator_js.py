"""The browser calculator (docs/powerlib.js) must agree with the Python reference implementation."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from scipy.stats import norm

from exptools import (mde_mean, mde_proportion, power_mean, power_proportion, sample_size_mean,
                      sample_size_proportion)

LIB = Path(__file__).resolve().parents[1] / "docs" / "powerlib.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def js(expr: str):
    out = subprocess.run(["node", "-e", f"const L=require({json.dumps(str(LIB))});console.log(JSON.stringify({expr}))"],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("x", [-6, -3.2, -1, 0, 0.5, 1.96, 2.99, 3.01, 5])
def test_norm_cdf(x):
    assert js(f"L.normCdf({x})") == pytest.approx(norm.cdf(x), rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("p", [1e-9, 0.001, 0.025, 0.2, 0.5, 0.8, 0.975, 0.999999])
def test_norm_inv(p):
    assert js(f"L.normInv({p})") == pytest.approx(norm.ppf(p), abs=1e-9)


CASES = [
    dict(p=0.10, mde=0.02, alpha=0.05, power=0.8, two=True, ratio=1.0, vr=0.0, rel=False),
    dict(p=0.0057, mde=0.2, alpha=0.05, power=0.9, two=True, ratio=1.0, vr=0.0, rel=True),
    dict(p=0.3, mde=0.03, alpha=0.01, power=0.8, two=False, ratio=2.0, vr=0.25, rel=False),
]


@pytest.mark.parametrize("c", CASES)
def test_proportion_functions(c):
    kw = dict(alpha=c["alpha"], power=c["power"], two_sided=c["two"], ratio=c["ratio"], variance_reduction=c["vr"])
    ref = sample_size_proportion(c["p"], c["mde"], relative=c["rel"], **kw)
    o = f"{{alpha:{c['alpha']},power:{c['power']},twoSided:{str(c['two']).lower()},ratio:{c['ratio']},varianceReduction:{c['vr']},relative:{str(c['rel']).lower()}}}"
    got = js(f"L.sampleSizeProportion({c['p']},{c['mde']},{o})")
    assert got["nControl"] == ref["n_control"] and got["nTreatment"] == ref["n_treatment"]
    o2 = o.replace("relative:" + str(c["rel"]).lower(), "relative:false")
    n = ref["n_control"]
    kwp = {k: v for k, v in kw.items() if k != "power"}
    assert js(f"L.powerProportion({c['p']},{ref['delta']},{n},{o2})") == pytest.approx(
        power_proportion(c["p"], ref["delta"], n, **kwp), abs=1e-9)
    assert js(f"L.mdeProportion({c['p']},{n},{o2})") == pytest.approx(mde_proportion(c["p"], n, **kw), rel=1e-6)


def test_mean_functions():
    kw = dict(alpha=0.05, power=0.8, two_sided=True, ratio=1.5, variance_reduction=0.3)
    o = "{alpha:0.05,power:0.8,twoSided:true,ratio:1.5,varianceReduction:0.3}"
    ref = sample_size_mean(11.6, 0.15, **kw)
    got = js(f"L.sampleSizeMean(11.6,0.15,{o})")
    assert got["nControl"] == ref["n_control"] and got["nTreatment"] == ref["n_treatment"]
    assert js(f"L.powerMean(11.6,0.15,{ref['n_control']},{o})") == pytest.approx(power_mean(11.6, 0.15, ref["n_control"], **{k: v for k, v in kw.items() if k != "power"}), abs=1e-9)
    assert js(f"L.mdeMean(11.6,5000,{o})") == pytest.approx(mde_mean(11.6, 5000, **kw), rel=1e-9)
