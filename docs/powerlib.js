/* Power / sample-size maths. Mirrors src/exptools/power.py (normal approximation).
   Plain functions, no dependencies; usable in the browser (window.PowerLib) and in Node. */
(function (root) {
  'use strict';

  // Standard normal CDF. Series for |x| <= 3, continued fraction for the tails (~1e-15 accuracy).
  function upperTail(x) { // P(Z > x) for x > 3, continued fraction
    var f = 0;
    for (var k = 80; k >= 1; k--) f = k / (x + f);
    return Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) / (x + f);
  }

  function normCdf(x) {
    if (x < -3) return upperTail(-x);  // computed directly so the lower tail keeps full precision
    if (x > 3) return 1 - upperTail(x);
    if (x < 0) return 1 - normCdf(-x);  // |x| <= 3 here, so no meaningful precision loss
    {
      // Phi(x) = 1/2 + phi(x) * sum_n x^(2n+1) / (1*3*5*...*(2n+1)); all terms positive, no cancellation
      var term = x, sum = x, n = 0, x2 = x * x;
      while (term > 1e-18 * sum && n < 300) {
        n += 1;
        term *= x2 / (2 * n + 1);
        sum += term;
      }
      return 0.5 + sum * Math.exp(-x2 / 2) / Math.sqrt(2 * Math.PI);
    }
  }

  // Inverse CDF: Acklam's rational approximation, polished with Halley steps.
  function normInv(p) {
    if (p <= 0 || p >= 1) throw new RangeError('p must be in (0,1)');
    var a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    var b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    var c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    var d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    var pl = 0.02425, x, q, r;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else if (p <= 1 - pl) {
      q = p - 0.5; r = q * q;
      x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    for (var i = 0; i < 2; i++) {
      var e = normCdf(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
      x = x - u / (1 + x * u / 2);
    }
    return x;
  }

  function zSum(alpha, power, twoSided) {
    return normInv(1 - alpha / (twoSided ? 2 : 1)) + normInv(power);
  }

  function opts(o) {
    o = o || {};
    return {
      alpha: o.alpha == null ? 0.05 : o.alpha,
      power: o.power == null ? 0.8 : o.power,
      twoSided: o.twoSided !== false,
      ratio: o.ratio == null ? 1 : o.ratio,         // n_treatment / n_control
      vr: o.varianceReduction || 0,                  // CUPED: share of variance removed
      relative: !!o.relative
    };
  }

  function result(nc, ratio, extra) {
    var nt = Math.ceil(nc * ratio);
    var r = { nControl: nc, nTreatment: nt, nTotal: nc + nt };
    for (var k in extra) r[k] = extra[k];
    return r;
  }

  function sampleSizeProportion(pBase, mde, o) {
    o = opts(o);
    var delta = o.relative ? pBase * mde : mde, pNew = pBase + delta;
    if (!(pBase > 0 && pBase < 1 && pNew > 0 && pNew < 1)) throw new RangeError('rates must lie strictly between 0 and 1');
    var k = 1 - o.vr, z = zSum(o.alpha, o.power, o.twoSided);
    var nc = Math.ceil((k * pBase * (1 - pBase) + k * pNew * (1 - pNew) / o.ratio) * z * z / (delta * delta));
    return result(nc, o.ratio, { delta: delta, pTreatment: pNew });
  }

  function sampleSizeMean(sd, mde, o, meanBase) {
    o = opts(o);
    var delta = o.relative ? meanBase * mde : mde, v = (1 - o.vr) * sd * sd, z = zSum(o.alpha, o.power, o.twoSided);
    var nc = Math.ceil((v + v / o.ratio) * z * z / (delta * delta));
    return result(nc, o.ratio, { delta: delta });
  }

  function powerFromSe(delta, se, alpha, twoSided) {
    var za = normInv(1 - alpha / (twoSided ? 2 : 1)), d = Math.abs(delta) / se;
    return normCdf(d - za) + (twoSided ? normCdf(-d - za) : 0);
  }

  function powerProportion(pBase, delta, nControl, o) {
    o = opts(o);
    var pNew = pBase + delta, k = 1 - o.vr;
    var se = Math.sqrt(k * (pBase * (1 - pBase) / nControl + pNew * (1 - pNew) / (nControl * o.ratio)));
    return powerFromSe(delta, se, o.alpha, o.twoSided);
  }

  function powerMean(sd, delta, nControl, o) {
    o = opts(o);
    var se = Math.sqrt((1 - o.vr) * sd * sd * (1 / nControl + 1 / (nControl * o.ratio)));
    return powerFromSe(delta, se, o.alpha, o.twoSided);
  }

  function mdeProportion(pBase, nControl, raw) {
    var o = opts(raw);
    var lo = 1e-9, hi = 1 - pBase - 1e-9;
    for (var i = 0; i < 100; i++) {
      var mid = (lo + hi) / 2;
      if (powerProportion(pBase, mid, nControl, raw) < o.power) lo = mid; else hi = mid;
    }
    return hi;
  }

  function mdeMean(sd, nControl, o) {
    o = opts(o);
    var se = Math.sqrt((1 - o.vr) * sd * sd * (1 / nControl + 1 / (nControl * o.ratio)));
    return zSum(o.alpha, o.power, o.twoSided) * se;
  }

  var api = { normCdf: normCdf, normInv: normInv, sampleSizeProportion: sampleSizeProportion, sampleSizeMean: sampleSizeMean,
              powerProportion: powerProportion, powerMean: powerMean, mdeProportion: mdeProportion, mdeMean: mdeMean };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.PowerLib = api;
})(typeof window !== 'undefined' ? window : this);
