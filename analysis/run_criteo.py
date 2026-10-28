"""Criteo uplift v2.1 (13.98M users): the same toolkit at scale, plus an assignment-validity audit.

Usage:  python analysis/run_criteo.py [--sample N]    (--sample = quick debug run on N random rows)
Writes docs/assets/criteo/results.json and figures.

Data is split into three disjoint parts so every nuisance model is evaluated out-of-sample:
  A (11%)  propensity model + CUPAC covariate model
  B (39%)  outcome / uplift models
  C (50%)  holdout where all estimates and evaluations are computed
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exptools import (aipw_ate, cuped_effect, diff_in_means, ipw_ate, mde_proportion, power_proportion,
                      sample_size_proportion, srm_test, standardized_mean_diff, weighted_smd)
from exptools.data import CRITEO_FEATURES, load_criteo
from exptools.uplift import (LEARNERS, ipw_gain, ipw_weights, qini_coefficient, qini_curve, targeting_curve,
                             uplift_by_group)

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "criteo"
SEED = 11
P_NOMINAL = 0.85  # overall treated share
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def clf(seed: int = SEED):
    return HGBC(max_depth=4, learning_rate=0.08, max_iter=150, min_samples_leaf=500, l2_regularization=1.0, random_state=seed)


def fit_t(XB, yB, tB, XC):
    """Outcome models per arm (trained on B) -> predictions under treatment and control for holdout rows."""
    m1 = clf().fit(XB[tB == 1], yB[tB == 1])
    m0 = clf().fit(XB[tB == 0], yB[tB == 0])
    return m1.predict_proba(XC)[:, 1], m0.predict_proba(XC)[:, 1]


def chunk_stat(fn, n: int, n_chunks: int = 20, seed: int = SEED) -> tuple[float, list[float]]:
    """Value on all rows + t-interval across random disjoint chunks."""
    idx = np.random.default_rng(seed).permutation(n)
    vals = np.array([fn(c) for c in np.array_split(idx, n_chunks)])
    half = stats.t.ppf(0.975, n_chunks - 1) * vals.std(ddof=1) / np.sqrt(n_chunks)
    v = float(fn(np.arange(n)))
    return v, [v - half, v + half]


def main(sample: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_criteo()
    if sample:
        df = df.sample(sample, random_state=SEED).reset_index(drop=True)
    X = df[CRITEO_FEATURES].to_numpy(np.float32)
    t = df["treatment"].to_numpy(np.int8)
    Y = {m: df[m].to_numpy(np.int8) for m in ("visit", "conversion")}
    expo = df["exposure"].to_numpy(np.int8)
    n = len(df)
    R: dict = {"n": n, "sample": sample, "treated_share": float(t.mean()), "n_treated": int(t.sum()), "n_control": int((1 - t).sum())}
    log(f"loaded {n:,} rows")

    perm = np.random.default_rng(SEED).permutation(n)
    nA, nB = int(0.11 * n), int(0.39 * n)
    A, B, C = perm[:nA], perm[nA:nA + nB], perm[nA + nB:]
    R["splits"] = {"A": len(A), "B": len(B), "C": len(C)}
    XC, tC = X[C], t[C].astype(float)
    YC = {m: Y[m][C].astype(float) for m in Y}

    # ------------------------------------------------------------------ 1. is assignment really random?
    R["srm"] = srm_test([R["n_control"], R["n_treated"]], [1 - P_NOMINAL, P_NOMINAL])
    prop_model = clf().fit(X[A], t[A])
    e = np.clip(prop_model.predict_proba(XC)[:, 1], 0.01, 0.99)
    auc, auc_ci = chunk_stat(lambda c: roc_auc_score(tC[c], e[c]), len(C))
    R["propensity"] = {"auc": auc, "auc_ci": auc_ci, "quantiles": dict(zip(["min", "p1", "p10", "p50", "p90", "p99", "max"],
                       np.quantile(e, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist())),
                       "effective_sample_size_ratio": float((1 / e[tC == 1]).sum() ** 2 / (1 / e[tC == 1] ** 2).sum() / (tC == 1).sum())}
    R["balance"] = [{"feature": f, "smd_raw": standardized_mean_diff(X[t == 1, i].astype(float), X[t == 0, i].astype(float)),
                     "smd_weighted": weighted_smd(XC[:, i].astype(float), tC, e)} for i, f in enumerate(CRITEO_FEATURES)]
    R["max_abs_smd_raw"] = max(abs(b["smd_raw"]) for b in R["balance"])
    R["max_abs_smd_weighted"] = max(abs(b["smd_weighted"]) for b in R["balance"])
    log(f"propensity AUC {auc:.4f}, max raw SMD {R['max_abs_smd_raw']:.3f}, weighted {R['max_abs_smd_weighted']:.4f}")

    # ------------------------------------------------------------------ 2. effect estimates: naive vs adjusted
    mu = {m: fit_t(X[B], Y[m][B], t[B], XC) for m in Y}
    log("outcome models fitted")
    est = []
    for m in Y:
        y = YC[m]
        naive = diff_in_means(y[tC == 1], y[tC == 0])
        est.append({"metric": m, "method": "Naive difference in means", "estimate": naive.estimate, "se": naive.se})
        eff, info = cuped_effect(y, XC.astype(float), tC.astype(bool))
        est.append({"metric": m, "method": "CUPED, linear in the 12 features", "estimate": eff.estimate, "se": eff.se,
                    "variance_reduction": info["variance_reduction"]})
        cm = clf().fit(X[A], Y[m][A])
        eff, info = cuped_effect(y, cm.predict_proba(XC)[:, 1], tC.astype(bool))
        est.append({"metric": m, "method": "CUPAC (ML covariate)", "estimate": eff.estimate, "se": eff.se,
                    "variance_reduction": info["variance_reduction"]})
        ip = ipw_ate(y, tC, e)
        est.append({"metric": m, "method": "IPW (estimated propensity)", "estimate": ip["estimate"], "se": ip["se"]})
        ai = aipw_ate(y, tC, e, *mu[m])
        est.append({"metric": m, "method": "AIPW (doubly robust)", "estimate": ai["estimate"], "se": ai["se"],
                    "variance_reduction": float(max(0.0, 1 - (ai["se"] / ip["se"]) ** 2))})
        R.setdefault("naive", {})[m] = naive.as_dict()
    R["estimates"] = est
    get = lambda m, meth: next(x for x in est if x["metric"] == m and x["method"].startswith(meth))

    # effect on the exposed (one-sided non-compliance: only treated users can see the ad)
    pi_naive = float(expo[C][tC == 1].mean())
    ex = ipw_ate(expo[C].astype(float), tC, e)
    R["exposure"] = {"rate_treated_naive": pi_naive, "rate_adjusted": ex["estimate"], "rate_control": float(expo[C][tC == 0].mean())}
    R["cace"] = {}
    for m in Y:
        a = get(m, "AIPW")
        z = 1.96 * a["se"]
        R["cace"][m] = {"estimate": a["estimate"] / ex["estimate"], "ci": [(a["estimate"] - z) / ex["estimate"], (a["estimate"] + z) / ex["estimate"]]}
    log("effects done")

    ratio = P_NOMINAL / (1 - P_NOMINAL)
    R["sample_size_for_adjusted_effect"] = []
    for m in Y:
        base, delta, vr = R["naive"][m]["mean_ctrl"], get(m, "AIPW")["estimate"], get(m, "AIPW")["variance_reduction"]
        R["sample_size_for_adjusted_effect"].append({
            "metric": m, "delta": delta, "base": base, "vr": vr,
            "total_raw": sample_size_proportion(base, delta, ratio=ratio)["n_total"],
            "total_adjusted": sample_size_proportion(base, delta, ratio=ratio, variance_reduction=vr)["n_total"]})

    # ------------------------------------------------------------------ 3. power: allocation and a resampling check
    R["allocation"] = []
    for m in Y:
        base = R["naive"][m]["mean_ctrl"]
        for total in (100_000, 1_000_000):
            for share_c in (0.5, 0.25, 0.15):
                nc = int(total * share_c)
                mde = mde_proportion(base, nc, ratio=(1 - share_c) / share_c)
                R["allocation"].append({"metric": m, "total": total, "control_share": share_c, "mde_abs": mde, "mde_rel": mde / base})
    rng = np.random.default_rng(SEED)
    grids = {"visit": [2_000, 5_000, 10_000, 20_000, 40_000], "conversion": [25_000, 50_000, 100_000, 200_000, 400_000]}
    reps = 400 if not sample else 60
    R["power_check"] = []
    for m in Y:
        y, nv = YC[m], R["naive"][m]
        for N in grids[m]:
            rej = 0
            for _ in range(reps):
                idx = rng.integers(0, len(C), N)
                tt = tC[idx] == 1
                v = y[idx]
                a, b = v[tt], v[~tt]
                rej += abs((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))) > 1.96
            R["power_check"].append({"metric": m, "n_total": N, "empirical": rej / reps,
                                     "theory": power_proportion(nv["mean_ctrl"], nv["estimate"], int(N * (1 - R["treated_share"])), ratio=ratio)})
    log("power check done")

    # ------------------------------------------------------------------ 4. uplift models (train B -> evaluate C with propensity weights)
    XB, tB = X[B], t[B]
    up: dict = {"n_train": len(B), "n_holdout": len(C), "outcomes": {}}
    scores: dict[str, dict[str, np.ndarray]] = {m: {} for m in Y}
    logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))

    def evaluate(tau, m):
        y = YC[m]
        q, ci = chunk_stat(lambda c: qini_coefficient(tau[c], y[c], tC[c], propensity=e[c]), len(C))
        top = np.argsort(np.argsort(-tau, kind="stable"), kind="stable") / len(tau) < 0.2
        return {"qini": q, "qini_ci": ci, "gain_top20": ipw_gain(y, tC, e, top), "gain_random20": 0.2 * ipw_gain(y, tC, e)}

    for m in ("visit", "conversion"):
        yB = Y[m][B]
        scores[m]["T-learner"] = mu[m][0] - mu[m][1]
        scores[m]["S-learner"] = LEARNERS["S-learner"](clf(), XB, yB, tB, XC)
        scores[m]["X-learner"] = LEARNERS["X-learner"](clf(), XB, yB, tB, XC, propensity=e)
        scores[m]["T-learner (logistic)"] = LEARNERS["T-learner"](logit(), XB, yB, tB, XC)
        up["outcomes"][m] = {"ate_ipw": ipw_gain(YC[m], tC, e), "learners": {}}
        for name, tau in scores[m].items():
            up["outcomes"][m]["learners"][name] = evaluate(tau, m)
            log(f"uplift {m} {name}: qini {up['outcomes'][m]['learners'][name]['qini']:.5f}")
    sur = "X-learner on visits (surrogate)"
    scores["conversion"][sur] = scores["visit"]["X-learner"]
    up["outcomes"]["conversion"]["learners"][sur] = evaluate(scores["visit"]["X-learner"], "conversion")
    for m in Y:
        o = up["outcomes"][m]
        o["primary"] = "X-learner"  # fixed in advance (built for unbalanced assignment)
        o["qini_curves"] = {k: qini_curve(s, YC[m], tC, 60, propensity=e).assign(n_total=len(C)).to_dict("list") for k, s in scores[m].items()}
        o["deciles"] = uplift_by_group(scores[m]["X-learner"], YC[m], tC, propensity=e).to_dict("records")

    costs = (0.0, 0.0003, 0.0006)  # cost per targeted user, in units of one conversion (ad cost / value per conversion)
    tc = targeting_curve(scores["conversion"]["X-learner"], YC["conversion"], tC, e, costs=costs)
    up["targeting"] = tc.to_dict("records")
    up["targeting_best"] = {str(c): {"best_fraction": float(s.loc[s.profit.idxmax(), "fraction"]), "best_profit": float(s.profit.max()),
                                     "all_profit": float(s[s.fraction == 1].profit.iloc[0])} for c, s in tc.groupby("cost")}

    # model-free heterogeneity screen: IPW-weighted conversion lift by quintile of each feature
    w = ipw_weights(tC, e) * YC["conversion"]
    het = []
    for i, f in enumerate(CRITEO_FEATURES):
        col = XC[:, i]
        edges = np.unique(np.quantile(col, np.linspace(0, 1, 6)))
        bins = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, len(edges) - 2)
        rows = [{"bin": b + 1, "n": int((bins == b).sum()), "uplift": float(w[bins == b].mean()), "se": float(w[bins == b].std(ddof=1) / np.sqrt((bins == b).sum()))}
                for b in range(len(edges) - 1) if (bins == b).sum() > 1000]
        if len(rows) >= 3:
            het.append({"feature": f, "spread": max(r["uplift"] for r in rows) - min(r["uplift"] for r in rows), "bins": rows})
    up["feature_heterogeneity"] = sorted(het, key=lambda h: -h["spread"])

    # learning curve: how much training data does the T-learner need?
    up["learning_curve"] = []
    for ntr in [s for s in (50_000, 200_000, 800_000, 3_000_000, len(B)) if s <= len(B)]:
        tau = LEARNERS["T-learner"](clf(), XB[:ntr], Y["conversion"][B][:ntr], tB[:ntr], XC)
        q, ci = chunk_stat(lambda c: qini_coefficient(tau[c], YC["conversion"][c], tC[c], propensity=e[c]), len(C))
        up["learning_curve"].append({"n_train": ntr, "qini": q, "qini_ci": ci})
        log(f"learning curve {ntr:,}: {q:.5f}")
    R["uplift"] = up

    R["runtime_seconds"] = time.time() - T0
    (OUT / f"results{'_debug' if sample else ''}.json").write_text(json.dumps(R, indent=1, default=float))
    log("wrote results")
    if not sample:
        import plots_criteo
        plots_criteo.make_all(R, OUT)
        log("wrote figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    main(ap.parse_args().sample)
