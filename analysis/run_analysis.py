"""Run the full Hillstrom analysis; writes docs/assets/results.{json,js} and figures.

Usage:  python analysis/run_analysis.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exptools import (cuped_effect, diff_in_means, mde_mean, mde_proportion, power_mean,
                      power_proportion, sample_size_proportion, srm_test, standardized_mean_diff)
from exptools.data import ARMS, feature_matrix, load_hillstrom
from exptools.uplift import (LEARNERS, bootstrap, crossfit_arm_outcomes, crossfit_uplift, ipw_gain,
                             qini_coefficient, qini_curve, targeting_curve, uplift_by_group)

import plots

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 7
METRICS = ["visit", "conversion", "spend"]
COVARIATES = ["history", "recency", "newbie", "mens", "womens"]  # all measured before the send


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def gbm(kind: str, **kw):
    cls = HGBC if kind == "binary" else HGBR
    return cls(max_depth=3, learning_rate=0.05, max_iter=100, min_samples_leaf=200, random_state=SEED, **kw)


def main() -> None:
    df = load_hillstrom()
    X = feature_matrix(df)
    arms = {name: df[df.segment == name] for name in ARMS}
    ctrl = arms["No E-Mail"]
    R: dict = {"n": len(df), "arm_counts": {k: len(v) for k, v in arms.items()}}

    # ---------------------------------------------------------------- 1. validity
    R["srm"] = srm_test([len(arms[k]) for k in ARMS])
    bal = []
    for c in COVARIATES:
        for arm in ("Mens E-Mail", "Womens E-Mail"):
            bal.append({"covariate": c, "arm": arm, "smd": standardized_mean_diff(arms[arm][c], ctrl[c])})
    R["balance"] = bal
    R["max_abs_smd"] = max(abs(b["smd"]) for b in bal)

    # ---------------------------------------------------------------- 2. A/B results
    ab, pvals = [], []
    for arm in ("Mens E-Mail", "Womens E-Mail"):
        for m in METRICS:
            e = diff_in_means(arms[arm][m], ctrl[m])
            ab.append({"arm": arm, "metric": m, **e.as_dict()})
            pvals.append(e.p_value)
    for row, p in zip(ab, holm(pvals)):
        row["p_holm"] = p
    R["ab"] = ab
    mw = {m: diff_in_means(arms["Mens E-Mail"][m], arms["Womens E-Mail"][m]).as_dict() for m in METRICS}
    R["mens_vs_womens"] = mw

    # ---------------------------------------------------------------- 3. CUPED
    cuped_rows = []
    for arm in ("Mens E-Mail", "Womens E-Mail"):
        sub = df[df.segment.isin([arm, "No E-Mail"])]
        treated = (sub.segment == arm).values
        Xs = X.loc[sub.index].values
        for m in ("visit", "spend"):
            y = sub[m].values
            # CUPAC: covariate = out-of-fold ML prediction of the metric from pre-treatment features
            pred = np.zeros(len(y))
            for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xs):
                pred[te] = (gbm("binary" if m == "visit" else "cont").fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
                            if m == "visit" else gbm("cont").fit(Xs[tr], y[tr]).predict(Xs[te]))
            variants = {"History only (classic CUPED)": sub[["history"]].values,
                        "All pre-treatment covariates": sub[COVARIATES].values,
                        "ML prediction (CUPAC)": pred}
            raw = diff_in_means(y[treated], y[~treated])
            cuped_rows.append({"arm": arm, "metric": m, "method": "Raw (no adjustment)", "estimate": raw.estimate,
                               "se": raw.se, "variance_reduction": 0.0, "se_ratio": 1.0,
                               "corr": 0.0, "p_value": raw.p_value})
            for name, cov in variants.items():
                eff, info = cuped_effect(y, cov, treated)
                rho = float(np.sqrt(max(info["variance_reduction"], 0)))
                cuped_rows.append({"arm": arm, "metric": m, "method": name, "estimate": eff.estimate, "se": eff.se,
                                   "variance_reduction": info["variance_reduction"], "se_ratio": info["se_ratio"],
                                   "corr": rho, "p_value": eff.p_value})
    R["cuped"] = cuped_rows

    # Monte Carlo: variance reduction should track rho^2; CIs should keep nominal coverage
    rng = np.random.default_rng(SEED)
    sim, n_sim, true_effect = [], 4000, 0.1
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9):
        covered = covered_raw = 0
        vr = []
        reps = 400
        for _ in range(reps):
            x = rng.normal(size=n_sim)
            tt = rng.random(n_sim) < 0.5
            y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n_sim) + true_effect * tt
            eff, info = cuped_effect(y, x, tt)
            covered += eff.ci_low <= true_effect <= eff.ci_high
            covered_raw += info["raw"].ci_low <= true_effect <= info["raw"].ci_high
            vr.append(info["variance_reduction"])
        sim.append({"rho": rho, "rho2": rho**2, "variance_reduction": float(np.mean(vr)),
                    "coverage_cuped": covered / reps, "coverage_raw": covered_raw / reps})
    R["cuped_sim"] = sim

    # ---------------------------------------------------------------- 4. power analysis
    n_arm = len(ctrl)
    power_tbl = []
    for arm in ("Mens E-Mail", "Womens E-Mail"):
        for m in METRICS:
            row = next(r for r in ab if r["arm"] == arm and r["metric"] == m)
            sd = float(np.sqrt((arms[arm][m].var() + ctrl[m].var()) / 2))
            base = float(ctrl[m].mean())
            if m == "spend":
                mde = mde_mean(sd, n_arm)
            else:
                mde = mde_proportion(base, n_arm)
            power_tbl.append({"arm": arm, "metric": m, "baseline": base, "sd": sd, "mde_abs": mde,
                              "mde_rel": mde / base, "observed_abs": row["estimate"],
                              "observed_rel": row["estimate"] / base})
    R["mde"] = power_tbl

    # Empirical power: repeatedly subsample the real data and count rejections
    grid = [250, 500, 1000, 2000, 4000, 8000]
    reps = 500
    emp = []
    for m in ("visit", "spend"):
        a, c = arms["Mens E-Mail"][m].values, ctrl[m].values
        delta, base = a.mean() - c.mean(), c.mean()
        sd = float(np.sqrt((a.var() + c.var()) / 2))
        for n in grid:
            rej = 0
            for _ in range(reps):
                s1, s0 = rng.choice(a, n, replace=False), rng.choice(c, n, replace=False)
                rej += stats.ttest_ind(s1, s0, equal_var=False).pvalue < 0.05
            theo = (power_proportion(base, delta, n) if m == "visit" else power_mean(sd, delta, n))
            emp.append({"metric": m, "n_per_arm": n, "empirical": rej / reps, "theoretical": theo})
    R["power_check"] = emp

    # Sample-size table for a typical conversion-rate test at the Hillstrom baseline
    base = float(ctrl["conversion"].mean())
    R["sample_size_table"] = [
        {"relative_lift": rl, "n_per_arm": sample_size_proportion(base, rl, relative=True)["n_control"],
         "n_per_arm_cuped30": sample_size_proportion(base, rl, relative=True, variance_reduction=0.3)["n_control"]}
        for rl in (0.05, 0.10, 0.20, 0.30, 0.50)]
    R["baseline_conversion"] = base

    # ---------------------------------------------------------------- 5. uplift: any e-mail vs none
    Xv = X.values
    t = df.treated.values
    p_treat = float(t.mean())
    seeds = range(5)  # different CV folds + model seeds; scores are averaged (each is out-of-fold)
    up = {"p_treat": p_treat, "n_seeds": len(seeds)}
    rank = lambda tau: np.argsort(np.argsort(-tau, kind="stable"), kind="stable") / len(tau)

    def evaluate(scores_by_seed: list[np.ndarray], y: np.ndarray) -> tuple[dict, np.ndarray]:
        avg = np.mean(scores_by_seed, axis=0)
        per_seed = [qini_coefficient(s, y, t) for s in scores_by_seed]
        lo, hi = bootstrap(lambda idx: qini_coefficient(avg[idx], y[idx], t[idx]), len(y), n_boot=150, seed=SEED)
        return {"qini": qini_coefficient(avg, y, t), "qini_ci": [lo, hi], "qini_seed_sd": float(np.std(per_seed)),
                "qini_seed_min": float(min(per_seed)), "qini_seed_max": float(max(per_seed)),
                "gain_top30": ipw_gain(y, t, p_treat, rank(avg) < 0.3),
                "gain_random30": 0.3 * ipw_gain(y, t, p_treat)}, avg

    averaged = {}
    for outcome in ("visit", "spend"):
        y = df[outcome].values
        kind = "binary" if outcome == "visit" else "cont"
        specs = [(name, fn, lambda s, k=kind: HGBC(max_depth=3, learning_rate=0.05, max_iter=100, min_samples_leaf=200, random_state=s)
                  if k == "binary" else HGBR(max_depth=3, learning_rate=0.05, max_iter=100, min_samples_leaf=200, random_state=s))
                 for name, fn in LEARNERS.items()]
        if outcome == "visit":
            specs.append(("T-learner (logistic)", LEARNERS["T-learner"],
                          lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))))
        res, avg_scores = {}, {}
        for name, fn, factory in specs:
            by_seed = [crossfit_uplift(Xv, y, t, fn, factory(s), seed=s) for s in seeds]
            res[name], avg_scores[name] = evaluate(by_seed, y)
        averaged[outcome] = avg_scores
        up[outcome] = {"ate_ipw": ipw_gain(y, t, p_treat), "learners": res}

    # Spend is ~99% zeros with a heavy tail, so models trained on it chase noise. Rank on the dense signal
    # (visit uplift) and *evaluate* on the business metric (spend). Randomization keeps this evaluation unbiased.
    ys = df["spend"].values
    sur_name = "X-learner on visits (surrogate)"
    up["spend"]["learners"][sur_name], _ = evaluate([averaged["visit"]["X-learner"]], ys)
    averaged["spend"][sur_name] = averaged["visit"]["X-learner"]
    up["spend"]["primary"] = sur_name
    up["visit"]["primary"] = "X-learner"  # fixed in advance: designed for unbalanced (2:1) assignment

    for outcome in ("visit", "spend"):
        y = df[outcome].values
        scores = averaged[outcome]
        primary = scores[up[outcome]["primary"]]
        up[outcome]["deciles"] = uplift_by_group(primary, y, t).to_dict("records")
        up[outcome]["qini_curves"] = {n: qini_curve(s, y, t, 60).assign(n_total=len(y)).to_dict("list") for n, s in scores.items()}
    tc = targeting_curve(averaged["spend"][sur_name], ys, t, p_treat, costs=(0.0, 0.25, 0.50))
    up["spend"]["targeting"] = tc.to_dict("records")
    up["spend"]["targeting_best"] = {}
    for c, sub in tc.groupby("cost"):
        up["spend"]["targeting_best"][str(c)] = {
            "best_fraction": float(sub.loc[sub.profit.idxmax(), "fraction"]), "best_profit": float(sub.profit.max()),
            "all_profit": float(sub[sub.fraction == 1].profit.iloc[0])}
    # Fixed in advance: e-mail the top 30% vs everyone. CI by bootstrapping customers (scores held fixed; they are out-of-fold).
    sur = averaged["spend"][sur_name]
    top = rank(sur) < 0.3
    up["spend"]["top30_vs_all"] = {}
    for c in (0.0, 0.25, 0.50):
        def gap(idx, c=c):
            return (ipw_gain(ys[idx], t[idx], p_treat, top[idx]) - c * 0.3) - (ipw_gain(ys[idx], t[idx], p_treat) - c)
        lo, hi = bootstrap(gap, len(ys), n_boot=300, seed=SEED)
        up["spend"]["top30_vs_all"][str(c)] = {"diff": float(gap(np.arange(len(ys)))), "ci": [lo, hi]}
    pv = averaged["visit"]["X-learner"]
    prof = X.assign(quintile=pd.qcut(pd.Series(pv).rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).values,
                    score=pv, channel=df["channel"].values)
    up["visit"]["profile"] = prof.groupby("quintile", observed=True).agg(
        recency=("recency", "mean"), history=("history", "mean"), newbie=("newbie", "mean"),
        mens=("mens", "mean"), womens=("womens", "mean"), score=("score", "mean")).reset_index().to_dict("records")
    R["uplift"] = up

    # ---------------------------------------------------------------- 6. which e-mail to send? (3-way policy)
    arm_ids = df.arm.values
    pol = {}
    for outcome in ("visit", "spend"):
        y = df[outcome].values
        kind = "binary" if outcome == "visit" else "cont"
        pred = crossfit_arm_outcomes(Xv, y, arm_ids, gbm(kind), seed=SEED)
        rules = {
            "Send nothing": np.zeros(len(y), int),
            "Mens e-mail to everyone": np.ones(len(y), int),
            "Womens e-mail to everyone": np.full(len(y), 2),
            "Match to stated interest": np.where(df.womens.values == 1, 2, 1),
            "Learned best-arm policy": pred.argmax(1),
        }
        # Policy value by inverse propensity weighting; each arm had probability 1/3
        def value(policy, idx):
            return float(np.mean((arm_ids[idx] == policy[idx]) * y[idx] * 3))
        rng2 = np.random.default_rng(SEED)
        boots = [rng2.integers(0, len(y), len(y)) for _ in range(300)]
        pol[outcome] = {}
        for name, policy in rules.items():
            bv = [value(policy, b) for b in boots]
            pol[outcome][name] = {"value": value(policy, np.arange(len(y))),
                                  "ci": [float(np.quantile(bv, .025)), float(np.quantile(bv, .975))]}
        learned = rules["Learned best-arm policy"]
        pol[outcome]["_share"] = {n: float(np.mean(learned == i)) for n, i in [("none", 0), ("mens", 1), ("womens", 2)]}
        diff = [value(rules["Learned best-arm policy"], b) - value(rules["Mens e-mail to everyone"], b) for b in boots]
        pol[outcome]["_learned_minus_mens"] = {"diff": float(np.mean(diff)), "ci": [float(np.quantile(diff, .025)), float(np.quantile(diff, .975))]}
    R["policy"] = pol

    # ---------------------------------------------------------------- write out
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
    (OUT / "results.js").write_text("window.RESULTS = " + json.dumps(R, default=float) + ";\n")
    plots.make_all(R, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
