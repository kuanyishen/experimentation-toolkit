"""Render docs/criteo.html from docs/assets/criteo/results.json (every number in the text comes from the analysis)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "docs" / "assets" / "criteo" / "results.json").read_text())
REPO = "https://github.com/kuanyishen/experimentation-toolkit"

pp = lambda x, d=2: f"{x * 100:.{d}f} pp"
pc = lambda x, d=0: f"{x * 100:.{d}f}%"
ci = lambda lo, hi, f: f"[{f(lo)}, {f(hi)}]"
per1k = lambda x, d=2: f"{x * 1000:.{d}f}"

est = lambda m, meth: next(e for e in R["estimates"] if e["metric"] == m and e["method"].startswith(meth))
out = R["uplift"]["outcomes"]
P = R["propensity"]
learners = lambda m: out[m]["learners"]
X_v, X_c = learners("visit")["X-learner"], learners("conversion")["X-learner"]
share20 = lambda m: out[m]["learners"]["X-learner"]["gain_top20"] / out[m]["ate_ipw"]
dec1 = lambda m: out[m]["deciles"][0]["observed_uplift"]
tb = R["uplift"]["targeting_best"]
alloc = lambda m, tot, sc: next(a for a in R["allocation"] if a["metric"] == m and a["total"] == tot and a["control_share"] == sc)
ss = {s["metric"]: s for s in R["sample_size_for_adjusted_effect"]}
naive_v, naive_c, aipw_v, aipw_c = est("visit", "Naive"), est("conversion", "Naive"), est("visit", "AIPW"), est("conversion", "AIPW")
adj_range = lambda m: (min(e["estimate"] for e in R["estimates"] if e["metric"] == m and not e["method"].startswith("Naive")),
                       max(e["estimate"] for e in R["estimates"] if e["metric"] == m and not e["method"].startswith("Naive")))
av, ac = adj_range("visit"), adj_range("conversion")


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def est_table():
    rows = []
    for m, name in (("visit", "Site visit"), ("conversion", "Conversion")):
        for e in [x for x in R["estimates"] if x["metric"] == m]:
            vr = e.get("variance_reduction")
            rows.append([name, e["method"], f"{e['estimate'] * 100:.3f}", f"{e['se'] * 100:.4f}",
                         f"{(e['estimate'] / R['naive'][m]['estimate'] - 1) * 100:+.0f}%" if not e["method"].startswith("Naive") else "–",
                         pc(vr, 0) if vr else "–"])
    return table(["Metric", "Estimator", "Lift (pp)", "Std. error (pp)", "vs naive", "Variance cut"], rows)


def alloc_table():
    rows = []
    for m, name in (("visit", "Site visit"), ("conversion", "Conversion")):
        for sc, lab in ((0.5, "50 / 50"), (0.25, "75 / 25"), (0.15, "85 / 15 (as run)")):
            a = alloc(m, 100_000, sc)
            b = alloc(m, 1_000_000, sc)
            rows.append([name, lab, pc(a["mde_rel"], 0), pc(b["mde_rel"], 1)])
    return table(["Metric", "Treatment / control split", "Relative MDE, 100k users", "Relative MDE, 1M users"], rows)


def learner_table(m):
    rows = []
    for name, v in learners(m).items():
        star = " ★" if name == out[m]["primary"] else ""
        rows.append([name + star, per1k(v["qini"], 2), ci(v["qini_ci"][0] * 1000, v["qini_ci"][1] * 1000, lambda x: f"{x:.2f}"),
                     per1k(v["gain_top20"]), per1k(v["gain_random20"])])
    unit = "visits" if m == "visit" else "conversions"
    return table(["Model", "Qini ×1000", "95% CI", f"Top 20%: extra {unit} / 1,000 users", "Random 20%"], rows)


def target_table():
    rows = []
    for c in ("0.0", "0.0003", "0.0006"):
        b = tb[c]
        rows.append([f"{float(c) * 1000:.1f}", per1k(b["all_profit"]), f"{pc(b['best_fraction'], 0)} → {per1k(b['best_profit'])}",
                     f"{(b['best_profit'] / b['all_profit'] - 1) * 100:+.0f}%"])
    return table(["Cost per targeted user (conversions per 1,000)", "Target everyone: net / 1,000", "Best audience → net / 1,000", "Gain vs everyone"], rows)


def balance_rows():
    return table(["Feature", "SMD, raw", "SMD, weighted"], [[b["feature"], f"{b['smd_raw']:+.3f}", f"{b['smd_weighted']:+.4f}"] for b in R["balance"]])


top_het = R["uplift"]["feature_heterogeneity"][:3]
lc = R["uplift"]["learning_curve"]
cu = lambda m, meth: est(m, meth)

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Criteo Uplift at Scale</title>
<meta name="description" content="14-million-user ad experiment: a randomization audit, propensity-adjusted effects, CUPAC variance reduction, power and allocation analysis, and uplift modeling.">
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav class="top"><div class="wrap">
  <a class="brand" href="index.html">Experimentation toolkit</a>
  <a href="index.html">Hillstrom</a><a class="on" href="criteo.html">Criteo 14M</a>
  <a href="#audit">Audit</a><a href="#effect">Effect</a><a href="#cuped">CUPAC</a><a href="#power">Power</a><a href="#uplift">Uplift</a>
  <a href="calculator.html">Calculator</a><a href="{REPO}">GitHub</a>
</div></nav>
<main class="wrap">
<header class="hero">
  <div class="eyebrow">Case study 2 · Experimentation at scale</div>
  <h1>The randomization wasn't quite random, and it changed the answer</h1>
  <p class="lead">The Criteo uplift benchmark: {R['n']:,} users, an ad campaign shown to 85% of them. Same toolkit as the e-mail case study, but at 200× the scale, and with a twist: the arms are measurably imbalanced, so the naive A/B read-out overstates the lift. I audit it, correct it, and then find the {pc(1 - 0.8, 0)} of users who carry most of the effect.</p>
  <div class="row"><a class="btn" href="#audit">Start with the audit</a><a class="btn ghost" href="index.html">First case study: Hillstrom e-mail</a></div>
</header>

<div class="kpis">
  <div class="kpi"><b>{naive_v['estimate'] * 100:.2f} → {aipw_v['estimate'] * 100:.2f} pp</b><span>site-visit lift: naive vs doubly-robust adjusted</span></div>
  <div class="kpi"><b>{P['auc']:.3f}</b><span>propensity AUC: the features faintly predict who was treated (0.5 = pure chance)</span></div>
  <div class="kpi"><b>{pc(share20('conversion'), 0)}</b><span>of incremental conversions come from the top 20% of users by predicted uplift</span></div>
  <div class="kpi"><b>{pc(cu('visit', 'CUPAC')['variance_reduction'], 0)}</b><span>variance CUPAC removes on visits (vs 0.1% for spend on the Hillstrom data)</span></div>
</div>

<div class="callout"><p><b>TL;DR.</b> The campaign works, but the textbook difference-in-means overstates the visit lift by ~{(naive_v['estimate'] / aipw_v['estimate'] - 1) * 100:.0f}% and the conversion lift by ~{(naive_c['estimate'] / aipw_c['estimate'] - 1) * 100:.0f}%, because the treated group was slightly tilted toward users who visit anyway. Propensity weighting, an outcome model and their doubly-robust combination agree and restore balance. After adjustment, uplift is extremely concentrated: the top decile of users shows a {pp(dec1('conversion'), 2)} conversion lift vs a {pp(out['conversion']['ate_ipw'], 2)} average, and targeting the top 20% captures ~{pc(share20('conversion'), 0)} of all incremental conversions.</p></div>

<h2 id="data">The experiment</h2>
<p>Criteo AI Lab's uplift dataset (v2.1; <a href="https://ar5iv.labs.arxiv.org/html/2111.10106">Diemert et al.</a>): {R['n']:,} users, 12 anonymised numeric features, a treatment flag ({pc(R['treated_share'], 0)} treated), and three outcomes: site <b>visit</b>, <b>conversion</b> and ad <b>exposure</b>. Only {pc(R['exposure']['rate_adjusted'], 1)} of treated users were actually exposed to an ad, so "treatment" really means <i>eligible to be shown ads</i>. Exposure is post-treatment and is never used as a feature.</p>
<p>To keep every model honest the data is split into three disjoint parts: <b>A</b> ({R['splits']['A']:,} users, propensity and CUPAC covariate models), <b>B</b> ({R['splits']['B']:,}, outcome and uplift models) and <b>C</b> ({R['splits']['C']:,}, a holdout where every estimate below is computed). The full pipeline runs in about {R['runtime_seconds'] / 60:.0f} minutes on 4 CPU cores.</p>

<h2 id="audit">1 · Was assignment actually random?</h2>
<p>The split passes a sample-ratio check against 85/15 (χ² p = {R['srm']['p_value']:.2f}). But with 14M rows, covariate balance is the sharper test:</p>
<figure><img src="assets/criteo/fig_balance.png" alt="Standardized mean differences before and after propensity weighting" style="max-width:640px"><figcaption>Treated minus control, in standard deviations. Every raw difference is below the usual 0.1 rule of thumb (max {R['max_abs_smd_raw']:.3f}), yet with 14M users they are highly significant, and the features predict outcomes strongly.</figcaption></figure>
<p>A classifier trained on part A to predict <i>treatment</i> from the 12 features scores AUC <b>{P['auc']:.3f}</b> on holdout C (95% CI {ci(P['auc_ci'][0], P['auc_ci'][1], lambda x: f'{x:.4f}')}). Under perfect randomization that would be 0.5. It is a faint signal, but the same features predict <i>visits</i> very well, and that is what matters: a faint tilt in who gets treated, multiplied by a strong outcome predictor, is a real bias. Estimated assignment probabilities range from {P['quantiles']['min']:.2f} to {P['quantiles']['max']:.2f} (median {P['quantiles']['p50']:.3f}), so overlap is excellent and weights are benign (effective sample size {pc(P['effective_sample_size_ratio'], 1)} of nominal). After weighting by the inverse propensity, the largest remaining imbalance falls to <b>{R['max_abs_smd_weighted']:.4f}</b>.</p>
<p class="sub">Why might this happen? Published descriptions say the data pool several incrementality tests; if their treated/control ratios differ, any feature that identifies the sub-test predicts assignment. The features are anonymised, so I can't verify that. Another possibility, that features were recorded <i>after</i> treatment and so were affected by it, would make adjusting for them wrong. The evidence below favours the first explanation (three independent adjustments agree and restore balance), but balance on observed features can't rule out unobserved confounding.</p>

<h2 id="effect">2 · What is the effect, really?</h2>
<figure><img src="assets/criteo/fig_estimates.png" alt="Effect estimates under five estimators"><figcaption>Five estimators on holdout C, 95% CIs. Naive = difference in means as if randomized. IPW = inverse-propensity weighting. AIPW = augmented IPW (doubly robust: consistent if either the propensity or the outcome model is right).</figcaption></figure>
{est_table()}
<p>The naive visit lift is <b>{naive_v['estimate'] * 100:.2f} pp</b>; every adjusted estimator lands between {av[0] * 100:.2f} and {av[1] * 100:.2f} pp, with the doubly-robust estimate at <b>{aipw_v['estimate'] * 100:.2f} pp</b> (95% CI {ci((aipw_v['estimate'] - 1.96 * aipw_v['se']) * 100, (aipw_v['estimate'] + 1.96 * aipw_v['se']) * 100, lambda x: f'{x:.2f}')}). For conversions the shift is smaller, {naive_c['estimate'] * 100:.3f} pp naive vs {aipw_c['estimate'] * 100:.3f} pp adjusted, but still around {(naive_c['se'] and (naive_c['estimate'] - aipw_c['estimate']) / aipw_c['se']):.1f} adjusted standard errors. Either way the campaign clearly works; the question is how much to credit it, which is what a budget decision depends on.</p>
<h3>Intent-to-treat vs the effect on people who saw an ad</h3>
<p>Since only {pc(R['exposure']['rate_adjusted'], 1)} of treated users were exposed (and control users never could be), the ITT lift is diluted. Scaling by the exposure rate (a Wald / complier-average estimate, assuming ads affect only people who see them) gives a conversion lift of <b>{pp(R['cace']['conversion']['estimate'], 1)}</b> per exposed user (95% CI {ci(R['cace']['conversion']['ci'][0] * 100, R['cace']['conversion']['ci'][1] * 100, lambda x: f'{x:.1f}')} pp) and {pp(R['cace']['visit']['estimate'], 0)} for visits.</p>

<h2 id="cuped">3 · CUPED, CUPAC and what they really do</h2>
<p>With no pre-experiment metric available, I use the same idea with the 12 pre-treatment features: a gradient-boosting model fitted on part A predicts each user's outcome, and CUPED subtracts that prediction (<b>CUPAC</b>). Unlike the Hillstrom data, where prior-year spend barely predicted spend (0.1% variance cut), these features are highly informative about visits:</p>
<ul class="tight">
<li><b>Visits:</b> CUPAC removes {pc(cu('visit', 'CUPAC')['variance_reduction'], 0)} of the variance (linear CUPED on the raw features: {pc(cu('visit', 'CUPED')['variance_reduction'], 0)}); AIPW's outcome model cuts the standard error by a similar amount relative to IPW ({pc(aipw_v['variance_reduction'], 0)} of variance).</li>
<li><b>Conversions:</b> only {pc(cu('conversion', 'CUPAC')['variance_reduction'], 0)}. A 0.3%-positive outcome is hard to predict.</li>
<li><b>In sample-size terms</b> (to detect the adjusted effect with 80% power at the actual 85/15 split): visits {ss['visit']['total_raw']:,} → {ss['visit']['total_adjusted']:,} users ({(1 - ss['visit']['total_adjusted'] / ss['visit']['total_raw']) * 100:.0f}% fewer); conversions {ss['conversion']['total_raw']:,} → {ss['conversion']['total_adjusted']:,} ({(1 - ss['conversion']['total_adjusted'] / ss['conversion']['total_raw']) * 100:.0f}% fewer).</li>
</ul>
<div class="callout warn"><p><b>A warning that only shows up with imbalanced arms.</b> CUPED's promise is "same expected effect, less variance". Here the CUPED and CUPAC point estimates <em>moved</em> ({cu('visit', 'Naive')['estimate'] * 100:.2f} → {cu('visit', 'CUPAC')['estimate'] * 100:.2f} pp for visits), far more than their standard errors shrank. That shift is not variance reduction; it is the covariates quietly correcting a confounded comparison. Treat an estimate that moves more than the variance reduction can explain as a randomization alarm, then fix it properly with IPW/AIPW rather than a single-coefficient adjustment (the two still differ from each other by about 0.1 pp).</p></div>

<h2 id="power">4 · Power and allocation</h2>
<p>The experiment was 85/15 by design: more exposure, but a small control group. The control group is what limits precision, so unequal splits cost power:</p>
{alloc_table()}
<p>At 100,000 users, detecting a conversion lift needs a {pc(alloc('conversion', 100000, 0.15)['mde_rel'], 0)} relative change with the 85/15 split vs {pc(alloc('conversion', 100000, 0.5)['mde_rel'], 0)} at 50/50, roughly <b>{(alloc('conversion', 100000, 0.15)['mde_rel'] / alloc('conversion', 100000, 0.5)['mde_rel']) ** 2:.1f}× more users</b> for the same power. The 14M-user campaign was far beyond what the average effect needs ({ss['conversion']['total_raw']:,} users); its scale pays off for <em>uplift modelling</em>, not for the headline A/B result. I checked the formulas by repeatedly subsampling the real data:</p>
<figure><img src="assets/criteo/fig_power.png" alt="Power formula vs resampled data"><figcaption>Lines: closed-form power at 85/15. Dots: share of 400 random subsamples with p &lt; 0.05. They agree except at the smallest conversion sample (25,000 users contain only ~7 control conversions, where the normal approximation is shaky).</figcaption></figure>
<p><a href="calculator.html">The live calculator</a> handles uneven splits: set "traffic to control" to 15%.</p>

<h2 id="uplift">5 · Uplift: who is the ad actually changing?</h2>
<p>Four learners (S, T, X and a logistic T-learner) are trained on part B ({R['uplift']['n_train']:,} users) and scored on holdout C. Because assignment depends on the features, evaluation uses <b>propensity-weighted</b> Qini curves and uplift-by-decile, an extension I added to the toolkit (and unit-tested on synthetic confounded data). Confidence intervals come from 20 random chunks of the holdout. The X-learner (★) was fixed in advance as the primary model.</p>
<h3>Site visits</h3>
{learner_table('visit')}
<h3>Conversions</h3>
{learner_table('conversion')}
<p>The S- and X-learners are best on conversions; the single-model S-learner beats the T-learner clearly, a typical pattern when the control arm is small and its positives are few. Ranking on <i>visit</i> uplift, a much denser signal, and evaluating on conversions (the "surrogate" row) works nearly as well as modelling conversions directly. Rank on the dense proxy, measure on the business metric.</p>
<figure><img src="assets/criteo/fig_qini.png" alt="Qini curves for visits and conversions"><figcaption>Propensity-weighted Qini curves. The steep rise is the point: a small slice of users produces nearly all of the incremental outcome.</figcaption></figure>
<figure><img src="assets/criteo/fig_deciles.png" alt="Measured lift by uplift decile"><figcaption>Measured lift inside each decile of predicted uplift (95% CI). The top decile lifts visits by {pp(dec1('visit'), 1)} and conversions by {pp(dec1('conversion'), 2)}; each of the bottom five deciles adds at most {max(d['observed_uplift'] for d in out['visit']['deciles'][5:]) * 100:.2f} pp to visits and {max(d['observed_uplift'] for d in out['conversion']['deciles'][5:]) * 100:.2f} pp to conversions.</figcaption></figure>
<p>Targeting the top 20% captures <b>{pc(share20('visit'), 0)}</b> of all incremental visits and <b>{pc(share20('conversion'), 0)}</b> of incremental conversions ({per1k(X_c['gain_top20'], 2)} of {per1k(out['conversion']['ate_ipw'], 2)} per 1,000 users), versus 20% for random targeting.</p>
<h3>What distinguishes high-uplift users?</h3>
<p>A model-free screen on holdout C (weighted conversion lift by bins of each feature) shows where the lift lives: low values of <code>{top_het[0]['feature']}</code>, <code>{top_het[1]['feature']}</code> and <code>{top_het[2]['feature']}</code>. The features are anonymised so there is no business interpretation, but the pattern is consistent and large:</p>
<figure><img src="assets/criteo/fig_heterogeneity.png" alt="Conversion lift by feature bin"><figcaption>Propensity-weighted conversion lift by feature bin on the four most heterogeneous features (many features take few distinct values, so bins are fewer than five). Screening 12 features is multiple comparisons, but the effects are many standard errors wide.</figcaption></figure>
<h3>Where to stop?</h3>
<p>Targeting has a price. Suppose each targeted user costs a fixed amount, expressed as a fraction of one conversion's value (e.g. ad cost $0.015 against a $50 conversion = 0.3 conversions per 1,000 users). The best audience shrinks as that cost grows:</p>
<figure><img src="assets/criteo/fig_targeting.png" alt="Net incremental conversions vs share targeted" style="max-width:660px"></figure>
{target_table()}
<p class="sub">Illustrative costs; Criteo provides no prices. The audience size is chosen on the same holdout, which is optimistic, but with ~7M users the curve's maximum is stable.</p>
<h3>How much data does uplift need?</h3>
<figure><img src="assets/criteo/fig_learning.png" alt="Learning curve" style="max-width:560px"><figcaption>Qini coefficient (T-learner, conversions) vs training-set size, with 95% CI. It is noisy below ~1M users ({lc[1]['qini'] * 1000:.2f} at {lc[1]['n_train'] // 1000}k, {lc[2]['qini'] * 1000:.2f} at {lc[2]['n_train'] // 1000}k) and reaches {lc[-2]['qini'] * 1000:.2f}–{lc[-1]['qini'] * 1000:.2f} at {lc[-2]['n_train'] / 1e6:.0f}–{lc[-1]['n_train'] / 1e6:.1f}M. Individual-level effects are far more data-hungry than average effects: this is why 14M users can matter even though ~0.1M suffice for the headline A/B.</figcaption></figure>

<h2>Limitations</h2>
<ul class="tight">
<li>All adjustments assume no <em>unmeasured</em> confounding; balance on 12 anonymised features cannot prove that.</li>
<li>Uplift ranks are evaluated with an estimated propensity; the propensity model is trained on a disjoint split, and the CIs ignore model-training variance.</li>
<li>One campaign, short outcome window; no revenue, so costs are illustrative.</li>
<li>Data was obtained from a public GitHub mirror because the official host was unreachable. Row count ({R['n']:,}) and overall rates match the published description, but I could not verify a checksum.</li>
</ul>
<h2>Reproduce</h2>
<pre><code>pip install -e ".[dev]"
python analysis/run_criteo.py     # downloads ~300 MB once; ~15 min, 16 GB RAM
python analysis/build_criteo.py   # renders this page</code></pre>
<footer>Built by Sky Shen · <a href="{REPO}">source</a> · <a href="index.html">Hillstrom case study</a> · <a href="calculator.html">calculator</a></footer>
</main>
</body>
</html>
"""
(ROOT / "docs" / "criteo.html").write_text(page)
print("wrote docs/criteo.html")
