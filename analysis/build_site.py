"""Render docs/index.html from docs/assets/results.json so every number in the text comes from the analysis."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "docs" / "assets" / "results.json").read_text())
REPO = "https://github.com/kuanyishen/experimentation-toolkit"

ab = {(r["arm"], r["metric"]): r for r in R["ab"]}
M, W = "Mens E-Mail", "Womens E-Mail"
pp = lambda x, d=1: f"{x * 100:+.{d}f} pp"
usd = lambda x, d=2: f"{'-' if x < 0 else ''}${abs(x):.{d}f}"
pc = lambda x, d=0: f"{x * 100:.{d}f}%"
pv = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
ci = lambda lo, hi, f: f"[{f(lo)}, {f(hi)}]"
up, pol = R["uplift"], R["policy"]
cu = [r for r in R["cuped"] if r["arm"] == M]
cu_all = lambda m: next(r for r in cu if r["metric"] == m and r["method"].startswith("All"))
cu_cupac = lambda m: next(r for r in cu if r["metric"] == m and r["method"].startswith("ML"))
mde = {(r["arm"], r["metric"]): r for r in R["mde"]}
best = up["spend"]["targeting_best"]
t30 = up["spend"]["top30_vs_all"]
L_v, L_s = up["visit"]["learners"], up["spend"]["learners"]
sur = L_s[up["spend"]["primary"]]
logi = L_v["T-learner (logistic)"]


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def ab_table():
    rows = []
    for arm, label in ((M, "Men's e-mail"), (W, "Women's e-mail")):
        for m, name, f in (("visit", "Site visit", lambda x: pp(x, 2)), ("conversion", "Purchase", lambda x: pp(x, 2)),
                           ("spend", "Spend / customer", usd)):
            r = ab[(arm, m)]
            rows.append([label, name, pc(r["mean_ctrl"], 2) if m != "spend" else usd(r["mean_ctrl"]),
                         pc(r["mean_treat"], 2) if m != "spend" else usd(r["mean_treat"]),
                         f"{f(r['estimate'])} ({ci(r['ci_low'], r['ci_high'], f)})", f"{r['relative_lift'] * 100:+.0f}%", pv(r["p_holm"])])
    return table(["Arm", "Metric", "Control", "E-mail", "Effect (95% CI)", "Relative", "p (Holm)"], rows)


def cuped_table():
    rows = []
    for m in ("visit", "spend"):
        for r in [x for x in cu if x["metric"] == m]:
            rows.append([m, r["method"], f"{r['corr']:.3f}" if r["corr"] else "–", pc(r["variance_reduction"], 1),
                         f"{r['se']:.4f}" if m == "visit" else f"{r['se']:.3f}", f"{(1 - r['se_ratio']) * 100:.1f}%"])
    return table(["Metric", "Covariate set", "√R²", "Variance cut", "Std. error", "SE shrink"], rows)


def mde_table():
    rows = []
    for m, name in (("visit", "Site visit"), ("conversion", "Purchase"), ("spend", "Spend / customer")):
        r = mde[(M, m)]
        f = (lambda x: usd(x)) if m == "spend" else (lambda x: f"{x * 100:.2f} pp")
        rows.append([name, (usd(r["baseline"]) if m == "spend" else pc(r["baseline"], 2)), f(r["mde_abs"]),
                     f"{r['mde_rel'] * 100:.0f}%", f(r["observed_abs"]), f"{r['observed_rel'] * 100:.0f}%"])
    return table(["Metric", "Control level", "MDE (80% power)", "MDE relative", "Observed (Mens)", "Observed relative"], rows)


def sample_table():
    rows = [[f"+{r['relative_lift'] * 100:.0f}%", f"{r['n_per_arm']:,}", f"{r['n_per_arm_cuped30']:,}"] for r in R["sample_size_table"]]
    return table(["Relative lift to detect", "Users per arm", "…with CUPED (ρ≈0.55)"], rows)


def learner_table(outcome):
    u = up[outcome]
    rows = []
    for name, v in u["learners"].items():
        star = " ★" if name == u["primary"] else ""
        f = (lambda x: f"{x:.4f}") if outcome == "visit" else (lambda x: f"{x:.3f}")
        rows.append([name + star, f(v["qini"]), ci(v["qini_ci"][0], v["qini_ci"][1], f),
                     f(v["gain_top30"]), f(v["gain_random30"])])
    unit = "visits" if outcome == "visit" else "$"
    return table(["Model", "Qini coefficient", "95% CI", f"Top-30% gain ({unit}/customer)", "Random 30%"], rows)


def policy_table(outcome):
    rows = []
    for k, v in pol[outcome].items():
        if k.startswith("_"):
            continue
        f = (lambda x: pc(x, 1)) if outcome == "visit" else usd
        rows.append([k, f(v["value"]), ci(v["ci"][0], v["ci"][1], f)])
    return table(["Policy", "Expected " + ("visit rate" if outcome == "visit" else "spend / customer"), "95% CI"], rows)


def profile_rows():
    rows = [[f"Q{int(r['quintile'])}", f"{r['score'] * 100:.1f} pp", pc(r["womens"]), pc(r["mens"]), pc(r["newbie"]),
             f"{r['recency']:.1f}", usd(r["history"], 0)] for r in up["visit"]["profile"]]
    return table(["Uplift quintile", "Predicted lift", "Buys women's", "Buys men's", "New customer", "Months since purchase", "Prior-year spend"], rows)


def targeting_rows():
    rows = []
    for c in ("0.0", "0.25", "0.5"):
        b, d = best[c], t30[c]
        rows.append([usd(float(c)), usd(b["all_profit"]), pc(b["best_fraction"]) + " → " + usd(b["best_profit"]),
                     f"{usd(d['diff'])} {ci(d['ci'][0], d['ci'][1], usd)}"])
    return table(["Cost per e-mail", "E-mail everyone", "Best audience share → net / customer*", "Top 30% minus everyone (95% CI)"], rows)


page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experimentation Toolkit</title>
<meta name="description" content="Analysing a randomized e-mail campaign with CUPED, power analysis and uplift modeling, plus a live sample-size calculator.">
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav class="top"><div class="wrap">
  <a class="brand" href="index.html">Experimentation toolkit</a>
  <a href="#validity">Validity</a><a href="#effect">Effect</a><a href="#cuped">CUPED</a><a href="#power">Power</a><a href="#uplift">Uplift</a><a href="#policy">Policy</a>
  <a href="calculator.html">Calculator</a><a href="{REPO}">GitHub</a>
</div></nav>
<main class="wrap">
<header class="hero">
  <div class="eyebrow">Case study · Causal inference &amp; experimentation</div>
  <h1>Did the e-mail work, and who should get it?</h1>
  <p class="lead">An end-to-end analysis of a real randomized e-mail campaign ({R['n']:,} customers, Hillstrom MineThatData): trustworthy A/B read-out, CUPED variance reduction, power analysis, and uplift models that decide <em>who to target</em>. Everything is reproducible Python, tested, and the sample-size calculator at the end runs live in your browser.</p>
  <div class="row"><a class="btn" href="calculator.html">Open the calculator</a><a class="btn ghost" href="{REPO}">View code</a></div>
</header>

<div class="kpis">
  <div class="kpi"><b>{pp(ab[(M, 'visit')]['estimate'], 1)}</b><span>site visits from the men's e-mail ({ab[(M, 'visit')]['relative_lift'] * 100:+.0f}% relative)</span></div>
  <div class="kpi"><b>{usd(ab[(M, 'spend')]['estimate'])}</b><span>extra spend per customer, men's e-mail (p {pv(ab[(M, 'spend')]['p_holm'])})</span></div>
  <div class="kpi"><b>{pc(cu_all('spend')['variance_reduction'], 1)}</b><span>variance CUPED removes from spend here: weak covariates</span></div>
  <div class="kpi"><b>+{(logi['gain_top30'] / logi['gain_random30'] - 1) * 100:.0f}%</b><span>more incremental visits from the top-30% uplift audience vs a random 30%</span></div>
</div>

<div class="callout"><p><b>TL;DR.</b> The campaign clearly works and the men's creative beats the women's ({usd(R['mens_vs_womens']['spend']['estimate'])} more spend per customer, p = {pv(R['mens_vs_womens']['spend']['p_value'])}). CUPED is nearly useless <em>on this dataset</em> because the only pre-experiment covariates barely predict spend, and I show with simulation exactly when it does pay off. Uplift models find real, but moderate, heterogeneity in visit response; they can't reliably learn it from raw spend, and a per-customer choice of creative does not beat simply sending the men's e-mail to everyone.</p></div>

<h2 id="data">The experiment</h2>
<p>In 2008 Kevin Hillstrom's retailer randomly assigned {R['n']:,} customers to a men's-merchandise e-mail, a women's-merchandise e-mail, or no e-mail, and tracked the following two weeks: site visit, purchase, and dollars spent. Covariates (all measured <em>before</em> the send): months since last purchase, prior-year spend, whether the customer buys men's/women's products, new-vs-existing, ZIP type and acquisition channel.</p>

<h2 id="validity">1 · Can I trust the randomization?</h2>
<p>Before reading any result: a sample-ratio-mismatch test on the arm sizes ({', '.join(f'{c:,}' for c in R['srm']['counts'])}) gives χ² p = {R['srm']['p_value']:.2f}, so no sign of broken assignment. Covariate balance is excellent: the largest standardized mean difference between any e-mail arm and control is {R['max_abs_smd']:.3f} (rule of thumb: worry above 0.1).</p>

<h2 id="effect">2 · Did it work?</h2>
<figure><img src="assets/fig_ab.png" alt="Forest plot of treatment effects"><figcaption>Difference in means with 95% Welch confidence intervals.</figcaption></figure>
{ab_table()}
<p class="sub">Six hypotheses (2 e-mails × 3 metrics) are tested, so p-values are Holm-adjusted. Both e-mails lift visits, purchases and spend; the men's e-mail beats the women's on all three (spend difference {usd(R['mens_vs_womens']['spend']['estimate'])}, 95% CI {ci(R['mens_vs_womens']['spend']['ci_low'], R['mens_vs_womens']['spend']['ci_high'], usd)}). Spend is the noisy one: 99% of customers spend nothing, so the standard deviation ({usd(mde[(M, 'spend')]['sd'], 0)}) dwarfs the mean.</p>

<h2 id="cuped">3 · CUPED: variance reduction from pre-experiment data</h2>
<p>CUPED (Deng et al., 2013) subtracts the part of the outcome that a pre-experiment covariate already explains: <code>Y_adj = Y − θ·(X − mean(X))</code>. It leaves the effect estimate unbiased and cuts variance by ρ², the squared correlation between covariate and outcome. I tried prior-year spend alone, all pre-treatment covariates together, and a cross-fitted gradient-boosting prediction (CUPAC).</p>
{cuped_table()}
<div class="callout warn"><p><b>Honest result:</b> on spend, CUPED shrinks standard errors by roughly {(1 - cu_all('spend')['se_ratio']) * 100:.1f}%, which is nothing, and even an ML covariate does no better ({pc(cu_cupac('spend')['variance_reduction'], 2)}). Prior-year spend correlates only ~0.02 with two-week spend. For visits the gain is a still-small {pc(cu_all('visit')['variance_reduction'], 1)}. CUPED is a tool for metrics with a strong pre-period analogue (e.g. last month's revenue to predict this month's), not magic.</p></div>
<figure><img src="assets/fig_cuped.png" alt="CUPED variance reduction versus correlation and CI coverage"><figcaption>Simulation (400 replications per point): variance reduction tracks ρ² almost exactly, CIs keep their nominal 95% coverage. The red squares are the Hillstrom estimates above.</figcaption></figure>

<h2 id="power">4 · Power analysis</h2>
<p>How big an effect could this experiment have detected? With {R['arm_counts']['No E-Mail']:,} customers per arm:</p>
{mde_table()}
<p>The site-visit lift was ~{ab[(M, 'visit')]['relative_lift'] / (mde[(M, 'visit')]['mde_rel']):.0f}× the minimum detectable effect, so visits were overwhelmingly powered. Spend is the opposite: the men's e-mail's {ab[(M, 'spend')]['relative_lift'] * 100:.0f}% lift is only {ab[(M, 'spend')]['relative_lift'] / mde[(M, 'spend')]['mde_rel']:.1f}× the {mde[(M, 'spend')]['mde_rel'] * 100:.0f}% MDE, and the women's e-mail's {ab[(W, 'spend')]['relative_lift'] * 100:.0f}% lift is barely above its {mde[(W, 'spend')]['mde_rel'] * 100:.0f}% MDE ({ab[(W, 'spend')]['relative_lift'] / mde[(W, 'spend')]['mde_rel']:.1f}×). That is why the spend p-values are weaker even though the dollar effects are large: a test with 80% power at the MDE will often miss effects that size. To check the formulas against reality I repeatedly subsampled the real data and counted how often the test fired:</p>
<figure><img src="assets/fig_power.png" alt="Power curves, formula versus resampled data"><figcaption>Lines: closed-form power. Dots: share of 500 random subsamples with p &lt; 0.05. They agree, so the formulas can be trusted for planning.</figcaption></figure>
<p>For planning a <em>new</em> test at this campaign's baseline purchase rate ({pc(R['baseline_conversion'], 2)}), sample sizes are brutal for rare events:</p>
{sample_table()}
<p><a href="calculator.html">Plug in your own numbers in the live calculator →</a></p>

<h2 id="uplift">5 · Uplift modeling: who should get the e-mail?</h2>
<p>The average effect says nothing about <em>which customers</em> respond. Uplift (CATE) models estimate the effect per customer; I compare S-, T- and X-learners with gradient boosting, plus a plain logistic-regression T-learner as a simple baseline. Because assignment was randomized, I can evaluate them without modelling assumptions:</p>
<ul class="tight">
<li><b>Out-of-fold scores only.</b> 5-fold cross-fitting, repeated over {up['n_seeds']} seeds and averaged, so no customer is scored by a model that saw them.</li>
<li><b>Qini curve:</b> rank customers by predicted uplift, then measure real treated-vs-control outcomes within the top-k slice. Area above the random-targeting line = Qini coefficient.</li>
<li><b>Model choice fixed in advance:</b> the X-learner (built for unbalanced 2:1 assignment: 2/3 of customers got an e-mail). Differences between models are within noise.</li>
</ul>
<h3>Target: site visits (dense, 15% positive)</h3>
{learner_table('visit')}
<p>Every model beats random targeting (CIs exclude 0), and the simple logistic model is as good as the boosted trees. Targeting the top 30% yields {logi['gain_top30']:.3f} extra visits per customer vs {logi['gain_random30']:.3f} from a random 30% (+{(logi['gain_top30'] / logi['gain_random30'] - 1) * 100:.0f}%).</p>
<figure><img src="assets/fig_qini.png" alt="Qini curves"><figcaption>Cumulative incremental outcome when e-mailing customers in order of predicted uplift. Dashed grey: random order.</figcaption></figure>
<figure><img src="assets/fig_deciles.png" alt="Observed lift by decile of predicted uplift"><figcaption>Observed treated-minus-control difference inside each decile of predicted uplift (95% CI). A good model shows a downward slope; decile 1 is clearly the most responsive.</figcaption></figure>
<h3>Target: spend (99% zeros, heavy tail)</h3>
{learner_table('spend')}
<p>Models trained directly on spend are indistinguishable from noise: their Qini coefficient flips sign from seed to seed (seed-to-seed s.d. ≈ {max(v['qini_seed_sd'] for k, v in L_s.items() if 'surrogate' not in k):.3f} vs. a signal of ~0). The workable approach is a <b>surrogate</b>: rank on the dense visit-uplift score and <i>evaluate</i> on dollars. That gives {usd(sur['gain_top30'])} per customer from the top 30% vs {usd(sur['gain_random30'])} at random, but its Qini CI {ci(sur['qini_ci'][0], sur['qini_ci'][1], lambda x: f'{x:.3f}')} still includes zero: suggestive, not proven, with 64k customers and this much noise.</p>
<h3>Who are the high-uplift customers?</h3>
{profile_rows()}
<p>The most responsive customers are those who buy <em>women's</em> merchandise and new customers, while men's-only buyers show the smallest predicted incremental response. Uplift is not the same as response rate: it ranks customers by how much the e-mail <em>changes</em> their behaviour, which need not be the customers who respond most in absolute terms.</p>

<h3>Where to stop? Targeting under a cost per e-mail</h3>
<figure><img src="assets/fig_targeting.png" alt="Net incremental revenue vs share of customers e-mailed" style="max-width:640px"><figcaption>Net incremental revenue per customer in the whole population = IPW-estimated uplift of the targeted slice minus cost × share e-mailed. Dots mark the best audience size.</figcaption></figure>
{targeting_rows()}
<p class="sub">*Audience size chosen on the same data, so it's optimistic; the fixed 30% column is the cleaner comparison. Revenue, not margin: scale the cost accordingly.</p>
<p>At a negligible cost per e-mail the answer is simply "e-mail everyone": the average effect (~{usd(up['spend']['ate_ipw'])}/customer) dwarfs it, and holding back 70% of customers forgoes {usd(-t30['0.0']['diff'])} each. Targeting only starts to look attractive once contact costs approach the average incremental revenue (expensive channels, discounts, unsubscribe cost). Even at $0.50 the estimated gain from the top 30% ({usd(t30['0.5']['diff'])}/customer) has a confidence interval that {'includes' if t30['0.5']['ci'][0] < 0 < t30['0.5']['ci'][1] else 'excludes'} zero, so this data supports the <em>mechanics</em> of cost-aware targeting better than a firm business conclusion.</p>

<h2 id="policy">6 · Which creative to send each customer?</h2>
<p>Three arms allow a bigger question: for each customer choose none / men's / women's e-mail. A T-learner per arm (cross-fitted) predicts the outcome under each and picks the best. Policies are valued with inverse-propensity weighting (each arm had probability 1/3), with bootstrap CIs.</p>
<figure><img src="assets/fig_policy.png" alt="Expected outcome under alternative policies"></figure>
{policy_table('spend')}
<div class="callout warn"><p><b>Negative result worth reporting:</b> the learned policy does <em>not</em> beat "send the men's e-mail to everyone" (spend difference {usd(pol['spend']['_learned_minus_mens']['diff'])}, 95% CI {ci(pol['spend']['_learned_minus_mens']['ci'][0], pol['spend']['_learned_minus_mens']['ci'][1], usd)}; visits {pp(pol['visit']['_learned_minus_mens']['diff'], 2)}, CI {ci(pol['visit']['_learned_minus_mens']['ci'][0] * 100, pol['visit']['_learned_minus_mens']['ci'][1] * 100, lambda x: f'{x:+.1f}')} pp). Personalization needs enough signal to pay for its estimation noise; here the single best creative captures nearly all of the value.</p></div>

<h2>Method notes &amp; limitations</h2>
<ul class="tight">
<li>"Treated" in the uplift section pools both e-mails (any e-mail vs none); the policy section separates them.</li>
<li>CUPED θ is estimated on pooled data and then treated as fixed (standard; negligible error at this n). The covariates are strictly pre-treatment.</li>
<li>Outcomes are a single two-week window; novelty, fatigue and long-run effects are invisible. Spend is revenue, not profit.</li>
<li>Uplift CIs bootstrap customers while holding the out-of-fold scores fixed, so they ignore model-training variability; I show seed-to-seed spread separately.</li>
<li>Data: <a href="https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html">Hillstrom, MineThatData E-Mail Analytics and Data Mining Challenge (2008)</a>.</li>
</ul>

<h2>Reproduce</h2>
<pre><code>git clone {REPO}.git &amp;&amp; cd experimentation-toolkit
pip install -e ".[dev]"
python analysis/run_analysis.py   # downloads data, writes results + figures (~3 min)
python analysis/build_site.py     # renders this page
pytest                            # statistical checks + JS/Python parity</code></pre>

<footer>Built by Sky Shen · <a href="{REPO}">source</a> · <a href="calculator.html">calculator</a></footer>
</main>
</body>
</html>
"""
(ROOT / "docs" / "index.html").write_text(page)
print("wrote docs/index.html")
