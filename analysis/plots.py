"""Figures for the write-up. One restrained palette (Okabe-Ito, colour-blind safe), direct labels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, RED, GREY, INK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8f98", "#1f2328"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8ebef", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left",
})
ARM_COLOR = {"Mens E-Mail": BLUE, "Womens E-Mail": ORANGE}


def _save(fig, out: Path, name: str):
    fig.savefig(out / name, facecolor="white")
    plt.close(fig)


def ab_effects(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    titles = {"visit": "Site visit rate (pp lift)", "conversion": "Purchase rate (pp lift)", "spend": "Spend per customer ($ lift)"}
    for ax, m in zip(axes, ["visit", "conversion", "spend"]):
        rows = [r for r in R["ab"] if r["metric"] == m]
        scale = 100 if m != "spend" else 1
        for i, r in enumerate(rows):
            c = ARM_COLOR[r["arm"]]
            ax.errorbar(r["estimate"] * scale, i, xerr=[[ (r["estimate"] - r["ci_low"]) * scale], [(r["ci_high"] - r["estimate"]) * scale]],
                        fmt="o", color=c, capsize=4, lw=2, ms=6)
            ax.text(r["estimate"] * scale, i + 0.28, f"{r['estimate'] * scale:+.2f}", ha="center", color=c, fontsize=9, fontweight="bold")
        ax.axvline(0, color=GREY, lw=1)
        ax.set_yticks(range(2), [r["arm"].replace(" E-Mail", "'s") + " e-mail" for r in rows])
        ax.set_ylim(-0.6, 1.7)
        ax.set_title(titles[m])
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig_ab.png")


def cuped_sim(R, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    s = pd.DataFrame(R["cuped_sim"])
    g = np.linspace(0, 1, 50)
    a.plot(g, g**2, color=GREY, lw=1.5, ls="--", label=r"theory: $\rho^2$")
    a.plot(s.rho, s.variance_reduction, "o-", color=BLUE, lw=2, label="simulated")
    for _, r in pd.DataFrame(R["cuped"]).query("method=='All pre-treatment covariates'").iterrows():
        pass
    hs = [r for r in R["cuped"] if r["method"] == "All pre-treatment covariates"]
    for r in hs:
        a.plot(r["corr"], r["variance_reduction"], "s", color=RED, ms=7, zorder=5)
    a.annotate("Hillstrom (visit, spend):\nvariance cut by only 0.1-3%", xy=(0.16, 0.025), xytext=(0.3, 0.02), color=RED,
              fontsize=9, arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    a.set_xlabel(r"correlation $\rho$ between pre-experiment covariate and metric")
    a.set_ylabel("variance reduction")
    a.set_title("CUPED removes ρ² of the variance")
    a.legend(frameon=False, loc="upper left")
    b.plot(s.rho, s.coverage_raw, "o-", color=GREY, lw=1.8, label="raw")
    b.plot(s.rho, s.coverage_cuped, "o-", color=BLUE, lw=1.8, label="CUPED")
    b.axhline(0.95, color=RED, lw=1.2, ls="--")
    b.text(0.9, 0.953, "nominal 95%", color=RED, ha="right", fontsize=8.5)
    b.set_ylim(0.90, 1.0)
    b.set_xlabel(r"$\rho$")
    b.set_ylabel("share of CIs containing the true effect")
    b.set_title("...and keeps CI coverage honest")
    b.legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    _save(fig, out, "fig_cuped.png")


def power_check(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    df = pd.DataFrame(R["power_check"])
    for ax, (m, title) in zip(axes, [("visit", "Site visits (Mens e-mail vs control)"), ("spend", "Spend per customer (Mens e-mail vs control)")]):
        d = df[df.metric == m]
        ax.plot(d.n_per_arm, d.theoretical, "-", color=BLUE, lw=2, label="formula")
        ax.plot(d.n_per_arm, d.empirical, "o", color=ORANGE, ms=7, label="re-sampled real data")
        ax.axhline(0.8, color=GREY, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(d.n_per_arm, [f"{n:,}" for n in d.n_per_arm], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("customers per arm")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("power (share of tests significant)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, out, "fig_power.png")


def qini(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    palette = {"S-learner": GREEN, "T-learner": ORANGE, "X-learner": BLUE, "T-learner (logistic)": RED,
               "X-learner on visits (surrogate)": "#CC79A7"}
    for ax, outcome, label in zip(axes, ("visit", "spend"), ("extra site visits", "extra revenue ($)")):
        curves = R["uplift"][outcome]["qini_curves"]
        first = next(iter(curves.values()))
        n = first["n_total"][0]
        ax.plot(first["fraction"], np.array(first["random"]) / n, color=GREY, ls="--", lw=1.5)
        for name, c in curves.items():
            ax.plot(c["fraction"], np.array(c["qini"]) / n, color=palette[name], lw=2.6 if name == R["uplift"][outcome]["primary"] else 1.2,
                    label=name, zorder=3 if name == R["uplift"][outcome]["primary"] else 2)
        ax.set_xlabel("share of customers e-mailed (highest predicted uplift first)")
        ax.set_ylabel(f"{label} per customer in population")
        ax.set_title(f"Qini curve: {'site visits' if outcome == 'visit' else 'spend'}")
        ax.axhline(0, color=INK, lw=0.6)
        ax.text(0.5, 0.0, "random targeting", color=GREY, fontsize=8.5, transform=ax.transAxes, ha="center", va="bottom")
    for ax in axes:
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig_qini.png")


def deciles(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, outcome, scale, unit in zip(axes, ("visit", "spend"), (100, 1), ("pp", "$")):
        d = pd.DataFrame(R["uplift"][outcome]["deciles"])
        ate = R["uplift"][outcome]["ate_ipw"] * scale
        ax.bar(d.group, d.observed_uplift * scale, color=BLUE, width=0.75, yerr=1.96 * d.se * scale, capsize=3, ecolor=GREY)
        ax.axhline(ate, color=RED, lw=1.3, ls="--")
        ax.text(10.4, ate, "average\neffect", color=RED, fontsize=8.5, va="center")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.4, 11.6)
        ax.set_xlabel("decile of predicted uplift (1 = most responsive)")
        ax.set_ylabel(f"observed lift ({unit})")
        ax.set_title(f"Measured lift by model decile: {'visits' if outcome == 'visit' else 'spend'}")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig_deciles.png")


def targeting(R, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    d = pd.DataFrame(R["uplift"]["spend"]["targeting"])
    cols = {0.0: BLUE, 0.25: ORANGE, 0.5: RED}
    for c, g in d.groupby("cost"):
        ax.plot(g.fraction, g.profit, color=cols[c], lw=2.2)
        i = g.profit.idxmax()
        ax.plot(g.loc[i, "fraction"], g.loc[i, "profit"], "o", color=cols[c], ms=8, zorder=4, mec="white")
        ax.text(1.01, g.profit.iloc[-1], f"${c:.2f} / e-mail", color=cols[c], va="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of customers e-mailed (ranked by predicted uplift)")
    ax.set_ylabel("net incremental revenue per customer ($)")
    ax.set_title("Where to stop: profit-maximising audience (dots)")
    fig.subplots_adjust(right=0.8)
    _save(fig, out, "fig_targeting.png")


def policy(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, outcome, unit in zip(axes, ("visit", "spend"), ("visit rate", "spend per customer ($)")):
        p = {k: v for k, v in R["policy"][outcome].items() if not k.startswith("_")}
        names = list(p)
        vals = np.array([p[k]["value"] for k in names])
        lo = vals - np.array([p[k]["ci"][0] for k in names])
        hi = np.array([p[k]["ci"][1] for k in names]) - vals
        colors = [GREY, BLUE, BLUE, BLUE, RED]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=colors, xerr=[lo, hi], ecolor=INK, capsize=3, height=0.65)
        ax.set_yticks(y, names, fontsize=8.5)
        ax.set_xlabel(f"expected {unit}")
        ax.set_title(f"Which e-mail to send: {'visits' if outcome == 'visit' else 'spend'}")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig_policy.png")


def make_all(R, out):
    for f in (ab_effects, cuped_sim, power_check, qini, deciles, targeting, policy):
        f(R, out)
