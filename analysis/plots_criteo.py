"""Figures for the Criteo page (same visual system as plots.py)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots import BLUE, GREEN, GREY, INK, ORANGE, RED, _save

PINK = "#CC79A7"


def balance(R, out):
    b = pd.DataFrame(R["balance"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(b))
    ax.axvline(0, color=INK, lw=0.8)
    ax.hlines(y, b.smd_raw, b.smd_weighted, color="#c9ced6", lw=1.2, zorder=1)
    ax.scatter(b.smd_raw, y, color=RED, s=46, zorder=3, label="raw (as randomized)")
    ax.scatter(b.smd_weighted, y, color=BLUE, s=46, zorder=3, label="after propensity weighting")
    ax.set_yticks(y, b.feature)
    ax.set_xlabel("standardized mean difference, treated − control")
    ax.set_title("Arms differ on the features; weighting fixes it")
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.065, 0.065)
    _save(fig, out, "fig_balance.png")


def estimates(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    cols = {"Naive": RED, "CUPED,": GREY, "CUPAC": GREY, "IPW": BLUE, "AIPW": GREEN}
    for ax, m, title in zip(axes, ("visit", "conversion"), ("Site visits (pp lift)", "Conversions (pp lift)")):
        rows = [r for r in R["estimates"] if r["metric"] == m][::-1]
        for i, r in enumerate(rows):
            c = next(v for k, v in cols.items() if r["method"].startswith(k))
            ax.errorbar(r["estimate"] * 100, i, xerr=1.96 * r["se"] * 100, fmt="o", color=c, capsize=3, lw=2, ms=6)
            ax.text(r["estimate"] * 100, i + 0.3, f"{r['estimate'] * 100:.3f}", ha="center", fontsize=8.5, color=c, fontweight="bold")
        ax.set_yticks(range(len(rows)), [r["method"].replace(" (estimated propensity)", "").replace(", linear in the 12 features", " (linear)") for r in rows], fontsize=8.5)
        ax.set_ylim(-0.6, len(rows) - 0.2)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig_estimates.png")


def power(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    d = pd.DataFrame(R["power_check"])
    for ax, m, title in zip(axes, ("visit", "conversion"), ("Site visits", "Conversions")):
        s = d[d.metric == m]
        ax.plot(s.n_total, s.theory, "-", color=BLUE, lw=2, label="formula (85/15 split)")
        ax.plot(s.n_total, s.empirical, "o", color=ORANGE, ms=7, label="re-sampled real data")
        ax.axhline(0.8, color=GREY, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(s.n_total, [f"{n:,}" for n in s.n_total], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("total users in the test")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("power")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, out, "fig_power.png")


def qini(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    pal = {"S-learner": GREEN, "T-learner": ORANGE, "X-learner": BLUE, "T-learner (logistic)": RED, "X-learner on visits (surrogate)": PINK}
    for ax, m, label, scale in zip(axes, ("visit", "conversion"), ("extra visits per 1,000 users", "extra conversions per 1,000 users"), (1000, 1000)):
        curves = R["uplift"]["outcomes"][m]["qini_curves"]
        first = next(iter(curves.values()))
        n = first["n_total"][0]
        ax.plot(first["fraction"], np.array(first["random"]) / n * scale, color=GREY, ls="--", lw=1.5)
        for name, c in curves.items():
            prim = name == R["uplift"]["outcomes"][m]["primary"]
            ax.plot(c["fraction"], np.array(c["qini"]) / n * scale, color=pal[name], lw=2.6 if prim else 1.2, label=name, zorder=3 if prim else 2)
        ax.set_xlabel("share of users targeted (highest predicted uplift first)")
        ax.set_ylabel(label)
        ax.set_title(f"Qini curve: {m}s")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig_qini.png")


def deciles(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, m in zip(axes, ("visit", "conversion")):
        o = R["uplift"]["outcomes"][m]
        d = pd.DataFrame(o["deciles"])
        ax.bar(d.group, d.observed_uplift * 100, color=BLUE, width=0.75, yerr=1.96 * d.se * 100, capsize=3, ecolor=GREY)
        ax.axhline(o["ate_ipw"] * 100, color=RED, lw=1.3, ls="--")
        ax.text(10.45, o["ate_ipw"] * 100, "average", color=RED, fontsize=8.5, va="center")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.4, 11.8)
        ax.set_xlabel("decile of predicted uplift (1 = most responsive)")
        ax.set_ylabel("measured lift (pp, propensity-weighted)")
        ax.set_title(f"Measured lift by model decile: {m}s")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig_deciles.png")


def targeting(R, out):
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    d = pd.DataFrame(R["uplift"]["targeting"])
    cols = {0.0: BLUE, 0.0003: ORANGE, 0.0006: RED}
    for c, g in d.groupby("cost"):
        ax.plot(g.fraction, g.profit * 1000, color=cols[c], lw=2.2)
        i = g.profit.idxmax()
        ax.plot(g.loc[i, "fraction"], g.loc[i, "profit"] * 1000, "o", color=cols[c], ms=8, mec="white", zorder=4)
        ax.text(1.01, g.profit.iloc[-1] * 1000, f"cost = {c * 1000:.1f} conv. per 1,000", color=cols[c], va="center", fontsize=8.5, fontweight="bold")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of users targeted (ranked by predicted uplift)")
    ax.set_ylabel("net incremental conversions per 1,000 users")
    ax.set_title("Where to stop targeting (dots = best audience)")
    fig.subplots_adjust(right=0.7)
    _save(fig, out, "fig_targeting.png")


def heterogeneity(R, out):
    top = R["uplift"]["feature_heterogeneity"][:4]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), sharey=True)
    for ax, h in zip(axes, top):
        b = pd.DataFrame(h["bins"])
        ax.errorbar(b.bin, b.uplift * 100, yerr=1.96 * b.se * 100, fmt="o-", color=BLUE, capsize=3, lw=1.8)
        ax.set_title(h["feature"])
        ax.set_xticks(b.bin)
        ax.set_xlabel("quintile of feature")
    axes[0].set_ylabel("conversion lift (pp)")
    fig.tight_layout()
    _save(fig, out, "fig_heterogeneity.png")


def learning(R, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    d = pd.DataFrame(R["uplift"]["learning_curve"])
    lo, hi = zip(*d.qini_ci)
    ax.fill_between(d.n_train, np.array(lo) * 1000, np.array(hi) * 1000, color=BLUE, alpha=0.18)
    ax.plot(d.n_train, d.qini * 1000, "o-", color=BLUE, lw=2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(d.n_train, [f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n // 1000}k" for n in d.n_train], fontsize=8.5)
    ax.minorticks_off()
    ax.set_xlabel("users used to train the uplift model")
    ax.set_ylabel("Qini coefficient (×1000)")
    ax.set_title("Uplift needs data: T-learner, conversions")
    _save(fig, out, "fig_learning.png")


def make_all(R, out: Path):
    for f in (balance, estimates, power, qini, deciles, targeting, heterogeneity, learning):
        f(R, out)
