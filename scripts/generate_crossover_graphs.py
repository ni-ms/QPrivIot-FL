#!/usr/bin/env python3
"""
generate_crossover_graphs.py — figures for the QPrivIoT-FL "client-count crossover" paper.

Central result: distributed Skellam DP under SecAgg reaches usable accuracy only in a
client-count WINDOW (N≈50-100); every naive correctly-noised baseline (per-client float
noise = fixed-dp-no-SecAgg; per-client integer noise = local Skellam) is broken at every N;
and the no-dp ceiling itself decays with N (data starvation), bounding the window from above.

Produces:
  figures/crossover_peakacc_vs_N.png   — peak val-acc vs N, one line per config (THE figure)
  figures/crossover_trajectories.png   — 2x2 small multiples, val-acc vs round at each N
  figures/crossover_dp_cost.png        — DP cost (no-dp ceiling − distributed) vs N

Usage: python3 scripts/generate_crossover_graphs.py [--eps 8.0] [--seed 42] [--dataset mnist]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "experiment_results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="mnist")
ap.add_argument("--eps", default="8.0")
ap.add_argument("--seed", default="42")
ap.add_argument("--Ns", default="10,50,100,200")
ARGS = ap.parse_args()
DS, EPS, SEED = ARGS.dataset, ARGS.eps, ARGS.seed
NS = [int(x) for x in ARGS.Ns.split(",")]

# config key -> (label, color, marker, filename builder).  Legacy N=10 runs have no nXXX suffix.
def fname(stem_no10, stem_nN, N):
    """stem_no10: filename for N=10 (legacy); stem_nN: format-string with {N} for N>10."""
    return DATA_DIR / (stem_no10 if N == 10 else stem_nN.format(N=N))

CONFIGS = {
    "no-dp": dict(
        label="No-DP ceiling", color="#2166ac", marker="o",
        f=lambda N: fname(f"results_{DS}_no-dp_seed{SEED}_eps{EPS}.json",
                          f"results_{DS}_no-dp_n{{N}}_seed{SEED}_eps{EPS}.json", N)),
    "distributed": dict(
        label="Distributed Skellam + SecAgg (Fix A)", color="#4dac26", marker="^",
        f=lambda N: fname(f"results_{DS}_fixed-dp_secagg_distributed_seed{SEED}_eps{EPS}.json",
                          f"results_{DS}_fixed-dp_secagg_distributed_n{{N}}_seed{SEED}_eps{EPS}.json", N)),
    "fixed-dp": dict(
        label="Fixed-DP, no SecAgg (corrected σ)", color="#d6604d", marker="s",
        f=lambda N: fname(f"results_{DS}_fixed-dp_seed{SEED}_eps{EPS}.json",
                          f"results_{DS}_fixed-dp_n{{N}}_seed{SEED}_eps{EPS}.json", N)),
    "local": dict(
        label="Local Skellam (per-client, SecAgg)", color="#f1a340", marker="P",
        f=lambda N: fname(f"results_{DS}_fixed-dp_secagg_local_seed{SEED}_eps{EPS}.json",
                          f"results_{DS}_fixed-dp_secagg_local_n{{N}}_seed{SEED}_eps{EPS}.json", N)),
}
ORDER = ["no-dp", "distributed", "fixed-dp", "local"]

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    pass
plt.rcParams.update({"savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.4,
                     "lines.linewidth": 2.0, "font.size": 11})


def load_traj(path):
    if not path.exists():
        return None
    rounds = json.load(open(path))["rounds"]
    xs = [r["round"] for r in rounds]
    ys = [r["val_accuracy"] for r in rounds]
    return xs, ys


def gather():
    """Return {cfg: {N: (xs, ys, peak)}} and report any missing/short runs."""
    data, warns = {}, []
    for cfg in ORDER:
        data[cfg] = {}
        for N in NS:
            t = load_traj(CONFIGS[cfg]["f"](N))
            if t is None:
                warns.append(f"  [missing] {cfg} N={N}: {CONFIGS[cfg]['f'](N).name}")
                continue
            xs, ys = t
            if len(xs) < 20:
                warns.append(f"  [short]   {cfg} N={N}: only {len(xs)} rounds")
            data[cfg][N] = (xs, ys, max(ys))
    return data, warns


def fig_crossover(data):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(0.10, color="gray", ls="--", lw=1.2, alpha=0.7)
    ax.text(NS[0], 0.105, "Random (10%)", color="gray", fontsize=9, va="bottom")
    for cfg in ORDER:
        c = CONFIGS[cfg]
        pts = [(N, data[cfg][N][2]) for N in NS if N in data[cfg]]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=c["color"], marker=c["marker"], markersize=8, label=c["label"])
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Number of clients N (full participation)")
    ax.set_ylabel("Peak validation accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{DS.upper()} (ε={EPS}, T=20): private FL has a usable client-count window")
    ax.legend(loc="center left", framealpha=0.92, fontsize=9)
    # shade the usable window
    ax.axvspan(50, 100, color="green", alpha=0.06)
    fig.tight_layout()
    p = FIG_DIR / "crossover_peakacc_vs_N.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p.relative_to(ROOT)}")


def fig_trajectories(data):
    n = len(NS)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.5 * nrow), sharex=True, sharey=True,
                             squeeze=False)
    for idx, N in enumerate(NS):
        ax = axes[idx // ncol][idx % ncol]
        ax.axhline(0.10, color="gray", ls="--", lw=1.0, alpha=0.6)
        for cfg in ORDER:
            if N not in data[cfg]:
                continue
            xs, ys, peak = data[cfg][N]
            c = CONFIGS[cfg]
            ax.plot(xs, ys, color=c["color"], marker=c["marker"], markevery=4, markersize=5,
                    label=f"{c['label']} ({peak*100:.0f}%)")
        ax.set_title(f"N = {N}")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
        if idx // ncol == nrow - 1:
            ax.set_xlabel("Communication round")
        if idx % ncol == 0:
            ax.set_ylabel("Validation accuracy")
    # hide any unused panels
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{DS.upper()} (ε={EPS}): distributed-Skellam learns only in the N≈50–100 window; "
                 f"naive DP never does", y=1.005, fontsize=12)
    fig.tight_layout()
    p = FIG_DIR / "crossover_trajectories.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p.relative_to(ROOT)}")


def fig_dp_cost(data):
    pts = [(N, data["no-dp"][N][2] - data["distributed"][N][2])
           for N in NS if N in data["no-dp"] and N in data["distributed"]]
    if not pts:
        print("  [skip] dp_cost: missing no-dp/distributed pairs")
        return
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([str(x) for x in xs], [y * 100 for y in ys], color="#762a83", alpha=0.85)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{y*100:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Number of clients N")
    ax.set_ylabel("DP cost: (no-dp ceiling − distributed)  [pp]")
    ax.set_title(f"{DS.upper()} (ε={EPS}): DP cost of Fix A is minimized at N≈100")
    fig.tight_layout()
    p = FIG_DIR / "crossover_dp_cost.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p.relative_to(ROOT)}")


def main():
    data, warns = gather()
    if warns:
        print("Data coverage warnings:")
        print("\n".join(warns))
    print(f"\nPeak accuracy table (ε={EPS}):")
    print(f"{'config':<38}" + "".join(f"N={N:<8}" for N in NS))
    for cfg in ORDER:
        row = "".join((f"{data[cfg][N][2]*100:>5.1f}%   " if N in data[cfg] else f"{'--':>8} ")
                      for N in NS)
        print(f"{CONFIGS[cfg]['label']:<38}{row}")
    print()
    fig_crossover(data)
    fig_trajectories(data)
    fig_dp_cost(data)


if __name__ == "__main__":
    main()