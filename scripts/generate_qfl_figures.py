#!/usr/bin/env python3
"""
generate_qfl_figures.py — figures for the fused "DP + Quantum FL" paper.

Headline: the private-FL client-count crossover is governed by model dimension d. A tiny-d VQC
incurs far lower DP cost than a large-d CNN at matched client count, under the SAME DP mechanism
and SAME calibrated sigma — demonstrated (a) across a matched VQC-vs-CNN run on the SAME task and
(b) controllably via a d-sweep.

Two comparison modes:
  --matched  : overlay a VQC json and a CNN json produced by _qfl_mnist.py / _qfl_cnn.py on the
               SAME class subset / partition / sigma (the clean, apples-to-apples contrast).
  --vs-cnn10 : overlay the 10-class VQC run against the project's original 262k-CNN MNIST result
               JSONs (cross-task anchor; messier, kept for reference).

Produces (suffixed by --tag):
  figures/qfl_crossover_vs_N_<tag>.png    — VQC vs CNN, peak acc + DP cost vs N
  figures/qfl_d_governs_crossover.png      — DP cost vs d at fixed N (the mechanism panel)

Usage:
  python3 scripts/generate_qfl_figures.py --matched \
      --vqc experiment_results/qfl_mnist_5class_eps8.0.json \
      --cnn experiment_results/qfl_cnn_5class_eps8.0.json --tag 5class_eps8
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
ap.add_argument("--matched", action="store_true", help="overlay a VQC json vs a CNN json (same task)")
ap.add_argument("--vs-cnn10", action="store_true", help="overlay 10-class VQC vs project 262k-CNN JSONs")
ap.add_argument("--vqc", default="experiment_results/qfl_mnist_eps8.0.json")
ap.add_argument("--cnn", default="experiment_results/qfl_cnn_5class_eps8.0.json")
ap.add_argument("--tag", default="8.0")
ap.add_argument("--seed", default="42")
ap.add_argument("--eps", default="8.0")
ap.add_argument("--Ns", default="10,50,100,200")
A = ap.parse_args()
if not (A.matched or getattr(A, "vs_cnn10")):
    A.matched = True   # default
NS = [int(x) for x in A.Ns.split(",")]
SEED, EPS = A.seed, A.eps

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    pass
plt.rcParams.update({"savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.4,
                     "lines.linewidth": 2.0, "font.size": 11})


def load_qfl(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {}, {}
    j = json.load(open(p))
    return j.get("meta", {}), j.get("runs", {})


def peak(runs, cfg, N):
    r = runs.get(f"{cfg}_N{N}")
    return r["best"] if r else None


# project 262k-CNN MNIST JSONs (for --vs-cnn10)
def cnn10_peak(cfg, N):
    stem = {"no-dp": "no-dp", "distributed": "fixed-dp_secagg_distributed",
            "local": "fixed-dp_secagg_local"}[cfg]
    name = (f"results_mnist_{stem}_seed{SEED}_eps{EPS}.json" if N == 10
            else f"results_mnist_{stem}_n{N}_seed{SEED}_eps{EPS}.json")
    p = DATA_DIR / name
    if not p.exists():
        return None
    return max(r["val_accuracy"] for r in json.load(open(p))["rounds"])


NAME = {"distributed": "Distributed Skellam (Fix A)", "no-dp": "No-DP ceiling",
        "local": "Local Skellam (naive)"}


def fig_crossover(vmeta, vruns, cmeta, cruns, cnn10):
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(13, 5.2))
    dv = vmeta.get("d", "?")
    dc = cmeta.get("d", "262k") if not cnn10 else "262k"
    nclass = len(vmeta.get("classes", range(10)))
    rand = 1.0 / max(nclass, 1)

    def cnn_get(cfg, N):
        return cnn10_peak(cfg, N) if cnn10 else peak(cruns, cfg, N)

    ax.axhline(rand, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax.text(NS[0], rand + 0.005, f"Random ({rand*100:.0f}%)", color="gray", fontsize=9, va="bottom")
    for cfg, ls, mk in [("no-dp", ":", "o"), ("distributed", "-", "^"), ("local", "--", "P")]:
        vp = [(N, peak(vruns, cfg, N)) for N in NS if peak(vruns, cfg, N) is not None]
        cp = [(N, cnn_get(cfg, N)) for N in NS if cnn_get(cfg, N) is not None]
        if vp:
            xs, ys = zip(*vp)
            ax.plot(xs, ys, color="#1b7837", ls=ls, marker=mk, ms=8, alpha=0.9,
                    label=f"VQC (d≈{dv}) — {NAME[cfg]}")
        if cp:
            xs, ys = zip(*cp)
            ax.plot(xs, ys, color="#b2182b", ls=ls, marker=mk, ms=8, alpha=0.9,
                    label=f"CNN (d≈{dc}) — {NAME[cfg]}")
    ax.set_xscale("log"); ax.set_xticks(NS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Number of clients N (full participation)")
    ax.set_ylabel("Peak validation accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Crossover: VQC (tiny d) vs CNN (large d)\n({nclass}-class MNIST, ε={EPS})")
    ax.legend(loc="upper right", framealpha=0.93, fontsize=7.5)

    # DP cost vs N panel
    axc.axhline(0, color="gray", lw=0.8)
    vpc = [(N, peak(vruns, "no-dp", N) - peak(vruns, "distributed", N)) for N in NS
           if peak(vruns, "no-dp", N) is not None and peak(vruns, "distributed", N) is not None]
    cpc = [(N, cnn_get("no-dp", N) - cnn_get("distributed", N)) for N in NS
           if cnn_get("no-dp", N) is not None and cnn_get("distributed", N) is not None]
    if vpc:
        xs, ys = zip(*vpc); axc.plot(xs, [y * 100 for y in ys], color="#1b7837", marker="^",
                                     ms=8, label=f"VQC (d≈{dv})")
    if cpc:
        xs, ys = zip(*cpc); axc.plot(xs, [y * 100 for y in ys], color="#b2182b", marker="s",
                                     ms=8, label=f"CNN (d≈{dc})")
    axc.set_xscale("log"); axc.set_xticks(NS)
    axc.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    axc.set_xlabel("Number of clients N")
    axc.set_ylabel("DP cost: (no-dp − distributed)  [pp]")
    axc.set_title(f"DP cost is far lower for the tiny-d VQC\n(largest gap at small N — cross-silo)")
    axc.legend(loc="upper right", framealpha=0.93, fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / f"qfl_crossover_vs_N_{A.tag}.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p.relative_to(ROOT)}")


def fig_d_governs(vmeta, vruns, cmeta, cruns):
    pts = []  # (d, dp_cost_pp, label)
    sweep_p = DATA_DIR / "qfl_dsweep.json"
    if sweep_p.exists():
        for row in json.load(open(sweep_p)).get("rows", []):
            pts.append((row["d"], row["dp_cost"] * 100, None))
    for meta, runs, lab in [(vmeta, vruns, f"VQC (d≈{vmeta.get('d','?')})"),
                            (cmeta, cruns, f"CNN (d≈{cmeta.get('d','?')})")]:
        nd, di = peak(runs, "no-dp", 10), peak(runs, "distributed", 10)
        if nd is not None and di is not None:
            pts.append((meta.get("d", 0), (nd - di) * 100, lab))
    if not pts:
        print("  [skip] d_governs: no data"); return
    pts.sort()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#762a83", marker="o", ms=7)
    for d, y, lab in pts:
        if lab:
            ax.annotate(lab, (d, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Model dimension d (trainable parameters)")
    ax.set_ylabel("DP cost at N=10: (no-dp − distributed)  [pp]")
    ax.set_title(f"DP cost grows monotonically with d at fixed N (ε={EPS}, N=10)")
    fig.tight_layout()
    p = FIG_DIR / "qfl_d_governs_crossover.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {p.relative_to(ROOT)}")


def main():
    vmeta, vruns = load_qfl(A.vqc)
    cnn10 = bool(getattr(A, "vs_cnn10"))
    cmeta, cruns = ({}, {}) if cnn10 else load_qfl(A.cnn)
    if not vruns:
        print(f"[warn] no VQC results at {A.vqc}")
    print(f"\nVQC vs CNN peak accuracy (tag={A.tag}, mode={'cnn10' if cnn10 else 'matched'}):")
    print(f"{'config':<22}" + "".join(f"N={N:<8}" for N in NS))
    cget = (lambda c, N: cnn10_peak(c, N)) if cnn10 else (lambda c, N: peak(cruns, c, N))
    for cfg in ("no-dp", "distributed", "local"):
        vr = "".join((f"{(peak(vruns,cfg,N) or 0)*100:>5.1f}%   " if peak(vruns,cfg,N) is not None
                      else f"{'--':>8} ") for N in NS)
        cr = "".join((f"{(cget(cfg,N) or 0)*100:>5.1f}%   " if cget(cfg,N) is not None
                      else f"{'--':>8} ") for N in NS)
        print(f"VQC {cfg:<18}{vr}")
        print(f"CNN {cfg:<18}{cr}")
    print()
    fig_crossover(vmeta, vruns, cmeta, cruns, cnn10)
    fig_d_governs(vmeta, vruns, cmeta, cruns)


if __name__ == "__main__":
    main()
