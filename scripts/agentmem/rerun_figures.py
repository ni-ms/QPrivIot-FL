"""
Regenerate the paper figures from the multi-seed rerun JSON.
Writes PNG + PDF to figures/. Read-only over experiment_results/<grid>/*.json.

  .venv/bin/python scripts/agentmem/rerun_figures.py --grid rerun_grid_rp

Figures:
  fig_utility_vs_eps       — LongMemEval oracle evidence-recall@5 vs eps, per K
  fig_leakage_drop         — MIA tail-AUC vs eps (leakage collapses to chance under DP)
  fig_projection_ablation  — the d-"crossover" is an artifact of the data-dependent PCA
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "experiment_results" / "rerun_grid_rp"   # headline grid; overridden by --grid
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

EPS_ORDER = ["inf (clean)", "inf", "eps=16", "eps=8", "eps=3", "eps=1"]
EPS_X = {"inf (clean)": 64.0, "inf": 64.0, "eps=16": 16.0, "eps=8": 8.0, "eps=3": 3.0, "eps=1": 1.0}


def _val(rows, label, key):
    """Fetch a metric; clean point is 'inf (clean)' or 'inf' depending on script."""
    labels = ["inf (clean)", "inf"] if label == "inf (clean)" else [label]
    for want in labels:
        for r in rows:
            if r["label"] == want and r.get(key) is not None:
                return r[key]
    return None


def load(grid=None):
    g = Path(grid) if grid else GRID
    if not g.is_absolute():
        g = ROOT / "experiment_results" / g
    if not g.exists():
        return {}
    return {p.stem: json.load(open(p)) for p in sorted(g.glob("*.json"))
            if not p.stem.endswith("_TABLE")}


def _series(rows, key):
    xs, ys, es = [], [], []
    for lab in EPS_ORDER:
        for r in rows:
            if r["label"] == lab and r.get(key) is not None and not np.isnan(r[key]):
                xs.append(EPS_X[lab]); ys.append(r[key]); es.append(r.get(key.replace("mean", "std"), 0) or 0)
    return np.array(xs), np.array(ys), np.array(es)


def fig_utility(data):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    items = sorted([(d["config"]["K"], tag, d) for tag, d in data.items()
                    if d["script"] == "longmemeval_probe" and d["config"]["d"] == 32])
    for K, tag, d in items:
        x, y, e = _series(d["rows"], "mean")
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=f"K={K}")
        ax.axhline(d["chance"], color="gray", ls=":", lw=0.6)
    ax.set_xscale("log"); ax.set_xticks([1, 3, 8, 16, 64])
    ax.set_xticklabels(["1", "3", "8", "16", "clean"])
    ax.set_xlabel("privacy budget epsilon (Skellam RDP)"); ax.set_ylabel("evidence-recall@5")
    ax.set_title("LongMemEval oracle: DP utility holds at tiny-K")
    ax.legend(title="buckets", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_utility_vs_eps.{ext}", dpi=150)
    plt.close(fig)


def fig_leakage(data):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    items = [(tag, d) for tag, d in data.items()
             if d["script"] == "longmemeval_leakage"]
    for tag, d in sorted(items, key=lambda t: t[1]["config"]["K"]):
        x, y, e = _series(d["rows"], "lc_auc_mean")
        ax.errorbar(x, y, yerr=e, marker="s", capsize=3, label=f"K={d['config']['K']}")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance (no leakage)")
    ax.set_xscale("log"); ax.set_xticks([1, 3, 8, 16, 64])
    ax.set_xticklabels(["1", "3", "8", "16", "clean"])
    ax.set_xlabel("privacy budget epsilon (Skellam RDP)"); ax.set_ylabel("MIA tail-AUC (low-count)")
    ax.set_title("LongMemEval oracle: low-count leakage collapses under DP")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_leakage_drop.{ext}", dpi=150)
    plt.close(fig)


# Categorical slots 1/2/3 of the validated palette (worst adjacent CVD dE 21.6, tritan).
# Contrast of slots 2-3 vs a light surface is < 3:1, so identity is ALSO carried by
# distinct markers + direct end-labels, never by hue alone.
PROJ_STYLE = [
    ("data-PCA (leaky)", "rerun_grid_abl_pca", "#2a78d6", "o", "-"),
    ("randproj (free)",  "rerun_grid_rp",      "#1baf7a", "s", "-"),
    ("publicPCA (free)", "rerun_grid_abl_pp",  "#eda100", "^", "--"),
]


def _st_series(data, key_probe, key_leak):
    """(ds, private-utility, clean tail-AUC) over the st-embedder d-sweep of one grid."""
    util = {v["config"]["d"]: v for v in data.values()
            if v.get("script") == "agentmem_probe" and v.get("embedder") == "st"}
    leak = {v["config"]["d"]: v for v in data.values()
            if v.get("script") == "agentmem_leakage" and v.get("embedder") == "st"}
    ds = sorted(set(util) & set(leak))
    priv = [_val(util[d]["rows"], "eps=8", key_probe) for d in ds]
    tail = [_val(leak[d]["rows"], "inf (clean)", key_leak) for d in ds]
    return ds, priv, tail


def fig_projection_ablation(_unused=None):
    """The d-'crossover' is an artifact of fitting the projection on user notes.

    Left  : PRIVATE utility vs d. Under data-PCA tiny-d wins; under either
            data-INDEPENDENT projection d=384 (no projection) wins.
    Right : pre-DP leakage vs d. Data-PCA shows a rising gradient; the honest
            projections are saturated. Post-DP, every d sits at chance.
    Two panels, one measure each -- never a dual axis.
    """
    series = []
    for name, grid, colour, marker, ls in PROJ_STYLE:
        d = load(grid)
        if not d:
            continue
        ds, priv, tail = _st_series(d, "topic_mean", "lc_auc_mean")
        if ds:
            series.append((name, ds, priv, tail, colour, marker, ls))
    if not series:
        print("  (skip fig_projection_ablation: no st-embedder configs)")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.9))
    # Direct-label at the LEFT end, where the three series are well separated; they
    # converge by construction at d=384 (identity projection), so labelling there collides.
    dy = {"data-PCA (leaky)": 9, "randproj (free)": 9, "publicPCA (free)": -13}
    for name, ds, priv, tail, colour, marker, ls in series:
        for ax, ys in ((axL, priv), (axR, tail)):
            ax.plot(ds, ys, marker=marker, linestyle=ls, color=colour, lw=2.0,
                    ms=6.5, mec="white", mew=1.0, label=name, zorder=3)
        axL.annotate(name, (ds[0], priv[0]), textcoords="offset points",
                     xytext=(6, dy.get(name, 8)), fontsize=7.5, color=colour, va="center")

    # post-DP floor: DP pins leakage at chance for EVERY d and every projection
    axR.axhspan(0.50, 0.57, color="0.85", alpha=0.6, zorder=0)
    axR.text(38, 0.535, "post-DP (all projections, all d)", fontsize=7, color="0.35")

    for ax in (axL, axR):
        ax.set_xscale("log", base=2)
        ax.set_xticks(series[0][1]); ax.set_xticklabels([str(x) for x in series[0][1]])
        ax.set_xlabel("projection dimension d")
        ax.grid(alpha=0.25, zorder=0)
        ax.axvline(384, color="0.6", ls=":", lw=1.2, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

    axL.set_ylabel("private utility (topic-acc @ epsilon~9.3)")
    axL.set_title("(a) tiny-d wins only under the leaky projection", fontsize=9.5)
    axL.set_xlim(right=520)
    axL.margins(y=0.14)
    axR.set_ylabel("pre-DP leakage (clean tail-AUC)")
    axR.set_title("(b) the leakage gradient is an artifact too", fontsize=9.5)
    axR.set_ylim(0.48, 1.02)
    axR.legend(fontsize=7.5, loc="center right", frameon=False)
    axR.text(384, 0.487, "identity proj.", fontsize=6.5, color="0.45", ha="center", va="bottom")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_projection_ablation.{ext}", dpi=150)
    plt.close(fig)


def fig_distilled_utility(data):
    items = [(tag, d) for tag, d in data.items() if d["script"] == "distilled_utility"]
    if not items:
        print("  (skip fig_distilled_utility: no distilled_utility configs)")
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for tag, d in sorted(items, key=lambda t: (("full" not in t[0]), t[1]["config"]["K"])):
        x, y, e = _series(d["rows"], "mean")
        scale = "500u" if "full" in tag else "100u"
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3,
                    label=f"{scale} K={d['config']['K']}")
    ax.axhline(next(iter(items))[1]["chance"], color="gray", ls=":", lw=0.8, label="chance")
    ax.set_xscale("log"); ax.set_xticks([3, 8, 16, 64]); ax.set_xticklabels(["3", "8", "16", "clean"])
    ax.set_xlabel("privacy budget epsilon (Skellam RDP)"); ax.set_ylabel("answer-recall@5")
    ax.set_title("Distilled-notes utility: density lifts DP retention (500u vs 100u)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_distilled_utility.{ext}", dpi=150)
    plt.close(fig)


def fig_distilled_leakage(data):
    """Raw vs distilled clean tail-AUC + DP collapse — distilled leaks MORE, DP kills both."""
    blocks = [d for d in data.values() if d["script"] == "distilled_leakage"]
    if not blocks:
        print("  (skip fig_distilled_leakage: no distilled_leakage configs)")
        return
    # representative config: largest-scale, largest K
    d = sorted(blocks, key=lambda b: (b["config"].get("distilled", "").count("pilot"),
                                      -b["config"]["K"]))[0]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for src, style in (("raw", dict(marker="o", color="tab:green")),
                       ("distilled", dict(marker="s", color="tab:purple"))):
        rows = d["sources"][src]["rows"]
        x, y, e = _series(rows, "lc_auc_mean")
        ax.errorbar(x, y, yerr=e, capsize=3, label=f"{src} (tail-AUC)", **style)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.set_xscale("log"); ax.set_xticks([3, 8, 16, 64]); ax.set_xticklabels(["3", "8", "16", "clean"])
    ax.set_xlabel("privacy budget epsilon (Skellam RDP)"); ax.set_ylabel("MIA tail-AUC (low-count)")
    ax.set_title(f"Distilled notes leak MORE than raw turns, DP kills both (K={d['config']['K']})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_distilled_leakage.{ext}", dpi=150)
    plt.close(fig)


def main():
    global GRID
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="rerun_grid_rp",
                    help="experiment_results/<grid> to draw the headline figures from")
    a = ap.parse_args()
    GRID = ROOT / "experiment_results" / a.grid
    data = load()
    if not data:
        print("no JSON in", GRID)
        return
    fig_utility(data)
    fig_leakage(data)
    fig_projection_ablation()
    fig_distilled_utility(data)
    fig_distilled_leakage(data)
    print("wrote figures ->", FIG)
    for p in sorted(FIG.glob("fig_*")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
