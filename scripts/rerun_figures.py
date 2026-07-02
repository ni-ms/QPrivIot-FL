"""
Regenerate the paper figures from the multi-seed rerun JSON.
Writes PNG + PDF to figures/. Read-only over experiment_results/rerun_grid/*.json.

  .venv/bin/python scripts/rerun_figures.py

Figures:
  fig_utility_vs_eps  — LongMemEval oracle evidence-recall@5 vs eps, per K (utility holds at tiny-K)
  fig_leakage_drop    — MIA tail-AUC vs eps (leakage collapses to chance under DP)
  fig_d_crossover     — real-embedding d-sweep: DP utility-retention AND pre-DP tail-AUC vs d
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "experiment_results" / "rerun_grid"
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


def load():
    return {p.stem: json.load(open(p)) for p in sorted(GRID.glob("*.json"))
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


def fig_crossover(data):
    """Real sentence-embeddings: as proj-dim d grows, pre-DP tail-AUC rises (more leakage)
    AND DP utility-retention falls -> tiny-d favored on both axes."""
    util = {d["config"]["d"]: d for _, d in data.items()
            if d["script"] == "agentmem_probe" and d["embedder"] == "st"}
    leak = {d["config"]["d"]: d for _, d in data.items()
            if d["script"] == "agentmem_leakage" and d["embedder"] == "st"}
    ds = sorted(set(util) & set(leak))
    if not ds:
        print("  (skip fig_d_crossover: no st-embedder configs yet)")
        return
    ret, tail = [], []
    for dd in ds:
        rows = util[dd]["rows"]
        clean = _val(rows, "inf (clean)", "topic_mean")
        e8 = _val(rows, "eps=8", "topic_mean")
        ret.append(100 * e8 / clean if clean else np.nan)
        tail.append(_val(leak[dd]["rows"], "inf (clean)", "lc_auc_mean"))
    fig, ax1 = plt.subplots(figsize=(6, 4.2))
    ax2 = ax1.twinx()
    l1 = ax1.plot(ds, ret, "o-", color="tab:blue", label="DP utility-retention @ e=8 (%)")
    l2 = ax2.plot(ds, tail, "s--", color="tab:red", label="pre-DP tail-AUC (leakage)")
    ax1.set_xlabel("projection dim d"); ax1.set_ylabel("retention @ e=8 (%)", color="tab:blue")
    ax2.set_ylabel("clean tail-AUC", color="tab:red")
    ax1.set_xscale("log", base=2); ax1.set_xticks(ds); ax1.set_xticklabels([str(x) for x in ds])
    ax1.set_title("Real embeddings: tiny-d wins on both axes")
    lns = l1 + l2
    ax1.legend(lns, [x.get_label() for x in lns], fontsize=8, loc="center right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_d_crossover.{ext}", dpi=150)
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
    data = load()
    if not data:
        print("no JSON in", GRID)
        return
    fig_utility(data)
    fig_leakage(data)
    fig_crossover(data)
    fig_distilled_utility(data)
    fig_distilled_leakage(data)
    print("wrote figures ->", FIG)
    for p in sorted(FIG.glob("fig_*")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
