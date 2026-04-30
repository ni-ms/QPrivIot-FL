"""
Generate all evaluation figures from test_artifacts/metrics_long.csv.

Outputs (300 DPI PNGs) into figures/:
  fig_learning_curves_<axis>.png   accuracy & loss vs round, faceted
  fig_communication_overhead.png   cumulative bytes per config
  fig_client_divergence.png        boxplot of per-client variance
  fig_privacy_utility.png          scatter, ε vs final accuracy
  fig_convergence_rate.png         rounds-to-target accuracy
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Publication style ─────────────────────────────────────────────────────────
sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
})

PALETTE = {
    "no-dp":    "#0173B2",
    "fixed-dp": "#DE8F05",
    "adapriv":  "#029E73",
    "device-only": "#CC78BC",
    "param-only":  "#CA9161",
    "round-only":  "#FBAFE4",
}
LABELS = {"no-dp": "No DP", "fixed-dp": "Fixed DP", "adapriv": "AdaPriv (Ours)"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _wide(df: pd.DataFrame, source_priority=("centralized", "distributed_eval",
                                              "project_json")) -> pd.DataFrame:
    """Pivot long→wide with deterministic source priority for each (run, round)."""
    src_rank = {s: i for i, s in enumerate(source_priority)}
    df = df.assign(_rank=df["source"].map(lambda s: src_rank.get(s, 99)))
    df = df.sort_values(["_rank"]).drop_duplicates(
        subset=["dataset", "config", "num_clients", "alpha", "seed",
                "epsilon", "round", "metric"])
    keys = ["dataset","config","num_clients","alpha","seed","epsilon","round"]
    return df.pivot_table(index=keys, columns="metric", values="value").reset_index()

def _ci95(x: pd.Series) -> float:
    x = x.dropna().to_numpy()
    if len(x) < 2: return 0.0
    return stats.sem(x) * stats.t.ppf(0.975, len(x) - 1)

# ── Figure 1 — Learning curves (accuracy & loss) ──────────────────────────────
def fig_learning_curves(df: pd.DataFrame, axis: str, out: pathlib.Path):
    """axis ∈ {'config', 'num_clients', 'alpha'}: hue dimension."""
    w = _wide(df)
    if "val_accuracy" not in w.columns or "avg_loss" not in w.columns:
        print("⚠️  missing val_accuracy/avg_loss"); return
    grp_keys = ["round", axis]
    agg = (w.groupby(grp_keys)
             .agg(acc_mean=("val_accuracy","mean"),
                  acc_ci=("val_accuracy", _ci95),
                  loss_mean=("avg_loss","mean"),
                  loss_ci=("avg_loss", _ci95))
             .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for level, sub in agg.groupby(axis):
        label = LABELS.get(str(level), str(level))
        color = PALETTE.get(str(level), None)
        for ax, mean_col, ci_col, ylab, title in [
            (axes[0], "acc_mean", "acc_ci", "Test Accuracy",  "Accuracy vs Round"),
            (axes[1], "loss_mean", "loss_ci", "Training Loss", "Loss vs Round"),
        ]:
            ax.plot(sub["round"], sub[mean_col], label=str(label), color=color, lw=2)
            ax.fill_between(sub["round"],
                            sub[mean_col] - sub[ci_col],
                            sub[mean_col] + sub[ci_col],
                            alpha=0.2, color=color)
            ax.set_xlabel("Round"); ax.set_ylabel(ylab); ax.set_title(title)
    axes[1].set_yscale("log")
    axes[0].legend(title=axis.replace("_"," ").title(), loc="lower right")
    fig.suptitle(f"Learning Curves by {axis} (95 % CI shading)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  ✓ {out}")

# ── Figure 2 — Communication overhead ─────────────────────────────────────────
def fig_communication_overhead(df: pd.DataFrame, out: pathlib.Path,
                               bytes_per_param: int = 4):
    """Cumulative bytes transferred per config = (params * 4) * num_clients * rounds."""
    w = _wide(df)
    # Approximate model size from any centralized run: assume Net (~135 KB / param count)
    # — replace with a measured value from your model.
    PARAMS = {"cifar10": 552_874, "femnist": 1_675_274, "iot": 396}
    rows = []
    for (cfg, n, ds, eps), g in w.groupby(["config","num_clients","dataset","epsilon"]):
        rounds = g["round"].max()
        params = PARAMS.get(ds, 0)
        total_bytes = params * bytes_per_param * n * rounds
        rows.append(dict(config=cfg, num_clients=n, dataset=ds,
                         epsilon=eps, total_mb=total_bytes / 1e6))
    bdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=bdf, x="num_clients", y="total_mb", hue="config",
                palette=PALETTE, ax=ax, errorbar=None)
    ax.set_xlabel("Number of Clients"); ax.set_ylabel("Cumulative Data (MB)")
    ax.set_title("Communication Overhead Across Configurations")
    ax.legend(title="Config")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  ✓ {out}")

# ── Figure 3 — Client divergence box plot ─────────────────────────────────────
def fig_client_divergence(per_client_csv: pathlib.Path, out: pathlib.Path):
    """
    Reads a per-client metrics CSV (columns: config, client_id, round, val_accuracy)
    produced by scripts/extract_per_client.py. Falls back to a synthetic
    proxy from seed-variance if per-client metrics are unavailable.
    """
    if not per_client_csv.exists():
        print(f"⚠️  {per_client_csv} missing — skipping client-divergence plot.")
        return
    pc = pd.read_csv(per_client_csv)
    last_round = pc["round"].max()
    final = pc[pc["round"] == last_round]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=final, x="config", y="val_accuracy", palette=PALETTE,
                ax=ax, fliersize=2)
    sns.stripplot(data=final, x="config", y="val_accuracy",
                  color="black", size=2.5, alpha=0.5, ax=ax)
    ax.set_xlabel("Configuration"); ax.set_ylabel("Per-client Final Accuracy")
    ax.set_title("Client Divergence at Final Round")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  ✓ {out}")

# ── Figure 4 — Privacy/utility scatter (bonus) ────────────────────────────────
def fig_privacy_utility(df: pd.DataFrame, out: pathlib.Path):
    w = _wide(df)
    if "val_accuracy" not in w.columns: return
    last = (w.sort_values("round").groupby(
              ["config","epsilon","seed"]).tail(1))
    agg = (last.groupby(["config","epsilon"])
                .agg(acc_mean=("val_accuracy","mean"),
                     acc_ci=("val_accuracy", _ci95))
                .reset_index())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg, sub in agg.groupby("config"):
        ax.errorbar(sub["epsilon"], sub["acc_mean"], yerr=sub["acc_ci"],
                    label=LABELS.get(cfg, cfg), color=PALETTE.get(cfg),
                    marker="o", capsize=3, lw=2)
    ax.set_xlabel("Privacy Budget (ε)"); ax.set_ylabel("Final Test Accuracy")
    ax.set_title("Privacy–Utility Tradeoff (95 % CI)")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  ✓ {out}")

# ── Figure 5 — Convergence rate ──────────────────────────────────────────────
def fig_convergence_rate(df: pd.DataFrame, target_acc: float,
                         out: pathlib.Path):
    w = _wide(df)
    if "val_accuracy" not in w.columns: return
    rows = []
    for (cfg, n, alpha, seed), g in w.groupby(["config","num_clients","alpha","seed"]):
        g = g.sort_values("round")
        hit = g[g["val_accuracy"] >= target_acc]
        rounds_to = int(hit["round"].iloc[0]) if len(hit) else int(g["round"].max() + 1)
        rows.append(dict(config=cfg, num_clients=n, alpha=alpha,
                         seed=seed, rounds_to=rounds_to))
    cdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=cdf, x="config", y="rounds_to", hue="num_clients",
                ax=ax, errorbar=("ci", 95), palette="viridis")
    ax.set_xlabel("Configuration")
    ax.set_ylabel(f"Rounds to reach acc ≥ {target_acc:.2f}")
    ax.set_title(f"Convergence Rate (target = {target_acc:.2f})")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    print(f"  ✓ {out}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="test_artifacts/metrics_long.csv")
    p.add_argument("--per-client-csv",
                   default="test_artifacts/per_client_metrics.csv")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--target-acc", type=float, default=0.60)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fig_learning_curves(df, axis="config",      out=out/"fig_learning_curves_config.png")
    if df["num_clients"].nunique() > 1:
        fig_learning_curves(df, axis="num_clients", out=out/"fig_learning_curves_n.png")
    if df["alpha"].nunique() > 1:
        fig_learning_curves(df, axis="alpha",       out=out/"fig_learning_curves_alpha.png")
    fig_communication_overhead(df,                  out=out/"fig_communication_overhead.png")
    fig_client_divergence(pathlib.Path(args.per_client_csv),
                                                    out=out/"fig_client_divergence.png")
    fig_privacy_utility(df,                         out=out/"fig_privacy_utility.png")
    fig_convergence_rate(df, args.target_acc,       out=out/"fig_convergence_rate.png")

if __name__ == "__main__":
    main()
