#!/usr/bin/env python3
"""
generate_graphs.py — Publication-quality figures for QPrivIot-FL paper.

Generates 6 figures from CIFAR-10 experiment results comparing:
  • no-dp      : No differential privacy baseline
  • fixed-dp   : Fixed uniform DP noise (σ=6.68, ε=3.0)
  • adapriv    : Adaptive per-layer DP (AdaPriv — paper contribution)
"""

import json
import csv
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "experiment_results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["no-dp", "fixed-dp", "adapriv"]
LABELS = {"no-dp": "No-DP", "fixed-dp": "Fixed-DP", "adapriv": "AdaPriv"}
COLORS = {"no-dp": "#2166ac", "fixed-dp": "#d6604d", "adapriv": "#4dac26"}
MARKERS = {"no-dp": "o", "fixed-dp": "s", "adapriv": "^"}

DEVICE_MAP = {
    "1": "Raspberry Pi 4",
    "2": "Raspberry Pi Zero",
    "3": "Smartphone",
    "4": "IoT Sensor",
}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    try:
        plt.style.use("seaborn-paper")
    except OSError:
        pass  # fall back to default

TITLE_FS = 14
LABEL_FS = 12
TICK_FS = 10
LEGEND_FS = 10

plt.rcParams.update({
    "font.size": TICK_FS,
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "lines.linewidth": 2.0,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(config: str) -> dict:
    path = DATA_DIR / f"results_cifar10_{config}_seed42_eps3.0.json"
    with open(path) as f:
        return json.load(f)


def load_csv(config: str) -> list[dict]:
    path = DATA_DIR / f"results_cifar10_{config}_seed42_eps3.0_per_client.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rounds_field(data: dict, field: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in data["rounds"]:
        val = r.get(field)
        if val is not None:
            xs.append(r["round"])
            ys.append(float(val))
    return xs, ys


def save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/{name}")


# ---------------------------------------------------------------------------
# Load all data up-front
# ---------------------------------------------------------------------------
print("Loading experiment data...")
json_data = {cfg: load_json(cfg) for cfg in CONFIGS}
csv_data = {cfg: load_csv(cfg) for cfg in CONFIGS}
print("  Done.\n")


# ---------------------------------------------------------------------------
# Graph 1 — Validation Accuracy vs Round
# ---------------------------------------------------------------------------
def graph1_accuracy():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Random-chance baseline
    ax.axhline(0.10, color="gray", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.text(0.5, 0.105, "Random (10%)", color="gray", fontsize=LEGEND_FS,
            transform=ax.get_yaxis_transform(), va="bottom")

    for cfg in CONFIGS:
        xs, ys = rounds_field(json_data[cfg], "val_accuracy")
        peak = max(ys) * 100
        label = f"{LABELS[cfg]} (peak: {peak:.1f}%)"
        markevery = [i for i, x in enumerate(xs) if x % 5 == 0]
        ax.plot(xs, ys, color=COLORS[cfg], marker=MARKERS[cfg],
                markevery=markevery, markersize=6, label=label)

    ax.set_xlabel("Communication Round", fontsize=LABEL_FS)
    ax.set_ylabel("Validation Accuracy", fontsize=LABEL_FS)
    ax.set_title(
        "CIFAR-10 Accuracy: No-DP vs Fixed-DP vs AdaPriv (ε=3.0, seed=42)",
        fontsize=TITLE_FS, pad=12
    )
    ax.set_xlim(1, 20)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig1_accuracy_vs_round.png")


# ---------------------------------------------------------------------------
# Graph 2 — Training Loss vs Round
# ---------------------------------------------------------------------------
def graph2_loss():
    fig, ax = plt.subplots(figsize=(10, 6))

    for cfg in CONFIGS:
        xs, ys = rounds_field(json_data[cfg], "avg_loss")
        markevery = [i for i, x in enumerate(xs) if x % 5 == 0]
        ax.plot(xs, ys, color=COLORS[cfg], marker=MARKERS[cfg],
                markevery=markevery, markersize=6, label=LABELS[cfg])

    ax.set_xlabel("Communication Round", fontsize=LABEL_FS)
    ax.set_ylabel("Average Training Loss", fontsize=LABEL_FS)
    ax.set_title("Training Loss Convergence", fontsize=TITLE_FS, pad=12)
    ax.set_xlim(1, 20)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig2_training_loss.png")


# ---------------------------------------------------------------------------
# Graph 3 — Privacy Budget Accumulation (DP configs only)
# ---------------------------------------------------------------------------
def graph3_privacy_budget():
    fig, ax = plt.subplots(figsize=(10, 6))

    dp_configs = ["fixed-dp", "adapriv"]
    for cfg in dp_configs:
        xs, ys = rounds_field(json_data[cfg], "total_epsilon")
        markevery = [i for i, x in enumerate(xs) if x % 5 == 0]
        ax.plot(xs, ys, color=COLORS[cfg], marker=MARKERS[cfg],
                markevery=markevery, markersize=6, label=LABELS[cfg])

    ax.axhline(3.0, color="black", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.text(0.5, 3.05, "Target ε = 3.0", color="black", fontsize=LEGEND_FS,
            transform=ax.get_yaxis_transform(), va="bottom")

    ax.set_xlabel("Communication Round", fontsize=LABEL_FS)
    ax.set_ylabel("Accumulated Privacy Budget (ε)", fontsize=LABEL_FS)
    ax.set_title("Privacy Budget Consumption Over Rounds", fontsize=TITLE_FS, pad=12)
    ax.set_xlim(1, 20)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig3_privacy_budget.png")


# ---------------------------------------------------------------------------
# Graph 4 — Per-Layer Sensitivity Heatmap (AdaPriv only, rounds 6–20)
# ---------------------------------------------------------------------------
def graph4_sensitivity_heatmap():
    rounds_data = json_data["adapriv"]["rounds"]
    # Collect layer names from first round that has sensitivities
    layer_names = None
    for r in rounds_data:
        s = r.get("sensitivities")
        if s:
            layer_names = list(s.keys())
            break

    if layer_names is None:
        print("  [skip] Graph 4: no sensitivity data found in adapriv results.")
        return

    # Build matrix: rows=layers, cols=rounds 6-20
    target_rounds = list(range(6, 21))
    matrix = np.full((len(layer_names), len(target_rounds)), np.nan)

    for r in rounds_data:
        rno = r["round"]
        if rno not in target_rounds:
            continue
        col = target_rounds.index(rno)
        s = r.get("sensitivities", {})
        for row, lname in enumerate(layer_names):
            val = s.get(lname)
            if val is not None:
                matrix[row, col] = float(val)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Sensitivity Score", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    ax.set_xticks(range(len(target_rounds)))
    ax.set_xticklabels([str(r) for r in target_rounds], fontsize=TICK_FS)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=TICK_FS)
    ax.set_xlabel("Communication Round", fontsize=LABEL_FS)
    ax.set_ylabel("Layer", fontsize=LABEL_FS)
    ax.set_title(
        "AdaPriv: Per-Layer Sensitivity Scores Over Training",
        fontsize=TITLE_FS, pad=12
    )
    fig.tight_layout()
    save(fig, "fig4_sensitivity_heatmap.png")


# ---------------------------------------------------------------------------
# Graph 5 — Per-Client Fairness by Device Tier
# ---------------------------------------------------------------------------
def graph5_fairness():
    device_order = ["1", "2", "3", "4"]
    device_labels = [DEVICE_MAP[d] for d in device_order]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    fig.suptitle(
        "Per-Device-Tier Accuracy Distribution (ε=3.0)",
        fontsize=TITLE_FS, y=1.01
    )

    for ax, cfg in zip(axes, CONFIGS):
        rows = csv_data[cfg]
        # Group val_accuracy by device_id
        groups = {d: [] for d in device_order}
        for row in rows:
            did = row["device_id"]
            acc = float(row["val_accuracy"])
            if did in groups:
                groups[did].append(acc)

        data_list = [groups[d] for d in device_order]
        medians = [np.median(g) if g else np.nan for g in data_list]

        vp = ax.violinplot(
            [g if g else [0] for g in data_list],
            positions=range(len(device_order)),
            showmedians=True,
            showextrema=True,
        )
        # Color violin bodies
        for body in vp["bodies"]:
            body.set_facecolor(COLORS[cfg])
            body.set_alpha(0.65)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.8)
        for part in ("cmins", "cmaxes", "cbars"):
            vp[part].set_color(COLORS[cfg])
            vp[part].set_alpha(0.8)

        # Annotate medians
        for i, med in enumerate(medians):
            if not np.isnan(med):
                ax.text(i, med + 0.01, f"{med:.2f}", ha="center",
                        va="bottom", fontsize=8, color="black")

        ax.set_xticks(range(len(device_order)))
        ax.set_xticklabels(device_labels, rotation=15, ha="right",
                           fontsize=TICK_FS - 1)
        ax.set_title(LABELS[cfg], fontsize=TITLE_FS - 1)
        ax.set_xlabel("Device Tier", fontsize=LABEL_FS - 1)
        if ax is axes[0]:
            ax.set_ylabel("Validation Accuracy", fontsize=LABEL_FS)

    fig.tight_layout()
    save(fig, "fig5_per_client_fairness.png")


# ---------------------------------------------------------------------------
# Graph 6 — Noise Multiplier & Clip Norm vs Round (AdaPriv only)
# ---------------------------------------------------------------------------
def graph6_noise_clip():
    rows = csv_data["adapriv"]

    # Aggregate per round: mean avg_noise and mean clip_norm
    round_data: dict[int, dict[str, list]] = {}
    for row in rows:
        rno = int(row["round"])
        noise = float(row.get("avg_noise") or 0)
        clip = float(row.get("clip_norm") or 0)
        if rno not in round_data:
            round_data[rno] = {"noise": [], "clip": []}
        round_data[rno]["noise"].append(noise)
        round_data[rno]["clip"].append(clip)

    sorted_rounds = sorted(round_data.keys())
    mean_noise = [np.mean(round_data[r]["noise"]) for r in sorted_rounds]
    mean_clip = [np.mean(round_data[r]["clip"]) for r in sorted_rounds]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    color_noise = "#7b2d8b"
    color_clip = "#e8601c"

    l1, = ax1.plot(sorted_rounds, mean_noise, color=color_noise, marker="D",
                   markersize=5, label="Avg Noise (σ)")
    l2, = ax2.plot(sorted_rounds, mean_clip, color=color_clip, marker="v",
                   markersize=5, linestyle="--", label="Clip Norm")

    ax1.set_xlabel("Communication Round", fontsize=LABEL_FS)
    ax1.set_ylabel("Average Noise Multiplier (σ)", color=color_noise,
                   fontsize=LABEL_FS)
    ax2.set_ylabel("Clip Norm", color=color_clip, fontsize=LABEL_FS)
    ax1.tick_params(axis="y", labelcolor=color_noise, labelsize=TICK_FS)
    ax2.tick_params(axis="y", labelcolor=color_clip, labelsize=TICK_FS)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(2))

    ax1.set_title(
        "AdaPriv: Noise Multiplier and Clip Norm Over Rounds",
        fontsize=TITLE_FS, pad=12
    )

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=LEGEND_FS,
               framealpha=0.9)

    fig.tight_layout()
    save(fig, "fig6_adapriv_noise_clip.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating figures...\n")

    graph1_accuracy()
    graph2_loss()
    graph3_privacy_budget()
    graph4_sensitivity_heatmap()
    graph5_fairness()
    graph6_noise_clip()

    generated = sorted(FIG_DIR.glob("*.png"))
    print(f"\n{'='*55}")
    print(f"Summary: {len(generated)} figure(s) saved to {FIG_DIR}")
    print("=" * 55)
    for p in generated:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<40}  {size_kb:>6.1f} KB")
    print("=" * 55)


if __name__ == "__main__":
    main()