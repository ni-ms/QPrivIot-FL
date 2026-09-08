#!/usr/bin/env python3
"""
generate_graphs.py — Publication-quality figures for QPrivIot-FL paper.

Generates up to 6 figures from MNIST experiment results comparing:
  • no-dp      : No differential privacy baseline
  • fixed-dp   : Fixed uniform DP noise
  • param-only : Per-layer adaptive DP (ablation)
  • adapriv    : Full adaptive per-layer + device + round DP (paper contribution)

Usage:
  python3 scripts/generate_graphs.py [--dataset mnist] [--eps 8.0] [--seed 42]

Missing configs are skipped gracefully (with a warning) rather than crashing,
so figures can be generated before the full experiment suite completes.
"""

import argparse
import json
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths / CLI
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "experiment_results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="Generate paper figures.")
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--eps", default="8.0", help="target epsilon used in result filenames")
parser.add_argument("--seed", default="42")
ARGS = parser.parse_args()

DATASET = ARGS.dataset
EPS = ARGS.eps
SEED = ARGS.seed

# Headline figure shows all four configs; ablation-only configs (param-only) are
# included when present. Missing configs are dropped at load time.
CONFIGS = ["no-dp", "fixed-dp", "param-only", "adapriv"]
LABELS = {
    "no-dp": "No-DP",
    "fixed-dp": "Fixed-DP",
    "param-only": "Param-Only",
    "adapriv": "AdaPriv",
}
COLORS = {
    "no-dp": "#2166ac",
    "fixed-dp": "#d6604d",
    "param-only": "#f1a340",
    "adapriv": "#4dac26",
}
MARKERS = {"no-dp": "o", "fixed-dp": "s", "param-only": "P", "adapriv": "^"}

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

DATASET_TITLE = DATASET.upper()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def json_path(config: str) -> Path:
    return DATA_DIR / f"results_{DATASET}_{config}_seed{SEED}_eps{EPS}.json"


def csv_path(config: str) -> Path:
    return DATA_DIR / f"results_{DATASET}_{config}_seed{SEED}_eps{EPS}_per_client.csv"


def load_json(config: str):
    path = json_path(config)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(config: str):
    path = csv_path(config)
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rounds_field(data: dict, field: str):
    xs, ys = [], []
    for r in data["rounds"]:
        val = r.get(field)
        if val is not None:
            xs.append(r["round"])
            ys.append(float(val))
    return xs, ys


def save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/{name}")


def markevery_for(xs, target_marks: int = 20):
    """Place ~target_marks markers regardless of round count (handles T=20 or T=100)."""
    if not xs:
        return None
    step = max(1, len(xs) // target_marks)
    return list(range(0, len(xs), step))


# ---------------------------------------------------------------------------
# Load all data up-front
# ---------------------------------------------------------------------------
print(f"Loading experiment data (dataset={DATASET}, eps={EPS}, seed={SEED})...")
json_data = {}
csv_data = {}
for cfg in CONFIGS:
    jd = load_json(cfg)
    if jd is None:
        print(f"  [skip] {cfg}: {json_path(cfg).name} not found")
        continue
    json_data[cfg] = jd
    cd = load_csv(cfg)
    if cd is not None:
        csv_data[cfg] = cd

if not json_data:
    raise SystemExit(
        f"No result JSONs found for dataset={DATASET}, eps={EPS}, seed={SEED} in {DATA_DIR}.\n"
        f"Run experiments first (e.g. bash scripts/run_paper_experiments.sh)."
    )

LOADED = [c for c in CONFIGS if c in json_data]

# Determine x-axis extent from the data (T=20 quick test or T=100 full run).
MAX_ROUND = max(
    (r["round"] for d in json_data.values() for r in d["rounds"]),
    default=20,
)
XTICK_STEP = 2 if MAX_ROUND <= 25 else 10
print(f"  Loaded: {', '.join(LOADED)}  (max round = {MAX_ROUND})\n")


# ---------------------------------------------------------------------------
# Graph 1 — Validation Accuracy vs Round
# ---------------------------------------------------------------------------
def graph1_accuracy():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Random-chance baseline (10 classes)
    ax.axhline(0.10, color="gray", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.text(0.5, 0.105, "Random (10%)", color="gray", fontsize=LEGEND_FS,
            transform=ax.get_yaxis_transform(), va="bottom")

    for cfg in LOADED:
        xs, ys = rounds_field(json_data[cfg], "val_accuracy")
        if not xs:
            continue
        peak = max(ys) * 100
        label = f"{LABELS[cfg]} (peak: {peak:.1f}%)"
        ax.plot(xs, ys, color=COLORS[cfg], marker=MARKERS[cfg],
                markevery=markevery_for(xs), markersize=6, label=label)

    ax.set_xlabel("Communication Round", fontsize=LABEL_FS)
    ax.set_ylabel("Validation Accuracy", fontsize=LABEL_FS)
    ax.set_title(
        f"{DATASET_TITLE} Accuracy: No-DP vs Fixed-DP vs AdaPriv (ε={EPS}, seed={SEED})",
        fontsize=TITLE_FS, pad=12
    )
    ax.set_xlim(1, MAX_ROUND)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(XTICK_STEP))
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig1_accuracy_vs_round.png")


# ---------------------------------------------------------------------------
# Graph 2 — Training Loss vs Round
# ---------------------------------------------------------------------------
def graph2_loss():
    fig, ax = plt.subplots(figsize=(10, 6))

    for cfg in LOADED:
        xs, ys = rounds_field(json_data[cfg], "avg_loss")
        if not xs:
            continue
        ax.plot(xs, ys, color=COLORS[cfg], marker=MARKERS[cfg],
                markevery=markevery_for(xs), markersize=6, label=LABELS[cfg])

    ax.set_xlabel("Communication Round", fontsize=LABEL_FS)
    ax.set_ylabel("Average Training Loss", fontsize=LABEL_FS)
    ax.set_title("Training Loss Convergence", fontsize=TITLE_FS, pad=12)
    ax.set_xlim(1, MAX_ROUND)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(XTICK_STEP))
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig2_training_loss.png")


# ---------------------------------------------------------------------------
# Graph 3 — Privacy Budget Accumulation (DP configs only)
# ---------------------------------------------------------------------------
def graph3_privacy_budget():
    dp_configs = [c for c in ("fixed-dp", "param-only", "adapriv") if c in json_data]
    if not dp_configs:
        print("  [skip] Graph 3: no DP configs loaded.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for cfg in dp_configs:
        xs, ys = rounds_field(json_data[cfg], "total_epsilon")
        if not xs:
            continue
        ax.plot(xs, ys, color=COLORS[cfg], marker=MARKERS[cfg],
                markevery=markevery_for(xs), markersize=6, label=LABELS[cfg])

    ax.axhline(float(EPS), color="black", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.text(0.5, float(EPS) * 1.01, f"Target ε = {EPS}", color="black",
            fontsize=LEGEND_FS, transform=ax.get_yaxis_transform(), va="bottom")

    ax.set_xlabel("Communication Round", fontsize=LABEL_FS)
    ax.set_ylabel("Accumulated Privacy Budget (ε)", fontsize=LABEL_FS)
    ax.set_title("Privacy Budget Consumption Over Rounds", fontsize=TITLE_FS, pad=12)
    ax.set_xlim(1, MAX_ROUND)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(XTICK_STEP))
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig3_privacy_budget.png")


# ---------------------------------------------------------------------------
# Graph 4 — Per-Layer Sensitivity Heatmap (AdaPriv, all rounds with data)
# ---------------------------------------------------------------------------
def graph4_sensitivity_heatmap():
    src = "adapriv" if "adapriv" in json_data else (
        "param-only" if "param-only" in json_data else None)
    if src is None:
        print("  [skip] Graph 4: no adaptive config loaded.")
        return

    rounds_data = json_data[src]["rounds"]
    layer_names = None
    for r in rounds_data:
        s = r.get("sensitivities")
        if s:
            layer_names = list(s.keys())
            break
    if layer_names is None:
        print("  [skip] Graph 4: no sensitivity data found.")
        return

    # Use every round that actually carries sensitivities (post-warmup).
    target_rounds = [r["round"] for r in rounds_data if r.get("sensitivities")]
    if not target_rounds:
        print("  [skip] Graph 4: no rounds with sensitivities.")
        return

    matrix = np.full((len(layer_names), len(target_rounds)), np.nan)
    round_to_col = {rno: i for i, rno in enumerate(target_rounds)}
    for r in rounds_data:
        col = round_to_col.get(r["round"])
        if col is None:
            continue
        s = r.get("sensitivities", {})
        for row, lname in enumerate(layer_names):
            val = s.get(lname)
            if val is not None:
                matrix[row, col] = float(val)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Sensitivity Score", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    # Subsample x tick labels so a 100-round run stays legible.
    tick_step = max(1, len(target_rounds) // 20)
    tick_idx = list(range(0, len(target_rounds), tick_step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(target_rounds[i]) for i in tick_idx], fontsize=TICK_FS)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=TICK_FS)
    ax.set_xlabel("Communication Round", fontsize=LABEL_FS)
    ax.set_ylabel("Layer", fontsize=LABEL_FS)
    ax.set_title(f"{LABELS[src]}: Per-Layer Sensitivity Scores Over Training",
                 fontsize=TITLE_FS, pad=12)
    fig.tight_layout()
    save(fig, "fig4_sensitivity_heatmap.png")


# ---------------------------------------------------------------------------
# Graph 5 — Per-Client Fairness by Device Tier
# ---------------------------------------------------------------------------
def graph5_fairness():
    cfgs = [c for c in LOADED if c in csv_data]
    if not cfgs:
        print("  [skip] Graph 5: no per-client CSVs loaded.")
        return

    device_order = ["1", "2", "3", "4"]
    device_labels = [DEVICE_MAP[d] for d in device_order]

    fig, axes = plt.subplots(1, len(cfgs), figsize=(5 * len(cfgs), 6), sharey=True,
                             squeeze=False)
    axes = axes[0]
    fig.suptitle(f"Per-Device-Tier Accuracy Distribution (ε={EPS})",
                 fontsize=TITLE_FS, y=1.01)

    for ax, cfg in zip(axes, cfgs):
        rows = csv_data[cfg]
        groups = {d: [] for d in device_order}
        for row in rows:
            did = row.get("device_id")
            try:
                acc = float(row["val_accuracy"])
            except (KeyError, ValueError):
                continue
            if did in groups:
                groups[did].append(acc)

        data_list = [groups[d] for d in device_order]
        medians = [np.median(g) if g else np.nan for g in data_list]

        vp = ax.violinplot(
            [g if g else [0] for g in data_list],
            positions=range(len(device_order)),
            showmedians=True, showextrema=True,
        )
        for body in vp["bodies"]:
            body.set_facecolor(COLORS[cfg])
            body.set_alpha(0.65)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.8)
        for part in ("cmins", "cmaxes", "cbars"):
            vp[part].set_color(COLORS[cfg])
            vp[part].set_alpha(0.8)

        for i, med in enumerate(medians):
            if not np.isnan(med):
                ax.text(i, med + 0.01, f"{med:.2f}", ha="center",
                        va="bottom", fontsize=8, color="black")

        ax.set_xticks(range(len(device_order)))
        ax.set_xticklabels(device_labels, rotation=15, ha="right", fontsize=TICK_FS - 1)
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
    src = "adapriv" if "adapriv" in csv_data else (
        "param-only" if "param-only" in csv_data else None)
    if src is None:
        print("  [skip] Graph 6: no adaptive per-client CSV loaded.")
        return

    rows = csv_data[src]
    round_data = {}
    for row in rows:
        try:
            rno = int(row["round"])
        except (KeyError, ValueError):
            continue
        noise = float(row.get("avg_noise") or 0)
        clip = float(row.get("clip_norm") or 0)
        round_data.setdefault(rno, {"noise": [], "clip": []})
        round_data[rno]["noise"].append(noise)
        round_data[rno]["clip"].append(clip)

    if not round_data:
        print("  [skip] Graph 6: no noise/clip data.")
        return

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
    ax1.set_ylabel("Average Noise Multiplier (σ)", color=color_noise, fontsize=LABEL_FS)
    ax2.set_ylabel("Clip Norm", color=color_clip, fontsize=LABEL_FS)
    ax1.tick_params(axis="y", labelcolor=color_noise, labelsize=TICK_FS)
    ax2.tick_params(axis="y", labelcolor=color_clip, labelsize=TICK_FS)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(XTICK_STEP))
    ax1.set_title(f"{LABELS[src]}: Noise Multiplier and Clip Norm Over Rounds",
                  fontsize=TITLE_FS, pad=12)

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right",
               fontsize=LEGEND_FS, framealpha=0.9)
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
    print(f"\n{'=' * 55}")
    print(f"Summary: {len(generated)} figure(s) saved to {FIG_DIR}")
    print("=" * 55)
    for p in generated:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<40}  {size_kb:>6.1f} KB")
    print("=" * 55)


if __name__ == "__main__":
    main()