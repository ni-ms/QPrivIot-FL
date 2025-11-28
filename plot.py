"""
Data plotting script for research paper visualizations.

This script:
1. Loads experimental results from three configuration files.
2. Creates comprehensive comparative visualizations (12 subplots) for the paper.
3. Handles missing metrics gracefully by using a default value (0.0).
4. Exits gracefully if no data files are found.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List
import os
import math

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

COLORS = {
    'no_dp': '#2c3e50',  # Dark blue-gray (No DP)
    'uniform': '#e74c3c',  # Red (Uniform DP)
    'adaptive': '#27ae60'  # Green (QPrivIoT)
}

MARKERS = {
    'no_dp': 'o',
    'uniform': 's',
    'adaptive': '^'
}


def load_data(filename):
    """
    Load data from JSON file. Exits gracefully if file is missing.

    Args:
        filename: Path to JSON file

    Returns:
        Dictionary with rounds data or empty dict if file missing
    """
    if os.path.exists(filename):
        print(f"✅ Loading {filename}...")
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if isinstance(data, list):
                result = {"rounds": data}
            elif isinstance(data, dict) and "rounds" in data:
                result = data
            else:
                result = {"rounds": []}

            num_rounds = len(result.get('rounds', []))
            print(f"   Loaded {num_rounds} rounds")

            if num_rounds > 0:
                sample = result['rounds'][0]
                print(f"   Available metrics: {list(sample.keys())}")

            return result

        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return {"rounds": []}
    else:
        print(f"⚠️  File not found: {filename}. Skipping.")
        return {"rounds": []}


def extract_metric(data, key, scale=1.0, cumsum=False, default_value=0.0):
    """
    Extract metric from data with proper error handling.

    Args:
        data: Dictionary with 'rounds' key
        key: Metric key to extract
        scale: Scaling factor
        cumsum: If True, return cumulative sum
        default_value: Value to use if metric missing

    Returns:
        List of metric values
    """
    rounds = data.get('rounds', [])

    if not rounds:
        return []

    values = []
    for r in rounds:
        val = r.get(key, default_value)
        values.append(val * scale)

    if cumsum and values:
        cumulative_sum = [sum(values[:i + 1]) for i in range(len(values))]
        return cumulative_sum

    return values


def plot_all():
    """
    Create comprehensive visualization with 12 subplots from loaded data.
    """
    print("\n" + "=" * 60)
    print("🎨 Creating Paper Figures")
    print("=" * 60 + "\n")

    no_dp = load_data("results_no_dp.json")
    uniform = load_data("results_uniform_dp.json")
    adaptive = load_data("results_adaptive_dp.json")

    lens = [len(d.get('rounds', [])) for d in [no_dp, uniform, adaptive]]

    if max(lens) == 0:
        print("\n❌ No data available in any file!")
        print("   Please ensure results files (results_no_dp.json, etc.) exist.")
        return

    max_rounds = max(lens)
    print(f"\n📈 Creating plots for up to {max_rounds} rounds...")

    fig = plt.figure(figsize=(24, 20))

    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1.2], hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    for name, data, key in [('No DP', no_dp, 'no_dp'),
                            ('Uniform DP', uniform, 'uniform'),
                            ('QPrivIoT (Ours)', adaptive, 'adaptive')]:

        vals = extract_metric(data, 'val_accuracy', scale=100)
        if vals:
            rounds = range(1, len(vals) + 1)
            ax1.plot(rounds, vals, marker=MARKERS[key], color=COLORS[key],
                     label=name, linewidth=2.5, markersize=8)

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Global Model Accuracy', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    ax2 = fig.add_subplot(gs[0, 1])
    for name, data, key in [('No DP', no_dp, 'no_dp'),
                            ('Uniform DP', uniform, 'uniform'),
                            ('QPrivIoT (Ours)', adaptive, 'adaptive')]:

        vals = extract_metric(data, 'val_loss')
        if vals:
            rounds = range(1, len(vals) + 1)
            ax2.plot(rounds, vals, marker=MARKERS[key], color=COLORS[key],
                     label=name, linewidth=2.5, markersize=8)

    ax2.set_xlabel('Round')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('(b) Convergence Stability', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])

    uni_eps = extract_metric(uniform, 'total_epsilon')
    adapt_eps = extract_metric(adaptive, 'total_epsilon')

    if uni_eps:
        ax3.plot(range(1, len(uni_eps) + 1), uni_eps, marker=MARKERS['uniform'],
                 color=COLORS['uniform'], label='Uniform DP', linewidth=2.5, markersize=8)
    if adapt_eps:
        ax3.plot(range(1, len(adapt_eps) + 1), adapt_eps, marker=MARKERS['adaptive'],
                 color=COLORS['adaptive'], label='QPrivIoT', linewidth=2.5, markersize=8)

    ax3.axhline(y=10.0, color='black', linestyle=':', linewidth=2, label='Budget (ε=10)')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Privacy Cost (ε)')
    ax3.set_title('(c) Privacy Budget Consumption', fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])

    client_lat_uni = extract_metric(uniform, 'avg_latency', cumsum=True)
    client_lat_adapt = extract_metric(adaptive, 'avg_latency', cumsum=True)

    if client_lat_uni:
        ax4.plot(range(1, len(client_lat_uni) + 1), client_lat_uni,
                 color=COLORS['uniform'], label='Uniform DP', linewidth=2.5)
    if client_lat_adapt:
        ax4.plot(range(1, len(client_lat_adapt) + 1), client_lat_adapt,
                 color=COLORS['adaptive'], label='QPrivIoT', linewidth=2.5)

    ax4.set_xlabel('Round')
    ax4.set_ylabel('Cumulative Time (s)')
    ax4.set_title('(d) Training Efficiency', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])

    if client_lat_uni and client_lat_adapt:
        acc_uni = extract_metric(uniform, 'val_accuracy', scale=100)
        acc_adapt = extract_metric(adaptive, 'val_accuracy', scale=100)

        min_len = min(len(acc_uni), len(client_lat_uni), len(acc_adapt), len(client_lat_adapt))
        acc_uni = acc_uni[:min_len]
        client_lat_uni = client_lat_uni[:min_len]
        acc_adapt = acc_adapt[:min_len]
        client_lat_adapt = client_lat_adapt[:min_len]

        eff_uni = [a / max(t, 0.01) for a, t in zip(acc_uni, client_lat_uni)]
        eff_adapt = [a / max(t, 0.01) for a, t in zip(acc_adapt, client_lat_adapt)]

        ax5.plot(range(1, len(eff_uni) + 1), eff_uni,
                 color=COLORS['uniform'], label='Uniform DP', linewidth=2.5)
        ax5.plot(range(1, len(eff_adapt) + 1), eff_adapt,
                 color=COLORS['adaptive'], label='QPrivIoT', linewidth=2.5)

    ax5.set_xlabel('Round')
    ax5.set_ylabel('Efficiency (Acc/Time)')
    ax5.set_title('(e) Training Efficiency', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])

    res_adapt = extract_metric(adaptive, 'avg_resource')
    if res_adapt:
        ax6.plot(range(1, len(res_adapt) + 1), res_adapt,
                 marker='o', color=COLORS['adaptive'], label='Avg Resource Score',
                 linewidth=2.5, markersize=8)

        eligibility_threshold = 0.15
        ax6.axhline(y=eligibility_threshold, color='red', linestyle='--', linewidth=2,
                    label='Eligibility Threshold')

    ax6.set_xlabel('Round')
    ax6.set_ylabel('Resource Score')
    ax6.set_title('(f) Device Participation', fontweight='bold')
    ax6.set_ylim([0, 1.1])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 0])

    sens = extract_metric(adaptive, 'avg_sensitivity')
    if sens:
        ax7.plot(range(1, len(sens) + 1), sens,
                 color='purple', linewidth=2.5)
        ax7.fill_between(range(1, len(sens) + 1), sens, alpha=0.3, color='purple')

    ax7.set_xlabel('Round')
    ax7.set_ylabel('Gradient Norm')
    ax7.set_title('(g) Gradient Sensitivity', fontweight='bold')
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])

    conv_score = extract_metric(adaptive, 'convergence_score')
    if conv_score:
        ax8.plot(range(1, len(conv_score) + 1), conv_score,
                 color='orange', linewidth=2.5)
        ax8.axhline(y=0.8, color='green', linestyle='--', linewidth=2,
                    label='Converged (>0.8)')

    ax8.set_xlabel('Round')
    ax8.set_ylabel('Convergence Score')
    ax8.set_title('(h) Model Convergence', fontweight='bold')
    ax8.set_ylim([0, 1.1])
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])

    clip_norms = extract_metric(adaptive, 'clip_norm')
    if clip_norms:
        ax9.plot(range(1, len(clip_norms) + 1), clip_norms,
                 color='brown', linewidth=2.5)
        initial_norm = 1.0
        ax9.axhline(y=initial_norm, color='red', linestyle=':', linewidth=2,
                    label=f'Initial Norm ({initial_norm})')

    ax9.set_xlabel('Round')
    ax9.set_ylabel('Clipping Norm')
    ax9.set_title('(i) Adaptive Clipping', fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    ax10 = fig.add_subplot(gs[3, 0])

    final_accs = []
    labels = []
    colors = []

    for name, data, key in [('No DP', no_dp, 'no_dp'),
                            ('Uniform DP', uniform, 'uniform'),
                            ('QPrivIoT', adaptive, 'adaptive')]:
        vals = extract_metric(data, 'val_accuracy', scale=100)
        if vals:
            final_accs.append(vals[-1])
            labels.append(name)
            colors.append(COLORS[key])

    if final_accs:
        bars = ax10.bar(labels, final_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax10.bar_label(bars, fmt='%.1f%%', fontsize=12, fontweight='bold')

    ax10.set_ylabel('Accuracy (%)')
    ax10.set_title('(j) Final Model Accuracy', fontweight='bold')
    ax10.set_ylim([0, max(final_accs) * 1.2] if final_accs else [0, 100])
    ax10.grid(True, alpha=0.3, axis='y')

    ax11 = fig.add_subplot(gs[3, 1])

    eps_uni = extract_metric(uniform, 'total_epsilon')
    eps_adapt = extract_metric(adaptive, 'total_epsilon')
    acc_uni = extract_metric(uniform, 'val_accuracy', scale=100)
    acc_adapt = extract_metric(adaptive, 'val_accuracy', scale=100)

    if eps_uni and acc_uni:
        ax11.scatter(eps_uni[-1], acc_uni[-1], s=300, color=COLORS['uniform'],
                     marker='s', edgecolor='black', linewidth=2, label='Uniform DP', zorder=3)
    if eps_adapt and acc_adapt:
        ax11.scatter(eps_adapt[-1], acc_adapt[-1], s=400, color=COLORS['adaptive'],
                     marker='^', edgecolor='black', linewidth=2, label='QPrivIoT', zorder=3)

    ax11.set_xlabel('Privacy Cost (ε)')
    ax11.set_ylabel('Accuracy (%)')
    ax11.set_title('(k) Privacy-Utility Trade-off', fontweight='bold')
    ax11.legend(loc='lower right')
    ax11.grid(True, alpha=0.3)

    ax12 = fig.add_subplot(gs[3, 2], polar=True)

    categories = ['Accuracy', 'Speed', 'Privacy', 'Stability']

    def get_normalized_stats(data):

        acc = extract_metric(data, 'val_accuracy')[-1] if extract_metric(data, 'val_accuracy') else 0.5

        lat = extract_metric(data, 'avg_latency', cumsum=True)[-1] if extract_metric(data, 'avg_latency') else 1.0

        eps = extract_metric(data, 'total_epsilon')[-1] if extract_metric(data, 'total_epsilon') else 10.0

        loss = extract_metric(data, 'val_loss')[-1] if extract_metric(data, 'val_loss') else 1.0

        return [
            acc,
            1.0 / (lat + 0.1),
            1.0 / (eps + 0.1),
            1.0 / (loss + 0.1)
        ]

    values_uni = get_normalized_stats(uniform)
    values_adapt = get_normalized_stats(adaptive)

    if all(len(v) == 4 for v in [values_uni, values_adapt]):
        all_vals = np.array([values_uni, values_adapt])
        mins = all_vals.min(axis=0)
        maxs = all_vals.max(axis=0)

        values_uni_norm = [(v - mi) / (ma - mi + 1e-6) for v, mi, ma in zip(values_uni, mins, maxs)]
        values_adapt_norm = [(v - mi) / (ma - mi + 1e-6) for v, mi, ma in zip(values_adapt, mins, maxs)]
    else:

        values_uni_norm = [0.5] * 5
        values_adapt_norm = [0.5] * 5

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_uni_norm += values_uni_norm[:1]
    values_adapt_norm += values_adapt_norm[:1]
    angles += angles[:1]

    ax12.plot(angles, values_uni_norm, 'o-', linewidth=2.5,
              color=COLORS['uniform'], label='Uniform DP')
    ax12.fill(angles, values_uni_norm, alpha=0.15, color=COLORS['uniform'])

    ax12.plot(angles, values_adapt_norm, '^-', linewidth=2.5,
              color=COLORS['adaptive'], label='QPrivIoT')
    ax12.fill(angles, values_adapt_norm, alpha=0.25, color=COLORS['adaptive'])

    ax12.set_xticks(angles[:-1])
    ax12.set_xticklabels(categories, fontsize=12)
    ax12.set_ylim(0, 1)
    ax12.set_title('(l) System Balance', fontweight='bold', pad=20)
    ax12.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax12.grid(True)

    plt.tight_layout()

    output_file = "paper_figures_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved: {output_file}")
    print(f"   Resolution: 300 DPI")
    print(f"   Size: {fig.get_size_inches()[0]:.1f} x {fig.get_size_inches()[1]:.1f} inches")

    plt.show()

    print("\n" + "=" * 60)
    print("✨ Plotting complete!")
    print("=" * 60)


if __name__ == "__main__":
    plot_all()
