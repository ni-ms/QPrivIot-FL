import json
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, List

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def load_results(no_dp_path: str, uniform_dp_path: str, adaptive_dp_path: str):
    """Load experiment results from JSON files."""
    with open(no_dp_path, 'r') as f:
        no_dp = json.load(f)
    with open(uniform_dp_path, 'r') as f:
        uniform_dp = json.load(f)
    with open(adaptive_dp_path, 'r') as f:
        adaptive_dp = json.load(f)

    return no_dp, uniform_dp, adaptive_dp


def plot_paper_figures(no_dp: List[Dict], uniform_dp: List[Dict], adaptive_dp: List[Dict],
                       save_path: str = "graphs.png"):
    """
    Create publication-ready figures for QPrivIoT-FL paper.

    Args:
        no_dp: Results from baseline (no differential privacy)
        uniform_dp: Results from uniform DP
        adaptive_dp: Results from adaptive per-layer DP
        save_path: Where to save the figure
    """

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    rounds = list(range(1, len(no_dp) + 1))

    # ========================================================================
    # FIGURE 1: Model Accuracy Progress (Main Result)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    no_dp_acc = [m['val_accuracy'] * 100 for m in no_dp]
    uniform_acc = [m['val_accuracy'] * 100 for m in uniform_dp]
    adaptive_acc = [m['val_accuracy'] * 100 for m in adaptive_dp]

    ax1.plot(rounds, no_dp_acc, 'o-', linewidth=2.5, markersize=8,
             color='#2E86AB', label='No DP (Baseline)', alpha=0.9)
    ax1.plot(rounds, uniform_acc, 's-', linewidth=2.5, markersize=8,
             color='#A23B72', label='Uniform DP', alpha=0.9)
    ax1.plot(rounds, adaptive_acc, '^-', linewidth=2.5, markersize=8,
             color='#F18F01', label='QPrivIoT-FL (Adaptive)', alpha=0.9)

    ax1.set_xlabel('Training Round', fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax1.set_title('(a) Model Accuracy Across Federated Rounds', fontweight='bold', loc='left')
    ax1.legend(loc='lower right', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.5, len(rounds) + 0.5)
    ax1.set_ylim(0, max(no_dp_acc) * 1.1)

    # Add annotations for final values
    ax1.annotate(f'{no_dp_acc[-1]:.1f}%',
                 xy=(rounds[-1], no_dp_acc[-1]),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=9, fontweight='bold', color='#2E86AB')
    ax1.annotate(f'{uniform_acc[-1]:.1f}%',
                 xy=(rounds[-1], uniform_acc[-1]),
                 xytext=(10, -10), textcoords='offset points',
                 fontsize=9, fontweight='bold', color='#A23B72')
    ax1.annotate(f'{adaptive_acc[-1]:.1f}%',
                 xy=(rounds[-1], adaptive_acc[-1]),
                 xytext=(10, 5), textcoords='offset points',
                 fontsize=9, fontweight='bold', color='#F18F01')

    # ========================================================================
    # FIGURE 2: Training Loss
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    no_dp_loss = [m['val_loss'] for m in no_dp]
    uniform_loss = [m['val_loss'] for m in uniform_dp]
    adaptive_loss = [m['val_loss'] for m in adaptive_dp]

    ax2.plot(rounds, no_dp_loss, 'o-', linewidth=2, markersize=6,
             color='#2E86AB', label='No DP', alpha=0.8)
    ax2.plot(rounds, uniform_loss, 's-', linewidth=2, markersize=6,
             color='#A23B72', label='Uniform DP', alpha=0.8)
    ax2.plot(rounds, adaptive_loss, '^-', linewidth=2, markersize=6,
             color='#F18F01', label='Adaptive DP', alpha=0.8)

    ax2.set_xlabel('Round', fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontweight='bold')
    ax2.set_title('(b) Validation Loss', fontweight='bold', loc='left')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # ========================================================================
    # FIGURE 3: Privacy Budget Consumption
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Calculate cumulative epsilon
    uniform_eps_cumulative = []
    adaptive_eps_cumulative = []
    eps_sum_uniform = 0
    eps_sum_adaptive = 0

    for u, a in zip(uniform_dp, adaptive_dp):
        eps_sum_uniform += u.get('avg_epsilon', 0)
        eps_sum_adaptive += a.get('avg_epsilon', 0)
        uniform_eps_cumulative.append(eps_sum_uniform)
        adaptive_eps_cumulative.append(eps_sum_adaptive)

    ax3.plot(rounds, uniform_eps_cumulative, 's-', linewidth=2.5, markersize=7,
             color='#A23B72', label='Uniform DP', alpha=0.9)
    ax3.plot(rounds, adaptive_eps_cumulative, '^-', linewidth=2.5, markersize=7,
             color='#F18F01', label='Adaptive DP', alpha=0.9)

    # Add target line
    target_epsilon = 10.0
    ax3.axhline(y=target_epsilon, color='red', linestyle='--', linewidth=2,
                label=f'Target ε = {target_epsilon}', alpha=0.7)

    ax3.set_xlabel('Training Round', fontweight='bold')
    ax3.set_ylabel('Cumulative ε', fontweight='bold')
    ax3.set_title('(c) Privacy Budget Consumption', fontweight='bold', loc='left')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0.5, len(rounds) + 0.5)

    # ========================================================================
    # FIGURE 4: Privacy-Utility Tradeoff (Scatter)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate final metrics
    methods = ['No DP', 'Uniform DP', 'Adaptive DP']
    epsilons = [float('inf'), uniform_eps_cumulative[-1], adaptive_eps_cumulative[-1]]
    accuracies = [no_dp_acc[-1], uniform_acc[-1], adaptive_acc[-1]]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']

    for i, (method, eps, acc, color, marker) in enumerate(zip(methods, epsilons, accuracies, colors, markers)):
        if eps == float('inf'):
            # Plot No DP at far right
            ax4.scatter([25], [acc], s=200, color=color, marker=marker,
                        alpha=0.8, edgecolors='black', linewidths=1.5, label=method)
            ax4.annotate(f'{method}\n{acc:.1f}%',
                         xy=(25, acc), xytext=(-30, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        else:
            ax4.scatter([eps], [acc], s=200, color=color, marker=marker,
                        alpha=0.8, edgecolors='black', linewidths=1.5, label=method)
            ax4.annotate(f'{method}\n{acc:.1f}%\nε={eps:.1f}',
                         xy=(eps, acc), xytext=(5, -20),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

    ax4.set_xlabel('Total Privacy Cost (ε)', fontweight='bold')
    ax4.set_ylabel('Final Accuracy (%)', fontweight='bold')
    ax4.set_title('(d) Privacy-Utility Tradeoff', fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(-1, 27)
    ax4.set_ylim(20, 60)

    # ========================================================================
    # FIGURE 5: Per-Layer Sensitivity (Your Innovation!)
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 2])

    # Extract sensitivity data from a few representative rounds
    rounds_to_show = [1, 5, 10]
    sensitivities = {round_num: adaptive_dp[round_num - 1].get('avg_sensitivity', 0)
                     for round_num in rounds_to_show}

    x = np.arange(len(rounds_to_show))
    width = 0.6

    colors_gradient = ['#FFA07A', '#FF6347', '#DC143C']
    bars = ax5.bar(x, [sensitivities[r] for r in rounds_to_show],
                   width, color=colors_gradient, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, round_num in zip(bars, rounds_to_show):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax5.set_xlabel('Training Round', fontweight='bold')
    ax5.set_ylabel('Avg. Model Sensitivity', fontweight='bold')
    ax5.set_title('(e) Model Parameter Sensitivity', fontweight='bold', loc='left')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'Round {r}' for r in rounds_to_show])
    ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax5.set_ylim(0, max(sensitivities.values()) * 1.2)

    # ========================================================================
    # Overall title
    # ========================================================================
    fig.suptitle('QPrivIoT-FL: Experimental Results on CIFAR-10',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"figures saved to: {save_path}")
    plt.close()


def print_summary_table(no_dp: List[Dict], uniform_dp: List[Dict], adaptive_dp: List[Dict]):
    """Print a summary table for the paper."""

    print("\n" + "=" * 80)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 80)
    print()
    print("| Method          | Final Acc | Total ε | Avg ε/Round | Improvement vs Uniform |")
    print("|-----------------|-----------|---------|-------------|------------------------|")

    no_dp_acc = no_dp[-1]['val_accuracy'] * 100
    uniform_acc = uniform_dp[-1]['val_accuracy'] * 100
    adaptive_acc = adaptive_dp[-1]['val_accuracy'] * 100

    uniform_total_eps = sum([m.get('avg_epsilon', 0) for m in uniform_dp])
    adaptive_total_eps = sum([m.get('avg_epsilon', 0) for m in adaptive_dp])

    improvement = adaptive_acc - uniform_acc

    print(f"| No DP           | {no_dp_acc:5.1f}%    | ∞       | -           | -                      |")
    print(
        f"| Uniform DP      | {uniform_acc:5.1f}%    | {uniform_total_eps:5.1f}   | {uniform_total_eps / len(uniform_dp):5.2f}        | -                      |")
    print(
        f"| QPrivIoT-FL     | {adaptive_acc:5.1f}%    | {adaptive_total_eps:5.1f}   | {adaptive_total_eps / len(adaptive_dp):5.2f}        | +{improvement:4.1f} pp ({improvement / uniform_acc * 100:.1f}%)        |")
    print()
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Load your experiment results
    no_dp, uniform_dp, adaptive_dp = load_results(
        no_dp_path="results_no_dp.json",
        uniform_dp_path="results_uniform_dp.json",
        adaptive_dp_path="results_adaptive_dp.json"
    )

    # Generate paper figures
    plot_paper_figures(no_dp, uniform_dp, adaptive_dp, save_path="graphs.png")

    # Print summary table
    print_summary_table(no_dp, uniform_dp, adaptive_dp)
