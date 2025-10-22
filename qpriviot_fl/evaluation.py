"""QPrivIot-FL: Evaluation and visualization module."""
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics(dataset_name="cifar10"):
    """Load training metrics from file."""
    metrics_file = f"training_metrics_{dataset_name}.npz"
    data = np.load(metrics_file)
    return {key: data[key] for key in data.files}


def plot_privacy_utility_tradeoff(metrics, output_dir="plots"):
    """Plot privacy-utility tradeoff curves."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    rounds = metrics["round"]

    ax1.plot(rounds, metrics["avg_loss"], 'b-o', label='Training Loss')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Rounds')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(rounds, metrics["avg_accuracy"], 'g-o', label='Training Accuracy')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy Over Rounds')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3_twin = ax3.twinx()
    ax3.plot(rounds, metrics["privacy_budget"], 'r-o', label='Privacy Budget')
    ax3_twin.plot(rounds, metrics["avg_epsilon"], 'purple', linestyle='--', marker='s', label='Epsilon Spent')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Privacy Budget', color='r')
    ax3_twin.set_ylabel('Epsilon', color='purple')
    ax3.set_title('Adaptive Privacy Budget Scheduling')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    ax4.plot(rounds, metrics["avg_energy"], 'orange', marker='D', label='Avg Energy Consumed')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Energy Consumed (%)')
    ax4.set_title('Average Energy Consumption Per Round')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/privacy_utility_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_dir}/privacy_utility_analysis.png")
    plt.close()


def plot_epsilon_accuracy_tradeoff(metrics, output_dir="plots"):
    """Plot epsilon vs accuracy tradeoff."""
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["avg_epsilon"], metrics["avg_accuracy"], 'bo-', markersize=8)

    for i, round_num in enumerate(metrics["round"]):
        if i % 2 == 0:
            plt.annotate(f'R{round_num}',
                         (metrics["avg_epsilon"][i], metrics["avg_accuracy"][i]),
                         textcoords="offset points", xytext=(5, 5), ha='left')

    plt.xlabel('Privacy Cost (Epsilon)')
    plt.ylabel('Model Accuracy')
    plt.title('Privacy-Utility Tradeoff: Epsilon vs Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/epsilon_accuracy_tradeoff.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_dir}/epsilon_accuracy_tradeoff.png")
    plt.close()


def generate_comparison_table(metrics):
    """Generate comparison table for paper."""
    final_round = -1

    table_data = {
        "Metric": [
            "Final Accuracy",
            "Final Loss",
            "Total Privacy Spent (ε)",
            "Initial Privacy Budget",
            "Final Privacy Budget",
            "Avg Energy per Round (%)",
            "Total Rounds",
        ],
        "Value": [
            f"{metrics['avg_accuracy'][final_round]:.4f}",
            f"{metrics['avg_loss'][final_round]:.4f}",
            f"{metrics['avg_epsilon'][final_round]:.4f}",
            f"{metrics['privacy_budget'][0]:.4f}",
            f"{metrics['privacy_budget'][final_round]:.4f}",
            f"{np.mean(metrics['avg_energy']):.4f}",
            f"{len(metrics['round'])}",
        ]
    }

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY TABLE")
    print("=" * 50)
    for metric, value in zip(table_data["Metric"], table_data["Value"]):
        print(f"{metric:.<40} {value}")
    print("=" * 50 + "\n")

    return table_data


def compare_baseline_vs_adaptive(adaptive_metrics, baseline_metrics=None):
    """Compare adaptive approach vs baseline uniform DP."""
    if baseline_metrics is None:
        print("\nNote: Baseline metrics not provided. Run baseline experiment for comparison.")
        return

    print("\n" + "=" * 60)
    print("ADAPTIVE VS BASELINE COMPARISON")
    print("=" * 60)

    comparison = {
        "Metric": ["Final Accuracy", "Final Epsilon", "Avg Energy"],
        "Baseline": [
            f"{baseline_metrics['avg_accuracy'][-1]:.4f}",
            f"{baseline_metrics['avg_epsilon'][-1]:.4f}",
            f"{np.mean(baseline_metrics['avg_energy']):.4f}",
        ],
        "Adaptive": [
            f"{adaptive_metrics['avg_accuracy'][-1]:.4f}",
            f"{adaptive_metrics['avg_epsilon'][-1]:.4f}",
            f"{np.mean(adaptive_metrics['avg_energy']):.4f}",
        ],
        "Improvement": [
            f"{(adaptive_metrics['avg_accuracy'][-1] - baseline_metrics['avg_accuracy'][-1]) * 100:.2f}%",
            f"{(baseline_metrics['avg_epsilon'][-1] - adaptive_metrics['avg_epsilon'][-1]):.4f}",
            f"{(baseline_metrics['avg_energy'].mean() - adaptive_metrics['avg_energy'].mean()):.2f}%",
        ]
    }

    for i, metric in enumerate(comparison["Metric"]):
        print(f"{metric}:")
        print(f"  Baseline:   {comparison['Baseline'][i]}")
        print(f"  Adaptive:   {comparison['Adaptive'][i]}")
        print(f"  Improvement: {comparison['Improvement'][i]}")
        print()


if __name__ == "__main__":
    dataset = "cifar10"

    print(f"Loading metrics for {dataset}...")
    metrics = load_metrics(dataset)

    print("Generating evaluation plots...")
    plot_privacy_utility_tradeoff(metrics)
    plot_epsilon_accuracy_tradeoff(metrics)

    print("Generating summary table...")
    generate_comparison_table(metrics)

    print("\nEvaluation complete!")
