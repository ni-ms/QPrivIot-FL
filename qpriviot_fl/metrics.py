"""Comprehensive metrics logging and visualization."""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from pathlib import Path


class MetricsLogger:
    """Logger for training metrics across rounds."""

    def __init__(self):
        self.metrics_history: List[Dict] = []

    def log_round(self, metrics: Dict):
        """Log metrics for a round."""
        self.metrics_history.append(metrics)

    def get_all_metrics(self) -> List[Dict]:
        """Get all logged metrics."""
        return self.metrics_history

    def save_to_file(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"Metrics saved to {filepath}")

    def load_from_file(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)


def plot_comprehensive_results(
        metrics: List[Dict],
        privacy_report: Dict,
        save_path: str = "results.png"
):
    """
    Create comprehensive visualization of training results.
    """
    if not metrics:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('QPrivIoT-FL Training Results', fontsize=16, fontweight='bold')

    rounds = [m.get("round", i + 1) for i, m in enumerate(metrics)]

    ax = axes[0, 0]
    val_acc = [m.get("val_accuracy", 0) * 100 for m in metrics]
    ax.plot(rounds, val_acc, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax.set_title('Model Accuracy Progress')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    train_loss = [m.get("train_loss", 0) for m in metrics]
    val_loss = [m.get("val_loss", 0) for m in metrics]
    if any(train_loss):
        ax.plot(rounds, train_loss, 'r-o', label='Train Loss', linewidth=2, markersize=6)
    if any(val_loss):
        ax.plot(rounds, val_loss, 'b-s', label='Val Loss', linewidth=2, markersize=6)
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    epsilons = [m.get("avg_epsilon", 0) for m in metrics]
    cumulative_epsilon = np.cumsum(epsilons)
    ax.plot(rounds, cumulative_epsilon, 'g-o', linewidth=2, markersize=6)
    ax.axhline(y=privacy_report["target_epsilon"], color='r', linestyle='--',
               label=f'Target ε={privacy_report["target_epsilon"]}')
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Cumulative ε', fontweight='bold')
    ax.set_title('Privacy Budget Consumption')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    conv_scores = [m.get("convergence_score", 0) for m in metrics]
    ax.plot(rounds, conv_scores, 'm-o', linewidth=2, markersize=6)
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Convergence Score', fontweight='bold')
    ax.set_title('Training Convergence')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    resource_scores = [m.get("resource_score", 0) for m in metrics]
    cpu = [m.get("cpu_percent", 0) for m in metrics]
    ram = [m.get("ram_percent", 0) for m in metrics]
    if any(resource_scores):
        ax.plot(rounds, resource_scores, 'c-o', label='Resource Score', linewidth=2, markersize=6)
    if any(cpu):
        ax.plot(rounds, np.array(cpu) / 100, 'r--', label='CPU Usage', linewidth=1.5)
    if any(ram):
        ax.plot(rounds, np.array(ram) / 100, 'b--', label='RAM Usage', linewidth=1.5)
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Score / Usage', fontweight='bold')
    ax.set_title('Device Resource Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if any(epsilons) and any(val_acc):
        ax.scatter(cumulative_epsilon, val_acc, c=rounds, cmap='viridis', s=100, alpha=0.7)
        ax.set_xlabel('Cumulative ε', fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
        ax.set_title('Privacy-Utility Tradeoff')
        cbar = plt.colorbar(ax.collections[0], ax=ax, label='Round')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive results plot saved to {save_path}")
    plt.close()


def generate_latex_table(metrics: List[Dict], save_path: str = "results_table.tex"):
    """Generate LaTeX table for paper."""
    if not metrics:
        return

    final = metrics[-1]

    latex_content = r"""
\begin{table}[h]
\centering
\caption{QPrivIoT-FL Final Results}
\begin{tabular}{|l|c|}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
Validation Accuracy (\%) & """ + f"{final.get('val_accuracy', 0) * 100:.2f}" + r""" \\
Validation Loss & """ + f"{final.get('val_loss', 0):.4f}" + r""" \\
Total Privacy Budget ($\epsilon$) & """ + f"{sum([m.get('avg_epsilon', 0) for m in metrics]):.2f}" + r""" \\
Convergence Score & """ + f"{final.get('convergence_score', 0):.2f}" + r""" \\
Avg Resource Score & """ + f"{final.get('resource_score', 0):.2f}" + r""" \\
\hline
\end{tabular}
\end{table}
"""

    with open(save_path, 'w') as f:
        f.write(latex_content)

    print(f"LaTeX table saved to {save_path}")
