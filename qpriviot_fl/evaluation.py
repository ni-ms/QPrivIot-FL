"""Experiment suite for paper - WITH BASELINE COMPARISONS."""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datetime import datetime


def run_experiment(
        experiment_name: str,
        dataset: str,
        num_rounds: int,
        use_dp: bool = True,
        use_adaptive_dp: bool = True,
        num_clients: int = 10,
        use_secagg: bool = True,
) -> Dict:
    """
    Run a single FL experiment.
    
    Args:
        experiment_name: Unique identifier for this experiment
        dataset: "cifar10", "femnist", or "iot"
        num_rounds: Number of training rounds
        use_dp: Whether to use differential privacy
        use_adaptive_dp: If True, use adaptive DP; if False, use standard fixed DP
        num_clients: Number of clients
        use_secagg: Whether to use secure aggregation
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'=' * 80}")
    print(f"🧪 Experiment: {experiment_name}")
    print(f"   Dataset: {dataset} | Rounds: {num_rounds} | Clients: {num_clients}")
    print(f"   DP: {use_dp} | Adaptive: {use_adaptive_dp} | SecAgg: {use_secagg}")
    print(f"{'=' * 80}\n")

    cmd = ["flwr", "run", "."]

    run_config = {
        "num-server-rounds": num_rounds,
        "dataset": dataset,
        "batch-size": 32,
        "local-epochs": 2,
        "learning-rate": 0.001,
        "use-dp": use_dp,
        "use-adaptive-dp": use_adaptive_dp,
        "num-supernodes": num_clients,
    }

    config_args = []
    for key, value in run_config.items():
        config_args.append(f"{key}={value}")

    cmd.extend(["--run-config", " ".join(config_args)])

    print(f"💻 Running command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

    metrics = parse_results(result.stdout)

    if result.returncode == 0:
        print(f"✅ Experiment completed!")
        print(f"   Final Accuracy: {metrics.get('final_accuracy', 'N/A'):.2%}")
        print(f"   Total Privacy (ε): {metrics.get('total_epsilon', 'N/A'):.2f}")
    else:
        print(f"❌ Experiment failed!")
        print(f"   Error: {result.stderr[:200]}")

    results = {
        "experiment_name": experiment_name,
        "dataset": dataset,
        "num_rounds": num_rounds,
        "use_dp": use_dp,
        "use_adaptive_dp": use_adaptive_dp,
        "num_clients": num_clients,
        "success": result.returncode == 0,
        **metrics,
        "timestamp": datetime.now().isoformat(),
    }

    return results


def parse_results(output: str) -> Dict:
    """
    Extract metrics from Flower output.
    
    Looks for patterns like:
    - "Val Accuracy: 30.96%"
    - "Total Privacy Spent (ε): 10.4109"
    - "accuracy': [(1, 0.1186), ..., (10, 0.3096)]"
    """
    metrics = {}

    accuracy_match = re.findall(r"'accuracy'.*?\((\d+),\s*([\d.]+)\)", output)
    if accuracy_match:
        final_round, final_acc = accuracy_match[-1]
        metrics["final_accuracy"] = float(final_acc)
        metrics["final_round"] = int(final_round)

    epsilon_match = re.search(r"Total Privacy Spent.*?ε.*?:\s*([\d.]+)", output)
    if epsilon_match:
        metrics["total_epsilon"] = float(epsilon_match.group(1))
    else:
        metrics["total_epsilon"] = 0.0

    loss_match = re.findall(r"round \d+:\s*([\d.]+)", output)
    if loss_match:
        metrics["final_loss"] = float(loss_match[-1])

    time_match = re.search(r"Run finished.*?in\s*([\d.]+)s", output)
    if time_match:
        metrics["training_time"] = float(time_match.group(1))

    return metrics


def run_baseline_comparison(dataset: str = "cifar10", num_rounds: int = 10):
    """
    Run the THREE experiments needed to prove your method works:
    1. No DP (upper bound)
    2. Standard DP (baseline)
    3. Your Adaptive DP (proposed method)
    """
    print(f"\n{'🔬' * 40}")
    print(f"BASELINE COMPARISON SUITE - {dataset.upper()}")
    print(f"{'🔬' * 40}\n")

    experiments = [
        {
            "experiment_name": f"{dataset}_no_dp",
            "dataset": dataset,
            "num_rounds": num_rounds,
            "use_dp": False,
            "use_adaptive_dp": False,
        },
        {
            "experiment_name": f"{dataset}_standard_dp",
            "dataset": dataset,
            "num_rounds": num_rounds,
            "use_dp": True,
            "use_adaptive_dp": False,
        },
        {
            "experiment_name": f"{dataset}_adaptive_dp",
            "dataset": dataset,
            "num_rounds": num_rounds,
            "use_dp": True,
            "use_adaptive_dp": True,
        },
    ]

    results = []
    for exp in experiments:
        result = run_experiment(**exp)
        results.append(result)

    return results


def create_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """Create comparison table for paper."""
    df = pd.DataFrame(results)

    cols = [
        "experiment_name",
        "dataset",
        "final_accuracy",
        "total_epsilon",
        "final_loss",
        "training_time",
        "success",
    ]

    df = df[cols]

    if len(df) >= 3:
        baseline_acc = df[df["experiment_name"].str.contains("standard")]["final_accuracy"].values[0]
        adaptive_acc = df[df["experiment_name"].str.contains("adaptive")]["final_accuracy"].values[0]
        improvement = ((adaptive_acc - baseline_acc) / baseline_acc) * 100

        print(f"\n📊 RESULTS SUMMARY:")
        print(f"{'=' * 60}")
        print(df.to_string(index=False))
        print(f"{'=' * 60}")
        print(f"\n🎯 KEY FINDING:")
        print(f"   Adaptive DP improves accuracy by {improvement:+.2f}% over Standard DP")
        print(f"   ({adaptive_acc:.2%} vs {baseline_acc:.2%})")
        print(f"{'=' * 60}\n")

    return df


def save_results(results: List[Dict], filename: str = "experiment_results.json"):
    """Save results to JSON file."""
    output_file = Path(filename)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to: {output_file}")


def main():
    """Run complete experimental suite."""

    print("\n" + "=" * 80)
    print("PHASE 1: CIFAR-10 BASELINE COMPARISON (10 rounds)")
    print("=" * 80)

    cifar10_results = run_baseline_comparison("cifar10", num_rounds=10)
    df_cifar10 = create_comparison_table(cifar10_results)
    save_results(cifar10_results, "results_cifar10_comparison.json")

    print("\n" + "=" * 80)
    print("PHASE 2: EXTENDED CIFAR-10 (20 rounds)")
    print("=" * 80)

    cifar10_extended = run_baseline_comparison("cifar10", num_rounds=20)
    df_extended = create_comparison_table(cifar10_extended)
    save_results(cifar10_extended, "results_cifar10_extended.json")

    print("\n" + "=" * 80)
    print("PHASE 3: FEMNIST VALIDATION")
    print("=" * 80)

    femnist_results = run_baseline_comparison("femnist", num_rounds=10)
    df_femnist = create_comparison_table(femnist_results)
    save_results(femnist_results, "results_femnist_comparison.json")

    print("\n" + "🎉" * 40)
    print("FINAL SUMMARY - ALL EXPERIMENTS")
    print("🎉" * 40 + "\n")

    all_results = cifar10_results + cifar10_extended + femnist_results
    df_all = pd.DataFrame(all_results)

    df_all.to_csv("all_results.csv", index=False)
    print("📊 All results saved to: all_results.csv")

    print("\n📝 LaTeX Table (copy to paper):")
    print("=" * 80)
    print(df_all[["experiment_name", "final_accuracy", "total_epsilon"]].to_latex(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
