"""Experiment evaluation script for paper results."""

import subprocess
import json
from pathlib import Path


def run_experiment(dataset: str, num_rounds: int, num_clients: int, use_secagg: bool = True):
    """Run a single FL experiment."""

    print(f"\n{'=' * 80}")
    print(f"Running experiment: {dataset}, {num_rounds} rounds, {num_clients} clients, SecAgg={use_secagg}")
    print(f"{'=' * 80}\n")

    cmd = [
        "flower-simulation",
        "--client-app", "qpriviot_fl.client:app",
        "--server-app", "qpriviot_fl.server:app",
        "--num-supernodes", str(num_clients),
    ]

    run_config = {
        "num-server-rounds": num_rounds,
        "dataset": dataset,
        "batch-size": 32,
        "local-epochs": 2,
        "learning-rate": 0.001,
        "fraction-fit": 0.8,
        "fraction-evaluate": 0.5,
        "min-fit-clients": 3,
        "min-available-clients": 3,
        "use-secagg": use_secagg,
        "num-shares": 3,
        "reconstruction-threshold": 2,
        "max-weight": 1000,
    }

    config_str = json.dumps(run_config)
    cmd.extend(["--run-config", config_str])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Experiment completed successfully!")
    else:
        print(f"❌ Experiment failed:")
        print(result.stderr)

    return result.returncode == 0


def run_all_experiments():
    """Run full experimental suite for paper."""

    experiments = [
        {"dataset": "cifar10", "num_rounds": 20, "num_clients": 10, "use_secagg": True},
        {"dataset": "cifar10", "num_rounds": 20, "num_clients": 10, "use_secagg": False},
        {"dataset": "femnist", "num_rounds": 15, "num_clients": 20, "use_secagg": True},
        {"dataset": "iot", "num_rounds": 10, "num_clients": 5, "use_secagg": True},
    ]

    results = []

    for exp in experiments:
        success = run_experiment(**exp)
        results.append({**exp, "success": success})

    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")

    for i, (exp, res) in enumerate(zip(experiments, results), 1):
        status = "✅ SUCCESS" if res["success"] else "❌ FAILED"
        print(f"{i}. {exp['dataset']} ({exp['num_rounds']} rounds, SecAgg={exp['use_secagg']}): {status}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    run_all_experiments()
