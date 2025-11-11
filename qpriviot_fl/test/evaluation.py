"""
SIMPLEST WORKING SOLUTION - Run experiments using your existing flwr setup.

Just call `flwr run` properly for each configuration.
"""

import subprocess
import time
from pathlib import Path
import json

def run_experiment(name: str, use_dp: str, use_adaptive_dp: str):
    """Run single experiment."""

    print(f"\n{'='*80}")
    print(f"🧪 {name}")
    print(f"   use-dp={use_dp}, use-adaptive-dp={use_adaptive_dp}")
    print(f"{'='*80}\n")

    cmd = [
        "flwr", "run", ".",
        "--run-config",
        f"num-server-rounds=10 dataset=cifar10 use-dp={use_dp} use-adaptive-dp={use_adaptive_dp}"
    ]

    print(f"Running: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    print(f"\n✅ Completed in {elapsed/60:.1f} minutes\n")
    return result.returncode == 0

def main():
    """Run all 3 experiments."""

    print("\n" + "🚀"*40)
    print("QPrivIoT-FL EXPERIMENTS")
    print("🚀"*40 + "\n")

    experiments = [
        ("No DP", "false", "false"),
        ("Uniform DP", "true", "false"),
        ("Adaptive DP", "true", "true"),
    ]

    for name, use_dp, use_adaptive_dp in experiments:
        success = run_experiment(name, use_dp, use_adaptive_dp)
        if not success:
            print(f"⚠️ {name} had issues")
        time.sleep(2)  # Brief pause between experiments

    print("\n" + "="*80)
    print("✅ ALL DONE!")
    print("="*80)
    print("\nCheck the output above for results.")
    print("Metrics should be in:")
    print("  - training_metrics_cifar10.json")
    print("  - results_cifar10_1.png")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
