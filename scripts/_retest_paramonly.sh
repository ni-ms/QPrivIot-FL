#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
for EPS in 3.0 8.0; do
  rm -f "experiment_results/results_mnist_param-only_seed42_eps${EPS}.json" \
        "experiment_results/results_mnist_param-only_seed42_eps${EPS}_per_client.csv"
  echo "[$(date '+%H:%M:%S')] START param-only eps=$EPS"
  flwr run . --run-config "dataset=\"mnist\" use-dp=false use-adaptive-dp=true ablation-mode=\"param-only\" seed=42 target-epsilon=$EPS num-server-rounds=20" 2>&1 \
    | grep -E "DP CALIB|Error|Traceback|Exception" || true
  echo "[$(date '+%H:%M:%S')] DONE param-only eps=$EPS"
done
echo "RETEST_COMPLETE"
