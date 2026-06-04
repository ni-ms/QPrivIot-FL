#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
run() {
  local cfg=$1 eps=$2 flags=$3
  rm -f "experiment_results/results_mnist_${cfg}_seed42_eps${eps}.json" \
        "experiment_results/results_mnist_${cfg}_seed42_eps${eps}_per_client.csv"
  echo "[$(date '+%H:%M:%S')] START $cfg eps=$eps"
  flwr run . --run-config "dataset=\"mnist\" $flags seed=42 target-epsilon=$eps num-server-rounds=20" 2>&1 \
    | grep -E "DP CALIB|Error|Traceback|Exception" || true
  echo "[$(date '+%H:%M:%S')] DONE $cfg eps=$eps"
}
for EPS in 3.0 8.0; do
  run fixed-dp    "$EPS" "use-dp=true  use-adaptive-dp=false"
  run device-only "$EPS" "use-dp=false use-adaptive-dp=true ablation-mode=\"device-only\""
done
echo "FAIRNESS_TEST_COMPLETE"
