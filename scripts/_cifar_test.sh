#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
run() {
  local cfg=$1 eps=$2 flags=$3
  echo "[$(date '+%H:%M:%S')] START $cfg eps=$eps"
  flwr run . --run-config "dataset=\"cifar10\" $flags seed=42 target-epsilon=$eps num-server-rounds=20" 2>&1 \
    | grep -E "DP CALIB|Error|Traceback|Exception" || true
  echo "[$(date '+%H:%M:%S')] DONE $cfg eps=$eps"
}
EPS=8.0
run no-dp      "$EPS" "use-dp=false use-adaptive-dp=false"
run fixed-dp   "$EPS" "use-dp=true  use-adaptive-dp=false"
run param-only "$EPS" "use-dp=false use-adaptive-dp=true ablation-mode=\"param-only\""
echo "CIFAR_TEST_COMPLETE"
