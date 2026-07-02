#!/usr/bin/env bash
# FEMNIST no-dp convergence diagnostic: find local-epochs/lr that actually trains the
# 62-class FemnistNet at T=20 (default le=1,lr=0.01 barely moves loss from random 4.13).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; mkdir -p logs
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run(){ local tag=$1 le=$2 lr=$3
  log "START $tag (le=$le lr=$lr)"
  flwr run . --federation-config "options.num-supernodes=50" \
    --run-config "dataset=\"femnist\" use-dp=false num-clients=50 min-fit-clients=50 min-available-clients=50 seed=42 target-epsilon=8.0 num-server-rounds=20 local-epochs=$le learning-rate=$lr" \
    > "logs/femnist_tune_${tag}.log" 2>&1
  log "DONE  $tag"; }
run le5_lr05 5 0.05
run le5_lr01 5 0.01
log "ALL DONE"
