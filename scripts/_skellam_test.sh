#!/usr/bin/env bash
# Fix A test: distributed vs local Skellam DP under SecAgg on MNIST (T=20, seed=42).
# Compares the new integer-domain distributed mechanism against local (√N more noise)
# and the legacy continuous-noise fixed-dp baseline.
set -uo pipefail
ROUNDS=20; SEED=42; DATASET=mnist
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local tag=$1; shift
    local cfg="$*"
    log "START $tag"
    flwr run . --run-config "$cfg" > "logs/skellam_${tag}.log" 2>&1
    log "DONE  $tag"
}

# no-DP ceiling (eps label arbitrary)
run "no-dp_eps8"        "dataset=\"$DATASET\" use-dp=false seed=$SEED target-epsilon=8.0 num-server-rounds=$ROUNDS"

for EPS in 8.0 3.0; do
    run "fixed_eps$EPS"       "dataset=\"$DATASET\" use-dp=true seed=$SEED target-epsilon=$EPS num-server-rounds=$ROUNDS"
    run "local_eps$EPS"       "dataset=\"$DATASET\" use-dp=true use-secagg=true secagg-dp-mode=\"local\"       seed=$SEED target-epsilon=$EPS num-server-rounds=$ROUNDS"
    run "distributed_eps$EPS" "dataset=\"$DATASET\" use-dp=true use-secagg=true secagg-dp-mode=\"distributed\" seed=$SEED target-epsilon=$EPS num-server-rounds=$ROUNDS"
done
log "ALL DONE"
