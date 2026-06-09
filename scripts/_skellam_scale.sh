#!/usr/bin/env bash
# Fix A scale test: does distributed Skellam DP under SecAgg cross into usable accuracy
# as the per-round client cohort N grows? Prediction: distributed mean noise ∝ σC/N,
# local ∝ σC/√N — so distributed should recover accuracy at much smaller N than local.
# MNIST, T=20, seed=42, ε=8, full participation (fraction-fit=1.0 → cohort = N).
set -uo pipefail
ROUNDS=20; SEED=42; DATASET=mnist; EPS=8.0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local tag=$1 N=$2 mode=$3
    local dpflags extra
    if [ "$mode" = "no-dp" ]; then
        dpflags='use-dp=false'
        extra=''
    else
        dpflags="use-dp=true use-secagg=true secagg-dp-mode=\"$mode\""
        extra=''
    fi
    log "START $tag (N=$N)"
    flwr run . \
      --federation-config "options.num-supernodes=$N" \
      --run-config "dataset=\"$DATASET\" $dpflags num-clients=$N min-fit-clients=$N min-available-clients=$N seed=$SEED target-epsilon=$EPS num-server-rounds=$ROUNDS" \
      > "logs/skellam_scale_${tag}.log" 2>&1
    log "DONE  $tag"
}

for N in 50 100 200; do
    run "local_n$N"       "$N" local
    run "distributed_n$N" "$N" distributed
done
run "nodp_n100" 100 no-dp   # ceiling reference at scale
log "ALL DONE"
