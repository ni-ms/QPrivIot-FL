#!/usr/bin/env bash
# Fix A scale sweep — GAP FILL after the first run was interrupted at local_n100.
# Already have (good, 20 rounds): distributed_n50=81.66% peak, local_n50=19.6% peak.
# This fills the load-bearing cells: distributed at N=100/200 (does the 1/N benefit keep
# stabilising?), local at N=100/200 (control — should stay broken), and a no-dp ceiling.
# Order: distributed first (most informative) in case of another interruption.
# MNIST, T=20, seed=42, ε=8, full participation (cohort = N).
set -uo pipefail
ROUNDS=20; SEED=42; DATASET=mnist; EPS=8.0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local tag=$1 N=$2 mode=$3
    local dpflags
    if [ "$mode" = "no-dp" ]; then
        dpflags='use-dp=false'
    else
        dpflags="use-dp=true use-secagg=true secagg-dp-mode=\"$mode\""
    fi
    log "START $tag (N=$N)"
    flwr run . \
      --federation-config "options.num-supernodes=$N" \
      --run-config "dataset=\"$DATASET\" $dpflags num-clients=$N min-fit-clients=$N min-available-clients=$N seed=$SEED target-epsilon=$EPS num-server-rounds=$ROUNDS" \
      > "logs/skellam_scale_${tag}.log" 2>&1
    log "DONE  $tag"
}

run "distributed_n100" 100 distributed
run "distributed_n200" 200 distributed
run "local_n100"       100 local
run "local_n200"       200 local
run "nodp_n100"        100 no-dp
log "ALL DONE"