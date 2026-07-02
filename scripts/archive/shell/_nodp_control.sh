#!/usr/bin/env bash
# No-DP control curve across client count N, to separate the two forces in the Fix A
# crossover: DP-noise dominance (small N) vs DATA STARVATION / non-IID (large N).
# If no-dp ALSO collapses at N=200, the upper-end distributed-DP collapse is statistical,
# not a privacy/mechanism artifact. N=10 no-dp is already known (98.65%).
# MNIST, T=20, seed=42, full participation (cohort = N).
set -uo pipefail
ROUNDS=20; SEED=42; DATASET=mnist
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local N=$1
    log "START nodp N=$N"
    flwr run . \
      --federation-config "options.num-supernodes=$N" \
      --run-config "dataset=\"$DATASET\" use-dp=false num-clients=$N min-fit-clients=$N min-available-clients=$N seed=$SEED target-epsilon=8.0 num-server-rounds=$ROUNDS" \
      > "logs/nodp_control_n${N}.log" 2>&1
    log "DONE  nodp N=$N"
}

for N in 50 100 200; do run "$N"; done
log "ALL DONE"