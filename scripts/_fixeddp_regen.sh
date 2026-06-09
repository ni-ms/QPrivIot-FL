#!/usr/bin/env bash
# Regenerate the no-SecAgg fixed-dp baseline at the CORRECT σ, after removing the
# /√num_samples deflation (FINDING 2). Legacy fixed-dp numbers (~98%) were non-private.
# Each client now adds calibrated N(0,(σ·C)²) per-client (no trusted server, no SecAgg),
# so the effective averaged noise is σC/√N — the honest "naive central-DP-via-local-noise"
# baseline against distributed-Skellam-under-SecAgg. MNIST, T=20, seed=42.
set -uo pipefail
ROUNDS=20; SEED=42; DATASET=mnist
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local N=$1 EPS=$2
    log "START fixed-dp N=$N eps=$EPS"
    flwr run . \
      --federation-config "options.num-supernodes=$N" \
      --run-config "dataset=\"$DATASET\" use-dp=true num-clients=$N min-fit-clients=$N min-available-clients=$N seed=$SEED target-epsilon=$EPS num-server-rounds=$ROUNDS" \
      > "logs/fixeddp_regen_n${N}_eps${EPS}.log" 2>&1
    log "DONE  fixed-dp N=$N eps=$EPS"
}

# ε=8 across the N grid (apples-to-apples with the distributed/local/no-dp curves)
for N in 10 50 100 200; do run "$N" 8.0; done
# ε=3 at N=10 to replace the invalid headline number
run 10 3.0
log "ALL DONE"