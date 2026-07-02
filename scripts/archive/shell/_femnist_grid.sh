#!/usr/bin/env bash
# FEMNIST crossover grid — writer-partitioned (one client = one writer, ~constant ~400
# samples/client regardless of N). This DECOUPLES client-count from data-per-client, the
# confound that bounded the MNIST window's upper edge. Key test: does distributed Skellam
# stay usable at N=200 on FEMNIST (no upper collapse), unlike MNIST?
# 62-class, T=20, seed=42, ε=8, full participation (cohort = N).
# Hyperparams local-epochs=5 / lr=0.05: the harness default (le=1, lr=0.01) barely trains
# 62-class FEMNIST (loss stuck near random 4.13); le=5/lr=0.05 reaches ~80% no-dp at T=20.
# Order: distributed (hero) → no-dp (ceiling) → local + fixed-dp (controls), so the
# load-bearing curves land first if interrupted.
set -uo pipefail
ROUNDS=20; SEED=42; DATASET=femnist; EPS=8.0; LE=5; LR=0.05
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    local tag=$1 N=$2 mode=$3
    local dpflags
    case "$mode" in
        no-dp)       dpflags='use-dp=false' ;;
        fixed-dp)    dpflags='use-dp=true' ;;                                   # no SecAgg
        local|distributed) dpflags="use-dp=true use-secagg=true secagg-dp-mode=\"$mode\"" ;;
    esac
    log "START $tag (N=$N)"
    flwr run . \
      --federation-config "options.num-supernodes=$N" \
      --run-config "dataset=\"$DATASET\" $dpflags num-clients=$N min-fit-clients=$N min-available-clients=$N seed=$SEED target-epsilon=$EPS num-server-rounds=$ROUNDS local-epochs=$LE learning-rate=$LR" \
      > "logs/femnist_${tag}_n${N}.log" 2>&1
    log "DONE  $tag (N=$N)"
}

for N in 10 50 100 200; do run distributed "$N" distributed; done
for N in 10 50 100 200; do run no-dp       "$N" no-dp;       done
for N in 10 50 100 200; do run local       "$N" local;       done
for N in 10 50 100 200; do run fixed-dp    "$N" fixed-dp;    done
log "ALL DONE"