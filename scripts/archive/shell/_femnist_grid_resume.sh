#!/usr/bin/env bash
# Resume the FEMNIST crossover grid after the 2026-06-10 interruption (process died during
# distributed N=200, round 4). Completed & KEPT (tuned le=5/lr=0.05): distributed N={10,50,100}
# and no-dp N=50. This script runs ONLY what is missing, at the SAME tuned hyperparams:
#   1. distributed N=200            (the KEY scientific test — does it stay usable, no upper collapse?)
#   2. no-dp   N={10,100,200}       (ceiling; N=50 already tuned, skipped)
#   3. local   N={10,50,100,200}    (naive-per-client control — expected to stay broken)
#   4. fixed-dp N={10,50,100,200}   (no-SecAgg control)
# The stale le=1 control files (4-9% acc) get overwritten with correct tuned runs.
# 62-class, T=20, seed=42, ε=8, full participation (cohort = N). N=200 runs ~50min each.
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

run distributed 200 distributed
for N in 10 100 200;     do run no-dp    "$N" no-dp;    done
for N in 10 50 100 200;  do run local    "$N" local;    done
for N in 10 50 100 200;  do run fixed-dp "$N" fixed-dp; done
log "ALL DONE"