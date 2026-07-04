#!/usr/bin/env bash
# Calibrated LiRA MIA + recon-decode extraction (upgrades the cosine proxy of §7.3).
# Offline LiRA with shadow releases; reports AUC + low-FPR TPR + extraction decode over an
# eps sweep, K-swept to mirror the oracle leakage table. Idempotent. GPU not required.
#
#   bash scripts/shell/rerun_lira.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY=.venv/bin/python; [[ -x "$PY" ]] || PY=.venv/Scripts/python.exe
export PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0
AGENTMEM=scripts/agentmem
SEEDS=${SEEDS:-0,1,2}
SHADOWS=${SHADOWS:-48}
TARGETS=${TARGETS:-1200}
OUT=experiment_results/lira
mkdir -p "$OUT" logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
  local tag=$1; shift
  local json="$OUT/${tag}.json"
  if [[ -f "$json" ]]; then log "SKIP  $tag (exists)"; return 0; fi
  log "RUN   $tag"
  if $PY "${AGENTMEM}/_agentmem_lira.py" "$@" --shadows "$SHADOWS" --targets "$TARGETS" \
        --seeds "$SEEDS" --json "$json" > "logs/${tag}.log" 2>&1; then
    log "OK    $tag"
  else
    log "FAIL  $tag (see logs/${tag}.log)"
  fi
}

log "=== LiRA grid | seeds=$SEEDS shadows=$SHADOWS targets=$TARGETS ==="
# oracle K-sweep @ d=32 (mirrors §7.3 tail-AUC table)
for K in 512 1024 2048; do
  run lira_oracle_K${K}_d32 --source oracle --K $K --d 32 --N 500 --pool 8000 --eps 16,8,3,1
done
# real ST embeddings, d-contrast at fixed K (mirrors §7.2 crossover)
run lira_st_K1024_d32  --source st --K 1024 --d 32  --N 100 --pool 4000 --eps 16,8,3,1
run lira_st_K1024_d384 --source st --K 1024 --d 384 --N 100 --pool 4000 --eps 16,8,3,1
log "=== done. JSON in $OUT/ ==="
