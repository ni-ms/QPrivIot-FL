#!/usr/bin/env bash
# At-scale generality check: the LongMemEval **s** variant (full haystacks,
# ~247k notes, ~96% distractors) — the paper's [PENDING] #1 run. Mirrors the
# oracle configs in rerun_grid.sh at --variant s. The first probe config embeds
# all ~247k notes once (GPU) and caches to experiment_results/_st_lme_s.npz;
# every later config reuses that cache. Idempotent: skips existing JSONs.
#
#   bash scripts/shell/rerun_svariant.sh            # seeds 0-4
#   SEEDS=0 bash scripts/shell/rerun_svariant.sh    # quick single-seed pass
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY=.venv/bin/python; [[ -x "$PY" ]] || PY=.venv/Scripts/python.exe   # Linux | Windows venv
AGENTMEM=scripts/agentmem
SEEDS=${SEEDS:-0,1,2,3,4}
OUT=experiment_results/rerun_grid_s
mkdir -p "$OUT" logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {  # run <tag> <script> <args...>
  local tag=$1; shift; local script=$1; shift
  local json="$OUT/${tag}.json"
  if [[ -f "$json" ]]; then log "SKIP  $tag (exists)"; return 0; fi
  log "RUN   $tag"
  if $PY "${AGENTMEM}/${script}" "$@" --seeds "$SEEDS" --json "$json" > "logs/${tag}.log" 2>&1; then
    log "OK    $tag"
  else
    log "FAIL  $tag (see logs/${tag}.log)"
  fi
}

log "=== rerun s-variant grid | seeds=$SEEDS ==="

# ---- utility (K-sweep @ d=32) — first config builds the 247k embedding cache ----
for K in 32 64 128 256; do
  run p6_probe_s_K${K}_d32 _longmemeval_probe.py --variant s --K $K --d 32 --topk 5 --release separate --eps 16,8,3
done

# ---- leakage (K-sweep @ d=32) ----
for K in 512 1024 2048; do
  run p6_leak_s_K${K}_d32 _longmemeval_leakage.py --variant s --K $K --d 32 --M 2000 --lowcount 3 --eps 16,8,3,1
done

log "=== done. JSON in $OUT/ ==="
