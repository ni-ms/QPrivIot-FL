#!/usr/bin/env bash
# Multi-seed final-metric reruns of the agent-memory paper grid (oracle / small configs).
# Expands the paper's 2-seed (0,1) tables to 5 seeds (0-4). All configs run on cached
# embeddings (no GPU needed). Idempotent: skips a config whose JSON already exists.
#
#   bash scripts/rerun_grid.sh            # full grid, seeds 0,1,2,3,4
#   SEEDS=0,1,2,3,4,5,6,7 bash scripts/rerun_grid.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY=.venv/bin/python
SEEDS=${SEEDS:-0,1,2,3,4}
OUT=experiment_results/rerun_grid
mkdir -p "$OUT" logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {  # run <tag> <script> <args...>
  local tag=$1; shift; local script=$1; shift
  local json="$OUT/${tag}.json"
  if [[ -f "$json" ]]; then log "SKIP  $tag (exists)"; return 0; fi
  log "RUN   $tag"
  if $PY "scripts/${script}" "$@" --seeds "$SEEDS" --json "$json" > "logs/${tag}.log" 2>&1; then
    log "OK    $tag"
  else
    log "FAIL  $tag (see logs/${tag}.log)"
  fi
}

log "=== rerun grid | seeds=$SEEDS ==="

# ---- Phase 0: synthetic TF-IDF utility (agentmem_probe) ----
run p0_A_highdim_N50_K64_d256   _agentmem_probe.py --N 50  --K 64 --d 256 --M 1000 --k 5 --alpha 0.3 --embedder tfidf --eps 8,3
run p0_B_tinyd_N200_K16_d64     _agentmem_probe.py --N 200 --K 16 --d 64  --M 1000 --k 5 --alpha 0.3 --embedder tfidf --eps 8,3
run p0_C_headline_N100_K32_d32  _agentmem_probe.py --N 100 --K 32 --d 32  --M 1000 --k 5 --alpha 0.3 --embedder tfidf --eps 16,8,3

# ---- Phase 2: LongMemEval oracle utility (K-sweep @ d=32) ----
for K in 32 64 128 256; do
  run p2_probe_oracle_K${K}_d32 _longmemeval_probe.py --variant oracle --K $K --d 32 --topk 5 --release separate --eps 16,8,3
done
# ---- Phase 2: LongMemEval oracle utility (d-sweep @ K=128) ----
for d in 64 128; do
  run p2_probe_oracle_K128_d${d} _longmemeval_probe.py --variant oracle --K 128 --d $d --topk 5 --release separate --eps 16,8,3
done

# ---- Phase 2: LongMemEval oracle leakage (K-sweep @ d=32) ----
for K in 512 1024 2048; do
  run p2_leak_oracle_K${K}_d32 _longmemeval_leakage.py --variant oracle --K $K --d 32 --M 2000 --lowcount 3 --eps 16,8,3,1
done

# ---- Phase 2: real sentence-embeddings (agentmem, --embedder st) ----
for d in 32 64 128 384; do
  run p2st_probe_N100_K32_d${d} _agentmem_probe.py   --N 100 --K 32   --d $d --M 1000 --k 5 --alpha 0.3 --embedder st --eps 16,8,3
  run p2st_leak_N100_K1024_d${d} _agentmem_leakage.py --N 100 --K 1024 --d $d --M 2000 --alpha 0.3 --lowcount 3 --embedder st --eps 16,8,3,1
done

# ---- Phase 4: synthetic TF-IDF leakage (K-sweep @ d=32) ----
for K in 256 1024 2048; do
  run p4_leak_tfidf_N100_K${K}_d32 _agentmem_leakage.py --N 100 --K $K --d 32 --M 2000 --alpha 0.3 --lowcount 3 --embedder tfidf --eps 16,8,3,1
done

# ---- Phase 5: LLM-distilled notes (vector-only utility + raw-vs-distilled leakage) ----
PILOT=experiment_results/_lme_distilled_oracle_pilot.json   # 100 users, 1784 distilled notes
FULL=experiment_results/_lme_distilled_oracle.json          # 500 users, 6252 distilled notes
# distilled UTILITY (answer-retrieval@5, vector-only release), K-sweep @ d=32
for K in 32 64 128; do
  run p5_distutil_pilot100_K${K}_d32 _longmemeval_distilled_utility.py --distilled "$PILOT" --K $K --d 32 --topk 5 --eps 16,8,3
  run p5_distutil_full500_K${K}_d32  _longmemeval_distilled_utility.py --distilled "$FULL"  --K $K --d 32 --topk 5 --eps 16,8,3
done
# distilled vs raw LEAKAGE, K-sweep @ d=32
for K in 256 512; do
  run p5_distleak_pilot100_K${K}_d32 _longmemeval_distilled_analysis.py --distilled "$PILOT" --K $K --d 32 --eps 16,8,3
done
for K in 512 1024; do
  run p5_distleak_full500_K${K}_d32 _longmemeval_distilled_analysis.py --distilled "$FULL" --K $K --d 32 --eps 16,8,3
done

log "=== done. JSON in $OUT/ ==="