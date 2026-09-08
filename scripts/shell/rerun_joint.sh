#!/usr/bin/env bash
# JOINT FRONTIER (paper §7.1): utility AND calibrated leakage at the SAME K.
#
# The headline grid (rerun_final.sh) measures utility only at K<=256 and leakage only at
# K>=512, so no table in the superseded draft showed both axes at one operating point.
# That gap is what let the abstract juxtapose "95% retention" (K=32) with "tail-AUC
# 0.93->0.53" (K>=512) as though they described one release. They do not.
#
# This driver fills the missing cells:
#   * utility  at K = 512, 1024, 2048   (_longmemeval_probe.py)
#   * leakage  at K = 32, 64, 128, 256  (_longmemeval_leakage.py, the cosine proxy)
#   * LiRA     at K = 32, 64, 128, 256  (_agentmem_lira.py, the calibrated attack)
#
# The LiRA cells are the load-bearing ones. The cosine proxy reads 0.504 -- chance -- at
# K=32 and thereby reports "nothing to protect" at the recommended operating point. The
# calibrated attack reads AUC 0.833 / TPR@1%FPR 0.175 / decode 0.771 on the same release.
# The proxy was blind, not the pool safe. See §7.1.
#
# It also prices the multi-round schedule of §5.7: --fl_sigma injects the raw sigma that a
# T-release deployment needs to stay inside a FIXED total eps, so retention can be read off
# directly rather than extrapolated.
#
#   bash scripts/shell/rerun_joint.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY=.venv/bin/python; [[ -x "$PY" ]] || PY=.venv/Scripts/python.exe
AM=scripts/agentmem
SEEDS=${SEEDS:-0,1,2,3,4}
LIRA_SEEDS=${LIRA_SEEDS:-0,1,2}
SHADOWS=${SHADOWS:-48}
TARGETS=${TARGETS:-1200}
CEPS=${CEPS:-0.1}
OUT=experiment_results/joint_rp
mkdir -p "$OUT" logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {  # run <tag> <script> <args...>
  local tag=$1; shift; local script=$1; shift
  local json="$OUT/${tag}.json"
  if [[ -f "$json" ]]; then log "SKIP  $tag"; return 0; fi
  log "RUN   $tag"
  if $PY "${AM}/${script}" "$@" --clip-eps "$CEPS" --json "$json" \
        > "logs/joint_${tag}.log" 2>&1; then log "OK    $tag"
  else log "FAIL  $tag (logs/joint_${tag}.log)"; fi
}

log "=== JOINT frontier grid | seeds=$SEEDS clip_eps=$CEPS ==="

# ---- missing utility cells: high K (where the leakage headline lives) ----
for K in 512 1024 2048; do
  run probe_oracle_K${K}_d32 _longmemeval_probe.py \
    --variant oracle --K $K --d 32 --topk 5 --release veconly --eps 16,8,3 \
    --proj randproj --seeds "$SEEDS"
done

# ---- missing leakage cells: low K (where the utility headline lives) ----
for K in 32 64 128 256; do
  run leak_oracle_K${K}_d32 _longmemeval_leakage.py \
    --variant oracle --K $K --d 32 --M 2000 --lowcount 3 --eps 16,8,3,1 \
    --proj randproj --seeds "$SEEDS"
done

# ---- the calibrated attack at the recommended operating points (the key runs) ----
for K in 32 64 128 256; do
  run lira_oracle_K${K}_d32 _agentmem_lira.py \
    --source oracle --K $K --d 32 --N 500 --pool 8000 --eps 16,8,3,1 \
    --shadows "$SHADOWS" --targets "$TARGETS" --seeds "$LIRA_SEEDS" --proj randproj
done

# ---- §5.7 multi-round: sigma a T-release deployment needs at FIXED total eps = 9.31 ----
# sigma solved by bisection on skellam_rdp_epsilon(..., releases=T); see _joint_frontier.py.
#   T=4  quarterly/yr  sigma 1.211      T=30  daily/month  sigma 3.317
#   T=12 monthly/yr    sigma 2.098      T=52  weekly/yr    sigma 4.367
for SG in 1.211 2.098 3.317 4.367; do
  run probe_K32_sigma${SG} _longmemeval_probe.py \
    --variant oracle --K 32 --d 32 --topk 5 --release veconly --eps 8 --fl_sigma "$SG" \
    --proj randproj --seeds "$SEEDS"
done

log "=== done. $OUT/ -> regenerate tables/figure with _joint_frontier.py ==="
