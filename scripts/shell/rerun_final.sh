#!/usr/bin/env bash
# FINAL mechanism (paper §4-§5):
#   * public-seed random projection      (--proj randproj)      -> data-independent, 0 eps
#   * DP-selected public clip bound C    (--clip-eps 0.1)       -> exponential mechanism, eps_c
#   * public quantiser bound B := C                             -> never clips, eps-invariant
#   * vector-only release                (--release veconly)    -> one clip quantile, one composition
#
# Writes the headline grids plus the two §7.2 ablation arms. The ablation arms differ from the
# headline ONLY in --proj, so the projection is the single isolated variable.
#
#   bash scripts/shell/rerun_final.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY=.venv/bin/python; [[ -x "$PY" ]] || PY=.venv/Scripts/python.exe
AM=scripts/agentmem
SEEDS=${SEEDS:-0,1,2,3,4}
CEPS=${CEPS:-0.1}
OUT=experiment_results/rerun_grid_rp
OUT_S=experiment_results/rerun_grid_s_rp
ABL_PCA=experiment_results/rerun_grid_abl_pca
ABL_PP=experiment_results/rerun_grid_abl_pp
mkdir -p "$OUT" "$OUT_S" "$ABL_PCA" "$ABL_PP" logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {  # run <outdir> <tag> <script> <args...>
  local out=$1; shift; local tag=$1; shift; local script=$1; shift
  local json="$out/${tag}.json"
  if [[ -f "$json" ]]; then log "SKIP  $tag"; return 0; fi
  log "RUN   $tag"
  if $PY "${AM}/${script}" "$@" --clip-eps "$CEPS" --seeds "$SEEDS" --json "$json" \
        > "logs/final_${tag}.log" 2>&1; then log "OK    $tag"
  else log "FAIL  $tag (logs/final_${tag}.log)"; fi
}

log "=== FINAL grid | seeds=$SEEDS clip_eps=$CEPS ==="

# ---- §7.1 oracle utility (vector-only) ----
for K in 32 64 128 256; do
  run "$OUT" p2_probe_oracle_K${K}_d32 _longmemeval_probe.py --variant oracle --K $K --d 32 --topk 5 --release veconly --eps 16,8,3 --proj randproj
done
for dd in 64 128; do
  run "$OUT" p2_probe_oracle_K128_d${dd} _longmemeval_probe.py --variant oracle --K 128 --d $dd --topk 5 --release veconly --eps 16,8,3 --proj randproj
done

# ---- §7.3 oracle leakage ----
for K in 512 1024 2048; do
  run "$OUT" p2_leak_oracle_K${K}_d32 _longmemeval_leakage.py --variant oracle --K $K --d 32 --M 2000 --lowcount 3 --eps 16,8,3,1 --proj randproj
done

# ---- §7.2 d-sweep: headline arm (randproj) + the two ablation arms ----
for dd in 32 64 128 384; do
  run "$OUT" p2st_probe_N100_K32_d${dd}  _agentmem_probe.py   --N 100 --K 32   --d $dd --M 1000 --k 5 --alpha 0.3 --embedder st --eps 16,8,3    --proj randproj
  run "$OUT" p2st_leak_N100_K1024_d${dd} _agentmem_leakage.py --N 100 --K 1024 --d $dd --M 2000 --alpha 0.3 --lowcount 3 --embedder st --eps 16,8,3,1 --proj randproj
  run "$ABL_PCA" p2st_probe_N100_K32_d${dd}  _agentmem_probe.py   --N 100 --K 32   --d $dd --M 1000 --k 5 --alpha 0.3 --embedder st --eps 16,8,3    --proj pca
  run "$ABL_PCA" p2st_leak_N100_K1024_d${dd} _agentmem_leakage.py --N 100 --K 1024 --d $dd --M 2000 --alpha 0.3 --lowcount 3 --embedder st --eps 16,8,3,1 --proj pca
  run "$ABL_PP"  p2st_probe_N100_K32_d${dd}  _agentmem_probe.py   --N 100 --K 32   --d $dd --M 1000 --k 5 --alpha 0.3 --embedder st --eps 16,8,3    --proj publicpca
  run "$ABL_PP"  p2st_leak_N100_K1024_d${dd} _agentmem_leakage.py --N 100 --K 1024 --d $dd --M 2000 --alpha 0.3 --lowcount 3 --embedder st --eps 16,8,3,1 --proj publicpca
done

# ---- §7.4 LLM-distilled notes ----
PILOT=experiment_results/_lme_distilled_oracle_pilot.json
FULL=experiment_results/_lme_distilled_oracle.json
for K in 32 64 128; do
  run "$OUT" p5_distutil_pilot100_K${K}_d32 _longmemeval_distilled_utility.py --distilled "$PILOT" --K $K --d 32 --topk 5 --eps 16,8,3 --proj randproj
  run "$OUT" p5_distutil_full500_K${K}_d32  _longmemeval_distilled_utility.py --distilled "$FULL"  --K $K --d 32 --topk 5 --eps 16,8,3 --proj randproj
done
for K in 256 512; do
  run "$OUT" p5_distleak_pilot100_K${K}_d32 _longmemeval_distilled_analysis.py --distilled "$PILOT" --K $K --d 32 --eps 16,8,3 --proj randproj
done
for K in 512 1024; do
  run "$OUT" p5_distleak_full500_K${K}_d32 _longmemeval_distilled_analysis.py --distilled "$FULL" --K $K --d 32 --eps 16,8,3 --proj randproj
done

# ---- §7.7 at-scale s-variant ----
for K in 32 64 128 256; do
  run "$OUT_S" p6_probe_s_K${K}_d32 _longmemeval_probe.py --variant s --K $K --d 32 --topk 5 --release veconly --eps 16,8,3 --proj randproj
done
for K in 512 1024 2048; do
  run "$OUT_S" p6_leak_s_K${K}_d32 _longmemeval_leakage.py --variant s --K $K --d 32 --M 2000 --lowcount 3 --eps 16,8,3,1 --proj randproj
done

log "=== done. $OUT/ $OUT_S/ $ABL_PCA/ $ABL_PP/ ==="
