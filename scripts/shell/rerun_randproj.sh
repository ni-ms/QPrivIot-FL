#!/usr/bin/env bash
# Re-run every config that feeds a §7 table, with the DATA-INDEPENDENT projection
# (--proj randproj) instead of the data-dependent PCA fit on user notes (paper §8).
#
# The random projection is drawn from a PUBLIC seed, so it costs zero privacy budget
# and the ε of §5.4 then accounts for the whole release.
#
# TF-IDF configs (p0_*, p4_*) are deliberately excluded: their TruncatedSVD path
# ignores --proj, so tagging them 'randproj' would be a lie.
#
#   bash scripts/shell/rerun_randproj.sh          # seeds 0-4, both grids
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY=.venv/bin/python; [[ -x "$PY" ]] || PY=.venv/Scripts/python.exe
AGENTMEM=scripts/agentmem
SEEDS=${SEEDS:-0,1,2,3,4}
PROJ=${PROJ:-randproj}
OUT=experiment_results/rerun_grid_rp
OUT_S=experiment_results/rerun_grid_s_rp
mkdir -p "$OUT" "$OUT_S" logs
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {  # run <outdir> <tag> <script> <args...>
  local out=$1; shift; local tag=$1; shift; local script=$1; shift
  local json="$out/${tag}.json"
  if [[ -f "$json" ]]; then log "SKIP  $tag"; return 0; fi
  log "RUN   $tag"
  if $PY "${AGENTMEM}/${script}" "$@" --proj "$PROJ" --seeds "$SEEDS" --json "$json" \
        > "logs/rp_${tag}.log" 2>&1; then log "OK    $tag"
  else log "FAIL  $tag (logs/rp_${tag}.log)"; fi
}

log "=== randproj rerun | seeds=$SEEDS proj=$PROJ ==="

# §7.1 oracle utility: K-sweep @ d=32, then d-sweep @ K=128
for K in 32 64 128 256; do
  run "$OUT" p2_probe_oracle_K${K}_d32 _longmemeval_probe.py --variant oracle --K $K --d 32 --topk 5 --release separate --eps 16,8,3
done
for d in 64 128; do
  run "$OUT" p2_probe_oracle_K128_d${d} _longmemeval_probe.py --variant oracle --K 128 --d $d --topk 5 --release separate --eps 16,8,3
done

# §7.3 oracle leakage
for K in 512 1024 2048; do
  run "$OUT" p2_leak_oracle_K${K}_d32 _longmemeval_leakage.py --variant oracle --K $K --d 32 --M 2000 --lowcount 3 --eps 16,8,3,1
done

# §7.2 real sentence-embedding d-crossover
for d in 32 64 128 384; do
  run "$OUT" p2st_probe_N100_K32_d${d}  _agentmem_probe.py   --N 100 --K 32   --d $d --M 1000 --k 5 --alpha 0.3 --embedder st --eps 16,8,3
  run "$OUT" p2st_leak_N100_K1024_d${d} _agentmem_leakage.py --N 100 --K 1024 --d $d --M 2000 --alpha 0.3 --lowcount 3 --embedder st --eps 16,8,3,1
done

# §7.4 LLM-distilled notes
PILOT=experiment_results/_lme_distilled_oracle_pilot.json
FULL=experiment_results/_lme_distilled_oracle.json
for K in 32 64 128; do
  run "$OUT" p5_distutil_pilot100_K${K}_d32 _longmemeval_distilled_utility.py --distilled "$PILOT" --K $K --d 32 --topk 5 --eps 16,8,3
  run "$OUT" p5_distutil_full500_K${K}_d32  _longmemeval_distilled_utility.py --distilled "$FULL"  --K $K --d 32 --topk 5 --eps 16,8,3
done
for K in 256 512; do
  run "$OUT" p5_distleak_pilot100_K${K}_d32 _longmemeval_distilled_analysis.py --distilled "$PILOT" --K $K --d 32 --eps 16,8,3
done
for K in 512 1024; do
  run "$OUT" p5_distleak_full500_K${K}_d32 _longmemeval_distilled_analysis.py --distilled "$FULL" --K $K --d 32 --eps 16,8,3
done

# §7.7 at-scale s-variant
for K in 32 64 128 256; do
  run "$OUT_S" p6_probe_s_K${K}_d32 _longmemeval_probe.py --variant s --K $K --d 32 --topk 5 --release separate --eps 16,8,3
done
for K in 512 1024 2048; do
  run "$OUT_S" p6_leak_s_K${K}_d32 _longmemeval_leakage.py --variant s --K $K --d 32 --M 2000 --lowcount 3 --eps 16,8,3,1
done

log "=== done. JSON in $OUT/ and $OUT_S/ ==="
