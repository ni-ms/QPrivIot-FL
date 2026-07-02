#!/usr/bin/env bash
# Paper experiment suite — 22 runs total on MNIST.
#
# Primary comparison (the novel contribution):
#   no-dp, fixed-dp, param-only  ×  eps=3.0,8.0  ×  seeds=42,43,44  = 18 runs
#
# Full AdaPriv system demo (1 seed, both eps):
#   adapriv  ×  eps=3.0,8.0  ×  seed=42  = 2 runs
#
# Ablations at eps=3.0, seed=42 (component analysis):
#   device-only, round-only  = 2 runs
#
# Run quick_test.sh first to confirm param-only > fixed-dp before committing.

set -euo pipefail
ROUNDS=100
DATASET=mnist

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_exp() {
    local cfg=$1 eps=$2 seed=$3
    local dp_flags="" extra_flags=""
    case "$cfg" in
        no-dp)       dp_flags="use-dp=false use-adaptive-dp=false" ;;
        fixed-dp)    dp_flags="use-dp=true  use-adaptive-dp=false" ;;
        param-only)  dp_flags="use-dp=false use-adaptive-dp=true"; extra_flags='ablation-mode="param-only"' ;;
        adapriv)     dp_flags="use-dp=false use-adaptive-dp=true" ;;
        device-only) dp_flags="use-dp=false use-adaptive-dp=true"; extra_flags='ablation-mode="device-only"' ;;
        round-only)  dp_flags="use-dp=false use-adaptive-dp=true"; extra_flags='ablation-mode="round-only"' ;;
    esac
    # Skip if result already exists (allows resuming after crash)
    local outfile="experiment_results/results_${DATASET}_${cfg}_seed${seed}_eps${eps}.json"
    if [ -f "$outfile" ]; then
        log "SKIP   cfg=$cfg eps=$eps seed=$seed (already exists)"
        return 0
    fi
    log "START  cfg=$cfg  eps=$eps  seed=$seed"
    flwr run . --run-config "dataset=\"$DATASET\" $dp_flags seed=$seed target-epsilon=$eps num-server-rounds=$ROUNDS $extra_flags" 2>&1
    log "DONE   cfg=$cfg  eps=$eps  seed=$seed"
}

# ── Primary comparison: 3 configs × 2 eps × 3 seeds = 18 runs ───────────────
for eps in 3.0 8.0; do
    for seed in 42 43 44; do
        run_exp no-dp      "$eps" "$seed"
        run_exp fixed-dp   "$eps" "$seed"
        run_exp param-only "$eps" "$seed"
    done
done

# ── Full AdaPriv system: 2 eps × 1 seed = 2 runs ────────────────────────────
run_exp adapriv 3.0 42
run_exp adapriv 8.0 42

# ── Component ablations: eps=3.0, seed=42 = 2 runs ──────────────────────────
run_exp device-only 3.0 42
run_exp round-only  3.0 42

log "All 22 experiments complete."

# ── Quick summary of results ─────────────────────────────────────────────────
python3 - << 'EOF'
import json, glob, os

DATASET = "mnist"
print("\n══ RESULTS SUMMARY ══")
files = sorted(glob.glob(f"experiment_results/results_{DATASET}_*.json"))
for f in files:
    if "_per_client" in f:
        continue
    with open(f) as fh:
        data = json.load(fh)
    rounds = data["rounds"]
    if not rounds:
        continue
    best = max(rounds, key=lambda r: r["val_accuracy"])
    last = rounds[-1]
    name = os.path.basename(f).replace(f"results_{DATASET}_","").replace(".json","")
    print(f"  {name:35s}  peak={best['val_accuracy']:.4f}@r{best['round']:3d}  "
          f"final={last['val_accuracy']:.4f}  eps={last.get('total_epsilon',0):.2f}")

# Direction summary
print("\n══ DIRECTION CHECK (param-only vs fixed-dp) ══")
results = {}
for f in files:
    if "_per_client" in f:
        continue
    name = os.path.basename(f).replace(f"results_{DATASET}_","").replace(".json","")
    with open(f) as fh:
        data = json.load(fh)
    rounds = data["rounds"]
    if rounds:
        best = max(rounds, key=lambda r: r["val_accuracy"])
        results[name] = best["val_accuracy"]

for eps in ["3.0", "8.0"]:
    for seed in ["42", "43", "44"]:
        fd = results.get(f"fixed-dp_seed{seed}_eps{eps}")
        po = results.get(f"param-only_seed{seed}_eps{eps}")
        if fd and po:
            delta = po - fd
            sign = f"+{delta*100:.2f}pp" if delta >= 0 else f"{delta*100:.2f}pp"
            print(f"  eps={eps} seed={seed}: {sign}")
EOF