#!/usr/bin/env bash
# Quick 20-round test on MNIST to validate the paper direction before full runs.
# Runs: no-dp, fixed-dp, param-only at eps=3.0 and eps=8.0, seed=42.
# Takes ~20-30 minutes.
# Gate: if param-only > fixed-dp at BOTH epsilon values → run full suite.

set -euo pipefail
ROUNDS=20
SEED=42
DATASET=mnist

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_exp() {
    local cfg=$1 eps=$2
    local dp_flags="" extra_flags=""
    case "$cfg" in
        no-dp)      dp_flags="use-dp=false use-adaptive-dp=false" ;;
        fixed-dp)   dp_flags="use-dp=true  use-adaptive-dp=false" ;;
        param-only) dp_flags="use-dp=false use-adaptive-dp=true"; extra_flags='ablation-mode="param-only"' ;;
    esac
    log "START  $cfg  eps=$eps"
    flwr run . --run-config "dataset=\"$DATASET\" $dp_flags seed=$SEED target-epsilon=$eps num-server-rounds=$ROUNDS $extra_flags" 2>&1 \
        | grep -E "Round [0-9]+|CENTRALIZED|EARLY|DP CALIB|Error|Traceback|Exception" || true
    log "DONE   $cfg  eps=$eps"
}

for EPS in 3.0 8.0; do
    run_exp no-dp      "$EPS"
    run_exp fixed-dp   "$EPS"
    run_exp param-only "$EPS"
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " QUICK TEST SUMMARY  (dataset=$DATASET, seed=$SEED, T=$ROUNDS)"
echo "════════════════════════════════════════════════════════════"
python3 - << 'EOF'
import json, os

results = {}
for eps in ["3.0", "8.0"]:
    for cfg in ["no-dp", "fixed-dp", "param-only"]:
        f = f"experiment_results/results_mnist_{cfg}_seed42_eps{eps}.json"
        if not os.path.exists(f):
            print(f"  eps={eps} {cfg}: MISSING")
            continue
        with open(f) as fh:
            data = json.load(fh)
        rounds = data["rounds"]
        if not rounds:
            print(f"  eps={eps} {cfg}: no rounds")
            continue
        best = max(rounds, key=lambda r: r["val_accuracy"])
        last = rounds[-1]
        key = (eps, cfg)
        results[key] = best["val_accuracy"]
        eps_spent = last.get("total_epsilon", 0)
        print(f"  eps={eps}  {cfg:12s}  peak={best['val_accuracy']:.4f}@r{best['round']:2d}"
              f"  final={last['val_accuracy']:.4f}  eps_spent={eps_spent:.2f}")

print("")
print("  ── Direction check ──")
for eps in ["3.0", "8.0"]:
    fd = results.get((eps, "fixed-dp"))
    po = results.get((eps, "param-only"))
    if fd is not None and po is not None:
        delta = po - fd
        sign = f"+{delta*100:.1f}pp" if delta >= 0 else f"{delta*100:.1f}pp"
        verdict = "✓ GOOD" if delta > 0.005 else ("~ marginal" if delta > -0.005 else "✗ FAIL")
        print(f"  eps={eps}: param-only vs fixed-dp  {sign}  {verdict}")

print("════════════════════════════════════════════════════════════")
EOF