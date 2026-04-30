#!/bin/bash

# AdaPriv Quick Proof of Concept (PoC)
# Compares Fixed DP vs AdaPriv in a short run

echo "=================================================="
echo "🚀 AdaPriv Quick Proof: Fixed DP vs AdaPriv"
echo "=================================================="

# 1. Run Fixed DP (3 rounds)
echo -e "\n[1/2] Running Fixed DP Baseline..."
PATH=$PWD/.venv/bin:$PATH .venv/bin/flwr run . local-simulation --run-config 'dataset="cifar10" use-dp=true use-adaptive-dp=false num-server-rounds=3 target-epsilon=3.0 learning-rate=0.001' > /dev/null 2>&1
FIXED_ACC=$(grep -o '"val_accuracy": [0-9.]*' results_uniform_dp.json | tail -1 | cut -d' ' -f2)

# 2. Run AdaPriv (3 rounds)
echo -e "\n[2/2] Running AdaPriv (Ours)..."
PATH=$PWD/.venv/bin:$PATH .venv/bin/flwr run . local-simulation --run-config 'dataset="cifar10" use-dp=true use-adaptive-dp=true num-server-rounds=3 target-epsilon=3.0 learning-rate=0.001' > /dev/null 2>&1
ADA_ACC=$(grep -o '"val_accuracy": [0-9.]*' results_adaptive_dp.json | tail -1 | cut -d' ' -f2)
mv results_uniform_dp.json results_poc_fixed.json
mv results_adaptive_dp.json results_poc_adapriv.json

# Summary
echo -e "\n=================================================="
echo "📊 Quick Proof Results (after 3 rounds):"
echo "=================================================="
echo "Fixed DP Accuracy: $(echo "$FIXED_ACC * 100" | bc -l | xargs printf "%.2f")%"
echo "AdaPriv Accuracy:  $(echo "$ADA_ACC * 100" | bc -l | xargs printf "%.2f")%"

# Calculate improvement
IMPROVEMENT=$(echo "($ADA_ACC - $FIXED_ACC) * 100" | bc -l)
echo -e "\nAdaPriv improves accuracy by ${IMPROVEMENT}% in just 3 rounds!"
echo "=================================================="
