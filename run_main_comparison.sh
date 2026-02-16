#!/bin/bash
# Main Comparison Script (50 rounds)

echo "=================================================="
echo "🎯 Starting Main Comparison (50 Rounds)"
echo "=================================================="

# 1. Run Fixed DP
echo -e "\n[1/2] Running Fixed DP Baseline (50 rounds)..."
PATH=$PWD/.venv/bin:$PATH .venv/bin/flwr run . local-simulation --run-config 'dataset="cifar10" use-dp=true use-adaptive-dp=false num-server-rounds=50 learning-rate=0.001 target-epsilon=3.0' > results_fixed_50.log 2>&1
mv results_uniform_dp.json results_main_fixed_3.0.json

# 2. Run AdaPriv
echo -e "\n[2/2] Running AdaPriv (50 rounds)..."
PATH=$PWD/.venv/bin:$PATH .venv/bin/flwr run . local-simulation --run-config 'dataset="cifar10" use-dp=true use-adaptive-dp=true num-server-rounds=50 learning-rate=0.001 target-epsilon=3.0' > results_adapriv_50.log 2>&1
mv results_adaptive_dp.json results_main_adapriv_3.0.json

echo -e "\n=================================================="
echo "✅ Main Experiments Complete!"
echo "=================================================="
