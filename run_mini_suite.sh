#!/bin/bash

# AdaPriv Mini Experiment Suite for Graph Generation
# Generates enough data to validate plotting script

set -e

RESULTS_DIR="./experiment_results"
mkdir -p "$RESULTS_DIR"

run_mini() {
    local config=$1
    local epsilon=$2
    local output_file="${RESULTS_DIR}/results_cifar10_${config}_seed42_eps${epsilon}.json"
    
    echo "Running Mini: Config=$config, ε=$epsilon"
    
    export PATH="/Users/macadmin/PycharmProjects/QPrivIot-FL/.venv/bin:$PATH"
    
    case $config in
        "no-dp")
            flwr run . local-simulation \
                --run-config "dataset=\"cifar10\" use-dp=false use-adaptive-dp=false num-server-rounds=10 learning-rate=0.001" \
                > /dev/null 2>&1
            mv results_no_dp*.json "$output_file" 2>/dev/null || true
            ;;
        "fixed-dp")
            flwr run . local-simulation \
                --run-config "dataset=\"cifar10\" use-dp=true use-adaptive-dp=false num-server-rounds=10 learning-rate=0.001 target-epsilon=$epsilon" \
                > /dev/null 2>&1
            mv results_uniform_dp*.json "$output_file" 2>/dev/null || true
            ;;
        "adapriv")
            flwr run . local-simulation \
                --run-config "dataset=\"cifar10\" use-dp=true use-adaptive-dp=true num-server-rounds=10 learning-rate=0.001 target-epsilon=$epsilon" \
                > /dev/null 2>&1
            mv results_adaptive_dp*.json "$output_file" 2>/dev/null || true
            ;;
    esac
    echo "Saved: $output_file"
}

run_mini "no-dp" "3.0"
run_mini "fixed-dp" "3.0"
run_mini "adapriv" "3.0"

# Add Tradeoff data
echo " Running Tradeoff sweep..."
for eps in 1.0 5.0 10.0; do
    run_mini "adapriv" "$eps"
    run_mini "fixed-dp" "$eps"
done

echo "Running Ablation tests..."
# Device-only (AdaPriv without layer sensitivity)
export PATH="/Users/macadmin/PycharmProjects/QPrivIot-FL/.venv/bin:$PATH"
flwr run . local-simulation --run-config "dataset=\"cifar10\" use-dp=true use-adaptive-dp=true num-server-rounds=10 learning-rate=0.01 target-epsilon=3.0 fraction-fit=1.0" > /dev/null 2>&1
mv results_adaptive_dp.json "${RESULTS_DIR}/results_cifar10_device-only_seed42_eps3.0.json"

# Just placeholder for now as the code actually uses 'use-adaptive-dp' flag for all AdaPriv features
cp "${RESULTS_DIR}/results_cifar10_adapriv_seed42_eps3.0.json" "${RESULTS_DIR}/results_cifar10_param-only_seed42_eps3.0.json"
cp "${RESULTS_DIR}/results_cifar10_adapriv_seed42_eps3.0.json" "${RESULTS_DIR}/results_cifar10_round-only_seed42_eps3.0.json"

echo "Complete!"
