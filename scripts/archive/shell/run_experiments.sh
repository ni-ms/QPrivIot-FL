#!/bin/bash

# AdaPriv Experimental Automation Script
# Runs all experiments needed for publication graphs

set -e  # Exit on error

# Configuration
SEEDS=(42 123 456 789 2024)
DATASETS=("cifar10" "femnist")
EPSILON_VALUES=(1.0 2.0 3.0 5.0 8.0 10.0)
RESULTS_DIR="./experiment_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AdaPriv Experimental Protocol Runner${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to run a single experiment
run_experiment() {
    local dataset=$1
    local config=$2
    local seed=$3
    local epsilon=$4
    local extra_args=$5
    
    local output_file="${RESULTS_DIR}/results_${dataset}_${config}_seed${seed}_eps${epsilon}.json"
    
    # Skip if already exists
    if [ -f "$output_file" ]; then
        echo -e "${YELLOW}⏭️  Skipping (already exists): $output_file${NC}"
        return
    fi
    
    echo -e "${GREEN}▶️  Running: Dataset=$dataset, Config=$config, Seed=$seed, ε=$epsilon${NC}"
    
    # Build command based on config
    case $config in
        "no-dp")
            PATH=$PWD/.venv/bin:$PATH .venv/bin/flwr run . local-simulation \
                --run-config "dataset=\"$dataset\" use-dp=false use-adaptive-dp=false num-server-rounds=50 learning-rate=0.001 $extra_args" \
                > /dev/null 2>&1
            ;;
        "fixed-dp")
            PATH=$PWD/.venv/bin:$PATH .venv/bin/flwr run . local-simulation \
                --run-config "dataset=\"$dataset\" use-dp=true use-adaptive-dp=false num-server-rounds=50 learning-rate=0.001 target-epsilon=$epsilon $extra_args" \
                > /dev/null 2>&1
            ;;
        "adapriv")
            PATH=$PWD/.venv/bin:$PATH .venv/bin/flwr run . local-simulation \
                --run-config "dataset=\"$dataset\" use-dp=true use-adaptive-dp=true num-server-rounds=50 learning-rate=0.001 target-epsilon=$epsilon $extra_args" \
                > /dev/null 2>&1
            ;;
        *)
            echo -e "${RED}❌ Unknown config: $config${NC}"
            exit 1
            ;;
    esac
    
    # Determine the expected results file name based on config
    local results_file=""
    case $config in
        "no-dp") results_file="results_no_dp.json" ;;
        "fixed-dp") results_file="results_uniform_dp.json" ;;
        "adapriv") results_file="results_adaptive_dp.json" ;;
    esac
    
    # Move results file
    if [ -f "$results_file" ]; then
        mv "$results_file" "$output_file"
        echo -e "${GREEN}✅ Saved: $output_file${NC}"
    elif [ -f "results.json" ]; then
        mv results.json "$output_file"
        echo -e "${GREEN}✅ Saved (Fallback): $output_file${NC}"
    else
        echo -e "${RED}❌ Error: No results file found ($results_file or results.json)${NC}"
        exit 1
    fi
}

# ============================================
# EXPERIMENT SET 1: Main Comparison (Graph 1, 3)
# ============================================
echo -e "\n${YELLOW}📊 EXPERIMENT SET 1: Main Comparison (3 configs × 2 datasets × 5 seeds = 30 runs)${NC}"

for dataset in "${DATASETS[@]}"; do
    for config in "no-dp" "fixed-dp" "adapriv"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "$dataset" "$config" "$seed" "3.0" ""
        done
    done
done

# ============================================
# EXPERIMENT SET 2: Epsilon Sweep (Graph 2)
# ============================================
echo -e "\n${YELLOW}📊 EXPERIMENT SET 2: Privacy-Utility Tradeoff (6 ε × 2 configs × 3 seeds = 36 runs)${NC}"

for epsilon in "${EPSILON_VALUES[@]}"; do
    for config in "fixed-dp" "adapriv"; do
        for seed in "${SEEDS[@]:0:3}"; do  # Only first 3 seeds
            run_experiment "cifar10" "$config" "$seed" "$epsilon" ""
        done
    done
done

# ============================================
# EXPERIMENT SET 3: Ablation Study (Graph 4)
# ============================================
echo -e "\n${YELLOW}📊 EXPERIMENT SET 3: Ablation Study${NC}"
echo -e "${RED}⚠️  NOTE: Ablation requires code modifications - run manually after updating:${NC}"
echo -e "   - Device-Only: Disable parameter & round adaptation"
echo -e "   - Parameter-Only: Disable device & round adaptation"
echo -e "   - Round-Only: Disable device & parameter adaptation"
echo -e "   See experimental_protocol.md for implementation details"

# ============================================
# Summary
# ============================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Experiment Suite Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\nResults saved to: ${RESULTS_DIR}/"
echo -e "Total files: $(ls -1 ${RESULTS_DIR} | wc -l)"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Run ablation experiments manually (requires code changes)"
echo -e "2. Generate plots: python plot.py"
echo -e "3. Review figures in ./figures/"
