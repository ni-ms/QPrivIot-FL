#!/bin/bash

# ======================================================================
# AdaPriv-FL: Master Comprehensive Test Suite
# ======================================================================
# This script automates:
# 1. Resource directory cleanup
# 2. Main Comparison Experiments (Accuracy/Loss vs. Rounds)
# 3. Privacy-Utility Tradeoff (Epsilon Sweeps)
# 4. JSON result management
# 5. Publication Graph Generation
# ======================================================================

set -e

# --- Configuration ---
SEEDS=(42 123 456 789 2024)
DATASETS=("cifar10" "femnist")
EPSILON_VALUES=(1.0 2.0 3.0 5.0 8.0 10.0)
ROUNDS=50
RESULTS_DIR="/Users/macadmin/PycharmProjects/QPrivIot-FL/experiment_results"
FIGURES_DIR="/Users/macadmin/PycharmProjects/QPrivIot-FL/figures"
VENV_PATH="/Users/macadmin/PycharmProjects/QPrivIot-FL/.venv"
PYTHON_BIN="$VENV_PATH/bin/python"
FLWR_BIN="$VENV_PATH/bin/flwr"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "$RESULTS_DIR"
mkdir -p "$FIGURES_DIR"

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}🚀 Starting AdaPriv-FL Comprehensive Test Suite     ${NC}"
echo -e "${GREEN}==================================================${NC}"

# Ensure we have the proper environment
if [ ! -f "$FLWR_BIN" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    exit 1
fi

# Function to run a single simulation
run_sim() {
    local dataset=$1
    local config=$2
    local seed=$3
    local eps=$4
    local rounds=$5
    
    local target_name="results_${dataset}_${config}_seed${seed}_eps${eps}.json"
    local target_path="$RESULTS_DIR/$target_name"

    if [ -f "$target_path" ]; then
        echo -e "${YELLOW}⏭️  Skipping existing: $target_name${NC}"
        return
    fi

    echo -e "${YELLOW}▶️  Running: $dataset | $config | Seed $seed | ε $eps | $rounds rounds${NC}"
    
    # Map config to flwr arguments
    local dp_args=""
    local results_source=""
    case $config in
        "no-dp")
            dp_args="use-dp=false use-adaptive-dp=false"
            results_source="results_no_dp.json"
            ;;
        "fixed-dp")
            dp_args="use-dp=true use-adaptive-dp=false target-epsilon=$eps"
            results_source="results_uniform_dp.json"
            ;;
        "adapriv")
            dp_args="use-dp=true use-adaptive-dp=true target-epsilon=$eps"
            results_source="results_adaptive_dp.json"
            ;;
        *)
            echo -e "${RED}Unknown config type: $config${NC}"
            return 1
            ;;
    esac

    # Execute simulation
    # We use 'local-simulation' federation defined in pyproject.toml
    "$FLWR_BIN" run . local-simulation \
        --run-config "dataset=\"$dataset\" num-server-rounds=$rounds learning-rate=0.001 $dp_args" \
        > /dev/null 2>&1

    # Move results to target folder
    if [ -f "$results_source" ]; then
        mv "$results_source" "$target_path"
        echo -e "${GREEN}✅ Saved: $target_name${NC}"
    else
        echo -e "${RED}❌ Error: Results file $results_source not generated!${NC}"
        exit 1
    fi
}

# ----------------------------------------------------------------------
# PHASE 1: Main Comparison (Graph 1, 3, 5, 6, 7)
# ----------------------------------------------------------------------
echo -e "\n${GREEN}[PHASE 1] Main Comparison Experiments${NC}"
for ds in "${DATASETS[@]}"; do
    for cfg in "no-dp" "fixed-dp" "adapriv"; do
        for s in "${SEEDS[@]}"; do
            run_sim "$ds" "$cfg" "$s" "3.0" "$ROUNDS"
        done
    done
done

# ----------------------------------------------------------------------
# PHASE 2: Privacy-Utility Epsilon Sweep (Graph 2)
# ----------------------------------------------------------------------
echo -e "\n${GREEN}[PHASE 2] Privacy-Utility Epsilon Sweep${NC}"
for eps in "${EPSILON_VALUES[@]}"; do
    for cfg in "fixed-dp" "adapriv"; do
        # Use first 3 seeds for the sweep to balance time/rigor
        for s in "${SEEDS[@]:0:3}"; do
            run_sim "cifar10" "$cfg" "$s" "$eps" "$ROUNDS"
        done
    done
done

# ----------------------------------------------------------------------
# PHASE 3: Graph Generation
# ----------------------------------------------------------------------
echo -e "\n${GREEN}[PHASE 3] Generating Publication Graphs${NC}"
if [ -f "plot.py" ]; then
    echo -e "${YELLOW}📊 Running plot.py...${NC}"
    "$PYTHON_BIN" plot.py
    echo -e "${GREEN}✅ All graphs generated in $FIGURES_DIR/${NC}"
else
    echo -e "${RED}Error: plot.py not found in current directory.${NC}"
fi

echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}✨ Comprehensive Suite Finished Successfully!       ${NC}"
echo -e "${GREEN}==================================================${NC}"
