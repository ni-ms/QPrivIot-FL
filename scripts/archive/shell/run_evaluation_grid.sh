#!/usr/bin/env bash
set -euo pipefail

DATASETS=(cifar10)
CONFIGS=(
  "use-dp=false use-adaptive-dp=false"        # no-dp
  "use-dp=true  use-adaptive-dp=false"        # fixed-dp
  "use-dp=true  use-adaptive-dp=true"         # adapriv
)
CONFIG_NAMES=(no-dp fixed-dp adapriv)

# Defaults (can be overridden by environment variables)
CLIENT_COUNTS=${CLIENT_COUNTS:-"2 5 10 20"}
ALPHAS=${ALPHAS:-"0.05 0.3 1.0"}
SEEDS=${SEEDS:-"1337 2024 9001"}
EPS=${EPS:-3.0}
ROUNDS=${ROUNDS:-50}
LR=${LR:-0.001}

OUT=experiment_results
LOGS=logs
mkdir -p "$OUT" "$LOGS"

for ds in "${DATASETS[@]}"; do
  for n in $CLIENT_COUNTS; do
    for alpha in $ALPHAS; do
      for seed in $SEEDS; do
        for i in "${!CONFIGS[@]}"; do
          name="${CONFIG_NAMES[$i]}"
          cfg="${CONFIGS[$i]}"
          tag="${ds}_${name}_n${n}_alpha${alpha}_seed${seed}_eps${EPS}"
          out_json="${OUT}/results_${tag}.json"
          [[ -f "$out_json" ]] && { echo "skip $tag (exists)"; continue; }
          echo "▶ $tag"
          flwr run . local-simulation \
            --federation-config "options.num-supernodes=${n}" \
            --run-config "dataset=\"${ds}\" num-server-rounds=${ROUNDS} \
                          learning-rate=${LR} target-epsilon=${EPS} \
                          dirichlet-alpha=${alpha} num-clients=${n} \
                          seed=${seed} ${cfg}" \
            2>&1 | tee "${LOGS}/${tag}.log"
          # Strategy emits results_<dp_mode>.json — rename to canonical tag
          for f in results_no_dp.json results_uniform_dp.json results_adaptive_dp.json; do
            if [[ -f "$f" ]]; then
              mv "$f" "$out_json"
              # Also rename per-client CSV if it exists
              f_pc="${f%.json}_per_client.csv"
              out_pc="${out_json%.json}_per_client.csv"
              [[ -f "$f_pc" ]] && mv "$f_pc" "$out_pc"
              break
            fi
          done
        done
      done
    done
  done
done
