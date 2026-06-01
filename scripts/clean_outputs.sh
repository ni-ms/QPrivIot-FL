#!/usr/bin/env bash
# Archive generated outputs into a dated folder instead of deleting them
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE="$ROOT/archived_outputs/$(date +%Y-%m-%d_%H-%M-%S)"

echo "Archiving generated outputs from $ROOT to $ARCHIVE ..."
mkdir -p "$ARCHIVE"

# Figures
shopt -s nullglob
figures=("$ROOT/figures"/*.png)
if [ ${#figures[@]} -gt 0 ]; then
    mkdir -p "$ARCHIVE/figures"
    mv -v "${figures[@]}" "$ARCHIVE/figures/"
else
    echo "  No figures to archive."
fi

# Experiment result JSONs and CSVs (all datasets)
result_files=("$ROOT/experiment_results"/results_*.json "$ROOT/experiment_results"/results_*.csv)
if [ ${#result_files[@]} -gt 0 ]; then
    mkdir -p "$ARCHIVE/experiment_results"
    mv -v "${result_files[@]}" "$ARCHIVE/experiment_results/"
else
    echo "  No experiment results to archive."
fi

# Root-level result JSONs and CSVs
root_results=("$ROOT"/results_*.csv "$ROOT"/results_*.json "$ROOT"/results.json)
if [ ${#root_results[@]} -gt 0 ]; then
    mv -v "${root_results[@]}" "$ARCHIVE/"
else
    echo "  No root-level results to archive."
fi
shopt -u nullglob

# test_artifacts directory
if [ -d "$ROOT/test_artifacts" ]; then
    mv -v "$ROOT/test_artifacts" "$ARCHIVE/test_artifacts"
else
    echo "  No test_artifacts/ to archive."
fi

# logs directory contents (keep the dir)
shopt -s nullglob
logs=("$ROOT/logs"/*)
if [ ${#logs[@]} -gt 0 ]; then
    mkdir -p "$ARCHIVE/logs"
    mv -v "${logs[@]}" "$ARCHIVE/logs/"
else
    echo "  No logs to archive."
fi
shopt -u nullglob

# Saved global model
if [ -f "$ROOT/global_model.npz" ]; then
    mv -v "$ROOT/global_model.npz" "$ARCHIVE/"
else
    echo "  No global_model.npz to archive."
fi

echo "Done. Files archived to: $ARCHIVE"