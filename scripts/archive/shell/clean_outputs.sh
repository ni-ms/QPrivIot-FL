#!/usr/bin/env bash
# Archive generated outputs into a dated folder BEFORE a fresh run.
#
# Moves the *results* of previous runs (figures, result JSON/CSV, logs, saved
# models) into archived_outputs/<timestamp>/ so a new run starts clean without
# losing history. Expensive-to-recompute embedding caches (experiment_results
# files starting with "_", e.g. _st_*.npz, _lme_*_emb.npz) are PRESERVED by
# default -- pass --purge-cache to archive those too.
#
# Usage:
#   scripts/clean_outputs.sh              # confirm, then archive
#   scripts/clean_outputs.sh -y           # skip confirmation
#   scripts/clean_outputs.sh --purge-cache  # also archive embedding caches
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE="$ROOT/archived_outputs/$(date +%Y-%m-%d_%H-%M-%S)"

ASSUME_YES=0
PURGE_CACHE=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=1 ;;
        --purge-cache) PURGE_CACHE=1 ;;
        -h|--help) sed -n '2,/^set -euo/{/^set -euo/!s/^# \{0,1\}//p;}' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

echo "About to archive previous run outputs:"
echo "  from : $ROOT"
echo "  to   : $ARCHIVE"
[ "$PURGE_CACHE" -eq 1 ] && echo "  (embedding caches WILL be archived)" \
                         || echo "  (embedding caches under experiment_results/_*.{npz,json} preserved)"
if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Proceed? [y/N] " reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

mkdir -p "$ARCHIVE"
shopt -s nullglob

archive_into() {  # archive_into <subdir> <files...>
    local sub="$1"; shift
    local files=("$@")
    if [ ${#files[@]} -gt 0 ]; then
        [ -n "$sub" ] && mkdir -p "$ARCHIVE/$sub"
        mv -v "${files[@]}" "$ARCHIVE/${sub:+$sub/}"
    fi
}

# Figures
archive_into figures "$ROOT/figures"/*.png

# Experiment result JSON/CSV (results_* and qfl_*); caches (_*) kept unless --purge-cache
archive_into experiment_results \
    "$ROOT/experiment_results"/results_*.json \
    "$ROOT/experiment_results"/results_*.csv \
    "$ROOT/experiment_results"/qfl_*.json \
    "$ROOT/experiment_results"/qfl_*.csv
if [ "$PURGE_CACHE" -eq 1 ]; then
    archive_into experiment_results/_cache \
        "$ROOT/experiment_results"/_*.npz \
        "$ROOT/experiment_results"/_*.json
fi

# Root-level result JSON/CSV (results.json added only if it exists; nullglob
# does not strip literal, non-glob paths)
root_results=("$ROOT"/results_*.csv "$ROOT"/results_*.json)
[ -f "$ROOT/results.json" ] && root_results+=("$ROOT/results.json")
[ ${#root_results[@]} -gt 0 ] && archive_into "" "${root_results[@]}"

# Root-level run logs (e.g. cnn5_eps3.log, vqc5.log)
archive_into logs_root "$ROOT"/*.log

# logs/ directory contents (keep the dir itself)
archive_into logs "$ROOT/logs"/*

# test_artifacts directory
if [ -d "$ROOT/test_artifacts" ]; then
    mv -v "$ROOT/test_artifacts" "$ARCHIVE/test_artifacts"
fi

# Saved global model
archive_into "" "$ROOT/global_model.npz"

shopt -u nullglob

# Drop the archive folder if nothing was actually moved
if [ -z "$(ls -A "$ARCHIVE" 2>/dev/null)" ]; then
    rmdir "$ARCHIVE"
    echo "Nothing to archive; no folder created."
else
    echo "Done. Files archived to: $ARCHIVE"
fi