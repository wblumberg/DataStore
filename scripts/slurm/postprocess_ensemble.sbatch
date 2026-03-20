#!/bin/bash
#SBATCH --job-name=datastore-postprocess
#SBATCH --output=logs/datastore-postprocess-%j.out
#SBATCH --error=logs/datastore-postprocess-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VARIABLE="${VARIABLE:-uh}"
OUTPUT_ZARR="${OUTPUT_ZARR:-/tmp/ensemble_diag.zarr}"

cd "$PROJECT_ROOT"
mkdir -p logs

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch postprocess_ensemble.sbatch <member1.zarr> [member2.zarr ...]"
  exit 2
fi

"$PYTHON_BIN" utils/postprocess_ensemble.py \
  --member-zarrs "$@" \
  --variable "$VARIABLE" \
  --output-zarr "$OUTPUT_ZARR" \
  --threshold "${VARIABLE}_gt_1=1.0" \
  --percentile 0.5
