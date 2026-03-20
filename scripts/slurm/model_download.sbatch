#!/bin/bash
#SBATCH --job-name=datastore-download
#SBATCH --output=logs/datastore-download-%j.out
#SBATCH --error=logs/datastore-download-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_TOKEN="${MODEL_TOKEN:-hrrr}"
MAX_WORKERS="${MAX_WORKERS:-4}"

cd "$PROJECT_ROOT"
mkdir -p logs

"$PYTHON_BIN" ingest_model/download_hrefnssl_models.py "$MODEL_TOKEN" --max-workers "$MAX_WORKERS"
