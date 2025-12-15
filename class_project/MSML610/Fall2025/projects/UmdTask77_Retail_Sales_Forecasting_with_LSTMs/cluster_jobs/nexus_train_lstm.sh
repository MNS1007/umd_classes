#!/bin/bash
#SBATCH --job-name=rsf_lstm
#SBATCH --output=logs/rsf_lstm_%j.out
#SBATCH --error=logs/rsf_lstm_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --qos=cml-scavenger

set -euo pipefail

# ---------------------------------------------------------------------------
# Customize these paths for your Nexus account/environment.
# ---------------------------------------------------------------------------
PROJECT_ROOT="$HOME/Documents/umd_classes/class_project/MSML610/Fall2025/projects/UmdTask77_Retail_Sales_Forecasting_with_LSTMs"
DATA_ROOT="/cmlscratch/$USER/store-sales-time-series-forecasting"
VENV_PATH="$HOME/.venvs/rsf/bin/activate"

# Load modules or activate conda/venv as needed for JAX/Flax.
if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
else
  echo "Missing virtualenv at $VENV_PATH" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

echo "Starting Retail Sales Forecasting (LSTM) run..."
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

python -m retail_sales_forecasting_with_lstms.example \
  --data-root "$DATA_ROOT" \
  --cell-type lstm \
  --epochs 50 \
  --batch-size 256 \
  --hidden-size 256 \
  --num-layers 2 \
  --dropout-rate 0.1 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --gradient-clip 1.0 \
  --lookback-days 180 \
  --horizon-days 28 \
  --train-ratio 0.85 \
  --include-external-regressors \
  --metrics mae rmse mape

echo "Completed run at $(date)"
