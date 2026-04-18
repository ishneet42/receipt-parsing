#!/bin/bash
#SBATCH --job-name=layoutlmv3-full
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ishneetkaur.chadha@colorado.edu

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
VENV_DIR="${VENV_DIR:-/scratch/alpine/$USER/receipt-parsing/.venv}"
RUN_DIR="${PROJECT_ROOT}/outputs/layoutlmv3-full/${SLURM_JOB_ID}"
RUN_LOG="${PROJECT_ROOT}/logs/layoutlmv3-full-${SLURM_JOB_ID}.log"

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Job started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Project root: ${PROJECT_ROOT}"

module purge || true
module load anaconda

cd "${PROJECT_ROOT}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL _CE_CONDA _CE_M
source "${VENV_DIR}/bin/activate"

export HF_HOME="/scratch/alpine/$USER/receipt-parsing/.hf-cache"
export TOKENIZERS_PARALLELISM=false

nvidia-smi || true

python train_layoutlmv3.py \
  --output-dir "${RUN_DIR}" \
  --num-train-epochs 50 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --logging-steps 25 \
  --eval-strategy epoch \
  --save-strategy epoch \
  --save-total-limit 2

echo "Job finished at $(date)"
