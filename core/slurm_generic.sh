#!/bin/bash
# slurm_generic.sh — Generic SLURM array template
# Works with any runner.py + cmd_*.txt setup.
#
# Required --export variable: CMD_FILE (path to the command list file)
# Optional: CONDA_ENV (default: qcl), CONDA_BASE (auto-detected)

#SBATCH --partition=standard
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dmarper2@upo.es

# ── 1. Conda activation ────────────────────────────────────────────────────
# Hercules: explicit path; fallback to auto-detection for other clusters.
HERCULES_CONDA="/lustre/software/easybuild/common/software/Miniconda3/4.9.2"
CONDA_ENV="${CONDA_ENV:-qcl}"

if [ -f "$HERCULES_CONDA/etc/profile.d/conda.sh" ]; then
    source "$HERCULES_CONDA/etc/profile.d/conda.sh"
else
    # Generic fallback: use conda info to find base
    CONDA_BASE_AUTO=$(conda info --base 2>/dev/null)
    CONDA_BASE_FALLBACK="${CONDA_BASE_AUTO:-$HOME/miniconda3}"
    if [ -f "$CONDA_BASE_FALLBACK/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE_FALLBACK/etc/profile.d/conda.sh"
    else
        echo "[ERROR] Cannot find conda. Set CONDA_BASE or install Miniconda."
        exit 1
    fi
fi

# Hercules requires 'source activate' instead of 'conda activate'
source activate "$CONDA_ENV" || { echo "[ERROR] Cannot activate env: $CONDA_ENV"; exit 1; }

# ── 2. Working directory ───────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR" || { echo "[ERROR] SLURM_SUBMIT_DIR not set."; exit 1; }

# ── 3. Command selection ───────────────────────────────────────────────────
if [ -z "$CMD_FILE" ]; then
    echo "[ERROR] CMD_FILE not set. Use --export=CMD_FILE=<path> in sbatch."
    exit 1
fi

if [ ! -f "$CMD_FILE" ]; then
    echo "[ERROR] CMD_FILE not found: $CMD_FILE"
    exit 1
fi

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMD_FILE")

if [ -z "$CMD" ]; then
    echo "[ERROR] No command at line $SLURM_ARRAY_TASK_ID in $CMD_FILE"
    exit 1
fi

# ── 4. Execution ───────────────────────────────────────────────────────────
echo "============================================================"
echo "  Job:    $SLURM_ARRAY_JOB_ID  |  Task: $SLURM_ARRAY_TASK_ID"
echo "  Node:   $SLURMD_NODENAME"
echo "  File:   $CMD_FILE"
echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo "CMD: $CMD"
echo "------------------------------------------------------------"

eval "$CMD ${EXTRA_ARGS} --machine-id hercules"

EXIT_CODE=$?
echo "------------------------------------------------------------"
echo "  Finish: $(date '+%Y-%m-%d %H:%M:%S')  |  Exit code: $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
