#!/bin/bash
# launch_pipeline.sh — Non-interactive full pipeline launcher for Hercules.
# Usage: sbatch core/launch_pipeline.sh   (master job that submits the chain)
#    or: bash core/launch_pipeline.sh     (direct submission from login node)
#
# Refreshes command files and submits all phases with sequential dependencies.

#SBATCH --partition=standard
#SBATCH --job-name=QCL_master
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

# ── 1. Conda activation ────────────────────────────────────────────────────
HERCULES_CONDA="/lustre/software/easybuild/common/software/Miniconda3/4.9.2"
CONDA_ENV="${CONDA_ENV:-qcl}"

if [ -f "$HERCULES_CONDA/etc/profile.d/conda.sh" ]; then
    source "$HERCULES_CONDA/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find conda at $HERCULES_CONDA"
    exit 1
fi

if ! source activate "$CONDA_ENV" 2>/dev/null; then
    echo "[WARN] Conda env '$CONDA_ENV' not found. Creating..."
    conda create -n "$CONDA_ENV" python=3.10 -y
    source activate "$CONDA_ENV"
    pip install -r requirements.txt
    echo "[OK] Environment $CONDA_ENV ready."
fi

# ── 2. Paths ───────────────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}"
mkdir -p logs results

# ── 3. Refresh command files ───────────────────────────────────────────────
echo "============================================================"
echo "  Refreshing command files..."
echo "============================================================"
python runner.py --config config.yaml --export-commands

# ── 4. Helper: count lines ─────────────────────────────────────────────────
count_lines() {
    local f="$1"
    if [ ! -f "$f" ]; then echo 0; return; fi
    grep -cve '^\s*$' "$f" 2>/dev/null || echo 0
}

# ── 5. Submit phases with sequential dependencies ──────────────────────────
PHASES=("cmds_1_topology_ideal.txt" "cmds_2_topology_noisy.txt" "cmds_3_crossdomain_ideal.txt" "cmds_4_crossdomain_noisy.txt")
NAMES=("topology_ideal" "topology_noisy" "crossdomain_ideal" "crossdomain_noisy")
IDS=("1" "2" "3" "4")

PREV_JOB=""
echo ""
echo "============================================================"
echo "  Submitting phased SLURM array jobs"
echo "============================================================"

for i in "${!PHASES[@]}"; do
    CMD_FILE="${PHASES[$i]}"
    NAME="${NAMES[$i]}"
    ID="${IDS[$i]}"
    N=$(count_lines "$CMD_FILE")

    if [ "$N" -eq 0 ]; then
        echo "[WARN] $CMD_FILE is empty, skipping phase $ID"
        continue
    fi

    if [ -n "$PREV_JOB" ]; then
        JOB_ID=$(sbatch --parsable \
            --job-name="QCL_${ID}_${NAME}" \
            --array="1-${N}%30" \
            --dependency=afterok:"${PREV_JOB}" \
            --export="CMD_FILE=${CMD_FILE}" \
            core/slurm_generic.sh)
    else
        JOB_ID=$(sbatch --parsable \
            --job-name="QCL_${ID}_${NAME}" \
            --array="1-${N}%30" \
            --export="CMD_FILE=${CMD_FILE}" \
            core/slurm_generic.sh)
    fi

    echo "  Phase [$ID] $NAME  →  Job ID: $JOB_ID  ($N tasks)"
    PREV_JOB="$JOB_ID"
done

echo "============================================================"
echo "  All phases submitted. Chain: ${IDS[@]}"
echo "  Monitor with: squeue -u \$USER"
echo "============================================================"
