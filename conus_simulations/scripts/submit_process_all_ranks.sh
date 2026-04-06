#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J process_conus
#SBATCH --array=0-159
#SBATCH -t 01:00:00
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -o logs/process_rank_%a.out
#SBATCH -e logs/process_rank_%a.err

echo "========================================================================"
echo "CONUS Snapshot Post-Processing - Rank $SLURM_ARRAY_TASK_ID"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================================================"

# Load Python module
module load cray-python

# Set working directory
cd /lustre/orion/lrn088/proj-shared/objective3/xxz/GGap/conus_simulations

# Create logs directory if it doesn't exist
mkdir -p logs

# Process this rank
RANK=$SLURM_ARRAY_TASK_ID

python scripts/process_snapshots.py \
    --snapshot_dir results/simulation/snapshots \
    --output_dir results/simulation \
    --ranks $RANK \
    --no_tree_data

EXIT_CODE=$?

echo "========================================================================"
echo "Processing Complete for Rank $RANK"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================================================"

exit $EXIT_CODE
