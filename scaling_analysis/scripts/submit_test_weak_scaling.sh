#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J test_weak_scaling
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -o ../logs/test_weak_scaling_%j.out
#SBATCH -e ../logs/test_weak_scaling_%j.err

unset SLURM_EXPORT_ENV

# Weak Scaling Test - Single Node (1-8 GPUs)
# Tests: 1, 2, 4, 8 GPUs with 8 sites per GPU (minimum for proper weak scaling)

echo "================================"
echo "Weak Scaling Test - Single Node"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

# Load Frontier modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

# Activate conda environment
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env

# Enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# Go to scripts directory
cd /lustre/orion/lrn088/proj-shared/objective3/xxz/GGap/scaling_analysis/scripts

# Create results directory
mkdir -p ../results
mkdir -p ../logs

# Configuration
# 1D slab decomposition: grid_height=4, each rank owns 4x1 column
# Cross-rank edges per rank = 24 (constant for 2+ GPUs)
SITES_PER_GPU=4
GRID_HEIGHT=4
NUM_GAPS=500
MAXTREES=1000
TICKS=20
MAX_BLOCKS_PER_SM=8

echo "Configuration:"
echo "  Sites per GPU: $SITES_PER_GPU (1D slab: ${GRID_HEIGHT}x1 block/rank)"
echo "  Grid height: $GRID_HEIGHT"
echo "  Gaps per site: $NUM_GAPS"
echo "  Trees per gap: $MAXTREES"
echo "  Ticks: $TICKS"
echo "  max_blocks_per_sm: $MAX_BLOCKS_PER_SM"
echo ""

# Set AMD GPU workaround (110 CUs for MI250X)
export SAGESIM_NUM_SMS=110

# Single CSV file for all GPU counts (rows appended per run)
CSV_FILE="../results/test_weak_scaling.csv"
rm -f $CSV_FILE

# Run tests for 1, 2, 4, 8 GPUs
for NGPUS in 1 2 4 8; do
    echo "================================"
    echo "Test: $NGPUS GPU(s)"
    echo "================================"
    echo "Start: $(date)"

    srun -N1 -n $NGPUS \
        --cpus-per-task=7 \
        --gpus-per-node=$NGPUS \
        --gpu-bind=closest \
        python -u test_weak_scaling_single_node.py \
            --sites-per-gpu $SITES_PER_GPU \
            --grid-height $GRID_HEIGHT \
            --num-gaps $NUM_GAPS \
            --maxtrees $MAXTREES \
            --ticks $TICKS \
            --max-blocks-per-sm $MAX_BLOCKS_PER_SM \
            --csv $CSV_FILE

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Success: $NGPUS GPU(s)"
    else
        echo "FAILED: $NGPUS GPU(s) with exit code $EXIT_CODE"
    fi

    echo "End: $(date)"
    echo ""
done

echo "================================"
echo "All tests complete"
echo "End time: $(date)"
echo "================================"
echo ""
echo "Results saved in: ../results/"
echo "Logs saved in: ../logs/"
