#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J weak_scaling
#SBATCH -N 20
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -o ../logs/weak_scaling_%j.out
#SBATCH -e ../logs/weak_scaling_%j.err

unset SLURM_EXPORT_ENV

# Weak Scaling: 1-20 Nodes (8-160 GPUs)
# 1D slab decomposition: grid_height=5, block=5x2 per rank
# 10 sites/GPU, 30 cross-rank edges/rank (matches CONUS ~25-30)

echo "========================================"
echo "Weak Scaling: 1-20 Nodes (8-160 GPUs)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Nodes allocated: $SLURM_NNODES"
echo ""

# Load Frontier modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

# Activate conda environment
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env

# Enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# AMD GPU workaround (110 CUs for MI250X)
export SAGESIM_NUM_SMS=110

# Go to scripts directory
cd /lustre/orion/lrn088/proj-shared/objective3/xxz/GGap/scaling_analysis/scripts

# Create output directories
mkdir -p ../results
mkdir -p ../logs

# Configuration
SITES_PER_GPU=10
GRID_HEIGHT=5
NUM_GAPS=500
MAXTREES=1000
TICKS=1000
MAX_BLOCKS_PER_SM=8

echo "Configuration:"
echo "  Sites per GPU: $SITES_PER_GPU (1D slab: ${GRID_HEIGHT}x2 block/rank)"
echo "  Grid height: $GRID_HEIGHT"
echo "  Cross-rank edges/rank: 30 (matches CONUS ~25-30)"
echo "  Gaps per site: $NUM_GAPS"
echo "  Trees per gap: $MAXTREES"
echo "  Ticks: $TICKS"
echo "  max_blocks_per_sm: $MAX_BLOCKS_PER_SM"
echo ""

# Single CSV file for all results
CSV_FILE="../results/weak_scaling.csv"
rm -f $CSV_FILE

# Run tests: 1, 2, 5, 10, 20 nodes (8, 16, 40, 80, 160 GPUs)
for NNODES in 1 2 5 10 20; do
    NGPUS=$((NNODES * 8))
    TOTAL_SITES=$((NGPUS * SITES_PER_GPU))

    echo "========================================"
    echo "Test: $NNODES node(s), $NGPUS GPUs, $TOTAL_SITES sites"
    echo "========================================"
    echo "Start: $(date)"

    srun -N $NNODES -n $NGPUS \
        --cpus-per-task=7 \
        --gpus-per-node=8 \
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
        echo "Success: $NNODES node(s), $NGPUS GPUs"
    else
        echo "FAILED: $NNODES node(s), $NGPUS GPUs (exit code $EXIT_CODE)"
    fi

    echo "End: $(date)"
    echo ""
done

echo "========================================"
echo "All tests complete"
echo "End time: $(date)"
echo "========================================"
echo ""
echo "Results: ../results/weak_scaling.csv"
echo "Logs: ../logs/weak_scaling_${SLURM_JOB_ID}.out"
