#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J strong_scaling
#SBATCH -N 32
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o ../logs/strong_scaling_%j.out
#SBATCH -e ../logs/strong_scaling_%j.err

unset SLURM_EXPORT_ENV

# Strong Scaling: Fixed 1024 sites, 1-32 nodes (8-256 GPUs)
# grid_height=4, width=256, cross-rank=24/rank

echo "========================================"
echo "Strong Scaling: 1024 sites fixed"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env
export MPICH_GPU_SUPPORT_ENABLED=1
export SAGESIM_NUM_SMS=110

cd /lustre/orion/lrn088/proj-shared/objective3/xxz/GGap/scaling_analysis/scripts
mkdir -p ../results ../logs

TOTAL_SITES=1024
GRID_HEIGHT=4
NUM_GAPS=500
MAXTREES=1000
TICKS=1000
MAX_BLOCKS_PER_SM=8

echo "Configuration:"
echo "  Total sites: $TOTAL_SITES (fixed)"
echo "  Grid: ${GRID_HEIGHT}x$((TOTAL_SITES / GRID_HEIGHT))"
echo "  Gaps/site: $NUM_GAPS, Trees/gap: $MAXTREES"
echo "  Ticks: $TICKS"
echo ""

CSV_FILE="../results/strong_scaling.csv"

for NNODES in 1 2 4 8 16 32; do
    NGPUS=$((NNODES * 8))
    SPG=$((TOTAL_SITES / NGPUS))

    echo "========================================"
    echo "Test: $NNODES node(s), $NGPUS GPUs, $SPG sites/GPU"
    echo "Start: $(date)"

    srun -N $NNODES -n $NGPUS \
        --cpus-per-task=7 \
        --gpus-per-node=8 \
        --gpu-bind=closest \
        python -u test_strong_scaling.py \
            --total-sites $TOTAL_SITES \
            --grid-height $GRID_HEIGHT \
            --num-gaps $NUM_GAPS \
            --maxtrees $MAXTREES \
            --ticks $TICKS \
            --max-blocks-per-sm $MAX_BLOCKS_PER_SM \
            --csv $CSV_FILE

    echo "Exit: $?, End: $(date)"
    echo ""
done

echo "Done. Results: $CSV_FILE"
