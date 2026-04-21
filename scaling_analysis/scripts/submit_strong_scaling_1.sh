#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J ss_1
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o ../logs/ss_1_%j.out
#SBATCH -e ../logs/ss_1_%j.err

unset SLURM_EXPORT_ENV

# Strong Scaling: 1 Node (8 GPUs) — baseline
# Fixed 2048 sites, grid_height=4, width=512
# 256 sites/GPU (heaviest workload)

echo "========================================"
echo "Strong Scaling: 1 Node (8 GPUs) — baseline"
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

TOTAL_SITES=2048
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
echo "  Sites/GPU: $((TOTAL_SITES / 8))"
echo ""

CSV_FILE="../results/strong_scaling.csv"

NNODES=1
NGPUS=8
SPG=$((TOTAL_SITES / NGPUS))

echo "========================================"
echo "Test: $NNODES node, $NGPUS GPUs, $SPG sites/GPU"
echo "Start: $(date)"

srun -N $NNODES -n $NGPUS \
    --cpus-per-task=7 \
    --gpus-per-node=8 \
    --gpu-bind=closest \
    python -u strong_scaling.py \
        --total-sites $TOTAL_SITES \
        --grid-height $GRID_HEIGHT \
        --num-gaps $NUM_GAPS \
        --maxtrees $MAXTREES \
        --ticks $TICKS \
        --max-blocks-per-sm $MAX_BLOCKS_PER_SM \
        --csv $CSV_FILE

echo "Exit: $?, End: $(date)"
echo ""
echo "Done. Results: $CSV_FILE"
