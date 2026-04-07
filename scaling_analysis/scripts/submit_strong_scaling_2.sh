#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J ss_2
#SBATCH -N 2
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -o ../logs/ss_2_%j.out
#SBATCH -e ../logs/ss_2_%j.err

unset SLURM_EXPORT_ENV

# Strong Scaling: 2 Nodes (16 GPUs)
# Fixed 2048 sites, grid_height=4, width=512
# 128 sites/GPU

echo "========================================"
echo "Strong Scaling: 2 Nodes (16 GPUs)"
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
echo "  Sites/GPU: $((TOTAL_SITES / 16))"
echo ""

CSV_FILE="../results/strong_scaling.csv"

NNODES=2
NGPUS=16
SPG=$((TOTAL_SITES / NGPUS))

echo "========================================"
echo "Test: $NNODES nodes, $NGPUS GPUs, $SPG sites/GPU"
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
