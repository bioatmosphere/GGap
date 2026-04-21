#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J ss_test
#SBATCH -N 16
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -o ../logs/ss_test_%j.out
#SBATCH -e ../logs/ss_test_%j.err

unset SLURM_EXPORT_ENV

# Strong Scaling Test: Fixed 512 sites, 1-16 nodes (8-128 GPUs)
# Quick test with 20 ticks before full 1000-tick run

echo "========================================"
echo "Strong Scaling Test (20 ticks)"
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
NUM_GAPS=100
MAXTREES=500
TICKS=20
MAX_BLOCKS_PER_SM=8

echo "Configuration:"
echo "  Total sites: $TOTAL_SITES (fixed)"
echo "  Grid: ${GRID_HEIGHT}x$((TOTAL_SITES / GRID_HEIGHT))"
echo "  Gaps/site: $NUM_GAPS, Trees/gap: $MAXTREES"
echo "  Ticks: $TICKS (test run)"
echo ""

CSV_FILE="../results/strong_scaling_test.csv"

for NNODES in 8 16; do
    NGPUS=$((NNODES * 8))
    SPG=$((TOTAL_SITES / NGPUS))

    echo "========================================"
    echo "Test: $NNODES node(s), $NGPUS GPUs, $SPG sites/GPU"
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
done

echo "Done. Results: $CSV_FILE"
