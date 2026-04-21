#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J test_wsb
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -o ../logs/test_wsb_%j.out
#SBATCH -e ../logs/test_wsb_%j.err

unset SLURM_EXPORT_ENV

# Weak Scaling B Test - Single Node (1-8 GPUs)
# 100 sites/GPU, grid_height=10, block=10x10/rank
# Cross-rank edges per rank = 60, cross-rank fraction = 7.5%

echo "================================"
echo "Weak Scaling B Test - Single Node"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env
export MPICH_GPU_SUPPORT_ENABLED=1
export SAGESIM_NUM_SMS=110

cd /lustre/orion/lrn088/proj-shared/objective3/xxz/GGap/scaling_analysis/scripts
mkdir -p ../results ../logs

SITES_PER_GPU=100
GRID_HEIGHT=10
NUM_GAPS=200
MAXTREES=500
TICKS=20
MAX_BLOCKS_PER_SM=8

echo "Configuration:"
echo "  Sites per GPU: $SITES_PER_GPU (1D slab: ${GRID_HEIGHT}x10 block/rank)"
echo "  Grid height: $GRID_HEIGHT"
echo "  Gaps per site: $NUM_GAPS"
echo "  Trees per gap: $MAXTREES"
echo "  Ticks: $TICKS"
echo "  max_blocks_per_sm: $MAX_BLOCKS_PER_SM"
echo ""

CSV_FILE="../results/test_weak_scaling_b.csv"
rm -f $CSV_FILE

for NGPUS in 1 2 4 8; do
    echo "================================"
    echo "Test: $NGPUS GPU(s)"
    echo "================================"
    echo "Start: $(date)"

    srun -N1 -n $NGPUS \
        --cpus-per-task=7 \
        --gpus-per-node=$NGPUS \
        --gpu-bind=closest \
        python -u weak_scaling.py \
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
