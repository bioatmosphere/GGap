#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J wsb_128
#SBATCH -N 128
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o ../logs/wsb_128_%j.out
#SBATCH -e ../logs/wsb_128_%j.err

unset SLURM_EXPORT_ENV

# Weak Scaling B: 128 Nodes (1024 GPUs)
# 100 sites/GPU, grid_height=10, block=10x10/rank

echo "========================================"
echo "Weak Scaling B: 128 Nodes (1024 GPUs)"
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

SITES_PER_GPU=100
GRID_HEIGHT=10
NUM_GAPS=200
MAXTREES=500
TICKS=1000
MAX_BLOCKS_PER_SM=8

CSV_FILE="../results/weak_scaling_b.csv"

NNODES=128
NGPUS=$((NNODES * 8))
echo "========================================"
echo "Test: $NNODES node(s), $NGPUS GPUs, $((NGPUS * SITES_PER_GPU)) sites"
echo "Start: $(date)"

srun -N $NNODES -n $NGPUS \
    --cpus-per-task=7 \
    --gpus-per-node=8 \
    --gpu-bind=closest \
    python -u weak_scaling.py \
        --sites-per-gpu $SITES_PER_GPU \
        --grid-height $GRID_HEIGHT \
        --num-gaps $NUM_GAPS \
        --maxtrees $MAXTREES \
        --ticks $TICKS \
        --max-blocks-per-sm $MAX_BLOCKS_PER_SM \
        --csv $CSV_FILE

echo "Exit: $?, End: $(date)"
echo ""
echo "Done. Results: $CSV_FILE"
