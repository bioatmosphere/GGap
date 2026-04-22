#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J ss_1to64
#SBATCH -N 64
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o ../logs/ss_%j.out
#SBATCH -e ../logs/ss_%j.err

unset SLURM_EXPORT_ENV

# Strong Scaling: 1-64 Nodes (8-512 GPUs)
# Fixed 2048 sites, 200 gaps, 500 trees

echo "========================================"
echo "Strong Scaling: 1-64 Nodes (8-512 GPUs)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env_xxz
export MPICH_GPU_SUPPORT_ENABLED=1
export SAGESIM_NUM_SMS=110

cd ${SLURM_SUBMIT_DIR}
mkdir -p ../results ../logs

TOTAL_SITES=2048
GRID_HEIGHT=4
NUM_GAPS=200
MAXTREES=500
TICKS=1000
MAX_BLOCKS_PER_SM=8

CSV_FILE="../results/strong_scaling.csv"

for NNODES in 1 2 4 8 16 32 64; do
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
