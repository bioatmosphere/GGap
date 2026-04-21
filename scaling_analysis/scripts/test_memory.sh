#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J memory_test
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -o ../logs/memory_%j.out
#SBATCH -e ../logs/memory_%j.err

unset SLURM_EXPORT_ENV

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

echo "Starting GGap memory capacity test..."
echo "Date: $(date)"
echo ""

# Run memory test on single GPU (use -u for unbuffered output)
srun -N1 -n1 --cpus-per-task=7 --gpus-per-node=1 --gpu-bind=closest python -u find_memory_limit.py

echo ""
echo "Memory test complete"
echo "Date: $(date)"
