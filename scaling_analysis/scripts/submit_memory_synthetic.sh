#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J mem_synthetic
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -o ../logs/mem_synthetic_%j.out
#SBATCH -e ../logs/mem_synthetic_%j.err

unset SLURM_EXPORT_ENV

echo "Memory test (synthetic 100 species)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env
export MPICH_GPU_SUPPORT_ENABLED=1
export SAGESIM_NUM_SMS=110

cd /lustre/orion/lrn088/proj-shared/objective3/xxz/GGap/scaling_analysis/scripts
mkdir -p ../logs

srun -N1 -n1 --gpus-per-node=1 python -u test_memory_synthetic.py

echo "End: $(date)"
