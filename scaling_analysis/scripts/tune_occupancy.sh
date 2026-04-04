#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J tune_occupancy
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -o ../logs/occupancy_%j.out
#SBATCH -e ../logs/occupancy_%j.err

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

echo "Starting GGap GPU Occupancy Tuning (Experiment 0b)..."
echo "Date: $(date)"
echo ""

# Create results directory
mkdir -p ../results

# Phase 1: Coarse occupancy sweep at 50 sites
# Goal: Find optimal range and maximum safe value
echo "========================================================================"
echo "Phase 1: Coarse Occupancy Sweep"
echo "Testing max_blocks_per_sm: 4, 8, 16, 32, 48, 64"
echo "Sites: 50 (from Exp 0)"
echo "Ticks: 20 per test"
echo "========================================================================"
echo ""

# Start GPU monitoring in background
rocm-smi --showuse --showmemuse --json > ../logs/gpu_usage_${SLURM_JOB_ID}.log 2>&1 &
GPU_MON_PID=$!
echo "Started GPU monitoring (PID: $GPU_MON_PID)"
echo ""

srun -N1 -n1 --cpus-per-task=7 --gpus-per-node=1 --gpu-bind=closest \
    python -u tune_occupancy.py \
        --sites 50 \
        --max_blocks_per_sm 4,8,16,32,48,64 \
        --ticks 20 \
        --csv ../results/exp0b_phase1_coarse_sweep.csv

# Stop GPU monitoring
kill $GPU_MON_PID 2>/dev/null || true
echo ""
echo "Stopped GPU monitoring"

echo ""
echo "Phase 1 complete!"
echo ""
echo "========================================================================"
echo "Next Steps:"
echo "========================================================================"
echo "1. Analyze Phase 1 results: python analyze_occupancy.py ../results/exp0b_phase1_coarse_sweep.csv"
echo "2. Identify optimal range from plots/output"
echo "3. Edit this script to uncomment Phase 2 with refined range"
echo "4. Run Phase 2 for fine-grained tuning"
echo "5. Run Phase 3 for site scaling validation"
echo ""

# Phase 2: Fine-grained refinement (UNCOMMENT AFTER PHASE 1)
# Goal: Pinpoint optimal max_blocks_per_sm value
#
# Example: If Phase 1 shows optimal around 16, test [12, 14, 16, 18, 20]
#
# echo "========================================================================"
# echo "Phase 2: Fine-Grained Refinement"
# echo "Testing max_blocks_per_sm: 12, 14, 16, 18, 20"
# echo "Sites: 50"
# echo "Ticks: 20 per test"
# echo "========================================================================"
# echo ""
#
# srun -N1 -n1 --cpus-per-task=7 --gpus-per-node=1 --gpu-bind=closest \
#     python -u tune_occupancy.py \
#         --sites 50 \
#         --max_blocks_per_sm 12,14,16,18,20 \
#         --ticks 20 \
#         --csv ../results/exp0b_phase2_fine_refinement.csv

# Phase 3: Site count scaling validation (UNCOMMENT AFTER PHASE 2)
# Goal: Prove optimal SM works across different agent counts
#
# Replace OPTIMAL_SM with the value from Phase 2
#
# echo "========================================================================"
# echo "Phase 3: Site Count Scaling Validation"
# echo "Testing sites: 10, 20, 30, 40, 50"
# echo "max_blocks_per_sm: OPTIMAL_SM (from Phase 2)"
# echo "Ticks: 20 per test"
# echo "========================================================================"
# echo ""
#
# OPTIMAL_SM=16  # TODO: Update with actual optimal value from Phase 2
#
# srun -N1 -n1 --cpus-per-task=7 --gpus-per-node=1 --gpu-bind=closest \
#     python -u tune_occupancy.py \
#         --sites 10,20,30,40,50 \
#         --max_blocks_per_sm $OPTIMAL_SM \
#         --ticks 20 \
#         --csv ../results/exp0b_phase3_site_scaling.csv

echo ""
echo "Occupancy tuning complete"
echo "Date: $(date)"
