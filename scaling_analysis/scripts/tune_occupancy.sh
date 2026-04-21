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

# Max blocks_per_sm Test - Confirm Safe Value
# Goal: Re-test max_blocks_per_sm=8 with diagnostic prints to confirm it works
echo "========================================================================"
echo "Testing max_blocks_per_sm = 8 (Confirmation test)"
echo "Sites: 20 (faster iteration)"
echo "Ticks: 20 (fused execution)"
echo ""
echo "Known results so far:"
echo "  - max_blocks_per_sm=2 (default): WORKS"
echo "  - max_blocks_per_sm=8: Worked in earlier test (no diagnostic prints)"
echo "  - max_blocks_per_sm=12: DEADLOCK"
echo "  - max_blocks_per_sm=16: DEADLOCK"
echo "  - max_blocks_per_sm=110: DEADLOCK"
echo ""
echo "Re-testing 8 with full diagnostics to confirm and get clean timing data"
echo "========================================================================"
echo ""

# Start continuous GPU monitoring in background
python -u monitor_gpu_usage.py --output ../logs/gpu_monitor_${SLURM_JOB_ID}.csv --interval 5 &
GPU_MON_PID=$!
echo "Started continuous GPU monitoring (PID: $GPU_MON_PID)"
echo "Sampling every 5 seconds -> ../logs/gpu_monitor_${SLURM_JOB_ID}.csv"
echo ""

srun -N1 -n1 --cpus-per-task=7 --gpus-per-node=1 --gpu-bind=closest \
    python -u tune_occupancy.py \
        --sites 20 \
        --max_blocks_per_sm 8 \
        --ticks 20 \
        --csv ../results/occupancy_test_sm8_confirm.csv

# Stop GPU monitoring
echo ""
echo "Stopping GPU monitoring..."
kill $GPU_MON_PID 2>/dev/null || true
wait $GPU_MON_PID 2>/dev/null || true
echo "GPU monitoring stopped"

echo ""
echo "max_blocks_per_sm=8 confirmation test complete!"
echo ""
echo "========================================================================"
echo "Results:"
echo "========================================================================"
echo "If job completed successfully (~13 minutes):"
echo "  - max_blocks_per_sm=8 CONFIRMED as safe"
echo "  - 8 blocks/CU = 880 total blocks = 112,640 threads (50% of hardware max)"
echo "  - 4x better than default (2 blocks/CU)"
echo "  - Next: Test 10 to find upper bound"
echo ""
echo "If job hangs/times out (>20 minutes):"
echo "  - Even 8 causes deadlock (surprising!)"
echo "  - Safe limit must be <8 (try 6 or 4)"
echo ""
echo "Test progress:"
echo "  - 2: WORKS ✓"
echo "  - 8: Confirming now..."
echo "  - 12: DEADLOCK ✗"
echo "  - 16: DEADLOCK ✗"
echo "  - 110: DEADLOCK ✗"
echo ""
echo "Review:"
echo "  - Log: ../logs/occupancy_${SLURM_JOB_ID}.out"
echo "  - CSV: ../results/occupancy_test_sm8_confirm.csv"
echo "  - GPU monitoring: ../logs/gpu_monitor_${SLURM_JOB_ID}.csv"
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
