#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J conus_test_nosave
#SBATCH -N 2
#SBATCH -p batch
#SBATCH -t 30:00
#SBATCH -o ../logs/conus_test_nosave_%j.out
#SBATCH -e ../logs/conus_test_nosave_%j.err

# Small test run for CONUS simulation WITHOUT periodic snapshots.
# Same scale as submit_conus_test.sh (2 nodes x 8 GPUs = 16 ranks)
# but calls simulate(years) in a single invocation with no GPU->CPU transfers
# and no disk saves inside the loop.
#
# Purpose:
#   1. Smoke-test the --no_snapshots code path before running the full 20-node job.
#   2. Fast A/B comparison of I/O overhead at small scale.
#
# Usage:
#   sbatch submit_conus_test_nosave.sh

echo "======================================================================"
echo "CONUS GGap Simulation - TEST RUN (NO SNAPSHOTS)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: 8 (GPUs)"
echo "Total ranks: $((SLURM_JOB_NUM_NODES * 8))"
echo "Start time: $(date)"
echo "======================================================================"

# Create output directories
mkdir -p ../logs
mkdir -p ../results/simulation_test_nosave

unset SLURM_EXPORT_ENV

# Load Frontier modules (matching working scaling_analysis configuration)
echo "Loading modules..."
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load metis/5.1.0

# Activate conda environment
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env

# Enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# Test parameters (match submit_conus_test.sh so the two logs are comparable)
NUM_GAPS=10
MAXTREES=100
YEARS=1000
REPORT_INTERVAL=500
DISPERSAL_FACTOR=2.0

echo ""
echo "======================================================================"
echo "Test Parameters (Scaled Down, no-snapshot mode):"
echo "  Dispersal factor: $DISPERSAL_FACTOR"
echo "  Gaps per site: $NUM_GAPS (vs 500 production)"
echo "  Max trees per gap: $MAXTREES (vs 1000 production)"
echo "  Simulation years: $YEARS"
echo "  Report interval: $REPORT_INTERVAL years (unused in no-snapshot mode)"
echo "  Sites: 1424 CONUS sites distributed across 16 ranks"
echo "  Snapshots: DISABLED (--no_snapshots)"
echo "======================================================================"
echo ""

# Run test with snapshots disabled
srun -n $((SLURM_JOB_NUM_NODES * 8)) \
     --ntasks-per-node=8 \
     --gpus-per-task=1 \
     --gpu-bind=closest \
     python run_conus.py \
     --dispersal_factor $DISPERSAL_FACTOR \
     --num_gaps $NUM_GAPS \
     --maxtrees $MAXTREES \
     --years $YEARS \
     --report_interval $REPORT_INTERVAL \
     --no_snapshots \
     --output_dir ../results/simulation_test_nosave \
     --test

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Test job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: No-snapshot test run completed successfully"
    echo "  Review logs in ../logs/conus_test_nosave_${SLURM_JOB_ID}.out"
    echo "  Compare against ../logs/conus_test_<jobid>.out (with snapshots)"
    echo "  If test passes, run production with: sbatch submit_conus_nosave.sh"
else
    echo "FAILED: No-snapshot test run failed with exit code $EXIT_CODE"
    echo "  Check error log: ../logs/conus_test_nosave_${SLURM_JOB_ID}.err"
fi
echo "======================================================================"

exit $EXIT_CODE
