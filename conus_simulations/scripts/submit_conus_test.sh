#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J conus_test
#SBATCH -N 2
#SBATCH -p batch
#SBATCH -t 30:00
#SBATCH -o ../logs/conus_test_%j.out
#SBATCH -e ../logs/conus_test_%j.err

# Small test run for CONUS simulation with 16 partitions (2 nodes × 8 GPUs)
#
# Usage:
#   sbatch submit_conus_test.sh

echo "======================================================================"
echo "CONUS GGap Simulation - TEST RUN"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: 8 (GPUs)"
echo "Total ranks: $((SLURM_JOB_NUM_NODES * 8))"
echo "Start time: $(date)"
echo "======================================================================"

# Create output directories
mkdir -p ../logs
mkdir -p ../results/simulation_test

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

# Test parameters (small scale)
NUM_GAPS=10
MAXTREES=100
YEARS=1000
REPORT_INTERVAL=500
DISPERSAL_FACTOR=2.0

echo ""
echo "======================================================================"
echo "Test Parameters (Scaled Down):"
echo "  Dispersal factor: $DISPERSAL_FACTOR"
echo "  Gaps per site: $NUM_GAPS (vs 500 production)"
echo "  Max trees per gap: $MAXTREES (vs 1000 production)"
echo "  Simulation years: $YEARS"
echo "  Report interval: $REPORT_INTERVAL years (2 data snapshots)"
echo "  Sites: 1424 CONUS sites distributed across 16 ranks"
echo "======================================================================"
echo ""

# Run test
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
     --output_dir ../results/simulation_test \
     --test

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Test job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Test run completed successfully"
    echo "  Review logs in ../logs/conus_test_${SLURM_JOB_ID}.out"
    echo "  If test passes, run production with: sbatch submit_conus.sh"
else
    echo "FAILED: Test run failed with exit code $EXIT_CODE"
    echo "  Check error log: ../logs/conus_test_${SLURM_JOB_ID}.err"
fi
echo "======================================================================"

exit $EXIT_CODE
