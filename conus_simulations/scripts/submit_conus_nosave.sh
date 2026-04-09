#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J conus_nosave
#SBATCH -N 20
#SBATCH -p batch
#SBATCH -t 30:00
#SBATCH -o ../logs/conus_nosave_%j.out
#SBATCH -e ../logs/conus_nosave_%j.err

# CONUS-wide GGap simulation WITHOUT periodic snapshots.
# Measures pure simulation cost (no GPU->CPU transfer, no disk save inside loop).
# Companion run to submit_conus.sh for SC2026 paper I/O overhead analysis.
#
# Usage:
#   sbatch submit_conus_nosave.sh

echo "======================================================================"
echo "CONUS GGap Simulation (NO SNAPSHOTS)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: 8 (GPUs)"
echo "Total ranks: $((SLURM_JOB_NUM_NODES * 8))"
echo "Start time: $(date)"
echo "======================================================================"

# Create output directories
mkdir -p ../logs
mkdir -p ../results/simulation_nosave

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

# Production parameters (match submit_conus.sh so timings are comparable)
NUM_GAPS=500
MAXTREES=1000
YEARS=1000
REPORT_INTERVAL=10
DISPERSAL_FACTOR=2.0

echo ""
echo "======================================================================"
echo "Production Parameters (no-snapshot mode):"
echo "  Dispersal factor: $DISPERSAL_FACTOR"
echo "  Gaps per site: $NUM_GAPS"
echo "  Max trees per gap: $MAXTREES"
echo "  Simulation years: $YEARS"
echo "  Report interval: $REPORT_INTERVAL years (timing prints only, no data save)"
echo "  Sites: 1424 CONUS sites distributed across 160 ranks"
echo "  Snapshots: DISABLED (--no_snapshots)"
echo "======================================================================"
echo ""

# Run simulation with snapshots disabled
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
     --output_dir ../results/simulation_nosave

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "No-snapshot job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: No-snapshot run completed successfully"
    echo "  Review logs in ../logs/conus_nosave_${SLURM_JOB_ID}.out"
    echo "  Compare against ../logs/conus_<jobid>.out (with snapshots)"
else
    echo "FAILED: No-snapshot run failed with exit code $EXIT_CODE"
    echo "  Check error log: ../logs/conus_nosave_${SLURM_JOB_ID}.err"
fi
echo "======================================================================"

exit $EXIT_CODE
