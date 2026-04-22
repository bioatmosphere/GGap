#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J conus_ggap
#SBATCH -N 20
#SBATCH -q debug
#SBATCH -t 1:00:00
#SBATCH -o ../logs/conus_%j.out
#SBATCH -e ../logs/conus_%j.err

# CONUS-wide GGap simulation with 160 partitions (20 nodes × 8 GPUs)
#
# Usage:
#   sbatch submit_conus.sh

echo "======================================================================"
echo "CONUS GGap Simulation"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: 8 (GPUs)"
echo "Total ranks: $((SLURM_JOB_NUM_NODES * 8))"
echo "Start time: $(date)"
echo "======================================================================"

# Create output directories
mkdir -p ../logs
mkdir -p ../results/simulation

unset SLURM_EXPORT_ENV

# Load Frontier modules (matching working scaling_analysis configuration)
echo "Loading modules..."
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load metis/5.1.0

# Activate conda environment
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env_xxz

# Enable GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# Production parameters
NUM_GAPS=500
MAXTREES=1000
YEARS=1000
REPORT_INTERVAL=10
DISPERSAL_FACTOR=2.0

echo ""
echo "======================================================================"
echo "Production Parameters:"
echo "  Dispersal factor: $DISPERSAL_FACTOR"
echo "  Gaps per site: $NUM_GAPS"
echo "  Max trees per gap: $MAXTREES"
echo "  Simulation years: $YEARS"
echo "  Report interval: $REPORT_INTERVAL years (100 data snapshots)"
echo "  Sites: 1424 CONUS sites distributed across 160 ranks"
echo "======================================================================"
echo ""

# Run production simulation
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
     --output_dir ../results/simulation

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Production job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Production run completed successfully"
    echo "  Review logs in ../logs/conus_${SLURM_JOB_ID}.out"
    echo "  Snapshots: ../results/simulation/snapshots/"
    echo "  To generate CSVs: python extract_last_species_data.py --snapshot_dir ../results/simulation/snapshots --metadata_dir ../results/simulation --output_dir ../results/last_year_species"
else
    echo "FAILED: Production run failed with exit code $EXIT_CODE"
    echo "  Check error log: ../logs/conus_${SLURM_JOB_ID}.err"
fi
echo "======================================================================"

exit $EXIT_CODE
