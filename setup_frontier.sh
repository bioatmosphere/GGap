#!/bin/bash
# Environment setup for GGap on OLCF Frontier
#
# Uses the shared sagesim_env directly.
#
# Usage: source setup_frontier.sh

module load cpe/26.03
module load PrgEnv-gnu
module load miniforge3
module load rocm/7.2.0
module load craype-accel-amd-gfx90a
module load metis/5.1.0

source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

echo "GGap environment ready."
