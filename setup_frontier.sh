#!/bin/bash
# Environment setup for GGap on OLCF Frontier
#
# Uses the shared sagesim_env directly.
#
# Usage: source setup_frontier.sh

module load miniforge3/23.11.0-0
module load cpe/25.09
module load rocm/6.4.2

source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env

echo "GGap environment ready."
