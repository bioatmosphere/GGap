#!/bin/bash
# Environment setup for GGap on OLCF Frontier
#
# Prerequisites: clone sagesim_env to your own prefix first:
#   conda create --prefix /path/to/your/env --clone /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env
#
# Usage: source setup_frontier.sh

module load miniforge3/23.11.0-0
module load cpe/25.09
module load rocm/6.4.2

source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env_xxz

echo "GGap environment ready."
