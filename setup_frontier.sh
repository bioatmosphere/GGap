#!/bin/bash
# Environment setup for GGap on OLCF Frontier
#
# Delegates to the project-wide runtime script (modules, ggap_dev conda env,
# LD_LIBRARY_PATH, cache dirs, MPICH_GPU_SUPPORT_ENABLED). Edit that file, not this one.
#
# Usage: source setup_frontier.sh

source /lustre/orion/proj-shared/cli215/setup.sh

echo "GGap environment ready: $(which python)"
