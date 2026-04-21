#!/bin/bash
#SBATCH -A m2467
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1

module load cudatoolkit/12.4
module load openmpi/5.0.7

cd /global/cfs/cdirs/m2467/wangb/GGap/gap
source /global/cfs/cdirs/m2467/wangb/GGap/.venv/bin/activate
mpirun -n 1 python run_one_site.py --years 500
