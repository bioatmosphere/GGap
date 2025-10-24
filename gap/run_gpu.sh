#!/bin/bash
#SBATCH -A m2467
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 10:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1

module load cuda/12.2
module load openmpi/5.0.7

cd /global/cfs/cdirs/m2467/wangb/GGap/gap
source /global/cfs/cdirs/m2467/wangb/GGap/.venv/bin/activate
srun -n 1 python run.py --num_trees 200 --years 100 --species_dist mixed
