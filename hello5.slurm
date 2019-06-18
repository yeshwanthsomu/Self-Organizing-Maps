#!/bin/bash
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=10
#SBATCH --time=00:45:00
#SBATCH --output=output_filename.out
#SBATCH --partition=parallel 
#SBATCH -A polarizedtarget


module load gcc/system
module load openmpi
module purge
module load singularity

containerdir=/home/za2hd/testdir/test4
singularity run --nv $containerdir/tensorflow-1.12.0-py36.simg analysis_2019.py
