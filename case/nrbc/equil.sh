#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=5:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=2
#SBATCH --tasks-per-node=24

# Specify a job name:
#SBATCH -J MyMPIJob

# Specify an output file
#SBATCH -o MyMPIJob-%j.out


# Run a command


srun --mpi=pmi2 /gpfs/scratch/xl24/yixiang/src_cascade/lmp_stam -in chan.equil.run
