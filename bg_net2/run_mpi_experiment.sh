#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A snic2022-22-368

# The name of the script is myjob
#SBATCH -J test_julia

# The partition
#SBATCH -p main

# 10 hours wall-clock time will be given to this job
#SBATCH -t 04:00:00

# Number of nodes
#SBATCH --nodes=4

# Number of MPI processes per node
#SBATCH --ntasks-per-node=64

# Run the executable named myexe
# and write the output into my_output_file
srun julia mpi_experiment.jl > /cfs/klemming/home/e/emilwa/Private/slurm_output12.txt 2> /cfs/klemming/home/e/emilwa/Private/slurm_stderr12.txt

