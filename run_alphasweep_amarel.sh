#!/bin/bash
#SBATCH --job-name=alpha_sweep_amarel      # Descriptive job name.
#SBATCH --partition=main                   # Use the main partition.
#SBATCH --nodes=1                          # Request one node.
#SBATCH --ntasks=1                         # One task (this process will spawn multiple threads).
#SBATCH --cpus-per-task=64                 # Request 64 CPU cores.
#SBATCH --mem=32G                        # Request 32 GB of memory.
#SBATCH --time=03:00:00                    # Set a time limit of 3 hours.
#SBATCH --output=alphasweep_amarel_%j.out   # Standard output (job ID appended).
#SBATCH --error=alphasweep_amarel_%j.err    # Standard error.

module purge
module load python/3.8
cd $SLURM_SUBMIT_DIR
python3 alphasweep3amarel.py