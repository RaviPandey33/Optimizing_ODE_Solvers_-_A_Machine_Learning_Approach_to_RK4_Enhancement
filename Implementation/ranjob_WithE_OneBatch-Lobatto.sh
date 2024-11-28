#!/bin/bash
#SBATCH --job-name=S2E-Original
#SBATCH --time=05:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=hpc-prf-mldens


export XLA_FLAGS="--xla_force_host_platform_device_count=100"
srun python With_Energy/OneBatch-Original_Lobatto_3A3B.py
