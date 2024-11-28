#!/bin/bash
#SBATCH --job-name=S2NE-Random
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --account=hpc-prf-mldens


export XLA_FLAGS="--xla_force_host_platform_device_count=100"
srun python Without_Energy/OneRandom-OneLobatto_WithoutE.py
