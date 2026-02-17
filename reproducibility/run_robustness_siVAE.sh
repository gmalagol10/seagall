#!/bin/bash
#SBATCH --job-name=rob_siVAE
#SBATCH --output=logs/robustness_siVAE.out
#SBATCH --error=logs/robustness_siVAE.err

#SBATCH --partition=slim16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ----------------------------
# Environment setup
# ----------------------------
CONDA_ENV=graph
ACTIVATE=/store24/project24/ladcol_012/miniconda3/bin/activate
source ${ACTIVATE} ${CONDA_ENV}

# Prevent CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optional: PyTorch safety
export CUDA_VISIBLE_DEVICES=""

# ----------------------------
# Run the script
# ----------------------------
echo "Starting job on $(hostname)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

python RobustnessMetric_siVAE.py
