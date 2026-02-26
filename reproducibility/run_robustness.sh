#!/bin/bash
#SBATCH --job-name=rob
#SBATCH --output=logs/robustness.out
#SBATCH --error=logs/robustness.err

#SBATCH --partition=slim16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ----------------------------
# Environment setup
# ----------------------------
module purge
module load miniconda   # or module load anaconda
source activate graph   # <-- your conda env name

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

python RobustnessMetric.py
