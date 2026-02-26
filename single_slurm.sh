#!/bin/bash
#SBATCH -N1
#SBATCH --qos=long
#SBATCH -G1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=10-0
#SBATCH --output=logs/loyo_%j.out

# --- Environment Setup ---
source ~/.bashrc
# conda activate your_env_name  # Uncomment and use your environment name
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib64:$LD_LIBRARY_PATH

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Run Training ---
# $1 is the year passed from the Python batch script
echo "Starting Stage 2 LOYO training for year: $1"

python /home/spotter5/fire_projections/train_loyo_node.py --year $1