#!/bin/bash
#SBATCH --job-name=xgb_loyo_array
#SBATCH --output=logs/loyo_%a.out    # Separate log file for each year
#SBATCH --error=logs/loyo_%a.err     # Separate error file for each year
#SBATCH --array=2001-2022            # Creates 22 parallel tasks (one per year)
#SBATCH --ntasks=1                   # Each array task is 1 process
#SBATCH -N 1                         # Each process stays on 1 node
#SBATCH --cpus-per-task=8            # CPUs for data loading/Parquet handling
#SBATCH --mem=64G                    # Memory needed per year (scaled down from 500G)
#SBATCH -G 1                         # 1 GPU per year/task
#SBATCH --qos=long                   # Retaining your QOS
#SBATCH --time=10-00:00:00           # Retaining your 10-day limit (though XGB is faster)

# --- ENVIRONMENT SETUP ---
# Update library paths if needed for your specific cluster setup
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Create logs directory if it doesn't exist
mkdir -p logs

# --- EXECUTION ---
# $SLURM_ARRAY_TASK_ID will automatically be the year (2001, 2002, etc.)
echo "Starting LOYO Training for Year: $SLURM_ARRAY_TASK_ID"
echo "Allocated GPU ID: $CUDA_VISIBLE_DEVICES"

# Call the Python script and pass the year as an argument
python /home/spotter5/fire_projections/train_loyo_node.py --year $SLURM_ARRAY_TASK_ID

echo "Finished Year: $SLURM_ARRAY_TASK_ID"