#!/bin/sh
##SBATCH --export=ALL
#SBATCH -N1
##SBATCH --nodelist=gpu018
##SBATCH --ntasks=1
#SBATCH --qos=long
#SBATCH --time=10-0
#SBATCH --cpus-per-task=1
#SBATCH -G4
#SBATCH --mem-per-cpu=300G

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# python /home/spotter5/fire_projections/ratio_curves.py
# python /home/spotter5/fire_projections/model_inference_thresh.py
python /home/spotter5/fire_projections/ratio_curves_auc_thresh.py









