#!/bin/sh

#SBATCH --account=edu
#SBATCH -J train-sentence-transformer
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gres=gpu:1                # Number of GPUs per node
#SBATCH --ntasks-per-node=1         # Number of processes per node (should be equal to the number of GPUs per node)
#SBATCH --time=80:00:00
#SBATCH --output=output_%j.log      # Standard output log file (%j is the job ID)
#SBATCH --error=error_%j.log        # Standard error log file

module load anaconda
module load cuda92/toolkit

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
source ~/.bashrc
conda activate myenv39           # Activate the virtual environment
#conda activate testenv

nvidia-smi

# Set PYTHONPATH to prioritize the virtual environment
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages:$PYTHONPATH

#export CUDA_VISIBLE_DEVICES=1  # Only use GPU 1

# Start nvidia-smi in the background to monitor GPU utilization every 10 seconds
#while true; do
 #   nvidia-smi >> gpu_utilization.log
 #   sleep 60
#done &


srun python /moto/home/ggn2104/beir/examples/retrieval/training/train_sbert4.py

nvidia-smi
# End of script

