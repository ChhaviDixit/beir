#!/bin/sh

#SBATCH -J train-sentence-transformer
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gres=gpu:1                # Number of GPUs per node
#SBATCH --ntasks-per-node=1         # Number of processes per node (should be equal to the number of GPUs per node)
#SBATCH --time=6:00:00
#SBATCH --qos=npl-48hr             # Requested QoS IMPORTANT: REPLACE WITH SBATCH --account=edu if using Terremoto cluster
#SBATCH --output=output_snowflake-arctic-embed-m-v1.5_ED-hotpotqa-lr1e-5-epochs10-temperature20_full_dev_2.out      # Standard output log file
#SBATCH --error=error_snowflake-arctic-embed-m-v1.5_ED-hotpotqa-lr1e-5-epochs10-temperature20_full_dev_2.out        # Standard error log file

# Terremoto Cluster
#module load anaconda
#module load cuda92/toolkit

# RPI Cluster
module load gcc/8.4.0/1
module load cuda/12.1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
source ~/.bashrc
source ~/barn/miniconda3x86/etc/profile.d/conda.sh  # Adjust path if needed (RPI Cluster only)

conda activate myenv39           # Activate the virtual environment for ED model training
#conda activate testenv          # Activate the virtual environment for CosSim model training

nvidia-smi

# Set PYTHONPATH to prioritize the virtual environment
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0  # Only use GPU 0

# Start nvidia-smi in the background to monitor GPU utilization every 10 seconds
#while true; do
 #   nvidia-smi >> gpu_utilization.log
 #   sleep 60
#done &


#srun python /moto/home/ggn2104/beir/examples/retrieval/training/train_sbert4.py
#srun python /moto/home/ggn2104/beir/examples/retrieval/training/train_sbert_latest.py
srun python /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/train_sbert_latest_2.py
nvidia-smi
# End of script

