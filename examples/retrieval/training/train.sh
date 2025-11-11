#!/bin/sh

#SBATCH -J train-sentence-transformer
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gres=gpu:1                # Number of GPUs per node
#SBATCH --ntasks-per-node=1         # Number of processes per node (should be equal to the number of GPUs per node)
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=16gb	# The memory the job will use per cpu core.
#SBATCH --qos=npl-48hr             # Requested QoS IMPORTANT: REPLACE WITH SBATCH --account=edu if using Terremoto cluster
#SBATCH --output=output_distilbert-base-uncased-dbpedia-entity-lr2e-5-epochs10-temperature20_full_dev_1GPU.out      # Standard output log file
#SBATCH --error=error_distilbert-base-uncased-dbpedia-entity-lr2e-5-epochs10-temperature20_full_dev_1GPU.out        # Standard error log file

# Terremoto Cluster
#module load anaconda
#module load cuda92/toolkit

module purge

# RPI Cluster
module load gcc/14.1.0
module load cuda

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
source ~/.bashrc
# source ~/barn/miniconda3x86/etc/profile.d/conda.sh  # Adjust path if needed (RPI Cluster only)
source /insomnia001/depts/edu/COMSE6998/cd3496/miniconda3/etc/profile.d/conda.sh


conda activate myenvED           # Activate the virtual environment for ED model training
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


#srun python /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/train_sbert4.py
#srun python /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/train_sbert_latest.py
# srun python /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/train_sbert_latest_2.py
srun python /insomnia001/home/cd3496/beir/examples/retrieval/training/train_sbert_latest_new.py
nvidia-smi
# End of script

