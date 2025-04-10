#!/bin/sh

#SBATCH -J train-sentence-transformer
#SBATCH --nodes=1                   # Single node
#SBATCH --gres=gpu:2                # 8 GPUs on this node
#SBATCH --ntasks-per-node=2         # One process per GPU
#SBATCH --cpus-per-task=8           # 8 CPU cores per task
#SBATCH --time=48:00:00            # Extended wall time
#SBATCH --qos=npl-48hr             # Requested QoS IMPORTANT: REPLACE WITH SBATCH --account=edu if using Terremoto cluster
#SBATCH --output=output_snowflake-arctic-embed-m-v1.5_ED-hotpotqa-lr1e-5-epochs10-temperature20_full_dev_3.out	  # Standard output log file (make sure correct LR and scale are set)
#SBATCH --error=error_snowflake-arctic-embed-m-v1.5_ED-hotpotqa-lr1e-5-epochs10-temperature20_full_dev_3.out        # Standard error log file (make sure correct LR and scale are set)

#RPI Cluster
module load gcc/8.4.0/1
module load cuda/12.2

# Terremoto Cluster
#module load anaconda
#module load cuda92/toolkit


# Activate Conda environment
source ~/.bashrc
source ~/barn/miniconda3x86/etc/profile.d/conda.sh #RPI Cluster only
conda activate myenv39

# Set Python path for correct environment
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=1,2,3,4

# export your rank 0 information (its address and port)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR


nvidia-smi
# Run training with torchrun for DDP
torchrun --nproc_per_node=2 /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/train_sbert_ddp_2.py
#torchrun --nproc_per_node=4 /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/test_ddp.py
