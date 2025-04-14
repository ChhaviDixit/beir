#!/bin/sh

#SBATCH -J eval-sentence-transformer-checkpoints
#SBATCH --nodes=1                   # Single node
#SBATCH --gres=gpu:5                # 5 GPUs on this node
#SBATCH --ntasks-per-node=5         # One process per GPU
#SBATCH --cpus-per-task=8           # 8 CPU cores per task
#SBATCH --time=7:00:00             # Extended wall time
#SBATCH --qos=npl-48hr              # Requested QoS
#SBATCH --output=output_snowflake-arctic-embed-m-v1.5_ED-hotpotqa-lr3e-5-epochs10-temperature20_full_dev_evaluation.out
#SBATCH --error=error_snowflake-arctic-embed-m-v1.5_ED-hotpotqa-lr3e-5-epochs10-temperature20_full_dev_evaluation.out

# RPI Cluster
module load gcc/8.4.0/1
module load cuda/12.2

# Activate Conda environment
source ~/.bashrc
source ~/barn/miniconda3x86/etc/profile.d/conda.sh # RPI Cluster only
conda activate myenv39

# Set Python path for correct environment
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages:$PYTHONPATH

# Set DDP master address and port
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR

nvidia-smi

# Run checkpoint evaluation using torchrun with 5 GPUs
#torchrun --nproc_per_node=5 /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/eval_checkpoints.py
python /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/eval_checkpoints.py

nvidia-smi

