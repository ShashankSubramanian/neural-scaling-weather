#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J <your_job_name>
#SBATCH -o %x_%j.out

# add your other #SBATCH flags as well

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WANDB_START_METHOD="thread"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Attach data and output directories as voulme mounts
DATAROOT="/path/to/data"
REGISTRY="/path/to/where/you/want/to/store/checkpoints"
OUTPUT="/path/to/where/you/want/to/store/results"
mkdir -p $OUTPUT

cmd="python inference.py $@"

set -x

# use your own run command that will mount the directories
srun -l <your_run_command_that_will_mount_dirs> 
    -V "$DATAROOT:/data;$OUTPUT:/expts;$REGISTRY:/registry" \
    bash -c "
    $cmd
    "

# or direct run with /data, /expts, /registry for the default paths
# srun -l bash -c "$cmd"
