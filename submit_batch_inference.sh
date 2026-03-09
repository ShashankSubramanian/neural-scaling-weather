#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH -C gpu
#SBATCH --account=m4790
#SBATCH -q regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J era5_expt
#SBATCH --image=registry.nersc.gov/dasrepo/shas1693/weather-pytorch:25.05-v2
#SBATCH --module=gpu,nccl-plugin
#SBATCH -o %x_%j.out

set -x

umask 002 # so proper permissions are set
export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WANDB_START_METHOD="thread"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Attach data and output directories as voulme mounts
DATAROOT="/pscratch/sd/s/shas1693/data/weather/era5"
REGISTRY="/pscratch/sd/s/shas1693/results/neuralscaling"
OUTPUT="/pscratch/sd/s/shas1693/results/neuralscaling/inference"
mkdir -p $OUTPUT

cmd="python inference.py $@"

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

set -x
srun -l shifter -V "$DATAROOT:/data;$OUTPUT:/expts;$REGISTRY:/registry" \
    bash -c "
    $cmd
    "
