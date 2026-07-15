#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH -p gh
#SBATCH -A CDA24017
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -J era5_expt
#SBATCH -o %x_%j.out

set -x

umask 002 # so proper permissions are set
module load tacc-apptainer

IMAGE="scigpt_25.06.sif"
DATAROOT="/scratch/10580/ssub/data"
REGISTRY="/scratch/10580/ssub/results/neuralscaling"
OUTPUT="/scratch/10580/ssub/results/neuralscaling"
mkdir -p $OUTPUT

echo "Enabling profiling..."
NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
PROFILE_DIR="$OUTPUT/profiles"
mkdir -p "$PROFILE_DIR"
export PROFILE_CMD="nsys profile $NSYS_ARGS -o $PROFILE_DIR/profile_%h_%p"

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WANDB_START_METHOD="thread"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=ERROR

cmd=$(cat <<EOF
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PYTHONNOUSERSITE=1

if [ -d /usr/local/cuda/compat/lib.real ]; then
  export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
else
  export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib
fi

$PROFILE_CMD python train.py "\$@"
EOF
)

srun -l apptainer exec --nv \
    --bind "$DATAROOT:/data,$OUTPUT:/expts,$REGISTRY:/registry" "$IMAGE" \
    bash -c "$cmd" bash "$@"
