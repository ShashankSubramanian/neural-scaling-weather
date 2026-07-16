#!/bin/bash
# Usage: sbatch submit_tests_vista.sh [test_target] [tp=1] [sp1=1] [sp2=1] [dp=1]
# Example: sbatch submit_tests_vista.sh test_te tp=2 sp1=2
#
# Set --nodes below (or override it on the sbatch command line). Vista has one
# GH200 per node, so --nodes must equal tp * sp1 * sp2 * dp.
#SBATCH --time=00:30:00
#SBATCH -p gh
#SBATCH -A CDA24017
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -J tests
#SBATCH -o %x_%j.out

set -euo pipefail
set -x

umask 002
module load tacc-apptainer

IMAGE="/scratch/10580/ssub/containers/scigpt_25.06.sif"
DATAROOT="/scratch/10580/ssub/data"

dp=1
tp=1
sp1=1
sp2=1
nodes="${SLURM_JOB_NUM_NODES:-1}"
test_path="tests/test_all.py"

set_test_from_arg() {
    local raw="$1"
    raw="${raw%.py}"
    if [[ $raw == tests/* ]]; then
        test_path="${raw}.py"
    else
        test_path="tests/${raw}.py"
    fi
}

for arg in "$@"; do
    if [[ $arg == tp=* ]]; then
        tp="${arg#*=}"
    elif [[ $arg == sp1=* ]]; then
        sp1="${arg#*=}"
    elif [[ $arg == sp2=* ]]; then
        sp2="${arg#*=}"
    elif [[ $arg == dp=* ]]; then
        dp="${arg#*=}"
    elif [[ $arg == test=* ]]; then
        set_test_from_arg "${arg#*=}"
    elif [[ $arg != *"="* ]] && [[ -n $arg ]]; then
        set_test_from_arg "$arg"
    fi
done

if (( tp * sp1 * sp2 * dp != nodes )); then
    echo "tp * sp1 * sp2 * dp must equal nodes ($nodes)." >&2
    exit 1
fi

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR="${SLURMD_NODENAME:-$(hostname)}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=ERROR

cmd=$(cat <<EOF
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PYTHONNOUSERSITE=1
export TP=$tp
export SP1=$sp1
export SP2=$sp2
export NVIDIA_TF32_OVERRIDE=0

if [ -d /usr/local/cuda/compat/lib.real ]; then
  export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
else
  export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib
fi

rank="\${SLURM_PROCID:-\${RANK:-\${PMI_RANK:-0}}}"
extra_pytest=""
if [ "\$rank" != "0" ]; then
  extra_pytest="--no-header --no-summary -qq --disable-warnings"
fi

python -m pytest -s \$extra_pytest "$test_path"
EOF
)

srun -l apptainer exec --nv \
    --bind "$DATAROOT:/data" "$IMAGE" \
    bash -c "$cmd"
