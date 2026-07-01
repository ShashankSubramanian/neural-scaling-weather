#!/bin/bash
# Usage: bash tests/run_tests.sh [test_target] [tp=1] [sp1=1] ...
#   test_target: base name (test_all, test_te) or path (tests/test_te, tests/test_te.py);
#                default tests/test_all.py. Also: test=test_te

image=registry.nersc.gov/dasrepo/shas1693/weather-pytorch:25.05-v2
dp=1
tp=1
sp1=1
sp2=1
nodes=1
test_path=tests/test_all.py

set_test_from_arg() {
    local raw="$1"
    raw="${raw%.py}"
    if [[ $raw == tests/* ]]; then
        test_path="${raw}.py"
    else
        test_path="tests/${raw}.py"
    fi
}

# parse args
for arg in "$@"; do
    if [[ $arg == tp=* ]]; then
        tp="${arg#*=}"
    elif [[ $arg == sp1=* ]]; then
        sp1="${arg#*=}"
    elif [[ $arg == sp2=* ]]; then
        sp2="${arg#*=}"
    elif [[ $arg == dp=* ]]; then
        dp="${arg#*=}"
    elif [[ $arg == nodes=* ]]; then
        nodes="${arg#*=}"
    elif [[ $arg == test=* ]]; then
        set_test_from_arg "${arg#*=}"
    elif [[ $arg != *"="* ]] && [[ -n $arg ]]; then
        set_test_from_arg "$arg"
    fi
done

# to test dataloader
DATAROOT="/pscratch/sd/s/shas1693/data/weather/era5"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

ngpu_per_node=$(( (${tp} * ${sp1} * ${sp2} * ${dp})/$nodes ))
export MASTER_ADDR=$(hostname)
srun --nodes $nodes \
     --ntasks-per-node $ngpu_per_node \
     --gpus-per-node $ngpu_per_node \
     -u \
     shifter --image=$image \
             --module=gpu,nccl-plugin \
             -V $DATAROOT:/data \
     bash -c '
         export TP='"$tp"'
         export SP1='"$sp1"'
         export SP2='"$sp2"'
         export NVIDIA_TF32_OVERRIDE=0
         rank="${SLURM_PROCID:-${RANK:-${PMI_RANK:-0}}}"
         extra_pytest=""
         if [ "$rank" != "0" ]; then
             extra_pytest="--no-header --no-summary -qq --disable-warnings"
         fi
         python -m pytest -s $extra_pytest '"$test_path"'
     '
