#!/bin/bash

image=registry.nersc.gov/dasrepo/shas1693/weather-pytorch:25.05-v2
dp=1
tp=1
sp1=1
sp2=1
nodes=1

# parse args
for arg in "$@"
do
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
         python -m pytest -s tests/test_all.py
     '

