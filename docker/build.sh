#!/bin/bash

# Launch this from the top-level dir of repo as bash docker/build.sh

# We use the NERSC private container registry here
# should be shared across the group following registry.nersc.gov/
# Set base nvcr.io pytorch container version with NVC_TAG
# To access the registry, do: podman-hpc login registry.nersc.gov
# See https://docs.nersc.gov/development/shifter/how-to-use/#using-registrynerscgov

NVC_TAG=25.06
BASE=registry.nersc.gov/dasrepo/shas1693/weather-pytorch
IMAGE=$BASE:$NVC_TAG

# build base image
set -x
podman-hpc build --build-arg nvc_tag=$NVC_TAG-py3 -t $IMAGE -f docker/Dockerfile .

podman-hpc push $IMAGE

