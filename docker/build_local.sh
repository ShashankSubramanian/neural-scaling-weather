#!/bin/bash

NVC_TAG=25.06-arm64
IMAGE=registry.gitlab.com/nersc/nesap/scigpt:$NVC_TAG

set -x
docker buildx build --platform linux/arm64 --build-arg nvc_tag=$NVC_TAG-py3 -t $IMAGE -f docker/Dockerfile_local --provenance=false --push .
