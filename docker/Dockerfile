# PyTorch:
#     use nvcr.io/nvidia/pytorch:$nvc_tag for base pytorch NGC
# DGL + PyTorch:
#     use nvcr.io/nvidia/dgl:$nvc_tag (which inherits from base pytorch NGC)
#     see https://docs.nvidia.com/deeplearning/frameworks/pdf/DGL-Release-Notes.pdf        
ARG nvc_tag
FROM nvcr.io/nvidia/pytorch:$nvc_tag

# Python packaging -----------------------------------------------------------
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Python packages -------------------------------------------------------------
RUN pip install --no-cache-dir \
    torchinfo \
    xarray zarr netCDF4 \
    h5py \
    wandb \
    black ruff typer \
    parameterized pytest \
    hydra-core \
    gcsfs \
    seaborn \
    fvcore \
    mpi4py

# torch-harmonics from source ------------------------------------------------
RUN cd /opt && \
    git clone https://github.com/NVIDIA/torch-harmonics.git && \
    cd torch-harmonics && \
    pip install --no-build-isolation -e .


# Clean up
RUN echo "Docker build finished successfully."

