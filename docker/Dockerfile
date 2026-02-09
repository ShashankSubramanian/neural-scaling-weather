# PyTorch:
#     use nvcr.io/nvidia/pytorch:$nvc_tag for base pytorch NGC
# DGL + PyTorch:
#     use nvcr.io/nvidia/dgl:$nvc_tag (which inherits from base pytorch NGC)
#     see https://docs.nvidia.com/deeplearning/frameworks/pdf/DGL-Release-Notes.pdf        
ARG nvc_tag
FROM nvcr.io/nvidia/pytorch:$nvc_tag

# System packages ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-dev \
        cdo \
        nco \
        netcdf-bin \
        build-essential \
        wget \
        mpich \
        libmpich-dev \
    && rm -rf /var/lib/apt/lists/*

# Cartopy (needs libgeos)
RUN python3 -m pip install --no-cache-dir --upgrade pip cartopy

# Parallel HDF5 (1.12.3) -----------------------------------------------------
RUN wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_12_3.tar.gz && \
    tar -xvf hdf5-1_12_3.tar.gz && \
    cd hdf5-hdf5-1_12_3 && \
    ./configure --enable-parallel --enable-shared --prefix=/opt/hdf5 && \
    make -j$(nproc) && make install && \
    cd / && rm -rf hdf5-hdf5-1_12_3 hdf5-1_12_3.tar.gz

# h5py built against MPI-enabled HDF5
RUN HDF5_MPI=ON CC=mpicc HDF5_DIR=/opt/hdf5 \
    pip install --no-deps h5py

# Python packages -------------------------------------------------------------
RUN pip install --no-cache-dir \
    timm \
    torchinfo \
    tensorly tensorly-torch \
    scikit-image \
    xarray zarr netCDF4 pyproj pygrib cdsapi \
    properscoring \
    ruamel.yaml \
    dask \
    wandb \
    black ruff typer \
    parameterized \
    hydra-core \
    gcsfs \
    seaborn \
    --upgrade

RUN cd /opt && \
    git clone https://github.com/NVIDIA/torch-harmonics.git && \
    cd torch-harmonics && \
    pip install --no-build-isolation -e .

RUN echo "Finished torch harmonics from source"

# Clean up
RUN echo "Docker build finished successfully."

