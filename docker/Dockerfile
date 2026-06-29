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
    "numpy>=1.22,<1.27" \
    "packaging<=24.2" \
    torchinfo \
    xarray zarr netCDF4 \
    wandb \
    black ruff typer \
    parameterized \
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
     
# Parallel HDF5 (1.12.3) -----------------------------------------------------
RUN wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_12_3.tar.gz && \
    tar -xzf hdf5-1_12_3.tar.gz && \
    cd hdf5-hdf5-1_12_3 && \
    ./configure --enable-parallel --enable-shared --prefix=/opt/hdf5 && \
    make -j$(nproc) && make install && \
    cd / && rm -rf hdf5-hdf5-1_12_3 hdf5-1_12_3.tar.gz

# h5py built against MPI-enabled HDF5
RUN HDF5_MPI=ON CC=mpicc HDF5_DIR=/opt/hdf5 \
    pip install --no-cache-dir --no-deps --no-binary=h5py --no-build-isolation h5py

# Clean up
RUN echo "Docker build finished successfully."

