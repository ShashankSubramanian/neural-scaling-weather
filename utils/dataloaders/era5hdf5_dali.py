from __future__ import annotations

import os
import logging
import glob
import numpy as np
import h5py
from datetime import datetime
from typing import List, Optional, Dict
import torch

# DALI imports
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
 
# distributed helpers
from distributed.helpers import compute_split_shapes_for_patching
from utils import comm

# helpers
from utils.misc_utils import (
    build_spherical_area_weights, 
    get_local_slice, 
    get_spatial_coords,
)


class ERA5HDF5DALIDataLoader(object):
    def __init__(
        self,
        dataset: Era5HDF5DatasetDALI,
        micro_batch_size: int = 1,
        num_data_workers: int = 1,
        seed: int = 333,
        mode: str = "train",
    ):
        """DALI data loader for ERA5 HDF5 temporal data.

        Args:
            dataset: Era5HDF5DatasetDALI dataset object
            micro_batch_size (int, optional): Micro batch size for training. Defaults to 1.
            num_data_workers (int, optional): Number of data loading worker processes. Defaults to 1.
            seed (int, optional): Random seed for reproducibility. Defaults to 333.
            mode (str, optional): Dataset mode - 'train', 'valid' or 'test'. Defaults to "train".
        """
        self.mode = mode
        self.global_seed = seed
        self.num_data_workers = num_data_workers
        self.device_index = torch.cuda.current_device()
        self.micro_batch_size = micro_batch_size
        self.dataset = dataset
        self.return_timestamp = getattr(dataset, "return_timestamp", False)

        # create pipeline
        self.pipeline = self.get_pipeline()
        self.pipeline.start_py_workers()
        self.pipeline.build()

        # create iterator
        output_map = ["data", "time"] if self.return_timestamp else ["data"]
        self.iterator = DALIGenericIterator(
            [self.pipeline],
            output_map=output_map,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP,
            prepare_first_batch=True,
        )

    def get_pipeline(self):
        """Create DALI pipeline for ERA5 data loading.

        The pipeline performs the following operations:
        1. Loads temporal ERA5 data using external source
        2. Transfers data to GPU
        3. Normalizes data using precomputed means and standard deviations

        Returns:
            Pipeline: DALI pipeline object configured for ERA5 data loading
        """
        pipeline = Pipeline(
            batch_size=self.micro_batch_size,
            num_threads=2,
            device_id=self.device_index,
            py_num_workers=self.num_data_workers,
            py_start_method="spawn",
            seed=self.global_seed,
        )

        with pipeline:
            num_outputs = 2 if self.return_timestamp else 1
            layout = ["FCHW", ""] if self.return_timestamp else "FCHW"
            outs = fn.external_source(
                source=self.dataset,
                num_outputs=num_outputs,
                layout=layout,
                batch=False,  # sample mode, pipeline will batch internally
                parallel=True,
                prefetch_queue_depth=self.num_data_workers,
            )

            # upload to GPU
            data = outs[0].gpu()

            # normalize data
            data = fn.normalize(
                data,
                device="gpu",
                axis_names="FHW",  # normalize over spatiotemporal
                batch=False,
                mean=self.dataset.means,
                stddev=self.dataset.stds,
            )

            if self.return_timestamp:
                pipeline.set_outputs(data, outs[1])
            else:
                pipeline.set_outputs(data)
        return pipeline

    def __len__(self):
        return self.dataset.num_steps_per_epoch

    def reset_pipeline(self):
        self.pipeline.reset()
        self.iterator.reset()

    def __iter__(self):
        for token in self.iterator:
            data = token[0]["data"]
            if self.return_timestamp:
                yield data, token[0]["time"]
            else:
                yield data


class Era5HDF5DatasetDALI:
    """
    DALI-based HDF5 dataset object serving ERA5 samples with temporal windows
    Incorporates double buffering functionality for improved performance
    """

    @classmethod
    def instantiate_from_cfg(cls, cfg, seed, mode):
        """Class method to instantiate the dataset in a given mode (train/valid/test) from a given hydra config"""
        return cls(
            name=cfg.data.name,
            tag=cfg.data.tag,
            mode=mode,
            dt_scale=cfg.data.dt_scale,
            temporal_context_window=cfg.train.temporal_context_window,
            num_rollout_steps=cfg.train.num_rollout_steps,
            model_patch_size=cfg.model.patch_size,
            limit_nsamples=cfg.data.limit_nsamples,
            valid_years=cfg.data.valid_years,
            test_years=cfg.data.test_years,
            limit_validation=cfg.data.limit_nsamples_val,
            invariants=cfg.data.invariants,
            micro_batch_size=cfg.parallelism.micro_batch_size,
            read_local=cfg.data.read_local,
            seed=seed,
        )

    def __init__(
        self,
        name: str,
        tag: str,
        mode: str,
        dt_scale: int = 1,
        temporal_context_window: int = 1,
        num_rollout_steps: int = 1,
        model_patch_size: int = 1,
        limit_nsamples: int = -1,
        valid_years: List = [None],
        test_years: List = [None],
        limit_validation: int = -1,
        invariants: Optional[List] = None,
        micro_batch_size: int = 1,
        read_local: bool = False,
        seed: int = 333,
        return_timestamp: bool = False,
    ):
        """
        Inputs:
            name: str
                dataset name under data directory tree
            tag: str
                tag for statistics and train/test split
            mode: str
                'train', 'valid', or 'test'
            dt_scale: int = 1
                dataset's dt * dt_scale is the model timestep
            temporal_context_window: int
                number of time steps to include in each sample
            num_rollout_steps: int
                number of rollout steps to include in each sample
            model_patch_size: int
                size of the model patch
            limit_nsamples: int
                limit total number of samples to a specific number; ignore if -1
            valid_years: List = [None]
                list of years to use for validation
            test_years: List = [None]
                list of years to use for testing
            limit_validation: int = -1
                limit total number of validation samples to a specific number; ignore if -1
            invariants: Optional[List] = None
                whether or not to include invariants like land-sea mask, orography
            micro_batch_size: int
                micro batch size for DALI pipeline
            read_local: bool = False
                whether to read only the local spatial data or global data
            seed: int
                random seed for shuffling (must be constant across workers)
        """
        self.location = "/data/" + name
        self.tag = tag
        self.mode = mode
        self.dt_scale = dt_scale
        self.temporal_context_window = temporal_context_window
        self.num_rollout_steps = num_rollout_steps
        self.model_patch_size = model_patch_size
        self.limit_nsamples = limit_validation if mode == "valid" else limit_nsamples
        self.valid_years = valid_years
        self.test_years = test_years
        self.invariants = invariants if invariants is not None else []
        self.micro_batch_size = micro_batch_size
        # self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.shuffle = True if mode == "train" else False
        self.read_local = read_local
        # If True, __call__ additionally emits the raw HDF5 time stamps for the
        # F-step window of each sample. Used for unit tests / debugging.
        self.return_timestamp = return_timestamp

        self._get_files_stats()
        self.means, self.stds = self._load_stats()

        # Per-worker sample buffer. Allocated lazily on first __call__ so it
        # lives in worker memory (not pickled across the spawn boundary).
        self.inp_buf = None
        self.time_buf = None

        if len(self.invariants) > 0:
            self.invar = self.get_invariants()
        else:
            self.invar = None

    def _load_stats(self):
        """Load dataset statistics from the appropriate stats h5 file"""
        stats_path = os.path.join(self.location, "stats", f"stats_{self.tag}.h5")
        with h5py.File(stats_path, "r") as f:
            mean = f["global_mean"][:]
            std = f["global_std"][:]

        # Expand singleton dims for spatial dims
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]

        return mean, std

    def get_metadata(self):
        """Get metadata for the dataset"""
        return {
            "sp_shapes": self.metadata["sp_shapes"], # shapes on all ranks
            "sp_slices": self.metadata["sp_slices"], # local slices
            "spatial_dims": self.metadata["spatial_dims"], # local image shapes
            "channels": self.metadata["channels"],
            "n_channels": self.metadata["n_channels"],
            "temporal_context_window": self.metadata["temporal_context_window"],
            "num_rollout_steps": self.metadata["num_rollout_steps"],
            "dt_hours": self.metadata["dt_hours"],
            "dt_scale": self.metadata["dt_scale"],
            "n_samples_total": self.n_samples_total,
            "n_samples_shard": self.n_samples_shard,
            "num_steps_per_epoch": self.num_steps_per_epoch,
            "read_local": self.read_local,
        }

    def _set_shapes_and_metadata(self):
        """ 
        Compute stats for sharding data and other metadata
        """
        # build area weights from global data
        area_weights = build_spherical_area_weights(self.latitudes)
        # get local slices for lat and lon
        sp1_shapes = compute_split_shapes_for_patching(len(self.latitudes), comm.get_size("sp1"), self.model_patch_size)
        sp2_shapes = compute_split_shapes_for_patching(len(self.longitudes), comm.get_size("sp2"), self.model_patch_size)
        lat_slice = get_local_slice(sp1_shapes, comm.get_rank("sp1"))
        lon_slice = get_local_slice(sp2_shapes, comm.get_rank("sp2"))
        latitudes_local = self.latitudes[lat_slice]
        longitudes_local = self.longitudes[lon_slice]
        # build metadata for model and loss
        self.metadata = {
            "sp_shapes": [sp1_shapes, sp2_shapes], # shapes on all ranks
            "sp_slices": [lat_slice, lon_slice], # local slices
            "spatial_dims": [len(latitudes_local), len(longitudes_local)], # local image shapes
            "channels": self.era5_channels,
            "n_channels": len(self.era5_channels),
            "coords": get_spatial_coords(locations=(latitudes_local, longitudes_local)), 
            "area_weights": area_weights[lat_slice, None], # [nlat, 1]
            "temporal_context_window": self.temporal_context_window,
            "num_rollout_steps": self.num_rollout_steps,
            "dt_hours": self.dt_hours,
            "dt_scale": self.dt_scale,
            "global_latitudes": self.latitudes,
            "global_longitudes": self.longitudes,
        }


    def _get_files_stats(self):
        """Scan directories and extract metadata for ERA5"""
        if not os.path.exists(self.location):
            raise FileNotFoundError(f"Data directory not found: {self.location}")

        self.era5_paths = glob.glob(os.path.join(self.location, "*.h5"))
        if not self.era5_paths:
            raise FileNotFoundError(f"No HDF5 files found in {self.location}")

        self.era5_paths = sorted(
            self.era5_paths, key=lambda x: int(os.path.basename(x).replace(".h5", ""))
        )

        holdout_yrs = self.valid_years + self.test_years
        if self.mode == "train":
            self.era5_paths = [
                x
                for x in self.era5_paths
                if int(os.path.basename(x).replace(".h5", "")) not in holdout_yrs
            ]
            self.years = [
                int(os.path.basename(x).replace(".h5", "")) for x in self.era5_paths
            ]
        elif self.mode == "valid":
            self.era5_paths = [
                x
                for x in self.era5_paths
                if int(os.path.basename(x).replace(".h5", "")) in self.valid_years
            ]
            self.years = [
                int(os.path.basename(x).replace(".h5", "")) for x in self.era5_paths
            ]
        else:
            self.era5_paths = [
                x
                for x in self.era5_paths
                if int(os.path.basename(x).replace(".h5", "")) in self.test_years
            ]
            self.years = [
                int(os.path.basename(x).replace(".h5", "")) for x in self.era5_paths
            ]

        self.n_years = len(self.era5_paths)

        # metadata from first file
        with h5py.File(self.era5_paths[0], "r") as f:
            self.era5_channels = [n.decode("utf-8") for n in f["channel"][:]]
            times = f["time"][:]
            data_dt = times[1] - times[0]
            self.dt_hours = np.timedelta64(int(data_dt), "h")
            self.latitudes = f["latitude"][:-1] # kill the southpole
            self.longitudes = f["longitude"][:]
        
        # Store file paths and initialize file handles as None for lazy loading
        # this is needed since DALI cannot pickle the h5py file objects
        self.ds_era5 = [None] * len(self.era5_paths)

        # Compute total samples
        self.n_samples_total = self.compute_total_samples()

        if self.limit_nsamples > 0:
            self.n_samples_total = min(self.n_samples_total, self.limit_nsamples)

        # Shard the data
        self.n_samples_shard = self.n_samples_total // comm.get_size("dp")
        self.shard_id = comm.get_rank("dp")

        # Number of steps per epoch
        # equivalent to n_samples_total // global_batch_size
        self.num_steps_per_epoch = self.n_samples_shard // self.micro_batch_size
        self.last_epoch = None
        self.index_permutation = None

        # for dataset checkpointing
        self.resume_skip_batches = 0
        self.iterations_to_skip = 0
        self.ckpt_epoch = 0

        # set shapes and metadata for splitting data in spatial dims and computing metadata
        self._set_shapes_and_metadata()

        # fake data; for profiling
        self.fake_data = False

    def set_dataset_state(self, checkpoint):
        """Set dataset state from checkpoint"""
        self.resume_skip_batches = checkpoint["n_mbs_in_epoch"]
        self.ckpt_epoch = checkpoint["epoch"]

    def compute_total_samples(self):
        """Compute total number of possible samples given temporal window.

        Stores ``self.all_times`` as a datetime64[h] concatenation across
        training years. Year offsets let __call__ map a global_idx ->
        (year_idx, local_idx) cheaply.
        """
        per_year_times = []
        for path in self.era5_paths:
            with h5py.File(path, "r") as ds:
                d = ds["time"][:]
                per_year_times.append(np.array(
                    [np.datetime64("1900-01-01") + np.timedelta64(int(ts), "h") for ts in d]
                ))

        self.n_samples_per_year = np.array([len(t) for t in per_year_times], dtype=np.int64)
        self.year_offsets = np.concatenate(([0], np.cumsum(self.n_samples_per_year)[:-1]))
        self.all_times = np.concatenate(per_year_times)

        return len(self.all_times) - (self.temporal_context_window + self.num_rollout_steps - 1) * self.dt_scale

    def __len__(self):
        return self.n_samples_shard

    def __del__(self):
        self.close()

    def close(self):
        """Explicitly close all file handles"""
        if hasattr(self, "ds_era5"):
            for f in self.ds_era5:
                if f is not None:
                    try:
                        f.close()
                    except:
                        pass  # File might already be closed
            self.ds_era5 = [None] * len(self.era5_paths)

    def _get_file_handle(self, year_idx):
        """Get file handle for a specific year, opening it if necessary"""
        if self.ds_era5[year_idx] is None:
            self.ds_era5[year_idx] = h5py.File(self.era5_paths[year_idx], "r")
        return self.ds_era5[year_idx]

    def _get_era5(self, timestamps):
        """Load ERA5 data for given timestamps"""
        # Group timestamps by year
        year_groups = {}
        for ts in timestamps:
            # convert numpy.datetime64 to datetime
            ts_dt = datetime.utcfromtimestamp(
                ts.astype("datetime64[s]").astype("int64")
            )
            year = ts_dt.year
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(ts)

        # Process each year's timestamps in batch
        data = []
        for year, year_timestamps in year_groups.items():
            year_idx = self.years.index(year)
            
            # Get file handle (lazy load if necessary)
            ds_handle = self._get_file_handle(year_idx)
            
            # Convert all timestamps for this year to hours since 1900-01-01
            hours_since_1900 = np.array(
                [
                    int((ts - np.datetime64("1900-01-01")) / np.timedelta64(1, "h"))
                    for ts in year_timestamps
                ]
            )
            time_indices = [
                np.where(ds_handle["time"][:] == h)[0][0] for h in hours_since_1900
            ]
            if self.read_local:
                # read only the local spatial data (incase it's too big)
                lat_slice = self.metadata["sp_slices"][0]
                lon_slice = self.metadata["sp_slices"][1]
                data.append(ds_handle["data"][time_indices, :, lat_slice, lon_slice])
            else:
                data.append(ds_handle["data"][time_indices, :, :, :])

        return data

    def _get_local_year_index(self, global_idx):
        """Map a global sample index to (local_idx_within_year, year_idx)."""
        year_idx = int(np.searchsorted(self.year_offsets, global_idx, side="right") - 1)
        local_idx = int(global_idx - self.year_offsets[year_idx])
        return local_idx, year_idx

    def _read_window_into(self, global_idx, buf):
        """Read tcw+nrs temporal steps starting at global_idx into buf.

        Uses h5py.read_direct with a strided slice per year so each step is read
        straight into the destination buffer with no intermediate allocation.
        If read_local is True, the spatial (sp1, sp2) slice is applied at read
        time (fewer bytes; best when slicing along H with sp2=1, since each
        per-(t,c) slab stays contiguous in the file). If read_local is False,
        the full spatial slab is read first and sliced in memory (better when
        sp2>1 makes the per-row reads strided and HDF5 would otherwise issue
        many small IOs).
        """
        n_steps = self.temporal_context_window + self.num_rollout_steps
        lat_slice = self.metadata["sp_slices"][0]
        lon_slice = self.metadata["sp_slices"][1]

        i = 0
        while i < n_steps:
            local_idx, year_idx = self._get_local_year_index(global_idx + i * self.dt_scale)
            n_year = int(self.n_samples_per_year[year_idx])
            # max number of additional steps that still fit inside this year file
            max_j = (n_year - 1 - local_idx) // self.dt_scale
            j_end = min(n_steps, i + max_j + 1)
            n_take = j_end - i

            t_sel = slice(local_idx, local_idx + n_take * self.dt_scale, self.dt_scale)
            ds_handle = self._get_file_handle(year_idx)

            if self.read_local:
                ds_handle["data"].read_direct(
                    buf,
                    np.s_[t_sel, :, lat_slice, lon_slice],
                    np.s_[i:j_end, ...],
                )
            else:
                # read the full spatial slab, then slice in memory
                arr = ds_handle["data"][t_sel, :, :, :]
                buf[i:j_end] = arr[..., lat_slice, lon_slice]

            i = j_end

    def __call__(self, sample_info):
        """DALI callable interface"""
        # Check if we need to shuffle again
        if sample_info.epoch_idx != self.last_epoch:
            if self.resume_skip_batches > 0:
                # we are starting from a dataloader checkpoint
                self.iterations_to_skip = self.resume_skip_batches
                # ensure skipping only happens once
                self.resume_skip_batches = 0  
            else:
                self.iterations_to_skip = 0
            if self.shuffle:
                # self.index_permutation = self.rng.permutation(self.n_samples_total)
                rng = np.random.default_rng(seed=(self.seed + self.ckpt_epoch + sample_info.epoch_idx))
                self.index_permutation = rng.permutation(self.n_samples_total)
            else:
                self.index_permutation = np.arange(self.n_samples_total)

            # shard the data
            start = self.n_samples_shard * self.shard_id
            end = start + self.n_samples_shard
            self.index_permutation = self.index_permutation[start:end]
            # update the last epoch
            self.last_epoch = sample_info.epoch_idx

        # Check if epoch is done
        if sample_info.iteration >= self.num_steps_per_epoch - self.iterations_to_skip:
            raise StopIteration

        sample_idx = (
            sample_info.idx_in_epoch
            + self.iterations_to_skip * self.micro_batch_size
        )
        global_idx = self.index_permutation[sample_idx]

        if self.inp_buf is None:
            self.inp_buf = np.empty(
                (
                    self.temporal_context_window + self.num_rollout_steps,
                    self.metadata["n_channels"],
                    self.metadata["spatial_dims"][0],
                    self.metadata["spatial_dims"][1],
                ),
                dtype=np.float32,
            )

        if self.fake_data:
            self.inp_buf = 0.001 * np.random.randn(
                self.temporal_context_window + self.num_rollout_steps,
                self.metadata["n_channels"],
                self.metadata["spatial_dims"][0],
                self.metadata["spatial_dims"][1],
            ).astype(np.float32)
        else:
            # read temporal window straight into inp_buf via h5py.read_direct
            self._read_window_into(global_idx, self.inp_buf)

        # careful: return a copy so each call has a fresh memory buffer
        # else, the pattern is unsafe, the buffer is overwritten 
        # on the next __call__ before DALI has a chance to consume it
        if self.return_timestamp:
            n_steps = self.temporal_context_window + self.num_rollout_steps
            if self.time_buf is None:
                self.time_buf = np.empty(n_steps, dtype=np.int64)
            t_idx = global_idx + np.arange(n_steps, dtype=np.int64) * self.dt_scale
            # DALI cannot carry datetime64 dtype; view as int64.
            self.time_buf[:] = self.all_times[t_idx].astype(np.int64)
            return [self.inp_buf.copy(), self.time_buf.copy()]
        return [self.inp_buf.copy()]

    def _normalize_era5(self, img):
        img -= self.means
        img /= self.stds
        return torch.as_tensor(img)

    def unnormalize_era5(self, img):
        img = img * self.stds
        img = img + self.means
        return img

    def get_samples(self, timestamps):
        """Get samples for given timestamps"""
        data = []
        for sample in self._get_era5(timestamps):
            data.append(self._normalize_era5(sample))
        lat_slice = self.metadata["sp_slices"][0]
        lon_slice = self.metadata["sp_slices"][1]
        data = torch.cat(data, dim=0)
        if not self.read_local:
            # split after the read
            data = data[..., lat_slice, lon_slice]
        return data

    def get_invariants(self):
        """
        Return the invariant fields (to be called once, before train/val loops, so they can be re-used instead of loaded each step)
        """
        if len(self.invariants) == 0:
            return None

        invar_to_use = list(filter(lambda x: x != "coslat", self.invariants))
        with h5py.File(os.path.join(self.location, "invar", "invariants.h5"), "r") as f:
            channel_names = [n.decode("utf-8") for n in f["channel"][:]]
            # Get indices of channels we want to use
            channel_indices = [
                i for i, nm in enumerate(channel_names) if nm in invar_to_use
            ]
            # Select only the channels we want
            invariant_array = f["data"][channel_indices, :-1, :]  # [channels, lat, lon]
            # Update channel names to only include selected ones
            channel_names = [channel_names[i] for i in channel_indices]

            if "orog" in invar_to_use:
                # normalize orog
                orog_idx = channel_names.index("orog")
                orog = invariant_array[orog_idx]
                low, high = orog.min(), orog.max()

                invariant_array[orog_idx] = (orog - low) / high

            if "coslat" in self.invariants:
                lats = np.cos(
                    np.deg2rad(self.latitudes)
                )  # proportional to area of cell
                lats = np.ones((1, 1, len(self.longitudes))) * lats[None, :, None]
                invariant_array = np.concatenate([invariant_array, lats])

            # local slice
            lat_slice = self.metadata["sp_slices"][0]
            lon_slice = self.metadata["sp_slices"][1]
            invariant_array = invariant_array[:, lat_slice, lon_slice]

            return invariant_array.astype(np.float32)