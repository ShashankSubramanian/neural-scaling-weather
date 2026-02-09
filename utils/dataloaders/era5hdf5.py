import os
import logging
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
import h5py
from typing import List, Optional
import calendar

# Add distributed helpers import
from distributed.helpers import compute_split_shapes_for_patching
from utils import comm
from utils.misc_utils import (
    build_spherical_area_weights, 
    get_local_slice, 
    get_spatial_coords,
)

class Era5HDF5Dataset(Dataset):
    """
    HDF5 dataset object serving ERA5 samples with temporal windows
    """

    @classmethod
    def instantiate_from_cfg(cls, cfg, mode):
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
            read_local=cfg.data.read_local,
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
        read_local: bool = False,
    ):
        """
        Inputs:
            name: str
                dataset name under data directory tree
            tag: str
                tag for statistics and train/test split
            mode: str
                'train', 'valid', or 'test'
            dt_scale: int
                scale factor for the time step
            temporal_context_window: int
                number of time steps to include in each sample
            num_rollout_steps: int
                number of rollout steps to include in each sample
            model_patch_size: int
                size of the model patch for spatial parallel processing
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
            read_local: bool
                whether to read the local spatial data (True) or the full global data (False)
        """
        super().__init__()
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
        self.read_local = read_local
        self._get_files_stats()
        self.means, self.stds = self._load_stats()

        if len(self.invariants) > 0:
            self.invar = self.get_invariants()
        else:
            self.invar = None

    def _load_stats(self):
        """Load dataset statistics from the appropriate stats h5 file"""
        stats_path = os.path.join(self.location, "stats", f"stats_{self.tag}.h5")
        with h5py.File(stats_path, 'r') as f:
            mean = f['global_mean'][:]
            std = f['global_std'][:]

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
            "dt_hours": self.metadata["dt_hours"],
            "dt_scale": self.metadata["dt_scale"],
            "n_samples_total": self.n_samples_total,
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
            "dt_hours": self.dt_hours,
            "dt_scale": self.dt_scale,
            "global_latitudes": self.latitudes,
            "global_longitudes": self.longitudes,
        }

    def _get_files_stats(self):
        """Scan directories and extract metadata for ERA5"""
        self.era5_paths = glob.glob(os.path.join(self.location, "*.h5"))
        self.era5_paths = sorted(
            self.era5_paths, key=lambda x: int(os.path.basename(x).replace(".h5", ""))
        )

        holdout_yrs = self.valid_years + self.test_years
        if self.mode == "train":
            self.era5_paths = [
                x for x in self.era5_paths
                if int(os.path.basename(x).replace(".h5", "")) not in holdout_yrs
            ]
            self.years = [
                int(os.path.basename(x).replace(".h5", "")) for x in self.era5_paths
            ]
        elif self.mode == "valid":
            self.era5_paths = [
                x for x in self.era5_paths
                if int(os.path.basename(x).replace(".h5", "")) in self.valid_years
            ]
            self.years = [
                int(os.path.basename(x).replace(".h5", "")) for x in self.era5_paths
            ]
        else:
            self.era5_paths = [
                x for x in self.era5_paths
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

        # Store file handles
        self.ds_era5 = [h5py.File(path, 'r') for path in self.era5_paths]
        
        self.n_samples_total = self.compute_total_samples()
        if self.limit_nsamples > 0:
            self.n_samples_total = min(self.n_samples_total, self.limit_nsamples)
        self._set_shapes_and_metadata()

        # for dataset checkpointing
        self.resume_skip_batches = 0
        self.iterations_to_skip = 0
        self.ckpt_epoch = 0

    def set_dataset_state(self, checkpoint):
        """Set dataset state from checkpoint"""
        self.resume_skip_batches = checkpoint["iters_in_epoch"]
        self.ckpt_epoch = checkpoint["epoch"]

    def compute_total_samples(self):
        """Compute total number of possible samples given temporal window"""
        self.all_times = []
        for ds in self.ds_era5:
            # convert integer timestamps to datetime64
            timestamps = np.array([np.datetime64('1900-01-01') + np.timedelta64(int(ts), 'h') for ts in ds['time'][:]])
            self.all_times.append(timestamps)

        self.all_times = np.concatenate(self.all_times)
        # # for non-overlapping windows, we need to divide the total available time steps
        # return len(self.all_times) // (self.temporal_context_window * self.dt)
        # Return number of possible starting points
        return len(self.all_times) - (self.temporal_context_window + self.num_rollout_steps) * self.dt_scale

    def __len__(self):
        return self.n_samples_total 

    def _normalize_era5(self, img):
        img -= self.means
        img /= self.stds
        return torch.as_tensor(img)

    def unnormalize_era5(self, img):
        img = img * self.stds
        img = img + self.means
        return img

    def _get_era5(self, timestamps):
        """Load ERA5 data for given timestamps"""
        # Group timestamps by year
        year_groups = {}
        for ts in timestamps:
            # convert numpy.datetime64 to datetime
            ts_dt = datetime.utcfromtimestamp(ts.astype("datetime64[s]").astype("int64"))
            year = ts_dt.year
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(ts)
        
        # Process each year's timestamps in batch
        data = []
        lat_slice = self.metadata["sp_slices"][0]
        lon_slice = self.metadata["sp_slices"][1]
        for year, year_timestamps in year_groups.items():
            year_idx = self.years.index(year)
            ds_handle = self.ds_era5[year_idx]
            # Convert all timestamps for this year to hours since 1900-01-01
            hours_since_1900 = np.array([int((ts - np.datetime64('1900-01-01')) / np.timedelta64(1, 'h')) 
                                        for ts in year_timestamps])
            time_indices = [np.where(ds_handle['time'][:] == h)[0][0] for h in hours_since_1900]
            if self.read_local:
                sample = ds_handle['data'][time_indices, :, lat_slice, lon_slice]  # [time, channels, lat, lon]
            else:
                sample = ds_handle['data'][time_indices, :, :, :]
            data.append(self._normalize_era5(sample))
        
        data = torch.cat(data, dim=0)
        if not self.read_local:
            # split after the read
            data = data[..., lat_slice, lon_slice]
        return data

    def __getitem__(self, global_idx):
        """Get a sample with temporal window, using dt for spacing between timesteps"""
        # # each window takes up temporal_context_window * dt timesteps
        # start_idx = global_idx * (self.temporal_context_window * self.dt)
        start_idx = global_idx
        window_indices = [start_idx + i * self.dt_scale for i in range(self.temporal_context_window + self.num_rollout_steps)]
        timestamps = self.all_times[window_indices]
        return self._get_era5(timestamps)

    def get_samples(self, timestamps):
        """Get samples for given timestamps"""
        data = []
        for sample in self._get_era5(timestamps):
            data.append(sample)
        data = torch.cat(data, dim=0)
        return data

    def get_invariants(self):
        """
        Return the invariant fields (to be called once, before train/val loops, so they can be re-used instead of loaded each step)
        """
        if len(self.invariants) == 0:
            return None

        invar_to_use = list(filter(lambda x: x != "coslat", self.invariants))
        with h5py.File(os.path.join(self.location, "invar", "invariants.h5"), 'r') as f:
            channel_names = [n.decode('utf-8') for n in f['channel'][:]]
            # Get indices of channels we want to use
            channel_indices = [i for i, nm in enumerate(channel_names) if nm in invar_to_use]
            # Select only the channels we want and the local spatial slice
            invariant_array = f['data'][channel_indices, :-1, :]  # [channels, lat, lon]
            # Update channel names to only include selected ones
            channel_names = [channel_names[i] for i in channel_indices]

            if "orog" in invar_to_use:
                # normalize orog
                orog_idx = channel_names.index("orog")
                orog = invariant_array[orog_idx]
                low, high = orog.min(), orog.max()
                invariant_array[orog_idx] = (orog - low) / high

            if "coslat" in self.invariants:
                lats = np.cos(np.deg2rad(self.latitudes))  # proportional to area of cell
                lats = np.ones((1, 1, len(self.longitudes))) * lats[None, :, None]
                invariant_array = np.concatenate([invariant_array, lats])

            # local slice
            lat_slice = self.metadata["sp_slices"][0]
            lon_slice = self.metadata["sp_slices"][1]
            invariant_array = invariant_array[:, lat_slice, lon_slice]

            return invariant_array.astype(np.float32) 
