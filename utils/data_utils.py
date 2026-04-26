import h5py
import logging
import numpy as np
import os
import types
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import List, Optional, Dict, Union, Tuple
from numpy.typing import NDArray
from utils.dataloaders.era5hdf5_dali import (
    ERA5HDF5DALIDataLoader,
    Era5HDF5DatasetDALI,
)
from utils.dataloaders.era5hdf5 import Era5HDF5Dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from utils import comm


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))

class ResumeSampler(Sampler):
    def __init__(self, sampler, resume_epoch, resume_iter):
        self.sampler = sampler
        self.resume_epoch = resume_epoch
        self.resume_iter = resume_iter
        self._did_skip = False
        self._current_epoch = None

    def set_epoch(self, epoch):
        self._current_epoch = epoch
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        indices = list(self.sampler)

        if (
            not self._did_skip
            and self._current_epoch == self.resume_epoch
            and self.resume_iter > 0
        ):
            indices = indices[self.resume_iter:]
            self._did_skip = True

        return iter(indices)

    def __len__(self):
        n = len(self.sampler)
        if (
            not self._did_skip
            and self._current_epoch == self.resume_epoch
        ):
            return max(0, n - self.resume_iter)
        return n

def get_dataset(cfg, mode="train"):
    if cfg.data.loader == "pytorch":
        # use pytorch dataloader
        if cfg.data.name in ["latlon_025deg_hdf5", "latlon_025deg_hdf5_1h"]:
            dataset = Era5HDF5Dataset.instantiate_from_cfg(cfg, mode)
        else:
            raise ValueError(f"Dataset {cfg.data.name} not found")
    elif cfg.data.loader == "dali":
        dataset = Era5HDF5DatasetDALI.instantiate_from_cfg(
            cfg=cfg,
            seed=42,
            mode=mode,
        )
    return dataset


def get_data_loader(cfg, dataset, mode="train"):
    if mode == "train" and cfg.data.limit_nsamples > 0:
        if cfg.data.limit_nsamples < cfg.data.batch_size:
            raise ValueError(
                f"Not enough training samples ({cfg.data.limit_nsamples}) "
                f"for requested batch size ({cfg.data.batch_size})"
            )
    elif mode in ["valid", "test"] and cfg.data.limit_nsamples_val > 0:
        if cfg.data.limit_nsamples_val < cfg.data.batch_size:
            raise ValueError(
                f"Not enough validation/test samples ({cfg.data.limit_nsamples_val}) "
                f"for requested batch size ({cfg.data.batch_size})"
            )

    if cfg.data.loader == "pytorch":
        if comm.get_size("dp") > 1:
            sampler = DistributedSampler(
                dataset,
                shuffle=(mode == "train"),
                num_replicas=comm.get_size("dp"),
                rank=comm.get_rank("dp"),
            )
        else:
            sampler = None

        if mode == "train" and dataset.resume_skip_batches > 0:
            if sampler is None:
                sampler = SequentialSampler(dataset)
            sampler = ResumeSampler(
                sampler=sampler,
                resume_epoch=dataset.ckpt_epoch,
                resume_iter=dataset.resume_skip_batches,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.parallelism.micro_batch_size,
            num_workers=cfg.data.num_data_workers,
            persistent_workers=True,
            shuffle=(sampler is None),
            sampler=sampler,
            worker_init_fn=worker_init,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
    elif cfg.data.loader == "dali":
        # use DALI dataloader: builds pipeline etc.
        dataloader = ERA5HDF5DALIDataLoader(
            dataset=dataset,
            micro_batch_size=cfg.parallelism.micro_batch_size,
            num_data_workers=cfg.data.num_data_workers,
            mode=mode,
            seed=42,
        )
        sampler = None  # DALI takes care of shuffling

    return dataloader, sampler


def get_preprocessor(cfg, dataset):
    return PreProcessor.instantiate_from_cfg(cfg, dataset)


class PreProcessor(nn.Module):
    """
    Data preprocessing class to handle concatenation of static features
    """

    @classmethod
    def instantiate_from_cfg(cls, cfg, dataset):
        """Class method to instantiate the preprocessor for a given dataset from a given hydra config"""
        return cls(
            dataset,
            invariants=cfg.data.invariants,
            send_to_device=(cfg.data.loader != "dali"),
            num_rollout_steps=cfg.train.num_rollout_steps,
        )

    def __init__(
        self,
        dataset,
        invariants=None,
        send_to_device=True,
        num_rollout_steps=1,
    ):
        super().__init__()

        if invariants is not None:
            self.add_invariants = True
            # expand invariants in batch and time dimensions (0 and 1)
            invar_tensor = torch.from_numpy(dataset.invar)
            invar_tensor = invar_tensor.unsqueeze(0).unsqueeze(0)
            self.register_buffer("invariants", invar_tensor)
        else:
            self.add_invariants = False
        self.send_to_device = send_to_device
        self.num_rollout_steps = num_rollout_steps

    def forward(
        self,
        batch: torch.Tensor,
        device: torch.device,
        mode: str = "train",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C, H, W = batch.shape
        if self.send_to_device:
            batch = batch.to(device)

        if self.add_invariants:
            invar_expanded = self.invariants.expand(
                B, T, -1, H, W
            )  # B, T, C_invar, H, W
            batch = torch.cat([batch, invar_expanded], dim=2)  # B, T, C+C_invar, H, W

        if mode == "train":
            inputs = batch[:, :-self.num_rollout_steps]
            targets = batch[:, -self.num_rollout_steps:, :C]
            return inputs, targets
        elif mode == "inference":
            # only inputs
            return batch
        else:
            raise ValueError(f"Invalid preprocessor mode: {mode}")
