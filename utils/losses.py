""" Losses and Metrics """

import torch
import torch.nn as nn
from typing import Tuple, List
import numpy as np
from utils import comm
from distributed.mappings import (
    allreduce_forward_identity_backward,
    allgather_forward_split_backward,
    split_forward_allgather_backward,
)
from torch_harmonics import RealSHT
import math
from utils.profiler_utils import profile_range

# Surface var weights used in graphcast
_GRAPHCAST_SFC = {
    "t2m": 1.0,
    "u10m": 0.1,
    "v10m": 0.1,
    "msl": 0.1,
    "tcwv": 0.1,
    "sp": 0.1,
}


def get_metric(metric, metadata, temporal_average=True):

    if metric == "weighted_rmse":
        return WeightedRMSE(weights=metadata["area_weights"], temporal_average=temporal_average)
    else:
        raise NotImplementedError("Metric %s is not implemented" % metric)


def get_loss(cfg, metadata):

    if cfg.loss.loss_func in ["mse", "mse_weighted"]:
        return WeightedLoss.instantiate_from_cfg(
            cfg, channels=metadata["channels"], weights=metadata["area_weights"]
        )
    elif cfg.loss.loss_func in ["spherical"]:
        # needs global latitudes and longitudes and not local
        return SphericalLoss.instantiate_from_cfg(
            cfg,
            latitudes=metadata["global_latitudes"],
            longitudes=metadata["global_longitudes"],
            sp_shapes=metadata["sp_shapes"],
        )
    elif cfg.loss.loss_func in ["amse"]:
        return AMSELoss.instantiate_from_cfg(
            cfg,
            latitudes=metadata["global_latitudes"],
            longitudes=metadata["global_longitudes"],
            sp_shapes=metadata["sp_shapes"],
        )
    else:
        raise NotImplementedError("Loss %s is not implemented" % cfg.loss.loss_func)


def build_ch_weights(channel_list, sfc_weights, pressure_weights):

    ch_idx = {var: channel_list.index(var) for var in channel_list}

    wt = np.ones((len(channel_list)))

    if sfc_weights:
        for var in channel_list:
            if var in _GRAPHCAST_SFC:
                wt[ch_idx[var]] *= _GRAPHCAST_SFC[var]

    if pressure_weights:
        for var in channel_list:
            if var not in _GRAPHCAST_SFC:
                level = int(var[1:])
                wt[ch_idx[var]] = 0.001 * level

    channel_weights = torch.from_numpy(wt)
    channel_weights = channel_weights[:, None, None] # CHW assumption
    channel_weights = channel_weights / torch.mean(channel_weights)
    return channel_weights


class WeightedLoss(nn.Module):

    @classmethod
    def instantiate_from_cfg(cls, cfg, channels, weights):
        return cls(
            sfc_weighting=cfg.loss.sfc_weights,
            pressure_weighting=cfg.loss.pressure_weights,
            area_weighting=cfg.loss.area_weights,
            channels=channels,
            weights=weights,
        )

    def __init__(
        self,
        sfc_weighting: bool = False,
        pressure_weighting: bool = False,
        area_weighting: bool = False,
        channels: List[str] = None,
        weights: torch.Tensor = None,
    ):
        """
        Args:
            sfc_weighting: bool, whether to weight surface variables
            pressure_weighting: bool, whether to weight pressure levels
            area_weighting: bool, whether to weight by area
            channels: List of channel names
            weights: Tensor of weights
        """

        super().__init__()

        # no area or channel weights by default
        self.area_weighting = area_weighting
        self.ch_weighting = pressure_weighting or sfc_weighting

        if area_weighting:
            assert weights is not None, "Weights needed for area weighting"
            self.register_buffer("weights", weights)

        if self.ch_weighting:
            assert channels is not None, "Channel list needed for ch weighting"
            self.register_buffer(
                "channel_weights",
                build_ch_weights(
                    channel_list=channels,
                    sfc_weights=sfc_weighting,
                    pressure_weights=pressure_weighting,
                ),
            )

    def forward(self, x, y):
        se = (x - y) ** 2
        if self.area_weighting:
            se *= self.weights
        if self.ch_weighting:
            se *= self.channel_weights
        # careful: taking care of spatial parallel scaling and reduce since
        # shards could be uneven; batch parallel scaling is handled by DDP
        H, W = se.shape[-2], se.shape[-1]
        num_pixels = H * W
        num_pixels = torch.tensor(num_pixels, dtype=torch.float32, device=se.device)
        num_pixels = allreduce_forward_identity_backward(num_pixels, comm_name="sp1-sp2")
        sse = torch.sum(se) 
        sse = allreduce_forward_identity_backward(sse, comm_name="sp1-sp2")
        scaling = num_pixels * math.prod(se.shape[:-2])
        return sse / scaling

class AMSELoss(nn.Module):

    @classmethod
    def instantiate_from_cfg(cls, cfg, latitudes, longitudes, sp_shapes):
        return cls(
            latitudes=latitudes,
            longitudes=longitudes,
            sp_shapes=sp_shapes,
            loss_type=cfg.loss.loss_func,
        )

    def __init__(
        self,
        latitudes: np.ndarray = None,
        longitudes: np.ndarray = None,
        sp_shapes: List[List[int]] = None,
        loss_type: str = None,
    ):
        """
        Args:
            latitudes: np.ndarray, latitudes : these are global latitudes not local
            longitudes: np.ndarray, longitudes : these are global longitudes not local
            sp_shapes: List[List[int]], spatial parallelism shapes for lat and lon
        """

        super().__init__()

        self.latitudes = latitudes
        self.longitudes = longitudes
        self.sp_shapes = sp_shapes
        self.loss_type = loss_type

        self.sht = RealSHT(len(latitudes), len(longitudes), grid="equiangular")

    def _allgather_helper(self, x):
        x_gather = allgather_forward_split_backward(
            x, dim=-1, comm_name="sp2", shapes=self.sp_shapes[1]
        )
        x_gather = allgather_forward_split_backward(
            x_gather, dim=-2, comm_name="sp1", shapes=self.sp_shapes[0]
        )
        return x_gather

    def _split_helper(self, x):
        x_split = split_forward_allgather_backward(
            x, dim=-2, comm_name="sp1", shapes=self.sp_shapes[0]
        )
        x_split = split_forward_allgather_backward(
            x_split, dim=-1, comm_name="sp2", shapes=self.sp_shapes[1]
        )
        return x_split

    def forward(self, x, y):
        # WAR: allgather x and y for now instead of distributed SHT 
        x_gather = self._allgather_helper(x)
        y_gather = self._allgather_helper(y)

        x_gather = x_gather.to(torch.float64)
        y_gather = y_gather.to(torch.float64)
        with torch.autocast(
            "cuda", enabled=False
        ):
            xcoeffs = self.sht(x_gather)
            ycoeffs = self.sht(y_gather)

        xcoeffssq = torch.square(torch.abs(xcoeffs))
        ycoeffssq = torch.square(torch.abs(ycoeffs))
        xycohcoeffssq = torch.real(xcoeffs * ycoeffs.conj())

        # reduce over m
        xnorm2 = xcoeffssq[..., 0] + 2 * torch.sum(xcoeffssq[..., 1:], dim=-1)
        ynorm2 = ycoeffssq[..., 0] + 2 * torch.sum(ycoeffssq[..., 1:], dim=-1)
        xycoh = xycohcoeffssq[..., 0] + 2 * torch.sum(xycohcoeffssq[..., 1:], dim=-1)

        # compute sqrt
        xnorm = torch.sqrt(xnorm2)
        ynorm = torch.sqrt(ynorm2)
        xycoh = xycoh / (xnorm * ynorm)

        # compute equation (6) from the paper
        loss = torch.square(xnorm - ynorm) + 2 * torch.maximum(xnorm2, ynorm2) * (1 - xycoh)

        # sum over l
        loss = torch.sum(loss, dim=-1)

        return loss.mean()

class WeightedRMSE(nn.Module):
    """Weighted RMSE function"""

    def __init__(
        self,
        weights: torch.Tensor = None,
        temporal_average: bool = True,
    ):
        super().__init__()
        assert weights is not None, "Weights needed!"
        self.temporal_average = temporal_average
        self.register_buffer("weights", weights)

    @profile_range("weighted_rmse")
    def forward(self, pred, target):
        se = self.weights * (pred - target) ** 2.0 
        se_sum = torch.sum(se, dim=(-1, -2))

        # spatial shards may be uneven
        spatial_count = torch.tensor(
            se.shape[-2] * se.shape[-1], dtype=torch.float32, device=se.device
        )
        global_se_sum = allreduce_forward_identity_backward(se_sum, comm_name="sp1-sp2")
        global_spatial_count = allreduce_forward_identity_backward(
            spatial_count, comm_name="sp1-sp2"
        )
        mse = global_se_sum / global_spatial_count
        rmse = torch.sqrt(mse)  # (b, t, c)

        # keep taking mean until we have a 1D tensor
        if self.temporal_average:
            while rmse.dim() > 1:
                rmse = torch.mean(rmse, dim=0) # return (c,)
        else:
            rmse = torch.mean(rmse, dim=0) # return (t, c)

        return rmse


class WeightedMSE(nn.Module):
    """Weighted MSE function"""

    def __init__(
        self,
        weights: torch.Tensor = None,
        temporal_average: bool = True,
    ):
        super().__init__()
        self.temporal_average = temporal_average
        if weights is None:
            self.weights = 1.
        else:
            self.weights = weights

    def forward(self, pred, target):
        se = self.weights * (pred - target) ** 2.0 
        se_sum = torch.sum(se, dim=(-1, -2))

        # spatial shards may be uneven
        spatial_count = torch.tensor(
            se.shape[-2] * se.shape[-1], dtype=torch.float32, device=se.device
        )
        global_se_sum = allreduce_forward_identity_backward(se_sum, comm_name="sp1-sp2")
        global_spatial_count = allreduce_forward_identity_backward(
            spatial_count, comm_name="sp1-sp2"
        )
        mse = global_se_sum / global_spatial_count

        # keep taking mean until we have a 1D tensor
        if self.temporal_average:
            while mse.dim() > 1:
                mse = torch.mean(mse, dim=0) # return (c,)
        else:
            mse = torch.mean(mse, dim=0) # return (t, c)

        return mse
