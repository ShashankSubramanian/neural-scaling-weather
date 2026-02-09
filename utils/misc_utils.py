"""
  misc utils for plotting and any application specific metrics
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import torch_harmonics as th
import torch
from utils import comm

def get_spatial_coords(locations):
    """
    Get coordinates for a given location
    """
    lat, lon = torch.tensor(locations[0]), torch.tensor(locations[1])
    lat, lon = torch.meshgrid(lat, lon, indexing='ij')
    # convert to radians
    lat_rad, lon_rad = torch.deg2rad(lat), torch.deg2rad(lon)
    # convert to cartesian coordinates
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    return torch.stack((x, y, z), dim=-1)  # shape (nlat, nlon, 3)

def get_local_slice(sizes, rank):
    """
    Get the local slice for a given rank
    inputs:
        sizes: list of ints, the size of each dimension
        rank: int, the rank to get the slice for
    returns:
        slice object
    """
    start = sum(sizes[:rank])
    end = start + sizes[rank]
    return slice(start, end)

def build_spherical_area_weights(latitudes):
    """
    Create a normalized grid-cell area weight tensor for use in losses, metrics, etc.
    inputs:
        latitudes: np.ndarray, shape (nlat,)
    returns:
        w: torch.Tensor, shape (nlat,)
    """
    cos_lat = np.cos(np.deg2rad(latitudes))  # proportional to area of cell
    cos_lat = cos_lat / cos_lat.mean()
    w = torch.from_numpy(cos_lat)
    return w

def create_dummy_metadata(H, W, C):
    """Create dummy domain metadata for testing"""
    return {
        "area_weights": torch.rand(H, W, dtype=torch.float32),
        "coords": torch.rand(H, W, 3, dtype=torch.float32),
        "sp_shapes": [[H], [W]],
        "sp_slices": [[slice(0, H)], [slice(0, W)]],
        "channels": [f"ch{i}" for i in range(C)],
        "n_channels": C,
        "spatial_dims": [H, W],
    }

def split_dummy_metadata(metadata, sp1_shapes, sp2_shapes):
    """Split dummy domain metadata into local shards"""
    sp1_slice = get_local_slice(sp1_shapes, comm.get_rank("sp1"))
    sp2_slice = get_local_slice(sp2_shapes, comm.get_rank("sp2"))
    return {
        "area_weights": metadata["area_weights"][sp1_slice, sp2_slice],
        "coords": metadata["coords"][sp1_slice, sp2_slice, :],
        "sp_shapes": [sp1_shapes, sp2_shapes],
        "sp_slices": [sp1_slice, sp2_slice],
        "channels": metadata["channels"],
        "n_channels": metadata["n_channels"],
        "spatial_dims": [sp1_shapes[comm.get_rank("sp1")], sp2_shapes[comm.get_rank("sp2")]],
    }

def show(u, ax, fig, rescale=None):
    """plot some output function to keep track of during logging"""
    h = ax.imshow(
        u.T, interpolation="nearest", cmap="rainbow", origin="lower", aspect="auto"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)


def vis_fields(fields):
    pred, tar = fields
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(pred, cmap="turbo", vmin=tar.min(), vmax=tar.max())
    ax[0].set_title("pred")
    ax[1].imshow(tar, cmap="turbo")
    ax[1].set_title("tar")
    fig.tight_layout()
    return fig


def compute_power_spectrum(x, sht=None):
    """
    Computes power spectrum using SHT
    Formula from https://en.wikipedia.org/wiki/Spherical_harmonics#Power_spectrum_in_signal_processing
    inputs:
        x: shape (b, c, lat, lon)
    returns:
        spectrum: shape (b, c, lat)
    """
    if len(x.shape) > 4:
        x = x[:,0]
    assert len(x.shape) == 4, "Power specturm computation only supports lat-lon grid"
    nlat, nlon = x.shape[-2:]
    if sht is None:
        sht = th.RealSHT(nlat, nlon, grid="equiangular").to(x.device)
    coeff = sht(x.to(torch.float64)) # some th versions show errors in fp32
    coeff = torch.view_as_real(coeff)
    coeff = (coeff[..., 0] ** 2 + coeff[..., 1] ** 2)
    power = coeff[..., :, :].sum(dim=-1) / (2 * torch.arange(nlat, device=x.device) + 1)

    return power


def plot_power_spectrum(psd_pred, psd_tar, label):
    """
    Plot power spectra of prediciton and target
    inputs:
        psd_pred: shape (nlat,)
        psd_tar: shape (nlat,)
    returns:
        fig: matplotlib fig
    """

    fig = plt.figure(figsize=(6, 4))
    plt.plot(psd_tar, "k-", label="tar")
    plt.plot(psd_pred, "r-", label="pred")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Wavenumber")
    plt.ylabel("Power")
    plt.title(f"Lead Time {label}h")
    plt.legend()

    return fig
