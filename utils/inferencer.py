import os
from typing import Dict, List, Union, Optional
import numpy as np
import torch
import torch.distributed as dist
from distributed.mappings import allgather_forward_split_backward
import logging
from utils.data_utils import (
    get_dataset,
    get_data_loader,
    get_preprocessor,
)
from models.helpers import get_model
import time
import wandb
from utils import comm
from utils.losses import get_metric
from utils.misc_utils import (
    compute_power_spectrum,
    vis_fields,
)
import pynvml
from collections import OrderedDict
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import torch_harmonics as th
import xarray as xr


# map btw weatherbench2 and ours
variable_mapping = {
    "10m_u_component_of_wind": "u10m",
    "10m_v_component_of_wind": "v10m",
    "2m_temperature": "t2m",
    "mean_sea_level_pressure": "msl",
    "geopotential": "z",
    "specific_humidity": "q",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}


class Inferencer:
    """inferencer class"""

    def __init__(self, cfg):
        """Initialize variables for inference and scoring."""
        # path vars
        self.cfg = cfg
        self.exp_dir = os.path.join("/expts", cfg.run_name, cfg.run_tag)

        # init the communicator groups (with model parallel if needed)
        comm.init(cfg)

        self.world_size = comm.get_world_size()
        self.local_rank = comm.get_local_rank()
        self.world_rank = comm.get_world_rank()

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda:%d" % self.local_rank)
        else:
            self.device = torch.device("cpu")

        pynvml.nvmlInit()
        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        self.log_to_screen = cfg.logging.log_to_screen and self.world_rank == 0
        self.log_to_wandb = cfg.logging.log_to_wandb and self.world_rank == 0
        self.cfg = cfg

    def init_exp_dir(self, exp_dir):
        self.checkpoint_file = self.cfg.inference.checkpoint

        # get the trained model's config
        hyperparams_path = self.cfg.inference.checkpoint_hyperparams
        if os.path.exists(hyperparams_path):
            trained_cfg = OmegaConf.load(hyperparams_path)
            # the model is the same as the trained config's model
            self.cfg.model = trained_cfg.model
            # use the same context windows as the trained config
            self.cfg.train = trained_cfg.train
            # no rollout steps in inference
            self.cfg.train.num_rollout_steps = 1
            if self.log_to_screen:
                logging.info(f"Loaded trained model's config from {hyperparams_path}")
        else:
            if self.log_to_screen:
                logging.info(
                    f"No hyperparameters.yaml found at {hyperparams_path}, using CLI only!"
                )

        if self.world_rank == 0:
            if not os.path.isfile(self.checkpoint_file) and not self.cfg.inference.run_benchmark:
                raise FileNotFoundError(
                    f"Checkpoint file not found at {self.checkpoint_file}"
                )
            wandb_dir = os.path.join(self.exp_dir, "wandb/")
            if not os.path.isdir(wandb_dir):
                os.makedirs(os.path.join(self.exp_dir, "wandb/"))

    def launch(self):
        self.init_exp_dir(self.exp_dir)
        if self.log_to_wandb:
            wandb_config = OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=True
            )
            wandb.init(
                dir=os.path.join(self.exp_dir, "wandb"),
                name=self.cfg.run_name + "_" + self.cfg.run_tag,
                config=wandb_config,
                group=self.cfg.run_name,
                project=self.cfg.logging.project_inference,
                entity=self.cfg.logging.entity,
                resume=False,
            )
        self.build_and_run()

    def build_and_run(self):
        if self.world_rank == 0:
            logging.info("------------------ Config ------------------")
            _ = [logging.info(l) for l in OmegaConf.to_yaml(self.cfg).split("\n")]
            logging.info("--------------------------------------------")

        # dump the full config to disk/wandb
        if self.world_rank == 0:
            OmegaConf.save(self.cfg, os.path.join(self.exp_dir, "hyperparameters.yaml"))
            if self.log_to_wandb:
                wandb.save(os.path.join(self.exp_dir, "hyperparameters.yaml"))

        # mode test will take 12 hourly data from the year (or the full year if dt > 12)
        self.test_dataset = get_dataset(self.cfg, mode="test")
        self.test_data_loader, self.test_sampler = get_data_loader(
            self.cfg,
            self.test_dataset,
            mode="test",
        )

        # inference metadata
        self.metadata = self.test_dataset.metadata
        self.n_channels = self.metadata["n_channels"]
        self.n_lat = len(self.metadata["global_latitudes"])
        self.n_lon = len(self.metadata["global_longitudes"])
        self.context_window = self.metadata["temporal_context_window"]
        self.dt_hours = int(self.metadata["dt_hours"] * self.metadata["dt_scale"] / np.timedelta64(1, "h"))
        self.lead_times = list(
            range(
                self.dt_hours,
                self.cfg.inference.time_horizon_in_hours + 1,
                self.dt_hours,
            )
        )
        self.n_inference_steps = len(self.lead_times)

        start_date = np.datetime64(self.cfg.inference.start_date)
        end_date = np.datetime64(self.cfg.inference.end_date)
        # ics are at 12h intervals between start and end date
        self.ics = np.arange(
            start_date, end_date + np.timedelta64(12, "h"), np.timedelta64(12, "h")
        )
        # self.ics = self.ics[0:8] # cut short for debugging
        self.total_ic_count = len(self.ics)
        # use dp for sharding IC list
        self.ics = self.ics[comm.get_rank("dp") :: comm.get_size("dp")]
        logging.info(
            f"rank {comm.get_rank('dp')} running inference for {len(self.ics)} ICs"
        )

        # logs
        self.set_log_channels()
        self.ic_count = 0

        # metrics
        self.weighted_rmse = get_metric(
            metric="weighted_rmse",
            metadata=self.metadata,
            temporal_average=False,
        ).to(self.device)
        self.mean = torch.as_tensor(self.test_dataset.means).to(self.device)
        self.std = torch.as_tensor(self.test_dataset.stds).to(self.device)
        self.rmse = torch.zeros(
            (len(self.ics), self.n_channels, len(self.lead_times)),
            dtype=torch.float32,
            device=self.device,
        )
        # spectra is for pred and target, organized by lead time
        # because we only log it for some lead times and not all
        self.spectra = {}
        for lead_time in self.spectra_times_in_hours:
            self.spectra[lead_time] = torch.zeros(
                (len(self.ics), 2, self.n_channels, self.n_lat),
                dtype=torch.float32,
                device=self.device,
            )
        self.sht = th.RealSHT(self.n_lat, self.n_lon, grid="equiangular").to(
            self.device
        )

        # benchmark data
        self.run_benchmark = self.cfg.inference.run_benchmark
        self.graphcast_path = "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr"
        self.hres_path = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"

        # init model
        if not self.run_benchmark:
            self.model = get_model(self.cfg, domain_metadata=self.metadata).to(
                self.device
            )
            self.model = comm.all_model_groups_exist(self.model)
            self.preprocessor = get_preprocessor(self.cfg, self.test_dataset).to(
                self.device
            )
            logging.info("Loading checkpoint %s" % self.checkpoint_file)
            self.restore_checkpoint(self.checkpoint_file)
            if self.log_to_screen:
                logging.info(self.model)
            self.model.eval()
        else:
            # get preds directly from gc
            if self.run_benchmark == "graphcast":
                self.benchmark = xr.open_zarr(self.graphcast_path)
            elif self.run_benchmark == "hres":
                self.benchmark = xr.open_zarr(self.hres_path)
            else:
                raise ValueError(f"Invalid benchmark: {self.run_benchmark}")
            self.benchmark_pred = torch.zeros(
                (len(self.lead_times), self.n_channels, self.n_lat, self.n_lon),
                dtype=torch.float32,
            )

        self.inference()

    def set_benchmark_pred(self, ic, lead_times):
        """Get the benchmark prediction for the given IC and lead time"""
        lead_time_deltas = [np.timedelta64(lead_time, "h") for lead_time in lead_times]
        forecast = self.benchmark.sel(time=ic, prediction_timedelta=lead_time_deltas)

        # zero out the pred tensor
        self.benchmark_pred.zero_()

        # benchmark has different variable names, map them correctly
        for channel_idx, channel_name in enumerate(self.metadata["channels"]):
            if isinstance(channel_name, bytes):
                channel_name = channel_name.decode("utf-8")

            # handle level-specific variables (like u50, z500)
            if len(channel_name) > 1 and channel_name[-1].isdigit():
                var_name = channel_name.rstrip(
                    "0123456789"
                )  # remove all trailing digits
                level_str = channel_name[len(var_name) :]  # get the level part
                level = int(level_str)

                # map to benchmark variable name
                benchmark_var = None
                for wb_var, model_var in variable_mapping.items():
                    if model_var == var_name:
                        benchmark_var = wb_var
                        break

                if benchmark_var is None or benchmark_var not in forecast.data_vars:
                    continue

                var_data = forecast[benchmark_var].sel(level=level).values
            else:
                # handle surface variables (no level)
                benchmark_var = None
                for wb_var, model_var in variable_mapping.items():
                    if model_var == channel_name:
                        benchmark_var = wb_var
                        break

                if benchmark_var is None or benchmark_var not in forecast.data_vars:
                    continue

                var_data = forecast[benchmark_var].values

            # benchamrk's shape is (leadtimes, lat, lon)
            # kill the last pixel for the benchmark as well
            self.benchmark_pred[:, channel_idx, :, :] = torch.flip(
                torch.from_numpy(var_data.astype(np.float32)), dims=(-2,)
            )[:, :-1]

    def all_gather_helper(self, x):
        with torch.no_grad():
            x_ag = allgather_forward_split_backward(
                x, dim=-1, comm_name="sp2", shapes=self.metadata["sp_shapes"][1]
            )
            x_ag = allgather_forward_split_backward(
                x_ag, dim=-2, comm_name="sp1", shapes=self.metadata["sp_shapes"][0]
            )
            return x_ag


    def set_log_channels(self):
        """Generate log channels from config: surface vars + atmospheric vars at specified levels"""
        log_channels = []
        # surface variables
        log_channels.extend(self.cfg.inference.log_sfc)
        # pressure level variables
        for var in self.cfg.inference.log_var:
            for level in self.cfg.inference.log_levels:
                log_channels.append(f"{var}{level}")

        # quick check that all log_channels exist in metadata["channels"]
        missing = [ch for ch in log_channels if ch not in self.metadata["channels"]]
        if self.log_to_screen:
            if missing:
                logging.info(
                    f"Some log channels are not there in the dataset: {missing}"
                )
            log_channels = [
                ch for ch in log_channels if ch in self.metadata["channels"]
            ]
            logging.info(f"Logging available channels: {log_channels}")
        self.log_channels = log_channels
        self.spectra_times_in_hours = self.cfg.inference.spectra_times_in_hours

    def log_memory_usage(self):
        """Logs the memory usage of the GPU"""
        if self.log_to_screen:
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (
                1024.0 * 1024.0 * 1024.0
            )
            max_reserved_gb = torch.cuda.max_memory_reserved(device=self.device) / (
                1024.0 * 1024.0 * 1024.0
            )
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (
                1024.0 * 1024.0 * 1024.0
            )
            logging.info(
                f"pynvml mem: {all_mem_gb} GB, torch allocated: {max_mem_gb} GB, torch reserved: {max_reserved_gb} GB"
            )

    def inference(self):
        get_samples_at_once = False
        with torch.inference_mode():
            visualize = True
            for ic in self.ics:
                inference_time = time.time()
                if self.log_to_screen:
                    logging.info(f"Running inference for IC {ic}")

                dt_hours = self.dt_hours
                past = self.context_window - 1
                if not get_samples_at_once:
                    ts_input_window = (
                        ic
                        - np.timedelta64(past * dt_hours, "h")
                        + np.timedelta64(dt_hours, "h") * np.arange(self.context_window)
                    )
                    input_window = (
                        self.test_dataset.get_samples(ts_input_window)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                else:
                    total_steps = past + self.n_inference_steps + 1
                    ts = (
                       ic
                       - np.timedelta64(past * dt_hours, "h")
                       + np.timedelta64(dt_hours, "h") * np.arange(total_steps) 
                    )
                    logging.info(f"Getting all samples for {len(ts)} timestamps")
                    all_samples = self.test_dataset.get_samples(ts)
                    input_window = (
                        all_samples[: self.context_window].unsqueeze(0).to(self.device)
                    )

                if self.run_benchmark:
                    logging.info(f"Getting benchmark prediction for IC {ic}")
                    self.set_benchmark_pred(ic, self.lead_times)
                    logging.info(
                        f"Benchmark prediction shape: {self.benchmark_pred.shape}"
                    )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                for i, lead_time in enumerate(self.lead_times):
                    if not get_samples_at_once:
                        target_ts = [ic + np.timedelta64(lead_time, "h")]
                        targets = self.test_dataset.get_samples(target_ts).unsqueeze(0).to(self.device)
                    else:
                        idx = self.context_window + i
                        targets = (
                            all_samples[idx : idx + 1].unsqueeze(0).to(self.device)
                        )  # target is next step always
                    
                    # unnormalize targets
                    targets = targets * self.std
                    targets = targets + self.mean

                    if self.run_benchmark:
                        gen = self.benchmark_pred[i].unsqueeze(0).unsqueeze(0).to(self.device)
                        pred = gen  # benchmark pred is already normalized
                    else:
                        # prepare inputs (append invars etc)
                        inputs = self.preprocessor(
                            input_window, device=self.device, mode="inference"
                        )
                        # predict
                        gen = self.model(inputs)

                        # unnormalize gen
                        pred = gen * self.std
                        pred = pred + self.mean

                    # compute metrics
                    step_rmse = self.weighted_rmse(pred, targets)[0]
                    self.rmse[self.ic_count, :, i] = step_rmse
                    # self.rmse[:, i] += step_rmse
                    if self.log_to_screen:
                        if i == 0:
                            self.log_memory_usage()
                        if i % 10 == 0:
                            logging.info(
                                f"RMSE for u10m at step {lead_time}h = {step_rmse[0].detach().cpu().numpy()}"
                            )

                    # compute spectra
                    if lead_time in self.spectra_times_in_hours:
                        if self.log_to_screen:
                            logging.info(
                                f"Computing spectra for lead time {lead_time}h"
                            )
                        # gather across sp ranks if needed
                        pred = self.all_gather_helper(pred)
                        targets = self.all_gather_helper(targets)
                        self.spectra[lead_time][self.ic_count, 0] = compute_power_spectrum(
                            pred, self.sht
                        )[0]
                        self.spectra[lead_time][self.ic_count, 1] = compute_power_spectrum(
                            targets, self.sht
                        )[0]

                        if visualize and comm.get_world_rank() == 0:
                            fields = [pred[0,0,0].cpu().numpy(), targets[0,0,0].cpu().numpy()]
                            # np.save(f"./pred_{ic}_{lead_time}h.npy", pred[0,0].cpu().numpy())
                            # np.save(f"./targets_{ic}_{lead_time}h.npy", targets[0,0].cpu().numpy())
                            fig = vis_fields(fields)
                            plt.savefig(os.path.join(self.exp_dir, f"vis_{ic}_{lead_time}h.png"))
                            plt.close(fig)

                    # update input window
                    input_window = torch.roll(input_window, shifts=-1, dims=1)
                    input_window[:, -1].copy_(gen[:, 0])

                inference_time = time.time() - inference_time
                if self.log_to_screen:
                    logging.info(
                        f"Inference time for IC {ic}: {inference_time} seconds"
                    )

                self.ic_count += 1
                visualize = False # only visualize the first IC

        self.compute_final_metrics()

        if self.log_to_wandb:
            wandb.finish()
        pynvml.nvmlShutdown()

    def compute_final_metrics(self):
        """Compute final averaged RMSE metrics per channel and step"""
        assert self.ic_count > 0, "No initial conditions processed"

        # some ics may be unstable - track them and remove them from the metrics
        unstable_count = 0
        for ic in range(self.ic_count):
            if (self.rmse[ic, 0] > 10).any():  
                unstable_count += 1
                self.rmse[ic] = 0  
                for lead_time in self.spectra:
                    self.spectra[lead_time][ic] = 0  
        
        self.ic_count -= unstable_count

        self.rmse = self.rmse.sum(dim=0)  # shape: [n_channels, n_steps]
        for lead_time in self.spectra:
            self.spectra[lead_time] = self.spectra[lead_time].sum(dim=0)  # shape: [2, n_channels, n_lat]

        # reduce from all procs
        dist.all_reduce(self.rmse, op=dist.ReduceOp.SUM, group=comm.get_group("dp"))
        for lead_time in self.spectra:
            dist.all_reduce(self.spectra[lead_time], op=dist.ReduceOp.SUM, group=comm.get_group("dp"))
        ic_count_tensor = torch.tensor(
            self.ic_count, dtype=torch.int32, device=self.device
        )
        dist.all_reduce(
            ic_count_tensor, op=dist.ReduceOp.SUM, group=comm.get_group("dp")
        )
        total_ic_count = ic_count_tensor.item()
        
        # Reduce unstable count across all processes and print
        unstable_tensor = torch.tensor(unstable_count, dtype=torch.int32, device=self.device)
        dist.all_reduce(unstable_tensor, op=dist.ReduceOp.SUM, group=comm.get_group("dp"))
        total_unstable = unstable_tensor.item()
        
        if self.log_to_screen:
            print(f"Number of unstable ICs (RMSE > 10): {total_unstable} out of {self.total_ic_count}")

        assert total_ic_count == self.total_ic_count - total_unstable, f"Total IC count mismatch: {total_ic_count} != {self.total_ic_count} - {total_unstable}"

        # average over all initial conditions across all processes
        self.rmse /= total_ic_count  # shape: [n_channels, n_steps]
        for lead_time in self.spectra:
            self.spectra[lead_time] /= total_ic_count  # shape: [2, n_channels, n_lat]

        # redo just the logging after averaging
        for i, lead_time in enumerate(self.lead_times):
            logs = {}
            for ch_idx, channel in enumerate(self.log_channels):
                if channel not in self.metadata["channels"]:
                    continue
                channel_idx = self.metadata["channels"].index(channel)
                # rmse
                logs[f"rmse_{channel}"] = self.rmse[channel_idx, i].item()

            if self.log_to_wandb:
                wandb.log(logs, step=int(lead_time), commit=True)

        if self.log_to_wandb:
            # save RMSE to numpy files for each logged channel
            for ch_idx, channel in enumerate(self.log_channels):
                if channel not in self.metadata["channels"]:
                    continue
                channel_idx = self.metadata["channels"].index(channel)
                rmse_values = self.rmse[channel_idx].detach().cpu().numpy()
                np.save(os.path.join(self.exp_dir, f"rmse_{channel}.npy"), rmse_values)

            # separate custom charts for spectra
            for lead_time in self.spectra_times_in_hours:
                for ch_idx, channel in enumerate(self.log_channels):
                    if channel not in self.metadata["channels"]:
                        continue
                    channel_idx = self.metadata["channels"].index(channel)
                    y_pred = (
                        self.spectra[lead_time][0, channel_idx].detach().cpu().numpy()
                    )
                    y_true = (
                        self.spectra[lead_time][1, channel_idx].detach().cpu().numpy()
                    )
                    x = np.arange(1, len(y_true) + 1)

                    # save spectra to numpy file (shape: 2 x n_lat, where 0=pred, 1=true)
                    spectra_data = np.stack([y_pred, y_true], axis=0)
                    np.save(os.path.join(self.exp_dir, f"spectra_{channel}_{lead_time}h.npy"), spectra_data)

                    spectrum_plot = wandb.plot.line_series(
                        xs=np.log10(x),
                        ys=[np.log10(y_true), np.log10(y_pred)],
                        keys=["true", "pred"],
                        title=f"Spectrum {channel} at {lead_time}h",
                        xname="log10(Wavenumber)",
                    )

                    wandb.log({f"spectrum_{channel}_{lead_time}h": spectrum_plot})

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cuda:{}".format(self.local_rank),
            weights_only=False,
        )
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
