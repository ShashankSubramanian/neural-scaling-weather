import os, sys, time
import numpy as np
import random
import torch
import math
from contextlib import nullcontext
from torch import GradScaler
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
import logging
from collections import OrderedDict
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import wandb
import pynvml

# our imports
from utils.data_utils import (
    get_dataset,
    get_data_loader,
    get_preprocessor,
)
from utils.optimizer_utils import set_scheduler, set_optimizer
from utils.misc_utils import (
    vis_fields,
    compute_power_spectrum,
    plot_power_spectrum,
    get_spatial_coords,
)
from utils.flops_utils import FlopsCalculator
from utils.losses import get_loss, get_metric
from utils import comm
from distributed.mappings import init_ddp_model_and_reduction_hooks
from distributed.helpers import init_params_for_shared_weights
from models.helpers import get_model
from utils.profiler_utils import init_profiler, profile_range, profile


def set_seed(cfg, world_size):
    if cfg.train.seed is None:
        seed = np.random.randint(10000)
    else:
        seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6


class Trainer:
    """trainer class"""

    def __init__(self, cfg):
        """init vars for distributed training (ddp) and logging"""
        self.exp_dir = os.path.join("/expts", cfg.run_name, cfg.run_tag)
        self.run_name, self.run_tag = cfg.run_name, cfg.run_tag
        self.auto_resume = not cfg.disable_auto_resume

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

        self.nvml_handle = None
        if self.device.type == "cuda":
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        self.log_to_screen = cfg.logging.log_to_screen and self.world_rank == 0
        self.log_to_wandb = cfg.logging.log_to_wandb and self.world_rank == 0
        self.cfg = cfg

        # Set gradient accumulation steps
        global_mbs = cfg.parallelism.micro_batch_size * comm.get_size("dp")
        self.gradient_accum_steps = cfg.data.batch_size // global_mbs
        if self.gradient_accum_steps <= 0:
            raise ValueError("gradient_accum_steps must be positive")
        if cfg.data.batch_size % global_mbs != 0:
            raise ValueError(
                "cfg.data.batch_size must be divisible by "
                "cfg.parallelism.micro_batch_size * data_parallel_size "
                f"({cfg.data.batch_size} % {global_mbs} != 0)"
            )

    def init_exp_dir(self, exp_dir):
        # top-level exp_dir should already exist, as hydra creates it to store the config
        # here we need to set up checkpoint dir if it's the first time, or verify the config matches if we're resuming
        if self.world_rank == 0:
            ckpt_dir, wandb_dir = os.path.join(exp_dir, "checkpoints/"), os.path.join(
                exp_dir, "wandb/"
            )

            if not os.path.isdir(ckpt_dir) or not os.path.isdir(wandb_dir):
                # First time, so create necessary subdirs
                os.makedirs(os.path.join(exp_dir, "checkpoints/"))
                os.makedirs(os.path.join(exp_dir, "wandb/"))

    def launch(self):
        # not sweeping, just a standard run
        self.init_exp_dir(self.exp_dir)
        if self.log_to_wandb:
            wandb_config = OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=True
            )
            wandb.init(
                dir=os.path.join(self.exp_dir, "wandb"),
                name=self.run_name + "_" + self.run_tag,
                config=wandb_config,
                group=self.run_name,
                project=self.cfg.logging.project,
                entity=self.cfg.logging.entity,
                resume=self.auto_resume,
            )
        self.build_and_run()

    def setup_checkpoint_strategy(self):
        """resume a run, branch from a previous run, or finetune from a previous run"""
        self.checkpoint_path = os.path.join(
            self.exp_dir,
            "checkpoints/ckpt.tar",
        )
        self.resuming = (
            True if os.path.isfile(self.checkpoint_path) and self.auto_resume else False
        )

        if self.resuming:
            # Make sure the config matches if we are auto-resuming
            prevcfg = OmegaConf.load(os.path.join(self.exp_dir, "hyperparameters.yaml"))
            assert OmegaConf.to_container(
                prevcfg, resolve=True
            ) == OmegaConf.to_container(self.cfg, resolve=True), (
                "Resuming run_name=%s, run_tag=%s failed: config saved to hyperparameters.yaml does not match current config"
                % (self.cfg.run_name, self.cfg.run_tag)
            )
            # resuming a run from a saved checkpoint (same name and tag)
            logging.info("Resuming checkpoint %s" % self.checkpoint_path)
            self.restore_checkpoint(
                checkpoint_path=self.checkpoint_path,
                restore_optimizer=True,
                restore_scheduler=True,
                restore_dataset=True,
            )
        elif self.cfg.train.branch_from is not None:
            # branching from a saved checkpoint (different name and tag) for cooldown etc.
            logging.info("Branching from checkpoint %s" % self.cfg.train.branch_from)
            # fresh scheduler state for cooldown etc.
            self.restore_checkpoint(
                checkpoint_path=self.cfg.train.branch_from,
                restore_optimizer=True,
                restore_scheduler=False,
                restore_dataset=True,
            )
            assert (
                self.iters == self.cfg.optimizer.cooldown_from_iter
            ), "must branch from an iteration where cooldown starts"
            
        elif self.cfg.train.finetune_from is not None:
            # finetuning from a saved checkpoint (different name and tag)
            logging.info("Finetuning from checkpoint %s" % self.cfg.train.finetune_from)
            # TODO: assumes finetuning uses the same model parallel resources; needs to change
            # fresh optimizer and scheduler state
            self.restore_checkpoint(
                checkpoint_path=self.cfg.train.finetune_from,
                restore_optimizer=False,
                restore_scheduler=False,
                restore_dataset=False,
            )

    def build_and_run(self):
        # initialize profiler
        self.profiler = init_profiler(self.cfg)

        if self.world_rank == 0:
            logging.info("------------------ Config ------------------")
            _ = [logging.info(l) for l in OmegaConf.to_yaml(self.cfg).split("\n")]
            logging.info("--------------------------------------------")

        set_seed(self.cfg, self.world_size)

        # init datasets
        self.train_dataset = get_dataset(self.cfg, mode="train")
        self.val_dataset = get_dataset(self.cfg, mode="valid")

        # get necessary info for model and loss from the dataset since they
        # deal with splitting, and getting all the data shapes + metadata
        self.metadata = self.train_dataset.metadata

        # setup flops calculator
        extra_inputs = len(self.cfg.data.invariants) if self.cfg.data.invariants else 0
        self.flops_calculator = FlopsCalculator(
            self.cfg,
            h=len(self.metadata["global_latitudes"]),
            w=len(self.metadata["global_longitudes"]),
            c_in=len(self.metadata["channels"]) + extra_inputs,
            c_out=len(self.metadata["channels"]),
        )

        # init model
        self.model = get_model(self.cfg, domain_metadata=self.metadata).to(self.device)
        # do some comm checks to make sure all groups are initialized
        self.model = comm.all_model_groups_exist(self.model)

        if comm.get_size("tp-sp1-sp2") > 1:
            # some weights can be shared across GPUs - make sure they are init the same
            init_params_for_shared_weights(self.model)

        # data preprocessor (adds static features)
        self.preprocessor = get_preprocessor(self.cfg, self.train_dataset).to(
            self.device
        )

        if self.cfg.train.compile:
            # TODO fix bugs with torch compilation
            self.model = torch.compile(self.model)
            self.preprocessor = torch.compile(self.preprocessor)

        # distributed wrapper for data parallel
        if dist.is_initialized():
            # wraps model in DDP and registers any extra comm hooks if needed
            self.model = init_ddp_model_and_reduction_hooks(
                self.model,
                device_ids=[self.local_rank],
                output_device=[self.local_rank],
                backend=self.cfg.parallelism.backend,
            )


        # init optimizer and learning rate scheduler
        self.optimizer = set_optimizer(self.cfg, self.model)
        self.scheduler = set_scheduler(
            self.cfg,
            self.optimizer,
        )

        self.gscaler = GradScaler("cuda", enabled=self.cfg.train.enable_amp)

        # init loss function
        self.loss_func = get_loss(self.cfg, self.metadata).to(self.device)
        # init metrics for tracking
        self.weighted_rmse = get_metric(
            metric="weighted_rmse", metadata=self.metadata
        ).to(self.device)

        self.iters_per_epoch = (
            self.train_dataset.n_samples_total // self.cfg.data.batch_size
        )
        if self.cfg.optimizer.scheduler == "cooldown":
            self.max_iters = self.cfg.optimizer.cooldown_to_iter
        else:
            self.max_iters = self.cfg.optimizer.max_iterations

        self.iters = 0
        self.n_mbs = 0
        self.n_mbs_in_epoch = 0
        self.start_epoch = 0
        self.end_epoch = math.ceil(self.max_iters / self.iters_per_epoch)

        # fresh start, checkpoint-restart, finetune, or branch
        self.setup_checkpoint_strategy()

        # set the dataloaders now after checkpointing
        self.train_data_loader, self.train_sampler = get_data_loader(
            self.cfg,
            self.train_dataset,
            mode="train",
        )

        self.val_data_loader, self.val_sampler = get_data_loader(
            self.cfg,
            self.val_dataset,
            mode="valid",
        )

        # dump the full config to disk/wandb
        if self.world_rank == 0:
            OmegaConf.save(self.cfg, os.path.join(self.exp_dir, "hyperparameters.yaml"))
            if self.log_to_wandb:
                wandb.save(os.path.join(self.exp_dir, "hyperparameters.yaml"))

        self.epoch = self.start_epoch
        self.logs = {}
        self.train_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.grad = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.train_time = 0.0
        self.flops_consumed = 0.0
        self.iters_last_log = self.iters
        self.n_mbs_last_log = self.n_mbs
        self.log_every = (
            self.cfg.train.log_every
            if self.cfg.train.log_every is not None
            else self.cfg.train.log_frequency * self.max_iters
        )

        n_params = count_parameters(self.model)
        if self.log_to_screen:
            logging.info(self.model)
            logging.info("number of model parameters (M): {}".format(n_params))
            logging.info(
                "number of training steps per epoch: {}".format(self.iters_per_epoch)
            )
            logging.info("max iterations: {}".format(self.max_iters))
            logging.info("approx number of epochs: {}".format(self.end_epoch))
            logging.info(f"number of training samples: {self.train_dataset.n_samples_total}")
            logging.info(f"number of validation samples: {self.val_dataset.n_samples_total}")

        if dist.is_initialized():
            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.device.index])
            else:
                dist.barrier()

        if self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except ValueError:
                pass

        # launch training
        self.train()

    def log_memory_usage(self):
        """Logs the memory usage of the GPU"""
        if self.log_to_screen and self.device.type == "cuda":
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

    def train(self):
        if self.log_to_screen:
            logging.info("starting training loop...")
        self.best_loss = getattr(self, "best_loss", np.inf)
        self.best_iter = getattr(self, "best_iter", 0)
        self.logs["best_iter"] = self.best_iter
        self.train_time = time.time()
        self.train_loss.zero_()
        self.grad.zero_()

        if self.log_to_screen:
            logging.info("----------- Memory before training ---------------")
            self.log_memory_usage()

        app_time = time.time()

        for epoch in range(self.start_epoch, self.end_epoch):
            start = time.time()
            self.epoch = epoch
            # Set current epoch in profiler
            self.profiler.set_epoch(epoch)

            if self.train_sampler is not None:
                # shuffles data before every epoch for pytorch dataloader
                self.train_sampler.set_epoch(epoch)

            # training
            epoch_time = time.time()
            nsteps = self.train_one_epoch()
            epoch_time = time.time() - epoch_time

            if self.log_to_screen:
                logging.info(
                    f"Epoch {epoch} completed; Training Steps: {nsteps}; Time: {epoch_time:.2f}s"
                )

        if self.log_to_wandb:
            wandb.finish()

        app_time = time.time() - app_time
        if self.log_to_screen:
            logging.info(f"Application time: {app_time:.2f}s")

        self.sync_devices()

        if self.nvml_handle is not None:
            pynvml.nvmlShutdown()

    def get_norm(self, model, vector_type="weights"):
        """
        logs the l2 norm of the gradient vector or weights
        type: "weights" or "grad"
        """
        norm = torch.zeros(1, dtype=torch.float32, device=self.device)
        for p in model.parameters():
            if vector_type == "weights":
                local_norm = p.detach().float().square().sum()
            elif vector_type == "grad":
                if p.grad is None:
                    continue
                local_norm = p.grad.detach().float().square().sum()
            else:
                raise ValueError(f"Invalid vector_type: {vector_type}")
            for sharded_comm in p.comm_metadata["sharded"]:
                if sharded_comm is not None:
                    dist.all_reduce(
                        local_norm,
                        op=dist.ReduceOp.SUM,
                        group=comm.get_group(sharded_comm),
                    )
            norm += local_norm
        return norm.sqrt().item()

    def clip_grad_norm(self, model, max_norm=1.0):
        """clips the gradient norm of the model"""
        if max_norm:
            norm = self.get_norm(model, vector_type="grad")
            if norm > max_norm:
                for p in model.parameters():
                    p.grad.mul_(max_norm / (norm + 1e-6))
        return

    def sync_devices(self):
        # sync all devices
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if dist.is_initialized():
            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.device.index])
            else:
                dist.barrier()

    def validate_and_checkpoint(self):
        """Validates the model and checkpoints the model"""
        self.sync_devices()
        self.logs["train_time"] = time.time() - self.train_time
        # get validation metrics
        self.logs["val_time"], self.logs["val_nsteps"], fields_to_plot = (
            self.val_one_epoch()
        )

        log_time = time.time()
        # mult by 3 for fwd and bwd pass
        self.flops_consumed = self.flops_calculator.flops() * 3
        self.flops_consumed *= self.cfg.train.num_rollout_steps
        self.flops_consumed *= self.n_mbs * comm.get_size("dp")
        self.logs["flops_consumed"] = self.flops_consumed
        self.logs["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self.logs["wt_norm"] = self.get_norm(self.model, vector_type="weights")
        # keep track of best model according to validaion loss
        if self.logs["val_loss"] <= self.best_loss:
            is_best_loss = True
            self.best_loss = self.logs["val_loss"]
            self.best_iter = self.iters
        else:
            is_best_loss = False
        self.logs["best_val_loss"] = self.best_loss
        self.logs["best_iter"] = self.best_iter
        # running average of train loss and grad
        n_mbs_since_log = max(1, self.n_mbs - self.n_mbs_last_log)
        iters_since_log = max(1, self.iters - self.iters_last_log)
        train_loss = self.train_loss / n_mbs_since_log
        grad = self.grad / iters_since_log
        # reduce metrics from all ranks
        with profile_range("train metrics reduce"):
            if dist.is_initialized():
                dist.all_reduce(
                    train_loss,
                    op=torch.distributed.ReduceOp.AVG,
                    group=comm.get_group("dp"),
                )
                dist.all_reduce(
                    grad,
                    op=torch.distributed.ReduceOp.AVG,
                    group=comm.get_group("dp"),
                )
        self.logs["train_loss"] = train_loss.detach().cpu().numpy()
        self.logs["grad"] = grad.detach().cpu().numpy()
        if self.log_to_wandb:
            wandb.log(self.logs, step=self.iters, commit=True)

        # write logs to file as well
        if comm.get_world_rank() == 0:
            self.write_logs()

        # save checkpoint (if best iter additionally save the best iter too)
        if self.cfg.train.save_checkpoint:
            with profile_range("save checkpoint"):
                self.save_checkpoint(self.checkpoint_path, is_best=is_best_loss)

        log_time = time.time() - log_time

        # some print statements
        if self.log_to_screen:
            logging.info(
                "##########################################################################"
            )
            logging.info(f"Iteration {self.iters} Summary:")
            logging.info(
                f"Timing: Train={self.logs['train_time']:.2f}s, Val={self.logs['val_time']:.2f}s, Log={log_time:.2f}s"
            )
            logging.info(
                f"Steps: Train={self.iters}, Mbs consumed={self.n_mbs}, Val={self.logs['val_nsteps']}, Grad Accum={self.gradient_accum_steps}"
            )
            logging.info(
                f"Losses: Train={float(self.logs['train_loss']):.6f}, Val={float(self.logs['val_loss']):.6f}"
            )
            logging.info(f"Training FLOPS consumed: {self.flops_consumed:.2f} TFlops")
            self.log_memory_usage()
            logging.info(
                "##########################################################################"
            )
        self.sync_devices()
        self.iters_last_log = self.iters
        self.n_mbs_last_log = self.n_mbs
        self.model.train()  # back to training

    def train_one_epoch(self):
        self.model.train()

        nsteps = 0
        nsteps_accum = 0  # number of steps that actually step the optimizer
        n_batches = len(self.train_data_loader)
        if self.cfg.data.loader == "dali" and self.epoch == self.train_dataset.ckpt_epoch:
            # dali does ckpt resume skipping inside the external source callback,
            # not through a sampler, so len(dataloader) still reports the full epoch.
            # adjust the accumulation cutoff for the resumed epoch to avoid a partial
            # grad accum stuff carrying into the next epoch.
            resume_skip_batches = max(
                self.train_dataset.resume_skip_batches,
                self.train_dataset.iterations_to_skip,
            )
            n_batches = max(0, n_batches - resume_skip_batches)
        n_batches_to_accum = n_batches - (n_batches % self.gradient_accum_steps)

        for batch_idx, batch in enumerate(self.train_data_loader):
            if torch.isnan(self.train_loss):
                logging.warn("NaN detected; stopping")
                break

            if self.iters >= self.max_iters:
                if self.log_to_screen:
                    logging.info(
                        f"Reached max iterations {self.iters}/{self.max_iters}; stopping"
                    )
                if not (self.iters % self.log_every == 0):
                    # breaking out an iter where we haven't checkpointed
                    # may happen in cooldown
                    self.validate_and_checkpoint()
                break

            # check when we should step the optimizer based on gradient accumulation
            is_optimizer_step = (batch_idx + 1) % self.gradient_accum_steps == 0

            # early stop if we finished the last grad accumulation step
            if (batch_idx + 1) > n_batches_to_accum:
                break

            nsteps += 1
            self.n_mbs += 1
            self.n_mbs_in_epoch += 1

            sync_context = (
                self.model.no_sync()
                if dist.is_initialized() and not is_optimizer_step
                else nullcontext()
            )
            with profile_range(f"iter {self.iters}"):
                with sync_context:
                    with profile_range("data preproc"):
                        inputs, targets = self.preprocessor(batch, device=self.device)

                    with profile_range("forward pass"):
                        # autocast for mixed precision
                        with torch.autocast(
                            "cuda", dtype=torch.float16, enabled=self.cfg.train.enable_amp
                        ):
                            gen = self.model(inputs)
                            loss = self.loss_func(gen, targets)
                            self.train_loss += loss.detach()  # for logging
                            # scale loss when gradient is accumulated
                            loss = loss / self.gradient_accum_steps

                    with profile_range("backward pass"):
                        self.gscaler.scale(loss).backward()

                with profile_range("optimizer"):
                    if is_optimizer_step:
                        nsteps_accum += 1
                        self.gscaler.unscale_(self.optimizer)
                        self.clip_grad_norm(
                            self.model, max_norm=self.cfg.train.clip_grad_norm
                        )
                        self.gscaler.step(self.optimizer)
                        self.gscaler.update()
                        self.iters += 1
                        if self.scheduler is not None:
                            self.scheduler.step()
                        with profile_range("grad norm"):
                            self.grad += self.get_norm(self.model, vector_type="grad")
                        self.model.zero_grad(set_to_none=True)

            if is_optimizer_step:
                # checkpoint and log every few optimizer iterations
                if self.iters % self.log_every == 0:
                    self.validate_and_checkpoint()
                    # log as averages since the last log
                    self.train_loss.zero_()
                    self.grad.zero_()
                    self.train_time = time.time()  # start timing and go back to training

                # save ckpts at the begining of cooldown in case we need to change the loss later
                if self.cfg.optimizer.scheduler == "cooldown":
                    cooldown_steps = int(self.cfg.optimizer.cooldown_to_iter * self.cfg.optimizer.cooldown_fraction)
                    cooldown_start = self.cfg.optimizer.cooldown_to_iter - cooldown_steps
                    if self.iters == cooldown_start:
                        ckpt_path = self.checkpoint_path.replace(".tar", "_base.tar")
                        self.save_checkpoint(ckpt_path, is_best=False)

                if self.iters >= self.max_iters:
                    if not (self.iters % self.log_every == 0):
                        self.validate_and_checkpoint()
                    break

        # reset here to avoid checkpointing issues
        self.n_mbs_in_epoch = 0

        return nsteps

    def val_one_epoch(self):
        self.model.eval()
        val_time = time.time()

        nc = self.metadata["n_channels"]
        mult = torch.as_tensor(self.train_dataset.stds[0, :, 0, 0]).to(self.device)
        valid_loss = torch.zeros((1), dtype=torch.float32, device=self.device)
        valid_weighted_rmse = torch.zeros((nc), dtype=torch.float32, device=self.device)
        nsteps = 0
        fields_to_plot = []
        with torch.inference_mode():
            for batch in self.val_data_loader:
                nsteps += 1
                with profile_range(f"val_step {nsteps}"):
                    with profile_range("val_data preproc"):
                        inputs, targets = self.preprocessor(batch, device=self.device)

                    with profile_range("val_forward pass"):
                        with torch.autocast(
                            "cuda", dtype=torch.float16, enabled=self.cfg.train.enable_amp
                        ):
                            gen = self.model(inputs)

                    valid_loss += self.loss_func(gen, targets)
                    valid_weighted_rmse += self.weighted_rmse(gen, targets)

        valid_loss /= nsteps
        valid_weighted_rmse /= nsteps
        valid_weighted_rmse *= mult

        with profile_range("val metrics reduce"):
            if dist.is_initialized():
                dist.all_reduce(
                    valid_loss,
                    op=torch.distributed.ReduceOp.AVG,
                    group=comm.get_group("dp"),
                )
                # rmse has sqrt so the metric computation will
                # take care of the reduce over sp1-sp2
                dist.all_reduce(
                    valid_weighted_rmse,
                    op=torch.distributed.ReduceOp.AVG,
                    group=comm.get_group("dp"),
                )

        self.logs["val_loss"] = valid_loss.detach().cpu().numpy()
        # track specific variables
        for var in self.cfg.data.track_channels:
            idx = self.val_dataset.era5_channels.index(var)
            self.logs.update(
                {f"val_rmse_{var}": valid_weighted_rmse[idx].detach().cpu().numpy()}
            )

        self.sync_devices()
        val_time = time.time() - val_time

        return val_time, nsteps, fields_to_plot

    def write_logs(self):
        """Writes the logs to a file"""
        log_file = os.path.join(self.exp_dir, "logs.txt")
        file_exists = os.path.exists(log_file)
        row = {"iter": self.iters}

        for k, v in self.logs.items():
            # convert tensors to scalars if needed
            row[k] = v.item() if hasattr(v, "item") else v

        with open(log_file, "a") as f:
            if not file_exists:
                header = ",".join(row.keys())
                f.write(header + "\n")
            values = ",".join(str(row[k]) for k in row.keys())
            f.write(values + "\n")

    def modify_checkpoint_path(self, checkpoint_path):
        """checkpoint filename based on the parallelization used"""
        tp_size = comm.get_size("tp")
        sp_size = comm.get_size("sp1") * comm.get_size("sp2")
        if tp_size > 1:
            # each tp gpu has a different checkpoint
            checkpoint_path = checkpoint_path.replace(
                "ckpt", "ckpt_tp{}".format(comm.get_rank("tp"))
            )
        if not getattr(self.cfg.model, "coord_pos_embed", False) and sp_size > 1:
            # each sp gpu has a different checkpoint since pos embed shards here
            checkpoint_path = checkpoint_path.replace(
                "ckpt",
                "ckpt_sp1{}_sp2{}".format(comm.get_rank("sp1"), comm.get_rank("sp2")),
            )
        return checkpoint_path

    def save_checkpoint(self, checkpoint_path, is_best=False, save_epoch=None):
        checkpoint_path = self.modify_checkpoint_path(checkpoint_path)
        checkpoint = {
            "iters": self.iters,
            "n_mbs": self.n_mbs,
            "n_mbs_in_epoch": self.n_mbs_in_epoch,
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "grad_scaler_state_dict": self.gscaler.state_dict(),
            "best_loss": self.best_loss,
            "best_iter": self.best_iter,
            "micro_batch_size": self.cfg.parallelism.micro_batch_size,
        }
        # only save from root ranks of each parallel group
        should_save = False
        if comm.get_rank("dp") == 0:
            if "tp" in checkpoint_path or comm.get_rank("tp") == 0:
                if "sp1" in checkpoint_path or comm.get_rank("sp1") == 0:
                    if "sp2" in checkpoint_path or comm.get_rank("sp2") == 0:
                        should_save = True

        if should_save:
            logging.info(
                f"Rank {comm.get_world_rank()} saving checkpoint to {checkpoint_path}"
            )
            torch.save(checkpoint, checkpoint_path)
            if is_best:
                torch.save(checkpoint, checkpoint_path.replace(".tar", "_best.tar"))
            torch.save(
                checkpoint,
                checkpoint_path.replace(".tar", f"_iter{self.iters}.tar"),
            )

    def restore_checkpoint(
        self,
        checkpoint_path,
        restore_optimizer=True,
        restore_scheduler=True,
        restore_dataset=False,
    ):
        checkpoint_path = self.modify_checkpoint_path(checkpoint_path)
        map_location = (
            "cuda:{}".format(self.local_rank) if self.device.type == "cuda" else "cpu"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            if not all(key.startswith("module.") for key in checkpoint["model_state"]):
                raise
            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key.removeprefix("module.")
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)

        if restore_optimizer:
            # restore the optimizer and lr scheduler state dicts
            # for finetuning, this is skipped, only model init
            self.iters = checkpoint["iters"]
            self.n_mbs = checkpoint["n_mbs"]
            self.n_mbs_in_epoch = checkpoint["n_mbs_in_epoch"]
            self.start_epoch = checkpoint["epoch"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.gscaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
            self.best_loss = checkpoint["best_loss"]
            self.best_iter = checkpoint["best_iter"]

        if restore_scheduler:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if restore_dataset:
            self.iters = checkpoint["iters"]
            self.n_mbs = checkpoint["n_mbs"]
            self.n_mbs_in_epoch = checkpoint["n_mbs_in_epoch"]
            self.start_epoch = checkpoint["epoch"]
            dataset_ckpt = {
                "n_mbs_in_epoch": self.n_mbs_in_epoch,
                "epoch": self.start_epoch,
                "micro_batch_size": int(checkpoint.get("micro_batch_size", 1)),
            }
            self.train_dataset.set_dataset_state(dataset_ckpt)

        if self.log_to_screen:
            logging.info(f"Restored checkpoint iteration = {self.iters}")
            logging.info(f"Restored checkpoint micro-batches = {self.n_mbs}")
            logging.info(f"Restored checkpoint epoch = {self.start_epoch}")
            logging.info(
                f"Restored checkpoint micro-batches within epoch = {self.n_mbs_in_epoch}"
            )
