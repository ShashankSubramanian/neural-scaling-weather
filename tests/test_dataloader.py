import torch
import torch.distributed as dist
import unittest
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import time
from parameterized import parameterized

from utils import comm
from utils.data_utils import PreProcessor, ResumeSampler
from utils.dataloaders.era5hdf5 import Era5HDF5Dataset
from utils.dataloaders.era5hdf5_dali import Era5HDF5DatasetDALI, ERA5HDF5DALIDataLoader
from test_helpers import setup_test_class


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_class(cls)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group(None)
    
    def tearDown(self):
        """Clean up after each test method"""
        # Clean up CUDA resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()

    # zarr not supported for now
    @parameterized.expand([
        # ["dali", "hdf5", 16, 4, ["lsm", "orog", "coslat"], False],
        # ["pytorch", "hdf5", 16, 4, ["lsm", "orog", "coslat"], False],
        ["dali", "hdf5", 4, 2, ["lsm", "orog", "coslat"], True],
        ["pytorch", "hdf5", 4, 2, ["lsm", "orog", "coslat"], True],
    ])
    def test_dataloader(self, dataloader_type, file_type, batch_size, 
                       temporal_context_window, invariants, read_local, generate_plots=True):
        """Test dataloader with different configurations; H,W are hardcoded for ERA5"""
        # micro_batch_size is the batch size per GPU (local in this case..)
        micro_batch_size = batch_size // comm.get_size("dp")
        limit_nsamples = 34
        start_epoch = 2
        checkpoint = {
            "iters_in_epoch": 4,
            "epoch": start_epoch,
            "micro_batch_size": micro_batch_size,
        }
        use_checkpoint = True
        # Create dataset
        if dataloader_type == "dali":
            dataset = Era5HDF5DatasetDALI(
                name="latlon_025deg_hdf5_1h",
                tag="v1.0",
                mode="train",
                dt_scale=6,
                temporal_context_window=temporal_context_window,
                model_patch_size=4,
                limit_nsamples=limit_nsamples,
                valid_years=[2018, 2019],
                test_years=[2020, 2021, 2022],
                invariants=invariants,
                micro_batch_size=micro_batch_size,
                seed=42,
                read_local=read_local,
            )
            if use_checkpoint:
                dataset.set_dataset_state(checkpoint)
            dataloader = ERA5HDF5DALIDataLoader(
                dataset=dataset,
                micro_batch_size=micro_batch_size,
                num_data_workers=2,
                mode="train",
                seed=42,
            )
            sampler = None
        else:
            if file_type == "hdf5":
                dataset = Era5HDF5Dataset(
                    name="latlon_025deg_hdf5_1h",
                    tag="v1.0",
                    mode="train",
                    dt_scale=6,
                    limit_nsamples=limit_nsamples,
                    temporal_context_window=temporal_context_window,
                    model_patch_size=4,
                    valid_years=[2018, 2019],
                    test_years=[2020, 2021, 2022],
                    invariants=invariants,
                )

            if use_checkpoint:
                dataset.set_dataset_state(checkpoint)

            if comm.get_size("dp") > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    shuffle=True,
                    num_replicas=comm.get_size("dp"),
                    rank=comm.get_rank("dp"),
                    drop_last=True,
                    seed=42,
                )
            else:
                sampler = None
            
            if use_checkpoint and dataset.resume_skip_batches > 0:
                sampler = ResumeSampler(
                    sampler=sampler,
                    resume_epoch=dataset.ckpt_epoch,
                    resume_iter=dataset.resume_skip_batches,
                )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=micro_batch_size,
                num_workers=2,
                shuffle=(sampler is None),
                sampler=sampler,
                drop_last=True,
                pin_memory=torch.cuda.is_available(),
            )

        
        # Create preprocessor
        preprocessor = PreProcessor(
            dataset=dataset,
            invariants=invariants,
            send_to_device=(dataloader_type != "dali"),
        ).to(self.device)
        
        if self.print_to_screen:
            print(f"Testing {dataloader_type.upper()} dataloader with {file_type} data")
            print(f"Dataset metadata: {dataset.get_metadata()}")
            print(f"Total samples: {len(dataset)}")

        # Get spatial dimensions
        H, W = dataset.metadata["spatial_dims"]
        
        # Process batches through dataloader and preprocessor
        all_inputs = []
        all_targets = []
        
        start_time = time.time()
        start_epoch = checkpoint["epoch"] if use_checkpoint else 0
        for ep in range(start_epoch, 4):
            batch_count = 0
            if sampler is not None:
                sampler.set_epoch(ep)
            for batch in dataloader:

                # do some quick norm checks
                norm = torch.norm(batch)
                if self.print_to_screen:
                    print(f"epoch = {ep}, batch = {batch_count}: norm = {norm}")

                # Check batch shape: [B, T, C, H, W]
                self.assertEqual(len(batch.shape), 5)
                self.assertEqual(batch.shape[0], micro_batch_size)
                self.assertEqual(batch.shape[1], temporal_context_window + 1)
                self.assertEqual(batch.dtype, torch.float32)
                
                # Process through preprocessor
                inputs, targets = preprocessor(batch, device=self.device)
                
                # Check preprocessor output shapes
                T, C = batch.shape[1], batch.shape[2] 
                n_invar = len(invariants) if invariants else 0
                
                # Inputs should be [B, T-1, C+C_invar, H, W]
                expected_input_shape = (micro_batch_size, T - 1, C + n_invar, H, W)
                self.assertEqual(inputs.shape, expected_input_shape)
                
                # Targets should be [B, T-1, C, H, W]
                expected_target_shape = (micro_batch_size, 1, C, H, W)
                self.assertEqual(targets.shape, expected_target_shape)
                
                if batch_count == 0 and ep == 3:
                    # save the last epoch's data for viz
                    all_inputs.append(inputs)
                    all_targets.append(targets)
                batch_count += 1
                
        
        processing_time = time.time() - start_time
        
        self.assertGreater(batch_count, 0, "No batches were loaded")
        
        if self.print_to_screen:
            print(f"Global batch size: {batch_size}")
            print(f"Micro batch size: {micro_batch_size}")
            print(f"Processed {batch_count} batches in {processing_time:.2f} seconds")
            print(f"Input shape: {all_inputs[0].shape}")
            print(f"Target shape: {all_targets[0].shape}")
        
        # Generate plots if requested
        if generate_plots and self.world_rank == 0:
            self._generate_plots(all_inputs, all_targets, dataset.era5_channels)

        
    
    def _generate_plots(self, inputs_list, targets_list, channels):
        """Generate PDF plots of the data"""
        # Create plots directory if it doesn't exist
        plot_dir = "tests/figs"
        os.makedirs(plot_dir, exist_ok=True)
            
        # Function to plot a page of 4x4 images
        def plot_page(data, start_idx, title_prefix, page_num, filename):
            fig, axes = plt.subplots(4, 4, figsize=(20, 20))
            fig.suptitle(f"{title_prefix} (Page {page_num})")
            
            for i in range(16):  # 4x4 grid
                if start_idx + i >= len(data):
                    # Remove unused subplots
                    for j in range(i, 16):
                        row, col = j // 4, j % 4
                        axes[row, col].remove()
                    break
                
                row, col = i // 4, i % 4
                im = axes[row, col].imshow(data[start_idx + i].cpu().numpy())
                axes[row, col].set_title(f"Channel {start_idx + i}")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                axes[row, col].set_xticklabels([])
                axes[row, col].set_yticklabels([])
                
                # Add colorbar to each subplot
                divider = make_axes_locatable(axes[row, col])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()
            return fig
            
        # Plot input channels for first batch
        if inputs_list:
            inputs = inputs_list[0]  # First batch
            batch_idx = 0
            time_idx = 0  # First timestep
            
            n_channels = inputs.shape[2]
            
            # Plot input channels
            for page in range((n_channels + 15) // 16):  # Ceiling division
                start_idx = page * 16
                filename = f"test_input_data_page_{page + 1}.pdf"
                plot_page(inputs[batch_idx, time_idx], start_idx, "Input Data", page + 1, filename)
            
            # Plot target channels
            if targets_list:
                targets = targets_list[0]  # First batch
                n_target_channels = targets.shape[2]
                
                for page in range((n_target_channels + 15) // 16):  # Ceiling division
                    start_idx = page * 16
                    filename = f"test_target_data_page_{page + 1}.pdf"
                    plot_page(targets[batch_idx, time_idx], start_idx, "Target Data", page + 1, filename)
            
            if self.print_to_screen:
                print(f"Generated plots in temporary directory: {plot_dir}")


if __name__ == "__main__":
    unittest.main()
