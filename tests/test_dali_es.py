"""DALI external-source dataloader: a single epoch must emit, across DP ranks,
exactly the ground-truth set of F-step sample windows for the training year(s).

The dataset is configured with return_timestamp=True so __call__ emits the
raw HDF5 time values for each sample's full F-step window; we build the
expected set of windows from the global time list and assert set equality
after an all-gather across DP. This catches duplicates, missing samples,
wrong stride, and wrong start positions in one shot.
"""

import gc
import glob
import os
import unittest

import h5py
import numpy as np
import torch
import torch.distributed as dist
from parameterized import parameterized

from utils import comm
from utils.dataloaders.era5hdf5_dali import (
    Era5HDF5DatasetDALI,
    ERA5HDF5DALIDataLoader,
)
from test_helpers import setup_test_class


TRAIN_YEAR = 2018
DT_SCALE = 6
TCW = 1
NRS = 1
LIMIT_NSAMPLES = 64  # keeps the test quick; divisible by all swept mbs


class TestDaliES(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_test_class(cls)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group(None)

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def _make_dataset(self, mbs):
        """Restrict training data to a single year by parking all other
        years in valid_years."""
        years = sorted(
            int(os.path.basename(p).replace(".h5", ""))
            for p in glob.glob("/data/latlon_025deg_hdf5_1h/*.h5")
        )
        held_out = [y for y in years if y != TRAIN_YEAR]
        return Era5HDF5DatasetDALI(
            name="latlon_025deg_hdf5_1h",
            tag="v1.0",
            mode="train",
            dt_scale=DT_SCALE,
            temporal_context_window=TCW,
            num_rollout_steps=NRS,
            model_patch_size=4,
            limit_nsamples=LIMIT_NSAMPLES,
            valid_years=held_out,
            test_years=[],
            invariants=[],
            micro_batch_size=mbs,
            seed=42,
            read_local=False,
            return_timestamp=True,
        )

    @staticmethod
    def _ground_truth_times(era5_paths):
        """Mirror ``Era5HDF5DatasetDALI.compute_total_samples``: read f['time']
        from each training file, lift to datetime64, concatenate, then cast
        to int64 the way the dataset does at the DALI emit boundary."""
        chunks = []
        for path in era5_paths:
            with h5py.File(path, "r") as f:
                d = f["time"][:]
                chunks.append(np.array(
                    [np.datetime64("1900-01-01") + np.timedelta64(int(ts), "h") for ts in d]
                ))
        return np.concatenate(chunks).astype(np.int64)

    @parameterized.expand([
        # (mbs, num_workers)
        [1, 1],
        [1, 2],
        [2, 2],
        [2, 8],
    ])
    def test_one_epoch_unique_and_covers(self, mbs, num_workers):
        dataset = self._make_dataset(mbs)
        dp_size = comm.get_size("dp")
        per_rank_count = (dataset.n_samples_shard // mbs) * mbs
        F = TCW + NRS

        # ground truth: for each valid global_idx in [0, n_samples_total),
        # the expected F-step window is global_times[idx + i*dt_scale] for
        # i in range(F). Build the full set of expected windows.
        global_times = self._ground_truth_times(dataset.era5_paths)
        starts = np.arange(dataset.n_samples_total, dtype=np.int64)
        offsets = np.arange(F, dtype=np.int64) * DT_SCALE
        expected_windows = global_times[starts[:, None] + offsets[None, :]]  # [n_total, F]
        expected_set = set(map(tuple, expected_windows.tolist()))

        # this test config is set up so drop_last loses nothing; otherwise
        # the set-equality check below would need to be relaxed to subset
        self.assertEqual(
            per_rank_count * dp_size, dataset.n_samples_total,
            f"test config must cover full ground truth "
            f"(per_rank_count={per_rank_count}, dp_size={dp_size}, "
            f"n_samples_total={dataset.n_samples_total})",
        )

        loader = ERA5HDF5DALIDataLoader(
            dataset=dataset,
            micro_batch_size=mbs,
            num_data_workers=num_workers,
            mode="train",
            seed=42,
        )

        # collect full F-step windows emitted to this rank
        local_chunks = []
        for data, time in loader:
            # data: [B, F, C, H, W], time: [B, F] int64
            local_chunks.append(time.detach().cpu().numpy().astype(np.int64))
        local_windows = (
            np.concatenate(local_chunks, axis=0)
            if local_chunks else np.zeros((0, F), dtype=np.int64)
        )

        # within-rank: right count, no duplicate windows
        self.assertEqual(
            local_windows.shape[0], per_rank_count,
            f"rank {self.world_rank}: expected {per_rank_count} samples, "
            f"got {local_windows.shape[0]} (mbs={mbs}, num_workers={num_workers})",
        )
        local_set = set(map(tuple, local_windows.tolist()))
        self.assertEqual(
            len(local_set), local_windows.shape[0],
            f"rank {self.world_rank}: duplicate windows within epoch "
            f"(mbs={mbs}, num_workers={num_workers})",
        )

        # all-gather windows across DP, then compare the union to ground truth
        if dp_size > 1:
            local_t = torch.from_numpy(local_windows).to(self.device)
            gathered = [torch.empty_like(local_t) for _ in range(dp_size)]
            dist.all_gather(gathered, local_t, group=comm.get_group("dp"))
            all_windows = np.concatenate([t.cpu().numpy() for t in gathered], axis=0)
        else:
            all_windows = local_windows

        self.assertEqual(
            all_windows.shape[0], dataset.n_samples_total,
            f"total emitted count {all_windows.shape[0]} != "
            f"n_samples_total {dataset.n_samples_total} "
            f"(mbs={mbs}, num_workers={num_workers})",
        )

        all_set = set(map(tuple, all_windows.tolist()))
        # if self.world_rank == 0:
        #     print(f"all_set: {sorted(all_set)[:10]}")
        #     print(f"expected_set: {sorted(expected_set)[:10]}")
        self.assertEqual(
            all_set, expected_set,
            f"emitted windows do not match ground truth "
            f"(missing={len(expected_set - all_set)}, "
            f"extra={len(all_set - expected_set)}, "
            f"mbs={mbs}, num_workers={num_workers})",
        )


if __name__ == "__main__":
    unittest.main()
