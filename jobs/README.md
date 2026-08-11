# Machine-specific job scripts

These are reference training scripts for the supported machines. The repository
root keeps `submit_batch.sh` configured for Perlmutter by default, and
`submit_batch_inference.sh` remains the default Perlmutter inference script.

To use a different machine, copy both of its reference files to the repository
root, replacing `submit_batch.sh` and `batchsub.py`. For example:

```bash
cp jobs/submit_batch_vista.sh submit_batch.sh
cp jobs/batchsub_vista.py batchsub.py
```

Run `batchsub.py` as usual after copying the files. Each reference defines a
user-editable `scheduler_selector`: `("-C", "gpu&hbm40g")` for Perlmutter and
`("-p", "gh")` for Vista. This command-line setting overrides the corresponding
default in `submit_batch.sh`.

To restore the default Perlmutter files:

```bash
cp jobs/submit_batch_perlmutter.sh submit_batch.sh
cp jobs/batchsub_perlmutter.py batchsub.py
```

Submit jobs from the repository root because these scripts invoke `train.py`
using a relative path. Machine-specific paths, accounts, images, and Slurm
resources may need to be updated before use.
