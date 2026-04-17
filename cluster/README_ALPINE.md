# Alpine Runbook

This repo is ready to run on the University of Colorado Alpine cluster with Slurm.

## 1. Put the repo on Alpine

Use a persistent directory on the login node. This path is simple and predictable:

```bash
mkdir -p ~/projects
cd ~/projects
git clone <YOUR_GIT_REPO_URL> receipt-parsing
cd receipt-parsing
```

If this code is not in a remote git repo yet, copy it from your laptop instead:

```bash
mkdir -p ~/projects/receipt-parsing
```

From your laptop:

```bash
rsync -av --progress "/Users/mani/Documents/Studies/Neural networks and deep learning/receipt-parsing/" <YOUR_IDENTIKEY>@alpine.rc.colorado.edu:~/projects/receipt-parsing/
```

After that, on Alpine:

```bash
cd ~/projects/receipt-parsing
```

## 2. Load the Slurm module and inspect GPU partitions

On the login node:

```bash
module load slurm/alpine
sinfo --Format Partition,Gres | grep -E 'aa100|ami100|al40|atesting_a100|atesting_mi100'
```

The official Alpine GPU partitions are:

- `atesting_a100` for short GPU smoke tests
- `aa100` for full A100 runs
- `ami100` for AMD MI100 runs
- `al40` for NVIDIA L40 runs

## 3. Create the Python environment once

From the repo root on Alpine:

```bash
bash cluster/setup_alpine_env.sh
```

This creates:

- `.venv`
- all Python dependencies from `requirements.txt`

## 4. Submit the smoke test first

Make sure you are in the repo root:

```bash
cd ~/projects/receipt-parsing
mkdir -p logs outputs
sbatch --account=<YOUR_ALLOCATION> cluster/alpine_smoke.sbatch
```

Why this job first:

- requests `atesting_a100`
- 1 GPU
- 20 training steps only
- 1 epoch cap
- no validation during the smoke run
- frequent logging

Useful commands after submission:

```bash
squeue -u $USER
tail -f logs/smoke-<JOB_ID>.log
tail -f slurm-<JOB_ID>.out
```

The smoke run writes checkpoints and trainer output under:

```bash
outputs/smoke/<JOB_ID>/
```

## 5. Submit the longer training run after the smoke test succeeds

```bash
cd ~/projects/receipt-parsing
sbatch --account=<YOUR_ALLOCATION> cluster/alpine_full.sbatch
```

This job uses:

- partition `aa100`
- 1 A100 GPU
- 30 epochs
- micro-batch size `1`
- gradient accumulation `2`
- effective batch size `2`

Logs:

```bash
tail -f logs/full-<JOB_ID>.log
tail -f slurm-<JOB_ID>.out
```

Outputs:

```bash
outputs/full/<JOB_ID>/
```

## 6. What to check if a job fails

Check queue status:

```bash
squeue -u $USER
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS
```

Check the logs:

```bash
tail -100 logs/smoke-<JOB_ID>.log
tail -100 logs/full-<JOB_ID>.log
tail -100 slurm-<JOB_ID>.out
```

Cancel a job:

```bash
scancel <JOB_ID>
```

## 7. Recommended order

1. Copy or clone the repo into `~/projects/receipt-parsing`
2. Run `bash cluster/setup_alpine_env.sh`
3. Submit the smoke job with `sbatch --account=<YOUR_ALLOCATION> cluster/alpine_smoke.sbatch`
4. Confirm the smoke log shows training progress
5. Submit the full job with `sbatch --account=<YOUR_ALLOCATION> cluster/alpine_full.sbatch`

## Notes

- The job scripts set `HF_HOME` to `./.hf-cache` inside the repo so model and dataset downloads are reused across runs.
- The smoke test uses the official Alpine GPU testing partition. According to the Alpine docs, `atesting_a100` is limited to 1 A100 MIG slice and a short runtime, which is appropriate for a workflow check.
- Full training is pointed at `aa100`, which is Alpine’s NVIDIA A100 partition.
