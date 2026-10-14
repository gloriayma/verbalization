# Inversion on Gefion (cluster-specific notes; this branch is never merged)

- **Layout:** code, `.venv` and `runs/` live in the main checkout `~/dev/verbalization`; only `run_grid.sbatch` comes
  from this branch (worktree `.claude/worktrees/gefion`). Submit from the main checkout so `$SLURM_SUBMIT_DIR` is it.
- **Data:** `INVERSION_DATA=/mnt/weka/home/gma/data/pile` (9 models, val, shards 0–1). Compute nodes have no local
  disk (`TmpDisk=0`); the loader does ~30 batches/s of 512 on Weka.
- **venv:** built on uv-managed Python 3.12 (`~/.local/share/uv/python`). `/opt/conda` is 3.12 on the login node but
  3.13 on the GPU nodes, so a venv on it breaks in jobs.
- **Slurm:** `h100-reserved`, QOS `normal`, no wall-time cap; the sbatch uses 7 days and self-requeues before that.

## Launched
- 2026-10-14, job `90192`: `s` family, layer 6, from main `0d879ef`:
  ```bash
  cd ~/dev/verbalization && FAMILY=s sbatch .claude/worktrees/gefion/inversion/run_grid.sbatch \
      --print_every 500 --val_every 2000 --ckpt_every 20000
  ```
  ~19 steps/s per run. At step 2000 every run is at 98–99% val token accuracy on its training source(s) and ~0% on
  the other seed.
