# Inversion on Gefion (cluster-specific notes; this branch is never merged)

- **Layout:** code, `.venv` and `runs/` live in the main checkout `~/dev/verbalization`; only `run_grid.sbatch` comes
  from this branch (worktree `.claude/worktrees/gefion`). Submit from the main checkout so `$SLURM_SUBMIT_DIR` is it.
- **Data:** `INVERSION_DATA=/mnt/weka/home/gma/data/pile` (9 models, val, shards 0–1). Compute nodes have no local
  disk (`TmpDisk=0`); the loader does ~30 batches/s of 512 on Weka.
- **venv:** built on uv-managed Python 3.12 (`~/.local/share/uv/python`). `/opt/conda` is 3.12 on the login node but
  3.13 on the GPU nodes, so a venv on it breaks in jobs.
- **Slurm:** `h100-reserved`, QOS `normal`, no wall-time cap; the sbatch uses 7 days and self-requeues before that.

## Launched
- 2026-10-14, job `90296` (running): the `90217` grid below with only the layer changed to **`final`** (after
  `final_layer_norm`), from main `66b4b7f` (same training code as `be87c95`), runs `runs/b128/*_Lfinal`:
  ```bash
  cd ~/dev/verbalization && FAMILY=s sbatch .claude/worktrees/gefion/inversion/run_grid.sbatch --layer final \
      --batch 128 --out_dir runs/b128 --log_every 10 --print_every 10 --print_examples 8 --val_every 250 \
      --val_dump_every 2500 --ckpt_every 50000
  .venv/bin/python -m inversion.watch --runs_dir runs/b128 --runs "*_Lfinal"   # live view
  ```
- 2026-10-14, job `90217` (running): `s` family, layer 6, **batch 128**, from main `be87c95`, outputs in `runs/b128/`:
  ```bash
  cd ~/dev/verbalization && FAMILY=s sbatch .claude/worktrees/gefion/inversion/run_grid.sbatch --batch 128 \
      --out_dir runs/b128 --log_every 10 --print_every 10 --print_examples 8 --val_every 250 --val_dump_every 2500 \
      --ckpt_every 50000
  .venv/bin/python -m inversion.watch --runs_dir runs/b128        # live view
  ```
  ~24–34 steps/s per run.
- 2026-10-14, job `90192` (cancelled after ~40k steps; batch 512 learned too fast to see dynamics; outputs in
  `runs/*_L6/`. It was cancelled before the sbatch SIGTERM trap existed, so it did not checkpoint on `scancel`): `s` family, layer 6, from main `0d879ef`:
  ```bash
  cd ~/dev/verbalization && FAMILY=s sbatch .claude/worktrees/gefion/inversion/run_grid.sbatch \
      --print_every 500 --val_every 2000 --ckpt_every 20000
  ```
  ~19 steps/s per run. At step 2000 every run is at 98–99% val token accuracy on its training source(s) and ~0% on
  the other seed.
