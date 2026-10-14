# Inversion on Gefion (cluster-specific notes; this branch is never merged)

- **Layout:** code, `.venv` and `runs/` live in the main checkout `~/dev/verbalization`; only `run_grid.sbatch` comes
  from this branch (worktree `.claude/worktrees/gefion`). Submit from the main checkout so `$SLURM_SUBMIT_DIR` is it.
- **Data:** `INVERSION_DATA=/mnt/weka/home/gma/data/pile` (9 models, val, shards 0–1). Compute nodes have no local
  disk (`TmpDisk=0`); the loader does ~30 batches/s of 512 on Weka.
- **venv:** built on uv-managed Python 3.12 (`~/.local/share/uv/python`). `/opt/conda` is 3.12 on the login node but
  3.13 on the GPU nodes, so a venv on it breaks in jobs.
- **Slurm:** `h100-reserved`, QOS `normal`, no wall-time cap; the sbatch uses 7 days and self-requeues before that.

## Launched
- 2026-10-14, job `90360` (running): random init, trained on a uniform mix of **s1..s7**, **s8, s9 held out**; batch
  512, layer 6, `runs/b512/rand_inv_mixs1to7_L6/`; val groups `s1-7` (seen) and `s8-9` (held out). From main `2c41f3f`:
  `RUN_ARGS="--init random --source mixs1to7 --val_groups s1,s2,s3,s4,s5,s6,s7;s8,s9" sbatch --array=0 -J inv-mix7-rand $G ${=F}`
  (same `F` as below). Jobs 90346/90347 (9-way mix) were cancelled at steps ~30k/29k and checkpointed.
- 2026-10-14, jobs `90346` / `90347` (running): 9-way mix of s1..s9 at layer 6, **batch 512**, outputs in `runs/b512/`;
  random init (val lumped as one `s1-9` group) and s1 init (val `s1` and lumped `s2-9`). From main `b5da069`:
  ```bash
  G=.claude/worktrees/gefion/inversion/run_grid.sbatch
  F="--batch 512 --out_dir runs/b512 --log_every 10 --print_every 50 --print_examples 8 --val_every 500 --val_dump_every 10000 --ckpt_every 20000"
  RUN_ARGS="--init random --source mixs1to9 --val_groups s1,s2,s3,s4,s5,s6,s7,s8,s9" sbatch --array=0 -J inv-mix9-rand $G ${=F}
  RUN_ARGS="--init s1 --source mixs1to9 --val_groups s1;s2,s3,s4,s5,s6,s7,s8,s9" sbatch --array=0 -J inv-mix9-s1 $G ${=F}
  ```
  (`${=F}`: the login shell is zsh, which does not word-split an unquoted `$F`.) ~13-18 steps/s.
- 2026-10-14, job `90217` (batch-128 `s` grid, `runs/b128/`) cancelled at steps ~97k-103k; it checkpointed on scancel.
- 2026-10-14, job `90296` (cancelled at steps ~85k-103k after ~1h16m; it checkpointed on scancel): the `90217` grid above with only the layer changed to **`final`** (after
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
