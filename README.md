# verbalization

## `olmo3_next_token.ipynb`
Raw next-token predictions from `allenai/Olmo-3-1025-7B` at revision `stage1-step1413814`, the last
pretraining-only checkpoint. There's no chat template, and nothing is added to your input (asserted in the notebook).

### Setup (already done on this cluster)
- **Python:** venv at `~/dev/verbalization/.venv` (system Python 3.10.12, torch 2.5.1+cu121, transformers 5.18).
  Jupyter kernel: "Python (verbalization .venv)".
- **Weights:** about 15 GB in `~/.cache/huggingface` (on `/mnt/nw`, shared with the GPU node).
  To re-download:
  ```bash
  ~/dev/verbalization/.venv/bin/hf download allenai/Olmo-3-1025-7B --revision stage1-step1413814
  ```
- **Hugging Face login:** not needed. The repo is public. `hf auth login` only raises download rate limits.

### Running on a GPU
The login VM has no GPU. GPUs are on the Slurm node `l40-worker` (8× L40, 48 GB). Start Jupyter there:

```bash
cd ~/dev/verbalization
srun --gres=gpu:1 -c 8 --mem=64G -t 8:00:00 --pty \
  .venv/bin/jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

Then do one of these:
- **Browser:** from your laptop, run `ssh -L 8888:l40-worker:8888 <login-host>`, then open the
  `http://127.0.0.1:8888/lab?token=...` URL that Jupyter printed.
- **VS Code (Remote-SSH to the login host):** open the notebook, choose "Select Kernel" → "Existing Jupyter Server",
  and paste `http://l40-worker:8888/lab?token=...`.

Run it headless:

```bash
srun --gres=gpu:1 -c 8 --mem=64G -t 30 .venv/bin/jupyter nbconvert --to notebook --execute \
  olmo3_next_token.ipynb --output /tmp/olmo3_out.ipynb
```
