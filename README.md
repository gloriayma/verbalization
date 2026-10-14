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

## `inversion/`: inverting PolyPythia activations
This trains a Pythia-160m (same architecture and parameter count) to recover N input tokens (default 10) from another
Pythia-160m's residual-stream activations at `--layer` (default 6). The activations replace the token embeddings
(`inputs_embeds`). Position i is trained with cross-entropy to output token i: no shift, causal attention as in pretraining.
The source model sees only the N tokens, starting at position 0 with no BOS.

Models are [PolyPythias](https://arxiv.org/abs/2503.09543) 160m runs at their final checkpoint `step143000`, in three
families. `download` fetches seeds 1–3 of each by default (`--models` adds more, e.g. `s4,s5`):

| family | Hub repos | short names | what differs (as documented) | what the checkpoints show |
|---|---|---|---|---|
| weight seeds | `pythia-160m-weight-seed{1,2,3}` | `w1`–`w3` | init only | **same init** as original Pythia (see below) |
| data seeds | `pythia-160m-data-seed{1,2,3}` | `d1`–`d3` | data order only | same init (= original pythia-160m's), different data order |
| main seeds | `pythia-160m-seed{1..9}` | `s1`–`s9` | init and data order | different init and data order |

**Caveat on the weight seeds** (checked 2026-10-14 by comparing weights). Each weight-seed repo's `step0` is a distinct
random init. But its `step1` is bit-identical to the original `EleutherAI/pythia-160m`'s `step1`, and its final
weights have zero cosine with its own `step0`. Meanwhile they have the same ~0.13 cosine with the original Pythia init
that every model trained from that init shows. So w1, w2 and w3 were trained from the *original* Pythia init, not
the different inits their `step0` suggests. They still drift apart, as much as the data seeds do: final weight
cosine is ~0.2 within and across the w and d families, and ~0 for any pair involving an s model. Whatever made them
diverge, "weight seed" here does not mean a different starting point. For truly different inits, use the s family.

For a family `F` in `w`, `d`, `s`, the grid is:

| run | inverter init | inverts activations of |
|---|---|---|
| `F1_inv_F1_L6` | F1 (pretrained) | F1 |
| `F1_inv_F2_L6` | F1 | F2 |
| `rand_inv_F2_L6` | random | F2 |
| `F1_inv_mixF12_L6` | F1 | F1 or F2, chosen per sample by a seeded fair coin (`--mix_seed`) |
| `rand_inv_mixF12_L6` | random | the same mix (same coin flips) |

Run names end in the layer (`_L6`, `_L12`, `_Lfinal`). `--layer 12` is the residual stream after the last block;
`--layer final` is after `final_layer_norm`, i.e. what the unembedding sees.

### Data and setup
```bash
.venv/bin/python -m inversion.download            # ~65 GB into /mnt/nw/home/g.ma/data/pile ($INVERSION_DATA)
.venv/bin/python -m inversion.download --shards 4 # add more training shards later (20 x 30 GB exist)
```
- **Train:** shards of `EleutherAI/pile-standard-pythia-preshuffled`, the exact Pythia-tokenized Pile tokens that
  PolyPythias trained on (data seeds only change the order). Each shard holds 15B tokens. Windows are drawn uniformly
  at random. A window containing `<|endoftext|>` is redrawn, so every window stays inside one document. The windows at
  step s depend only on `(--data_seed, s)`, so every run sees identical batches and a resumed run picks up exactly where
  it left off.
- **Val:** `EleutherAI/pile_val_test` val split (held out from training), tokenized into `val_tokens.bin`. There are
  `--n_val` fixed windows (default 8192).
- The Hub repos only ship `pytorch_model.bin`, which transformers 5.x won't `torch.load` on torch 2.5. So `download`
  converts each model once to safetensors under `$INVERSION_DATA/models/`, and training loads only those.
- The random init uses HF's GPT-NeoX init (normal, std 0.02), not GPT-NeoX's small_init/wang_init.

### Training
```bash
mkdir -p runs/slurm && sbatch inversion/run_grid.sbatch         # all 5 runs, 1 GPU each, until you scancel them
sbatch --array=3 inversion/run_grid.sbatch                      # just w1_inv_mixw12_L6
FAMILY=d sbatch inversion/run_grid.sbatch --layer final         # data-seed grid on final-layer activations
.venv/bin/python -m inversion.train --init w1 --source mixw12   # directly (e.g. on RunPod)
```
Runs train until stopped. The LR warms up linearly over 1000 steps, then stays at 6e-4 (AdamW, bf16 autocast,
batch 512). On SIGTERM, SIGUSR1 or Ctrl-C (and therefore on `scancel`), a run checkpoints before exiting. Re-running the
same command resumes from `runs/<name>/ckpt_*.pt`. Slurm jobs requeue themselves 5 minutes before the 24h QOS limit.
Useful flags: `--layer`, `--n_tokens`, `--print_every` (default 25: prints *every* example of that step's batch),
`--val_every` (default 500), `--n_val`, `--steps` (stop early), and `--input_norm layernorm`.

On a fresh GPU box (e.g. RunPod):
```bash
pip install -r inversion/requirements.txt
export INVERSION_DATA=/workspace/pile                   # wherever the ~65 GB should go
python -m inversion.download --shards 1                 # 1 shard (15B tokens) is plenty for a smoke test; w/d/s seeds 1-3
# smoke: short mixed run, then resume it, then check throughput at full batch size
python -m inversion.train --init w1 --source mixw12 --out_dir smoke --steps 200 --batch 32 --n_val 256 --val_every 100 --print_every 50 --ckpt_every 100
python -m inversion.train --init w1 --source mixw12 --out_dir smoke --steps 300 --batch 32 --n_val 256 --val_every 100 --print_every 50 --ckpt_every 100
python -m inversion.train --init random --source mixw12 --out_dir smoke --name speed --steps 400 --val_every 200 --print_every 100
```

### Outputs (`runs/<name>/`)
- `train_log.jsonl`: one line per step with `loss`, `tok_acc` and `grad_norm`, plus `loss_<src>`, `tok_acc_<src>`
  and `n_<src>` for each source. In the mix runs, that's the training loss split by w1 vs w2 samples.
- `val_log.jsonl`: every val pass evaluates the inverter on **both** w1 and w2 activations, whatever it trained on.
  Each line has loss, token accuracy, exact-sequence match and per-position accuracy.
- `val_samples/step_<step>_<src>.jsonl.gz`: every val input and inverter output at that step
  (`zcat ... | head`).
- `train_samples.log` (also on stdout): `[step S][w2] 7/10  IN : '...'  OUT: '...'` for every example of every
  `--print_every`-th batch.
- `.venv/bin/python -m inversion.plot` writes `runs/curves.png`: training loss split by sample source, plus val
  loss and accuracy split by activation source.
