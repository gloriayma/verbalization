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
Setting this up on a new machine? Follow [`inversion/HANDOFF.md`](inversion/HANDOFF.md).

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
.venv/bin/python -m inversion.download            # ~65 GB into $INVERSION_DATA (required; no default)
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
RUN_ARGS="--init random --source mixs1to9" sbatch --array=0 inversion/run_grid.sbatch   # any single run instead
sbatch --array=3 inversion/run_grid.sbatch                      # just w1_inv_mixw12_L6
FAMILY=d sbatch inversion/run_grid.sbatch --layer final         # data-seed grid on final-layer activations
.venv/bin/python -m inversion.train --init w1 --source mixw12   # directly (e.g. on RunPod)
```
Runs train until stopped. The LR warms up linearly over 1000 steps, then stays at 6e-4 (AdamW, bf16 autocast,
batch 512). On SIGTERM, SIGUSR1 or Ctrl-C (and therefore on `scancel`), a run checkpoints before exiting. Re-running the
same command resumes from `runs/<name>/ckpt_*.pt`. Slurm jobs requeue themselves 5 minutes before the 24h QOS limit.
Useful flags: `--layer`, `--n_tokens`, `--batch`, `--log_every` (default 1: one `train_log.jsonl` row per step; N =
one row per N steps, averaged), `--print_every` (default 25) with `--print_examples` (default 0 = the whole batch),
`--val_every` (default 500), `--n_val`, `--val_groups` (e.g. `'s1;s2,s3,s4,s5,s6,s7,s8,s9'`: one val row per group,
a multi-source group lumped by drawing each val window's source from it), `--val_examples` (default 4), `--val_dump_every` (default 0 = every val pass),
`--steps` (stop early), and `--input_norm layernorm`.

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
- `train_log.jsonl`: one line per `--log_every` steps with `loss`, `tok_acc` and `grad_norm` averaged over those
  steps, plus `loss_<src>`, `tok_acc_<src>` and `n_<src>` for each source, `n_steps`, and wall-clock `time`. `step` is
  the last step of the window. In the mix runs, that's the training loss split by w1 vs w2 samples.
- `val_log.jsonl`: every val pass evaluates the inverter on **both** w1 and w2 activations, whatever it trained on.
  Each line has loss, token accuracy, exact-sequence match, per-position accuracy, and `examples`: the first
  `--val_examples` val windows with the inverter's output (the same windows every pass).
- `val_samples/step_<step>_<src>.jsonl.gz`: every val input and inverter output at that step, every
  `--val_dump_every` steps (`zcat ... | head`).
- `train_samples.log` (also on stdout): a header plus `[step S][w2] 7/10  IN : '...'  OUT: '...'` for
  `--print_examples` examples of every `--print_every`-th batch.
- `.venv/bin/python -m inversion.plot` writes `runs/curves.png`: training loss split by sample source, plus val
  loss and accuracy split by activation source. Every logged point is a marker, and the titles give the logging
  cadence (`--smooth N` adds an EMA line).
- `inversion/plots.ipynb` (kernel "Python (verbalization .venv)") has all the plotting logic for editing: it reads
  `config.json`, `train_log.jsonl` and `val_log.jsonl` from `RUNS_DIR` directly (no repo imports) into pandas tables,
  with the colors, labels and panels in one cell and the step range in each figure cell.
- `.venv/bin/python -m inversion.watch --runs '*_L6'` is a live terminal view: it redraws on every val pass with each
  run's latest train stats and, per val source, the val metrics and the fixed val examples (input → output).

## `cycles/`: greedy token cycles in a frozen Pythia-160m
Two experiments on frozen PolyPythia 160m models (final checkpoint, loaded with `inversion.model.model_dir`, so
`INVERSION_DATA` must be set). Both use greedy decoding over the real vocab (50,277 tokens) and start from `"I"` (token 42).

- **`python -m cycles.context_free`** (A, default `--models s1,...,s9`): feed one token alone (position 0, no BOS), take
  the argmax, then feed *that* token alone in a fresh context, and repeat. With a one-token context this is a fixed map
  f: vocab → vocab. So f is computed for every token in one batched pass, then decomposed into all of its cycles and
  their basins. It is then walked from "I": when the chain reaches a visited token, it jumps to a random unvisited one, until the
  whole vocab is covered. Writes `f.npy`, `cycles.json` (by basin size) and `walk.jsonl` to
  `runs/cycles/<model>/context_free/`, plus a cross-seed `runs/cycles/context_free_compare.json`.
- **`python -m cycles.autoregressive`** (B, default `--model s1`): ordinary greedy generation with the full context
  until the output ends in one block repeated `--repeats` (5) times, or until it fills the 2048-token context. Then it
  restarts from a fresh context at a random token not yet generated. Writes
  `runs/cycles/<model>/autoregressive/segments.jsonl`.

A takes about a minute for all nine seeds on one GPU. B takes up to about 2048 decoding steps per segment (`--segments`, default 100).
