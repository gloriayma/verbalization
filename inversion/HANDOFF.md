# Handoff: PolyPythia activation-inversion experiments

The step-by-step runbook for standing these experiments up on a new machine or cluster. For the reference on the code,
flags and outputs, see the `inversion/` section of the [README](../README.md).

## 1. What this is
We train a Pythia-160m (random init, or pretrained weights of a PolyPythia) to take another PolyPythia-160m's residual
stream at layer L (default 6) for an N-token window (default N=10), fed in place of the token embeddings, and output
those N tokens. The loss is unshifted cross-entropy: position i predicts token i, with causal attention. For each
model family F (`w` weight seeds, `d` data seeds, `s` main seeds) there are 5 runs:

| run | inverter init | inverts activations of | question |
|---|---|---|---|
| `F1_inv_F1_L6` | F1 | F1 | can a model invert itself? |
| `F1_inv_F2_L6` | F1 | F2 | does a sibling's pretrained init help invert a different seed? |
| `rand_inv_F2_L6` | random | F2 | the baseline for the row above |
| `F1_inv_mixF12_L6` | F1 | F1 or F2, chosen per sample by a seeded coin | can one inverter handle both? (loss logged per source) |
| `rand_inv_mixF12_L6` | random | the same mix, same coin flips | baseline |

## 2. Status (as of 2026-10-14)
- **Written and checked on CPU:**
  - Layer-L extraction equals HF `hidden_states[L]`, and `--layer final` equals `last_hidden_state`.
  - Inverter parameter counts match the source (162.3M).
  - Training windows decode to clean Pile text, and the max token id is 50276.
  - Batches are identical for the same (seed, step).
  - The data loader manages ~27 batches/s of 512 on the old cluster's network FS.
- **Never run on a GPU:** the training loop, resume, signal checkpointing, val logging, plotting, and real throughput.
  Section 5 is the first GPU run; expect to fix small things.

## 3. Requirements
- **GPU:** one per run. A 160M model at batch 512 × 10 tokens is small, so any ≥24 GB GPU is plenty. A family grid
  is 5 GPUs.
- **Software:** Python ≥ 3.10 and a CUDA build of torch ≥ 2.5, plus `inversion/requirements.txt`.
- **Disk** under `$INVERSION_DATA`, about 70 GB in total:
  - 30 GB per Pile training shard (default 2 shards).
  - 1.4 GB raw val jsonl plus 0.7 GB tokenized val.
  - About 0.7 GB per model: the HF cache copy plus a converted copy, 9 models by default.
- **Disk per run:** about 4 GB of checkpoints (the last 2 kept), plus logs (see section 7).
- **Network:** only the Hugging Face Hub. Everything is public, so no token is needed, though `HF_TOKEN` raises rate limits.

## 4. Setup
```bash
git clone https://github.com/gloriayma/verbalization.git && cd verbalization
python -m venv .venv && source .venv/bin/activate
pip install torch                                   # the right CUDA build for this cluster
pip install -r inversion/requirements.txt
export INVERSION_DATA=/fast/local/disk/pile         # also set this in job scripts (sbatch passes the env through)
python -m inversion.download                        # 9 models + val + 2 shards; ~10-30 min depending on bandwidth
```
`download` is idempotent: rerun it to resume, add `--shards 4` for more data, or `--models s4,s5` for more models.
Decide the shard count before launching. Windows are drawn from all shards on disk, so adding shards mid-experiment
changes which windows later steps see, and runs started before and after would no longer share batches.
It produces:
- `$INVERSION_DATA/models/pythia-160m-*-step143000/model.safetensors`. The Hub repos only ship `pytorch_model.bin`,
  which transformers 5.x refuses to `torch.load` on torch < 2.6, so each one is converted once. Training only ever
  loads these.
- `$INVERSION_DATA/val_tokens.bin`: the Pile val split, 341M tokens, uint16, documents separated by token 0.
- `$INVERSION_DATA/standard/document-0000{0,1}-of-00020.bin`: exactly 30,000,000,000 bytes each (the script asserts it).

Put `INVERSION_DATA` on local disk if you can. Training does random 20-byte reads, ~2 ms each on the old cluster's
NFS. The loader uses `os.pread` with 32 threads, because `np.memmap` held the GIL and capped it at 2 batches/s.

## 5. Smoke test (do this before launching anything long)
Run these in order from the repo root on one GPU:
```bash
S="--out_dir smoke --batch 32 --n_val 256 --val_every 100 --print_every 50 --ckpt_every 100"
# a. short mixed run
python -m inversion.train --init w1 --source mixw12 --steps 200 $S
# b. resume
python -m inversion.train --init w1 --source mixw12 --steps 300 $S
# c. second mixed run, which must see the same per-sample sources
python -m inversion.train --init random --source mixw12 --steps 60 $S
python - <<'EOF'
import json
a = [json.loads(l) for l in open("smoke/w1_inv_mixw12_L6/train_log.jsonl")][:60]
b = [json.loads(l) for l in open("smoke/rand_inv_mixw12_L6/train_log.jsonl")]
print("same coin flips:", [r["n_w1"] for r in a] == [r["n_w1"] for r in b])
EOF
# d. stop signal -> checkpoint
timeout -s TERM 90 python -m inversion.train --init random --source w2 --name sigtest $S --ckpt_every 100000; ls smoke/sigtest
# e. throughput at full size
python -m inversion.train --init random --source mixs12 --out_dir smoke --name speed --steps 400 --val_every 200 --print_every 100
```
Pass criteria:
- **a.** Loss falls. Every 50 steps it prints a `[step S] loss … (w1 … w2 …) … steps/s` header followed by 32
  `[step S][w1|w2] k/10  IN : '…'  OUT: '…'` lines, and `[val step …]` lines appear for w1 and w2.
  `smoke/w1_inv_mixw12_L6/` holds `train_log.jsonl` (with `loss_w1`, `loss_w2`, `n_w1`, `n_w2`), `val_log.jsonl`,
  `val_samples/step_*_w1.jsonl.gz` and `train_samples.log`.
- **b.** It prints `resumed smoke/w1_inv_mixw12_L6 at step 200`, and `train_log.jsonl` ends up with exactly 300 lines,
  steps 0–299.
- **c.** It prints `same coin flips: True`.
- **d.** It prints `got signal 15; checkpointing at step …`, and a `ckpt_*.pt` file exists.
- **e.** Note the steps/s from the headers and the duration of a val pass (2 × 8192 windows), and check `nvidia-smi`
  memory. Sanity math: steps/s × 512 × 86400 = windows per day; 2 shards ≈ 3B windows.

An optional overfit check: one fixed batch with a random init should reach a loss near 0 within a few hundred steps.
```python
import numpy as np, torch, torch.nn.functional as F
from inversion.data import TokenWindows, val_path
from inversion.model import SourceModel, build_inverter, invert
tokens = torch.from_numpy(TokenWindows([val_path()], 10).sample(np.random.default_rng(0), 64)).cuda()
acts = SourceModel("w2", "6", "cuda")(tokens)
inv = build_inverter("random", 0, "cuda")
opt = torch.optim.AdamW([p for p in inv.parameters() if p.requires_grad], lr=3e-4)
for step in range(301):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = invert(inv, acts)
    loss = F.cross_entropy(logits.float().flatten(0, 1), tokens.flatten())
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0: print(step, loss.item())
```

## 6. Launching
Runs train **until stopped**. Ctrl-C, SIGTERM or `scancel` makes a run checkpoint and exit. Re-running the same
command resumes it from `runs/<name>/`; a run refuses to resume if you change a setting that affects training.

**With Slurm:** first edit the `#SBATCH` lines in `inversion/run_grid.sbatch` (partition/QOS, `--mem`, `--time`)
for the new cluster, then:
```bash
mkdir -p runs/slurm
FAMILY=s sbatch inversion/run_grid.sbatch                  # 5 runs, layer 6
FAMILY=d sbatch inversion/run_grid.sbatch --layer final    # extra train.py flags go after the script
```
Five minutes before `--time`, the job gets SIGUSR1 (`--signal=B:USR1@300`), checkpoints, and runs `scontrol requeue`
on itself. That needs requeue to be allowed on the cluster; if it isn't, resubmit by hand and the run resumes. The
script uses `$SLURM_SUBMIT_DIR/.venv/bin/python` unless you set `PY=...`.

**Without Slurm** (one node, 5 GPUs):
```bash
F=s; LAYER=6
CFGS=("--init ${F}1 --source ${F}1" "--init ${F}1 --source ${F}2" "--init random --source ${F}2"
      "--init ${F}1 --source mix${F}12" "--init random --source mix${F}12")
mkdir -p runs/logs
for i in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$i nohup python -m inversion.train ${CFGS[$i]} --layer $LAYER > runs/logs/${F}_${i}_L${LAYER}.out 2>&1 &
done
```

**Suggested order:**
1. `s` at layer 6. It's the only family whose seeds really start from different inits (see section 8).
2. `d` at layer 6: same init, different data order.
3. `w`: see the caveat in section 8 before spending GPUs on it.
4. `--layer final` (or `12`) for whichever family turns out interesting.

## 7. Monitoring and outputs
Each run writes `runs/<name>/`:
- `train_log.jsonl`: every step, with per-source loss, accuracy and count.
- `val_log.jsonl`: every `--val_every` steps (default 500), on both seed-1 and seed-2 activations of the family.
- `val_samples/step_<step>_<src>.jsonl.gz`: every val input/output pair.
- `train_samples.log`
- `ckpt_*.pt`

To plot and inspect:
```bash
python -m inversion.plot --runs '*_L6' --out runs/curves_L6.png
zcat runs/s1_inv_s2_L6/val_samples/step_000010000_s2.jsonl.gz | head -3
```
**Watch the disk.** With the defaults, every 25 steps each run prints all 512 examples of the batch, about 75 KB, to
both stdout and `train_samples.log`. Each val pass writes about 2 × 8192 gzipped JSON lines. Once you know steps/s from
smoke test (e), work out GB/day per run and raise `--print_every` / `--val_every` if needed. Changing these does not
block resuming.

## 8. Findings so far (verified by comparing checkpoints)
- **The weight seeds were not trained from different inits.** Each `pythia-160m-weight-seed{1,2,3}` repo's `step0` is
  a distinct random init, but its `step1` is **bit-identical** to the original `EleutherAI/pythia-160m`'s `step1`.
  Its final weights have cosine ≈ 0 with its own `step0` and ≈ 0.13 with the original Pythia init, the same as every
  model trained from that init. So w1, w2 and w3 all started from the original Pythia init, as did d1–d3, whose `step0`
  is bit-identical to pythia-160m's. The w models still diverge (w1 vs w2 weight cosine: 1.00 at step1, 0.99 at step1000,
  0.48 at step10000, ~0.22 final), but not because of the init.

| family | init | data order | final weight cosine between seeds | layer-6 activation cosine (seed 1 vs 2) |
|---|---|---|---|---|
| w (`weight-seed`) | shared (original Pythia's) | same | ~0.2 | ~0.6 |
| d (`data-seed`) | shared (original Pythia's) | different | ~0.2 | ~0.55 |
| s (`seed`) | different | different | ~0 | ~0 |

  Every w model and every d model is about 0.2 similar to every other, across families too. The s models are
  orthogonal to everything.
- **Large first-position activation:** at layer 6, position 0 has norm ~280 versus ~15 elsewhere (no BOS; it's the
  attention-sink token). Activations are fed in raw by default. If training is unstable, try `--input_norm layernorm`.

## 9. Defaults

| setting | default | flag |
|---|---|---|
| layer | 6 (0–12, or `final` = after the final LayerNorm) | `--layer` |
| window length | 10 tokens, starting at position 0, no BOS, never crossing a document boundary | `--n_tokens`, `--allow_eod` |
| batch | 512 | `--batch` |
| optimizer | AdamW β = (0.9, 0.95), weight decay 0.01 (none on biases/LN), grad clip 1.0, bf16 autocast | `--weight_decay`, `--grad_clip` |
| LR | 6e-4 with 1000 warmup steps, then constant | `--lr`, `--warmup` |
| seeds | data 0 (which windows), mix 0 (coin flips), init 0 (random init), val 0 | `--data_seed`, `--mix_seed`, `--init_seed`, `--val_seed` |
| val | 8192 Pile-val windows, every 500 steps, on seeds 1 and 2 of the source's family | `--n_val`, `--val_every`, `--val_sources` |
| printing | all examples of the batch, every 25 steps | `--print_every` (0 = off) |
| checkpoints | every 2000 steps, plus on any stop signal | `--ckpt_every` |
| random init | HF GPT-NeoX init (normal, std 0.02), not GPT-NeoX's small_init/wang_init | none |

## 10. Open questions for the user
- Given section 8, is the w grid still worth running? Its init comparison is effectively "same init".
- Which layers to sweep beyond 6 (e.g. 1, 12, final)?
- Should inverters also be evaluated across families, e.g. a d-trained inverter on s activations? This works with
  `--val_sources d1,d2,s1,s2`, but the extra sources have to be loaded on the GPU.
