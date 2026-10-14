"""Train a Pythia-160m inverter: a source PolyPythia's activations at --layer for an N-token window -> those N tokens.

    .venv/bin/python -m inversion.train --init w1 --source mixw12      # trains until stopped (Ctrl-C / SIGTERM / scancel)

Re-running the same command resumes from the latest checkpoint in runs/<name>/. Outputs in runs/<name>/:
    config.json             arguments (+ git commit)
    train_log.jsonl         one line per step: loss, token accuracy, grad norm, and per-source loss/acc/count
    val_log.jsonl           one line per (val step, source): loss, token acc, exact match, per-position acc
    val_samples/step_*.jsonl.gz   every val input and inverter output at that step, for each val source
    train_samples.log       every example of the batch every --print_every steps (also printed to stdout)
    ckpt_*.pt               last two checkpoints (model + optimizer)
"""

import argparse
import glob
import gzip
import json
import os
import signal
import subprocess
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from inversion.data import TokenWindows, TrainBatches, shard_paths, val_windows
from inversion.model import MODELS, REVISION, SOURCE_SETS, SourceModel, build_inverter, invert, model_dir

# Changing any of these makes a run directory's checkpoints incompatible / its curves incomparable.
FIXED_ARGS = ["source", "init", "layer", "n_tokens", "batch", "lr", "warmup", "weight_decay", "grad_clip",
              "data_seed", "mix_seed", "init_seed", "allow_eod", "input_norm"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=sorted(SOURCE_SETS),
                   help="model whose activations are inverted; mixw12 = per-sample seeded 50/50 mix of w1 and w2")
    p.add_argument("--init", required=True, choices=["random", *MODELS], help="inverter init: pretrained model or random")
    p.add_argument("--name", help="run directory name (default: <init>_inv_<source>_L<layer>, init 'random' -> 'rand')")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--layer", default="6", choices=[*map(str, range(13)), "final"],
                   help="residual stream after this block (0 = token embeddings, 12 = after the last block); "
                        "'final' = after final_layer_norm, i.e. the unembedding's input")
    p.add_argument("--n_tokens", type=int, default=10)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=1000, help="linear warmup steps, then constant LR")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=0, help="stop after this many steps; 0 = train until stopped")
    p.add_argument("--val_every", type=int, default=500, help="0 = no validation")
    p.add_argument("--n_val", type=int, default=8192, help="held-out windows from the Pile val set")
    p.add_argument("--val_sources", help="every val pass evaluates on activations of each of these "
                                         "(default: seeds 1 and 2 of the source's family, e.g. d1,d2 for --source d2)")
    p.add_argument("--print_every", type=int, default=25,
                   help="print every example of the batch (step, source, input, output) every this many steps; 0 = off")
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--data_seed", type=int, default=0, help="which windows each step sees (shared by all runs)")
    p.add_argument("--mix_seed", type=int, default=0, help="per-sample source choice in mixed runs")
    p.add_argument("--init_seed", type=int, default=0, help="random-init seed")
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument("--allow_eod", action="store_true", help="allow windows that cross a document boundary")
    p.add_argument("--input_norm", choices=["none", "layernorm"], default="none",
                   help="normalize activations before the inverter (default: feed them raw)")
    return p.parse_args()


STOP = {"signal": None}


def request_stop(signum, frame):
    STOP["signal"] = signum


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        return None


def checkpoints(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))


def save_checkpoint(run_dir, step, inverter, opt):
    path = os.path.join(run_dir, f"ckpt_{step:09d}.pt")
    torch.save({"step": step, "model": inverter.state_dict(), "opt": opt.state_dict()}, path + ".tmp")
    os.replace(path + ".tmp", path)
    for old in checkpoints(run_dir)[:-2]:
        os.remove(old)
    print(f"saved {path}", flush=True)


def truncate_log(path, before_step):
    """Drop log lines from steps that were trained/validated after the checkpoint we're resuming from."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        keep = [line for line in f if line.strip() and json.loads(line)["step"] < before_step]
    with open(path + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(path + ".tmp", path)


def gather_acts(sources, names, tokens, src):
    """Activations [B, N, D] where sample b comes from source names[src[b]]."""
    if len(names) == 1:
        return sources[names[0]](tokens)
    acts = None
    for k, name in enumerate(names):
        idx = (src == k).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        a = sources[name](tokens[idx])
        if acts is None:
            acts = a.new_empty(*tokens.shape, a.shape[-1])
        acts[idx] = a
    return acts


def main():
    args = parse_args()
    name = args.name or f"{'rand' if args.init == 'random' else args.init}_inv_{args.source}_L{args.layer}"
    family = SOURCE_SETS[args.source][0][0]
    args.val_sources = args.val_sources or f"{family}1,{family}2"
    run_dir = os.path.join(args.out_dir, name)
    os.makedirs(os.path.join(run_dir, "val_samples"), exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, request_stop)

    config_path = os.path.join(run_dir, "config.json")
    config = {**vars(args), "name": name, "revision": REVISION, "train_shards": shard_paths(), "git_commit": git_commit()}
    if os.path.exists(config_path):
        with open(config_path) as f:
            old = json.load(f)
        changed = {k: (old.get(k), config[k]) for k in FIXED_ARGS if old.get(k) != config[k]}
        if changed:
            raise SystemExit(f"{run_dir} was started with different settings {changed}; use another --name")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_names = SOURCE_SETS[args.source]
    tok = AutoTokenizer.from_pretrained(model_dir(train_names[0]))  # same tokenizer for every PolyPythia
    val_names = [n for n in args.val_sources.split(",") if n] if args.val_every else []
    sources = {n: SourceModel(n, args.layer, device) for n in dict.fromkeys(train_names + val_names)}
    inverter = build_inverter(args.init, args.init_seed, device, config_from=train_names[0])
    params = [p for p in inverter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if p.ndim >= 2], "weight_decay": args.weight_decay},
         {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True,
    )

    start_step = 0
    if checkpoints(run_dir):
        state = torch.load(checkpoints(run_dir)[-1], map_location=device)
        inverter.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        print(f"resumed {run_dir} at step {start_step}", flush=True)
    train_log_path = os.path.join(run_dir, "train_log.jsonl")
    val_log_path = os.path.join(run_dir, "val_log.jsonl")
    truncate_log(train_log_path, start_step)
    truncate_log(val_log_path, start_step)
    n_params = sum(p.numel() for p in inverter.parameters())
    print(f"run {name}: init={args.init} source={args.source} {train_names} layer={args.layer} N={args.n_tokens} "
          f"inverter params={n_params / 1e6:.1f}M", flush=True)

    val_tokens = None
    if val_names:
        val_tokens = torch.from_numpy(val_windows(args.n_tokens, args.n_val, args.val_seed, args.allow_eod)).to(device)

    def decode(ids):
        return tok.batch_decode(ids.tolist())

    @torch.no_grad()
    def validate(step):
        inverter.eval()
        for src_name in val_names:
            losses, preds = [], []
            for i in range(0, len(val_tokens), args.batch):
                t = val_tokens[i : i + args.batch]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = invert(inverter, sources[src_name](t), args.input_norm)
                losses.append(F.cross_entropy(logits.float().flatten(0, 1), t.flatten(), reduction="none").view(t.shape))
                preds.append(logits.argmax(-1))
            loss, pred = torch.cat(losses), torch.cat(preds)
            correct = pred == val_tokens
            row = {"step": step, "source": src_name, "loss": loss.mean().item(),
                   "tok_acc": correct.float().mean().item(), "exact_match": correct.all(1).float().mean().item(),
                   "pos_acc": correct.float().mean(0).tolist()}
            with open(val_log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[val step {step}][{src_name}] loss {row['loss']:.4f}  tok_acc {row['tok_acc']:.4f}  "
                  f"exact {row['exact_match']:.4f}  pos_acc {' '.join(f'{a:.2f}' for a in row['pos_acc'])}", flush=True)

            in_text, out_text = decode(val_tokens), decode(pred)
            in_ids, out_ids, corr = val_tokens.tolist(), pred.tolist(), correct.int().tolist()
            ex_loss = loss.mean(1).tolist()
            path = os.path.join(run_dir, "val_samples", f"step_{step:09d}_{src_name}.jsonl.gz")
            with gzip.open(path + ".tmp", "wt") as f:
                for i in range(len(in_ids)):
                    f.write(json.dumps({
                        "step": step, "source": src_name, "idx": i, "loss": ex_loss[i],
                        "n_correct": sum(corr[i]), "correct": corr[i],
                        "input_text": in_text[i], "output_text": out_text[i],
                        "input_ids": in_ids[i], "output_ids": out_ids[i],
                    }) + "\n")
            os.replace(path + ".tmp", path)
            for i in range(4):
                print(f"[val step {step}][{src_name}] {sum(corr[i])}/{args.n_tokens}  "
                      f"IN : {in_text[i]!r}  OUT: {out_text[i]!r}", flush=True)
        inverter.train()

    batches = TrainBatches(TokenWindows(shard_paths(), args.n_tokens, args.allow_eod), args.batch,
                           args.data_seed, args.mix_seed, len(train_names), start_step)
    train_log = open(train_log_path, "a", buffering=1)  # line-buffered: the log never lags the checkpoints
    samples_log = open(os.path.join(run_dir, "train_samples.log"), "a")
    t_last, step_last = time.time(), start_step
    inverter.train()
    for step, tokens_np, src_np in batches:
        if args.steps and step >= args.steps:
            if val_names:
                validate(step)
            save_checkpoint(run_dir, step, inverter, opt)
            break
        if val_names and step % args.val_every == 0:
            validate(step)

        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        tokens = torch.from_numpy(tokens_np).to(device, non_blocking=True)
        src = torch.from_numpy(src_np).to(device, non_blocking=True)
        acts = gather_acts(sources, train_names, tokens, src)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = invert(inverter, acts, args.input_norm)
        tok_loss = F.cross_entropy(logits.float().flatten(0, 1), tokens.flatten(), reduction="none").view(tokens.shape)
        loss = tok_loss.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()

        # One host sync per step for all the stats.
        pred = logits.detach().argmax(-1)
        correct = (pred == tokens).float()
        sample_loss, sample_acc = tok_loss.detach().mean(1), correct.mean(1)
        stats = [loss.detach(), correct.mean(), grad_norm]
        for k in range(len(train_names)):
            m = (src == k).float()
            n = m.sum()
            stats += [(sample_loss * m).sum() / n.clamp(min=1), (sample_acc * m).sum() / n.clamp(min=1), n]
        stats = torch.stack(stats).tolist()
        row = {"step": step, "lr": lr, "loss": stats[0], "tok_acc": stats[1], "grad_norm": stats[2]}
        for k, src_name in enumerate(train_names):
            row[f"loss_{src_name}"], row[f"tok_acc_{src_name}"], row[f"n_{src_name}"] = stats[3 + 3 * k : 6 + 3 * k]
            row[f"n_{src_name}"] = int(row[f"n_{src_name}"])
        train_log.write(json.dumps(row) + "\n")

        if args.print_every and step % args.print_every == 0:
            now = time.time()
            per_src = "  ".join(f"{n} {row[f'loss_{n}']:.4f}" for n in train_names)
            header = (f"[step {step}] loss {row['loss']:.4f} ({per_src})  tok_acc {row['tok_acc']:.4f}  "
                      f"grad_norm {row['grad_norm']:.2f}  lr {lr:.2e}  "
                      f"{(step - step_last) / max(now - t_last, 1e-9):.2f} steps/s")
            t_last, step_last = now, step
            in_text, out_text = decode(tokens), decode(pred)
            n_correct = correct.sum(1).int().tolist()
            lines = [header] + [
                f"[step {step}][{train_names[s]}] {c}/{args.n_tokens}  IN : {i!r}  OUT: {o!r}"
                for s, c, i, o in zip(src_np.tolist(), n_correct, in_text, out_text)
            ]
            text = "\n".join(lines) + "\n"
            print(text, end="", flush=True)
            samples_log.write(text)
            samples_log.flush()

        done = step + 1
        if STOP["signal"] is not None:
            print(f"got signal {STOP['signal']}; checkpointing at step {done} and exiting", flush=True)
            save_checkpoint(run_dir, done, inverter, opt)
            break
        if done % args.ckpt_every == 0:
            save_checkpoint(run_dir, done, inverter, opt)

    train_log.close()
    samples_log.close()
    batches.close()


if __name__ == "__main__":
    main()
