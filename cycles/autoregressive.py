"""Experiment B: ordinary greedy autoregressive generation, restarted from a fresh context whenever it loops.

A segment starts from one token (the first from "I"; later ones from a uniformly random token never generated or used
as a start in an earlier segment) and generates greedily with the full context until either
  - "cycle": the last repeats*p tokens are one length-p block repeated `--repeats` (default 5) times, smallest p wins;
  - "context_full": the sequence reaches the model's context window (2048, what Pythia was trained on). We then record
    the most repeats of any block the sequence ends with (2+ counts), so long-period loops that can't fit 5 copies
    still show up.

    python -m cycles.autoregressive [--model s1] [--segments 100] [--repeats 5] [--out runs/cycles]
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from cycles.common import V, default_device, load, pieces, start_token


def find_cycle(seq, repeats, min_repeats=None):
    """Smallest p such that seq ends with a length-p block repeated `repeats` times, as (p, repeats); else None.

    With min_repeats set, instead return the (p, r) with the most repeats r >= min_repeats over all p (smallest p on
    ties), for the end-of-context report.
    """
    seq = np.asarray(seq)
    n = len(seq)
    need = min_repeats or repeats
    best = None
    for p in range(1, n // need + 1):
        if seq[-1] != seq[-1 - p]:
            continue
        r = 1
        while (r + 1) * p <= n and np.array_equal(seq[n - (r + 1) * p:n - r * p], seq[n - p:]):
            r += 1
            if min_repeats is None and r >= repeats:
                return p, r
        if min_repeats is not None and r >= min_repeats and (best is None or r > best[1]):
            best = (p, r)
    return best


@torch.no_grad()
def generate_segment(model, start, repeats, max_len, device):
    seq = [start]
    out = model(input_ids=torch.tensor([[start]], device=device), use_cache=True)
    while True:
        nxt = out.logits[0, -1, :V].argmax().item()
        seq.append(nxt)
        hit = find_cycle(seq, repeats)
        if hit:
            return seq, "cycle", hit
        if len(seq) >= max_len:
            return seq, "context_full", find_cycle(seq, repeats, min_repeats=2)
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=out.past_key_values,
                    use_cache=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="s1")
    p.add_argument("--segments", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max_len", type=int, default=None, help="default: the model's max_position_embeddings (2048)")
    p.add_argument("--out", default="runs/cycles")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    model, tok = load(args.model, args.device)
    max_len = args.max_len or model.config.max_position_embeddings
    out = os.path.join(args.out, args.model, "autoregressive")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    seen = np.zeros(V, dtype=bool)
    start = start_token(tok)
    rows = []
    with open(os.path.join(out, "segments.jsonl"), "w") as fh:
        for k in range(args.segments):
            seq, end, hit = generate_segment(model, start, args.repeats, max_len, args.device)
            seen[seq] = True
            period, reps = hit or (None, None)
            row = {"segment": k, "start": start, "len": len(seq), "end": end, "period": period, "repeats": reps,
                   "block": pieces(seq[-period:], tok) if period else None, "ids": seq, "text": tok.decode(seq)}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)
            if k == 0:
                print(f'=== segment 0 from "I": {len(seq)} tokens, {end}, period {period} x{reps} ===')
                print(row["text"])
                print("===")
            print(f"[{k:3d}] start {tok.decode([start])!r:16} len {len(seq):5d}  {end:12s}  "
                  f"p={period} x{reps}  block {repr(tok.decode(seq[-period:]))[:60] if period else ''}", flush=True)
            start = int(rng.choice(np.flatnonzero(~seen)))

    ends = Counter(r["end"] for r in rows)
    lens = np.array([r["len"] for r in rows])
    print(f"\n{len(rows)} segments: {dict(ends)}; length median {np.median(lens):.0f}, mean {lens.mean():.0f}, "
          f"max {lens.max()}; distinct tokens generated {seen.sum()}")
    print("cycle periods:", dict(sorted(Counter(r["period"] for r in rows if r["end"] == "cycle").items())))


if __name__ == "__main__":
    main()
