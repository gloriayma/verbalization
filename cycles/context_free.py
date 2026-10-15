"""Experiment A: the context-free greedy chain.

Feed one token alone (position 0, no BOS), take the greedy next token X, then feed X alone in a fresh context, and so
on. Because the context is a single token, this is a fixed map f: vocab -> vocab (a functional graph: one out-edge per
token), so we compute f for every token in one batched pass and then:
  - decompose it exactly into cycles, each with its basin (the tokens that eventually fall into it);
  - walk it from "I": a segment ends when the next token was already visited (a "new_cycle" if it closed on itself,
    "merged" if it ran into an earlier segment), then restart at a uniformly random unvisited token, until every token
    has been visited;
  - compare f and its cycles across models (default: the nine main seeds s1..s9).

    python -m cycles.context_free [--models s1,...,s9] [--out cycles/runs]
"""

import argparse
import json
import os
from collections import Counter
from itertools import combinations

import numpy as np
import torch

from cycles.common import RUNS, V, default_device, load, pieces, start_token


@torch.no_grad()
def next_token_map(model, device, batch=4096):
    """f[t] = argmax over the real vocab of the logits after the one-token context [t]."""
    f = torch.empty(V, dtype=torch.long)
    for lo in range(0, V, batch):
        ids = torch.arange(lo, min(lo + batch, V), device=device)[:, None]
        f[lo:lo + len(ids)] = model(input_ids=ids).logits[:, -1, :V].argmax(-1).cpu()
    # Batching length-1 sequences needs no padding, so this must agree with one-at-a-time forward passes.
    for t in (0, 42, 309, V // 2, V - 1):
        assert model(input_ids=torch.tensor([[t]], device=device)).logits[0, -1, :V].argmax().item() == f[t], t
    return f.numpy()


def decompose(f):
    """Cycles of the functional graph f, in traversal order, and per token its cycle id and distance to the cycle."""
    n = len(f)
    cycle_of = np.full(n, -1)
    tail = np.full(n, -1)
    cycles = []
    state = np.zeros(n, dtype=np.int8)  # 0 unvisited, 1 on the current path, 2 done
    for s in range(n):
        if state[s]:
            continue
        path, t = [], s
        while state[t] == 0:
            state[t] = 1
            path.append(t)
            t = f[t]
        if state[t] == 1:  # closed a new cycle: the path from t's first occurrence on
            cyc = path[path.index(t):]
            for c in cyc:
                cycle_of[c], tail[c] = len(cycles), 0
            cycles.append(cyc)
            path = path[:path.index(t)]
        for u in reversed(path):  # everything before drains into the cycle that f[u] belongs to
            cycle_of[u], tail[u] = cycle_of[f[u]], tail[f[u]] + 1
        state[path] = 2
        state[cycles[cycle_of[s]]] = 2
    return cycles, cycle_of, tail


def walk(f, start, cycle_of, seed):
    """Follow f from `start`, jumping to a random unvisited token whenever the chain reaches a visited one."""
    rng = np.random.default_rng(seed)
    visited = np.zeros(len(f), dtype=bool)
    segments, t = [], start
    while True:
        ids = []
        while not visited[t]:
            visited[t] = True
            ids.append(int(t))
            t = f[t]
        segments.append({"start": ids[0], "ids": ids, "next": int(t),
                         "end": "new_cycle" if t in ids else "merged", "cycle": int(cycle_of[t])})
        unseen = np.flatnonzero(~visited)
        if not len(unseen):
            return segments
        t = rng.choice(unseen)


def canon(cyc):
    """Rotation-invariant key for a cycle (rotate so the smallest id leads)."""
    i = cyc.index(min(cyc))
    return tuple(cyc[i:] + cyc[:i])


def run_model(name, args, device):
    model, tok = load(name, device)
    out = os.path.join(args.out, name, "context_free")
    os.makedirs(out, exist_ok=True)
    f = next_token_map(model, device)
    del model
    np.save(os.path.join(out, "f.npy"), f)

    cycles, cycle_of, tail = decompose(f)
    basin = np.bincount(cycle_of, minlength=len(cycles))
    assert basin.sum() == V and all(f[c[-1]] == c[0] for c in cycles)
    order = np.argsort(-basin, kind="stable")
    cyc_rows = [{"id": int(i), "period": len(cycles[i]), "basin": int(basin[i]), "ids": [int(x) for x in cycles[i]],
                 "pieces": pieces(cycles[i], tok)} for i in order]
    with open(os.path.join(out, "cycles.json"), "w") as fh:
        json.dump(cyc_rows, fh, indent=1)

    start = start_token(tok)
    segments = walk(f, start, cycle_of, args.seed)
    assert sum(len(s["ids"]) for s in segments) == V
    with open(os.path.join(out, "walk.jsonl"), "w") as fh:
        for s in segments:
            fh.write(json.dumps({**s, "text": tok.decode(s["ids"]), "pieces": pieces(s["ids"], tok)}) + "\n")

    periods = Counter(len(c) for c in cycles)
    print(f"\n=== {name}: {len(cycles)} cycles, {periods[1]} fixed points, longest tail {tail.max()} ===")
    print("cycle periods:", dict(sorted(periods.items())))
    print(f"walk: {len(segments)} segments, {sum(s['end'] == 'new_cycle' for s in segments)} closed new cycles")
    i_seg = segments[0]
    print(f'"I" chain ({len(i_seg["ids"])} tokens, then -> {tok.decode([i_seg["next"]])!r}, {i_seg["end"]}):')
    print("   ", repr(tok.decode(i_seg["ids"])))
    print("top cycles by basin:")
    for r in cyc_rows[:args.show]:
        print(f"    basin {r['basin']:6d}  period {r['period']:3d}  {r['pieces']}")
    print(f"first {args.show} walk segments:")
    for s in segments[:args.show]:
        print(f"    [{s['end']:9s}] {tok.decode(s['ids'])!r} -> {tok.decode([s['next']])!r}")
    return {"f": f, "cycles": cycles, "basin": basin, "tok": tok, "i_chain": i_seg}


def compare(results, out):
    names = list(results)
    tok = results[names[0]]["tok"]
    agree = {f"{a}-{b}": float((results[a]["f"] == results[b]["f"]).mean()) for a, b in combinations(names, 2)}
    shared = Counter()
    for r in results.values():
        shared.update({canon(c) for c in r["cycles"]})
    per_model = {n: {"cycles": len(r["cycles"]), "fixed_points": sum(len(c) == 1 for c in r["cycles"]),
                     "largest_basin": int(r["basin"].max()),
                     "i_chain": tok.decode(r["i_chain"]["ids"]) + " -> " + tok.decode([r["i_chain"]["next"]])}
                 for n, r in results.items()}
    shared_rows = [{"n_models": k, "period": len(c), "ids": list(map(int, c)), "pieces": pieces(c, tok),
                    "models": [n for n, r in results.items() if c in {canon(x) for x in r["cycles"]}]}
                   for c, k in shared.most_common() if k > 1]
    with open(os.path.join(out, "context_free_compare.json"), "w") as fh:
        json.dump({"f_agreement": agree, "per_model": per_model, "shared_cycles": shared_rows}, fh, indent=1)

    print("\n=== comparison ===")
    print(f"{'model':6} {'cycles':>6} {'fixed':>6} {'max basin':>9}  \"I\" chain")
    for n, m in per_model.items():
        print(f"{n:6} {m['cycles']:6d} {m['fixed_points']:6d} {m['largest_basin']:9d}  {m['i_chain']!r}")
    vals = np.array(list(agree.values()))
    print(f"pairwise f agreement: mean {vals.mean():.3f}, min {vals.min():.3f}, max {vals.max():.3f}")
    print(f"cycles found in >1 model: {len(shared_rows)}; in all {len(names)}: "
          f"{sum(r['n_models'] == len(names) for r in shared_rows)}")
    for r in shared_rows[:20]:
        print(f"    {r['n_models']} models  {r['pieces']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(f"s{i}" for i in range(1, 10)))
    p.add_argument("--out", default=RUNS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show", type=int, default=20)
    p.add_argument("--device", default=default_device())
    args = p.parse_args()
    results = {name: run_model(name, args, args.device) for name in args.models.split(",")}
    if len(results) > 1:
        compare(results, args.out)


if __name__ == "__main__":
    main()
