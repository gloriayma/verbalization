"""Live view of training runs: the latest train stats, and for each val source the latest val metrics and a fixed set
of val examples (input -> inverter output).

    .venv/bin/python -m inversion.watch [--runs_dir runs] [--runs "*_L6"] [--once]

Redraws whenever any run logs a new val pass. The examples are the first --val_examples windows of the run's fixed val
set (stored in each val_log.jsonl row), so the same inputs appear on every redraw and you can watch the outputs change.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time


def tail_rows(path, n_bytes):
    """The JSON rows in the last n_bytes of a jsonl file (the first, possibly partial, line is dropped)."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - n_bytes))
        lines = f.read().decode(errors="replace").split("\n")
    if size > n_bytes:
        lines = lines[1:]
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # empty, or a line still being written
            pass
    return rows


def run_state(run_dir):
    train = tail_rows(os.path.join(run_dir, "train_log.jsonl"), 8192)
    val = {}
    for r in tail_rows(os.path.join(run_dir, "val_log.jsonl"), 1 << 16):
        val[r["source"]] = r  # rows are in step order, so the last one per source wins
    return train, val


def render(run_dirs, width):
    out = [time.strftime("%H:%M:%S") + f"  {len(run_dirs)} runs"]
    for run_dir in run_dirs:
        train, val = run_state(run_dir)
        line = f"== {os.path.basename(run_dir)}"
        if train:
            t = train[-1]
            per_src = "  ".join(f"{k[5:]} {v:.4f}" for k, v in t.items() if k.startswith("loss_"))
            line += f"   step {t['step']}  loss {t['loss']:.4f} ({per_src})  tok_acc {t['tok_acc']:.4f}"
            if len(train) > 1 and "time" in t and "time" in train[0] and t["time"] > train[0]["time"]:
                line += f"  {(t['step'] - train[0]['step']) / (t['time'] - train[0]['time']):.1f} steps/s"
        out.append(line)
        for src, r in sorted(val.items()):
            out.append(f"   [val step {r['step']} on {src}] loss {r['loss']:.4f}  tok_acc {r['tok_acc']:.4f}  "
                       f"exact {r['exact_match']:.4f}")
            for ex in r.get("examples", []):
                n = len(r["pos_acc"])
                out.append(f"     {ex['n_correct']:>2}/{n}  IN : {ex['input_text']!r}  OUT: {ex['output_text']!r}")
    return "\n".join(line[:width] for line in out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--runs", default="*", help="glob of run names to include, e.g. '*_L6'")
    p.add_argument("--poll", type=float, default=2.0, help="seconds between checks for a new val pass")
    p.add_argument("--once", action="store_true", help="print one frame and exit")
    args = p.parse_args()

    def run_dirs():
        return sorted(d for d in glob.glob(os.path.join(args.runs_dir, args.runs))
                      if os.path.exists(os.path.join(d, "config.json")))

    if args.once:
        print(render(run_dirs(), shutil.get_terminal_size().columns))
        return
    last = None
    try:
        while True:
            dirs = run_dirs()
            key = tuple((d, os.path.getsize(os.path.join(d, "val_log.jsonl"))
                         if os.path.exists(os.path.join(d, "val_log.jsonl")) else 0) for d in dirs)
            if key != last:  # a val pass landed (or a run appeared)
                frame = render(dirs, shutil.get_terminal_size().columns)
                sys.stdout.write("\033[H\033[2J" + frame + "\n")
                sys.stdout.flush()
                last = key
            time.sleep(args.poll)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
