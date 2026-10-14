"""Loss / accuracy curves for all runs in runs/.

    .venv/bin/python -m inversion.plot [--runs "*_L6"] [--smooth 200] [--out runs/curves.png]

Color = run; line style = which source model's samples (solid seed 1, dashed seed 2), so the mixed runs' training loss is
shown split by source, and val curves show each inverter evaluated on both sources.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLES = {"1": "-", "2": "--", "3": ":"}  # by seed number: w1/d1/s1 solid, w2/d2/s2 dashed


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows


def ema(x, span):
    out, acc, a = np.empty(len(x)), None, 2 / (span + 1)
    for i, v in enumerate(x):
        acc = v if acc is None else a * v + (1 - a) * acc
        out[i] = acc
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--runs", default="*", help="glob of run names to include, e.g. '*_L6' or '*mixd12*'")
    p.add_argument("--smooth", type=int, default=200, help="EMA span (steps) for training curves")
    p.add_argument("--out", help="default: <runs_dir>/curves.png")
    args = p.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs_dir, args.runs)) if os.path.exists(os.path.join(d, "config.json")))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_train, ax_vloss, ax_vacc = axes
    for i, run_dir in enumerate(run_dirs):
        name, color = os.path.basename(run_dir), COLORS[i % len(COLORS)]
        train = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
        if train:
            steps = np.array([r["step"] for r in train])
            sources = [k[len("loss_"):] for k in train[0] if k.startswith("loss_")]
            for s in sources:
                # Steps where a mixed batch had no samples of this source carry no information for it.
                keep = np.array([r[f"n_{s}"] > 0 for r in train])
                y = ema(np.array([r[f"loss_{s}"] for r in train])[keep], args.smooth)
                ax_train.plot(steps[keep], y, STYLES.get(s[1:], "-."), color=color, lw=1.5, label=f"{name} [{s} samples]")
        val = read_jsonl(os.path.join(run_dir, "val_log.jsonl"))
        for s in sorted({r["source"] for r in val}):
            rows = [r for r in val if r["source"] == s]
            vs = [r["step"] for r in rows]
            ax_vloss.plot(vs, [r["loss"] for r in rows], STYLES.get(s[1:], "-."), color=color, lw=1.5, label=f"{name} on {s}")
            ax_vacc.plot(vs, [r["tok_acc"] for r in rows], STYLES.get(s[1:], "-."), color=color, lw=1.5, label=f"{name} on {s}")

    ax_train.set(title=f"train loss (EMA {args.smooth} steps), by sample source", xlabel="step", ylabel="cross-entropy (nats)", yscale="log")
    ax_vloss.set(title="val loss, by activation source", xlabel="step", ylabel="cross-entropy (nats)", yscale="log")
    ax_vacc.set(title="val token accuracy, by activation source", xlabel="step", ylabel="token accuracy", ylim=(0, 1))
    for ax in axes:
        ax.grid(True, color="#e6e5e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = args.out or os.path.join(args.runs_dir, "curves.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
