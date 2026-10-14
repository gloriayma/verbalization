"""Loss / accuracy curves for all runs in runs/.

    .venv/bin/python -m inversion.plot [--runs_dir runs] [--runs "*_L6"] [--smooth 0] [--log_y] [--out runs/curves.png]

Every logged point is drawn as a marker (train: one train_log.jsonl row, i.e. a mean over --log_every steps; val: one
val pass), and the panel titles say how often each was logged. Color = run; line style = which source model's samples
(solid seed 1, dashed seed 2), so the mixed runs' training loss is shown split by source, and val curves show each
inverter evaluated on both sources.
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
POINTS = dict(marker="o", ms=2, lw=0.6)


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


def cadence(configs, key, default=None):
    """The runs' shared value of a config key, or 'varies'."""
    values = {c.get(key, default) for c in configs}
    return values.pop() if len(values) == 1 else "varies"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--runs", default="*", help="glob of run names to include, e.g. '*_L6' or '*mixd12*'")
    p.add_argument("--smooth", type=int, default=0,
                   help="also draw an EMA over this many logged points of each training curve; 0 = points only")
    p.add_argument("--log_y", action="store_true", help="log-scale y axis for the loss panels (default: linear)")
    p.add_argument("--out", help="default: <runs_dir>/curves.png")
    args = p.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs_dir, args.runs)) if os.path.exists(os.path.join(d, "config.json")))
    configs = []
    for d in run_dirs:
        with open(os.path.join(d, "config.json")) as f:
            configs.append(json.load(f))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_train, ax_vloss, ax_vacc = axes
    for i, run_dir in enumerate(run_dirs):
        name, color = os.path.basename(run_dir), COLORS[i % len(COLORS)]
        train = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
        if train:
            steps = np.array([r["step"] for r in train])
            sources = [k[len("loss_"):] for k in train[0] if k.startswith("loss_")]
            for s in sources:
                # Rows where a mixed batch had no samples of this source carry no information for it.
                keep = np.array([r[f"n_{s}"] > 0 for r in train])
                y = np.array([r[f"loss_{s}"] for r in train])[keep]
                style = STYLES.get(s[1:], "-.")
                ax_train.plot(steps[keep], y, style, color=color, alpha=0.5 if args.smooth else 1,
                              label=f"{name} [{s} samples]", **POINTS)
                if args.smooth:
                    ax_train.plot(steps[keep], ema(y, args.smooth), style, color=color, lw=1.5)
        val = read_jsonl(os.path.join(run_dir, "val_log.jsonl"))
        for s in sorted({r["source"] for r in val}):
            rows = [r for r in val if r["source"] == s]
            vs, style = [r["step"] for r in rows], STYLES.get(s[1:], "-.")
            ax_vloss.plot(vs, [r["loss"] for r in rows], style, color=color, label=f"{name} on {s}", **POINTS)
            ax_vacc.plot(vs, [r["tok_acc"] for r in rows], style, color=color, label=f"{name} on {s}", **POINTS)

    log_every, batch = cadence(configs, "log_every", 1), cadence(configs, "batch")
    val_every, n_val = cadence(configs, "val_every"), cadence(configs, "n_val")
    train_note = f"each point = mean over {log_every} step(s) × batch {batch}"
    if args.smooth:
        train_note += f"; line = EMA over {args.smooth} points"
    val_note = f"each point = one val pass, every {val_every} steps, {n_val} windows per source"
    ax_train.set(title=f"train loss, by sample source\n{train_note}", xlabel="step", ylabel="cross-entropy (nats)",
                 yscale="log" if args.log_y else "linear")
    ax_vloss.set(title=f"val loss, by activation source\n{val_note}", xlabel="step", ylabel="cross-entropy (nats)",
                 yscale="log" if args.log_y else "linear")
    ax_vacc.set(title=f"val token accuracy, by activation source\n{val_note}", xlabel="step", ylabel="token accuracy",
                ylim=(0, 1))
    for ax in axes:
        ax.title.set_fontsize(10)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = args.out or os.path.join(args.runs_dir, "curves.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
