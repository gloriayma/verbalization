"""Download PolyPythia 160m models (weight-, data- and combined seeds), Pile training shards, and the Pile val set.

    .venv/bin/python -m inversion.download              # models + 2 train shards + val
    .venv/bin/python -m inversion.download --shards 4   # add more shards later (already-downloaded ones are skipped)
    .venv/bin/python -m inversion.download --skip_val --shards 0 --models s4,s5  # more models

The train shards are raw uint16 token streams (Pythia tokenizer, documents separated by <|endoftext|> = 0), i.e. the
exact tokens PolyPythias trained on; data seeds only change the order. The val set is tokenized here into the same format.
"""

import argparse
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from inversion.data import DATA_DIR, EOD, shard_paths, val_path
from inversion.model import DEFAULT_DOWNLOAD, MODELS, REVISION, model_dir

TRAIN_REPO = "EleutherAI/pile-standard-pythia-preshuffled"
VAL_REPO = "EleutherAI/pile_val_test"
SHARD_BYTES = 30_000_000_000


def download_models(names):
    import shutil

    import torch
    from safetensors.torch import save_file

    for name in names:
        repo = MODELS[name]
        out = model_dir(name)
        if os.path.exists(os.path.join(out, "model.safetensors")):
            print(f"model {name}: already at {out}", flush=True)
            continue
        snap = snapshot_download(repo, revision=REVISION)
        # The repos only ship pytorch_model.bin; convert once so loading never needs torch.load (blocked on torch<2.6).
        state = torch.load(os.path.join(snap, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        state = {k: v.contiguous() for k, v in state.items()
                 if not k.endswith(("attention.bias", "attention.masked_bias", "rotary_emb.inv_freq"))}
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(snap):
            if f.endswith(".json"):
                shutil.copy(os.path.join(snap, f), out)
        save_file(state, os.path.join(out, "model.safetensors.tmp"), metadata={"format": "pt"})
        os.replace(os.path.join(out, "model.safetensors.tmp"), os.path.join(out, "model.safetensors"))
        print(f"model {name}: {repo}@{REVISION} -> {out}", flush=True)


def download_shards(n):
    for i in range(n):
        fname = f"document-{i:05d}-of-00020.bin"
        path = hf_hub_download(TRAIN_REPO, fname, repo_type="dataset", local_dir=os.path.join(DATA_DIR, "standard"))
        size = os.path.getsize(path)
        assert size == SHARD_BYTES, f"{path}: {size} bytes, expected {SHARD_BYTES}"
        print(f"shard {fname}: {size / 1e9:.0f} GB ({size // 2 / 1e9:.1f}B tokens)", flush=True)


def tokenize_val():
    out = val_path()
    if os.path.exists(out):
        print(f"val tokens already at {out}", flush=True)
        return
    from transformers import AutoTokenizer

    jsonl = hf_hub_download(VAL_REPO, "val.jsonl", repo_type="dataset", local_dir=os.path.join(DATA_DIR, "val"))
    tok = AutoTokenizer.from_pretrained(MODELS["w1"], revision=REVISION)  # tokenizer files only; no torch.load
    with open(jsonl) as f:
        texts = [json.loads(line)["text"] for line in f]
    print(f"tokenizing {len(texts)} val documents...", flush=True)
    chunks = []
    for i in range(0, len(texts), 4096):
        for ids in tok(texts[i : i + 4096], add_special_tokens=False)["input_ids"]:
            chunks.append(np.asarray(ids + [EOD], dtype=np.uint16))
    tokens = np.concatenate(chunks)
    tokens.tofile(out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"val: {len(tokens) / 1e6:.0f}M tokens -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=int, default=2, help="number of the 20 train shards (30 GB each) to download")
    p.add_argument("--models", default=",".join(DEFAULT_DOWNLOAD), help=f"any of {','.join(MODELS)}")
    p.add_argument("--skip_models", action="store_true")
    p.add_argument("--skip_val", action="store_true")
    args = p.parse_args()
    if not args.skip_models:
        download_models(args.models.split(","))
    if not args.skip_val:
        tokenize_val()
    download_shards(args.shards)
    print("train shards on disk:", shard_paths(), flush=True)


if __name__ == "__main__":
    main()
