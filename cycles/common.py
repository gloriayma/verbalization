"""Frozen PolyPythia-160m loading and token formatting shared by the two cycle experiments."""

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from inversion.model import model_dir

V = 50277  # real vocab size; unembedding rows past this (up to 50304) are padding and never picked
START = "I"  # token 42, no leading space


def load(name, device):
    """Frozen GPTNeoXForCausalLM (fp32, eval) and its tokenizer from `inversion.download`'s local copy."""
    path = model_dir(name)
    model = GPTNeoXForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval().requires_grad_(False)
    return model, AutoTokenizer.from_pretrained(path)


def start_token(tok):
    ids = tok.encode(START)
    assert len(ids) == 1, ids
    return ids[0]


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def pieces(ids, tok):
    """Each token decoded on its own, so whitespace/newline tokens stay visible in a repr."""
    return [tok.decode([i]) for i in ids]
