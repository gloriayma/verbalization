"""PolyPythia source models (frozen, produce activations) and the Pythia-160m inverter (activations -> tokens)."""

import os

import torch
import torch.nn.functional as F
from transformers import AutoConfig, GPTNeoXForCausalLM

from inversion.data import DATA_DIR

REVISION = "step143000"  # final checkpoint (same weights as `main`)
MODELS = {
    **{f"w{i}": f"EleutherAI/pythia-160m-weight-seed{i}" for i in (1, 2, 3)},
    **{f"d{i}": f"EleutherAI/pythia-160m-data-seed{i}" for i in (1, 2, 3)},
}
# A run's --source names one of these; mixes pick a source per sample (see data.source_assignment).
SOURCE_SETS = {**{k: [k] for k in MODELS}, "mixw12": ["w1", "w2"]}


def model_dir(name):
    """Local safetensors copy made by `inversion.download` (the Hub repos only ship pytorch_model.bin, which
    transformers refuses to torch.load on torch < 2.6)."""
    return os.path.join(DATA_DIR, "models", f"{MODELS[name].split('/')[1]}-{REVISION}")


class SourceModel:
    """Frozen PolyPythia truncated after block `layer`; returns the residual stream there (before any final LayerNorm).

    layer=0 is the token embedding itself.
    """

    def __init__(self, name, layer, device):
        model = GPTNeoXForCausalLM.from_pretrained(model_dir(name), dtype=torch.float32)
        assert 0 <= layer <= model.config.num_hidden_layers, layer
        self.neox = model.gpt_neox.to(device).eval().requires_grad_(False)
        self.neox.layers = self.neox.layers[:layer]
        self.layer = layer
        self._out = None
        if layer > 0:
            self.neox.layers[-1].register_forward_hook(lambda mod, inp, out: setattr(self, "_out", out))

    @torch.no_grad()
    def __call__(self, tokens):
        if self.layer == 0:
            return self.neox.embed_in(tokens)
        self.neox(input_ids=tokens)
        out, self._out = self._out, None
        return out


def build_inverter(init, init_seed, device):
    """Pythia-160m: pretrained weights of a PolyPythia (init='w1', ...) or freshly initialized (init='random').

    Random init uses HF's GPTNeoX init (normal std=initializer_range), not GPT-NeoX's small_init/wang_init.
    """
    if init == "random":
        config = AutoConfig.from_pretrained(model_dir("w1"))
        torch.manual_seed(init_seed)
        model = GPTNeoXForCausalLM(config)
    else:
        model = GPTNeoXForCausalLM.from_pretrained(model_dir(init), dtype=torch.float32)
    # embed_in stays in the model (same parameter count) but is never used: activations go in via inputs_embeds.
    model.gpt_neox.embed_in.requires_grad_(False)
    return model.to(device)


def invert(inverter, acts, input_norm="none"):
    """Logits [B, N, V]; position i is trained to output token i (unshifted, causal attention as in pretraining)."""
    if input_norm == "layernorm":
        acts = F.layer_norm(acts, acts.shape[-1:])
    return inverter(inputs_embeds=acts).logits
