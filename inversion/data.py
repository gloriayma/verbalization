"""N-token windows sampled from raw uint16 token streams (Pile shards / tokenized Pile val).

Every batch is a pure function of (seed, step), so all runs see identical windows in identical order and resuming from a
checkpoint at step s reproduces batch s exactly, without saving any RNG state.
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DATA_DIR = os.environ.get("INVERSION_DATA", "/mnt/nw/home/g.ma/data/pile")
EOD = 0  # <|endoftext|>, separates documents in the token streams


def shard_paths():
    return sorted(glob.glob(os.path.join(DATA_DIR, "standard", "document-*-of-00020.bin")))


def val_path():
    return os.path.join(DATA_DIR, "val_tokens.bin")


class TokenWindows:
    """Uniformly random N-token windows from one or more uint16 token files (concatenated, windows never span files)."""

    def __init__(self, paths, n_tokens, allow_eod=False):
        assert paths, "no token files found; run `python -m inversion.download` first"
        # os.pread rather than np.memmap: memmap page faults hold the GIL, so prefetch threads couldn't overlap the
        # random reads (~2 ms each on /mnt/nw); pread releases it (~25x more batches/s with 16 threads).
        self.fds = [os.open(p, os.O_RDONLY) for p in paths]
        self.n = n_tokens
        self.allow_eod = allow_eod
        self.cum_starts = np.cumsum([os.path.getsize(p) // 2 - n_tokens + 1 for p in paths])

    def _window(self, start):
        f = int(np.searchsorted(self.cum_starts, start, side="right"))
        local = start - (int(self.cum_starts[f - 1]) if f else 0)
        return np.frombuffer(os.pread(self.fds[f], 2 * self.n, 2 * local), dtype=np.uint16)

    def sample(self, rng, batch):
        out = np.empty((batch, self.n), dtype=np.int64)
        filled = 0
        while filled < batch:
            # Windows containing <|endoftext|> are rejected (unless allow_eod), keeping every window inside one document.
            for start in rng.integers(0, self.cum_starts[-1], size=batch - filled):
                w = self._window(int(start))
                if self.allow_eod or not (w == EOD).any():
                    out[filled] = w
                    filled += 1
        return out


def source_assignment(mix_seed, step, batch, n_sources):
    """Per-sample index into the run's source models; for 2 sources this is a seeded fair coin per sample."""
    if n_sources == 1:
        return np.zeros(batch, dtype=np.int64)
    return np.random.default_rng([mix_seed, step]).integers(0, n_sources, size=batch)


class TrainBatches:
    """Yields (step, tokens[B,N], source_idx[B]) from `start_step` on, prefetching in background threads."""

    def __init__(self, windows, batch, data_seed, mix_seed, n_sources, start_step, prefetch=32):
        self.windows, self.batch = windows, batch
        self.data_seed, self.mix_seed, self.n_sources = data_seed, mix_seed, n_sources
        self.step = start_step
        self.pool = ThreadPoolExecutor(max_workers=prefetch)
        self.pending = [self.pool.submit(self._make, s) for s in range(start_step, start_step + prefetch)]

    def _make(self, step):
        tokens = self.windows.sample(np.random.default_rng([self.data_seed, step]), self.batch)
        return tokens, source_assignment(self.mix_seed, step, self.batch, self.n_sources)

    def __iter__(self):
        return self

    def __next__(self):
        tokens, src = self.pending.pop(0).result()
        self.pending.append(self.pool.submit(self._make, self.step + len(self.pending) + 1))
        step, self.step = self.step, self.step + 1
        return step, tokens, src

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def val_windows(n_tokens, n_val, val_seed=0, allow_eod=False):
    """Fixed held-out windows from the tokenized Pile val set."""
    return TokenWindows([val_path()], n_tokens, allow_eod).sample(np.random.default_rng([val_seed]), n_val)
