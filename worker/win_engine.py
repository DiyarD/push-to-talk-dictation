"""Windows fast loader for Phonon-2 CUDA (Apache-2.0, derives from
fermionresearch/phonon docker-phonon2/phonon2_cuda_engine.py).

Identical decode to the vendored Transcriber: same HF_CONFIG graph, same
bf16 values (same fp32->bf16 rounding at cache build), same greedy TDT +
CUDA graphs + GPU mel. Only the LOAD is faster: dense bf16 safetensors
mapped straight to GPU (~2s) instead of numpy five_value expansion (~15s)
+ fp32 copies + CPU random-init (~10s).

Falls back to the stock loader for packed/Triton or non-bf16 configs.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONTAINER_SHA = "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e"

from phonon2_cuda_engine import (  # noqa: E402
    HF_CONFIG,
    WIN,
    Transcriber as _Base,
    log,
    mel_filters,
)


def _cache_paths(model_dir: Path):
    model_dir = Path(model_dir)
    return model_dir / "dense-bf16.safetensors", model_dir / "dense-bf16.json"


def cache_valid(model_dir: Path) -> bool:
    cache, pin = _cache_paths(model_dir)
    if not (cache.is_file() and pin.is_file()):
        return False
    try:
        pin = json.loads(pin.read_text())
        if pin.get("container_sha256") != CONTAINER_SHA:
            return False
        want = pin.get("model_fermion_sha256")
        if not want:
            return False
    except Exception:
        return False
    # the premise of bit-identity: the container is still the pinned bytes
    container = Path(model_dir) / "model.fermion"
    if not container.is_file():
        return False
    digest = hashlib.sha256()
    with open(container, "rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == want


class CachedTranscriber(_Base):
    def __init__(self, model_key: str, model_dir: Path):
        try:
            import torch
        except ImportError:
            raise SystemExit("[phonon-cuda] PyTorch is not installed")
        if not torch.cuda.is_available():
            raise SystemExit("[phonon-cuda] no CUDA device available")
        from transformers import ParakeetForTDT, ParakeetTDTConfig

        self.torch = torch
        self.model_key = model_key
        self.model_dir = Path(model_dir)
        self.entry = {"name": "Phonon-2", "repo": "FermionResearch/Phonon-2", "profile": "five-value",
                      "aliases": ("phonon-2", "phonon2", "phonon")}
        self.accepted_names = {"", "phonon", "phonon-cuda", "phonon-2", "phonon2", "fermionresearch/phonon-2"}
        self.path_kind = os.environ.get("PHONON2_CUDA_PATH", "dense").lower()
        self.pad_s = float(os.environ.get("PHONON2_CUDA_PAD_S", "5.0"))
        self.K = int(os.environ.get("PHONON2_CUDA_K", "16"))
        self.dtype = getattr(torch, os.environ.get("PHONON2_CUDA_DTYPE", "bfloat16"))
        self.packed = False

        use_cache = (self.path_kind == "dense" and self.dtype == torch.bfloat16
                     and cache_valid(self.model_dir))
        if not use_cache:
            log("fast cache unusable for this config; stock loader ...")
            super().__init__(model_key, model_dir)
            return

        started = time.perf_counter()
        log(f"loading Phonon-2 from fast cache in {self.model_dir} ...")
        from accelerate import init_empty_weights
        from safetensors.torch import load_file

        cache, _ = _cache_paths(self.model_dir)
        cfg = ParakeetTDTConfig(**HF_CONFIG)
        with init_empty_weights():
            model = ParakeetForTDT(cfg)
        state = load_file(str(cache), device="cuda:0")
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise SystemExit(f"[phonon-cuda] cache load: missing {missing[:5]} unexpected {list(unexpected)[:5]}")
        del state
        model = model.eval()
        self.model = model
        self.cfg = cfg
        self._finish_init(torch, started)

    def _finish_init(self, torch, started: float) -> None:
        # mirror of Transcriber.__init__ tail (vendored file, same version)
        import math

        import numpy as np

        cfg = self.cfg
        model = self.model
        self.blank = cfg.blank_token_id
        self.V = cfg.vocab_size
        self.durations = torch.tensor(cfg.durations, device="cuda", dtype=torch.long)
        self.max_sym = getattr(cfg, "max_symbols_per_step", 10) or 10
        lstm = model.decoder.lstm
        self.L = lstm.num_layers
        self.H = lstm.hidden_size
        self.Wih = [getattr(lstm, f"weight_ih_l{l}") for l in range(self.L)]
        self.Whh = [getattr(lstm, f"weight_hh_l{l}") for l in range(self.L)]
        self.bih = [getattr(lstm, f"bias_ih_l{l}", None) for l in range(self.L)]
        self.bhh = [getattr(lstm, f"bias_hh_l{l}", None) for l in range(self.L)]
        vocab = json.loads((self.model_dir / "config.json").read_text())["joint"]["vocabulary"]
        self.vocab = list(vocab)
        self.window = torch.hann_window(WIN, periodic=False, device="cuda")
        self.melf = torch.from_numpy(mel_filters()).cuda()
        self.graphs = {}
        self.device_name = torch.cuda.get_device_name(0)
        self.load_s = time.perf_counter() - started
        mode = os.environ.get("PHONON2_CUDA_WARMUP", "all").lower()
        t_w = time.perf_counter()
        if mode == "all" and self.pad_s > 0:
            nb = int(math.ceil(30.0 / self.pad_s))
            for k in range(1, nb + 1):
                self.transcribe(np.zeros(int(k * self.pad_s * 16000) - 1, dtype=np.float32))
            self.warm_s = time.perf_counter() - t_w
            self.warm_buckets = nb
        else:
            self.transcribe(np.zeros(int(2 * 16000), dtype=np.float32))
            self.warm_s = time.perf_counter() - t_w
            self.warm_buckets = 1
        log(f"loaded in {self.load_s:.1f}s on {self.device_name} (fast cache, "
            f"{str(self.dtype).split('.')[-1]}, CUDA graphs K={self.K}, pad bucket {self.pad_s:.0f} s; "
            f"warmed {self.warm_buckets} bucket(s) in {self.warm_s:.1f}s)")
