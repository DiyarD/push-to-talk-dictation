"""One-time: expand Phonon-2 five-value container -> dense bf16 safetensors.

The stock loader path (numpy five_value expansion + fp32 copies + CPU model
init) costs ~30s on a laptop. This cache pays it once; loads afterwards read
1.2 GB of bf16 straight to GPU in ~2s with bit-identical values to the
gated dense path (same fp32->bf16 rounding, same shapes incl. conv [...,1]).

Cache layout: models/phonon-2/dense-bf16.safetensors + dense-bf16.json pin.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

CONTAINER_SHA = "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e"
MODEL_DIR = HERE.parent / "models" / "phonon-2"
CACHE = MODEL_DIR / "dense-bf16.safetensors"
PIN = MODEL_DIR / "dense-bf16.json"


def build() -> None:
    import re
    import torch
    from safetensors.torch import save_file

    from fermion_container import read_container

    print("[cache] expanding container (one-time, ~30s) ...", flush=True)
    t = time.perf_counter()
    tensors, _ = read_container(str(MODEL_DIR / "model.fermion"))
    sd = {}
    for k, v in tensors.items():
        if k.endswith("num_batches_tracked"):
            v32 = float(np.asarray(v, dtype=np.float32).reshape(-1)[0])
            sd[k] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64)
        else:
            arr = np.ascontiguousarray(v)
            if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", k):
                arr = arr[:, :, None]
            t16 = torch.from_numpy(arr.astype(np.float16) if arr.dtype != np.float16 else arr.copy())
            sd[k] = t16.to(torch.bfloat16)
    del tensors
    print(f"[cache] expanded in {time.perf_counter()-t:.1f}s, saving ...", flush=True)
    t = time.perf_counter()
    save_file(sd, CACHE)
    digest = hashlib.sha256()
    with open(MODEL_DIR / "model.fermion", "rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""):
            digest.update(chunk)
    PIN.write_text(json.dumps({"container_sha256": CONTAINER_SHA,
                               "model_fermion_sha256": digest.hexdigest(),
                               "tensors": len(sd)}))
    print(f"[cache] saved {CACHE} ({CACHE.stat().st_size/1e9:.2f} GB) in {time.perf_counter()-t:.1f}s", flush=True)


if __name__ == "__main__":
    build()
