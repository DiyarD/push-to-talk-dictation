"""Profile Phonon-2 CUDA load stages (vendored engine untouched)."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
t0 = time.perf_counter()

import numpy as np

t = time.perf_counter()
import torch
print(f"import torch: {time.perf_counter()-t:.2f}s", flush=True)

t = time.perf_counter()
print(f"cuda available: {torch.cuda.is_available()}", flush=True)
torch.cuda.init()
print(f"cuda init: {time.perf_counter()-t:.2f}s", flush=True)

from transformers import ParakeetForTDT, ParakeetTDTConfig
from phonon2_cuda_engine import HF_CONFIG
from fermion_container import read_container

model_dir = HERE.parent / "models" / "phonon-2"

t = time.perf_counter()
tensors, index, raw = read_container(str(model_dir / "model.fermion"), with_raw=True)
print(f"read_container+expand: {time.perf_counter()-t:.2f}s", flush=True)

t = time.perf_counter()
import re
cfg = ParakeetTDTConfig(**HF_CONFIG)
model = ParakeetForTDT(cfg)
print(f"model init: {time.perf_counter()-t:.2f}s", flush=True)

t = time.perf_counter()
sd = {}
for k, v in tensors.items():
    if k.endswith("num_batches_tracked"):
        v32 = float(np.asarray(v, dtype=np.float32).reshape(-1)[0])
        sd[k] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64)
    else:
        arr = np.ascontiguousarray(v.astype(np.float32))
        sd[k] = torch.from_numpy(arr)
print(f"numpy->torch: {time.perf_counter()-t:.2f}s", flush=True)

t = time.perf_counter()
model.load_state_dict(sd, strict=False)
print(f"load_state_dict: {time.perf_counter()-t:.2f}s", flush=True)

t = time.perf_counter()
model = model.to(torch.bfloat16).cuda().eval()
print(f"to(bf16).cuda: {time.perf_counter()-t:.2f}s", flush=True)
print(f"TOTAL: {time.perf_counter()-t0:.2f}s", flush=True)
