# models/

Model weights live here. **They are not in git** — this directory is ignored
(1.4 GB).

The app expects a single model directory:

```
models/phonon-2/
├── config.json
├── dense-bf16.json          # fast-loader descriptor
├── dense-bf16.safetensors   # bf16 weights
├── model.fermion            # packed container
└── packed_manifest.json
```

`hotkey.exe` passes `--model-dir models/phonon-2` to the worker on every spawn,
so if this directory is missing, dictation fails at press time rather than
downloading anything.

## Getting the model

Download the Phonon-2 release archive from its Hugging Face distribution and
unpack it so that the files above sit directly in `models/phonon-2/` (not nested
one level deeper).

To fetch and verify a published model yourself, the bundled entrypoint knows how
to do this — it downloads from Hugging Face and checks the release's published
SHA-256 pin before unpacking:

```powershell
python worker\win_serve.py serve --model phonon-2 --port 8765
```

Downloads land in the cache root, not here:

- `%USERPROFILE%\.cache\phonon` by default
- `$env:PHONON_HOME` if set

Copy or symlink the unpacked result into `models/phonon-2` for the app to use it.

## Verify it

```powershell
# starts the server; you want to see "serving phonon-2 on http://127.0.0.1:8765"
python worker\win_serve.py serve --model-dir models\phonon-2 --port 8765
```

Check `worker/dictation-worker.log` for the load line — on a CUDA machine it
should name your GPU and report a bfloat16 fast-cache load in a few seconds.