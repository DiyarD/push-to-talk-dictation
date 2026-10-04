# Push-to-Talk Dictation

**Hold a hotkey, talk, and the words land in your cursor. Everything runs on
your own GPU. Nothing is uploaded.**

Press <kbd>Win</kbd>+<kbd>Shift</kbd>+<kbd>D</kbd>, speak, press it again — the
transcript is pasted wherever your cursor was. A small glowing orb shows what
the app is doing at every moment.

There is no window to focus, no button to click, and no cloud service in the
loop. Speech is captured, transcribed locally, and pasted.

---

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Using it](#using-it)
- [The orb](#the-orb)
- [Configuration](#configuration)
- [Autostart](#autostart)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Credits and licensing](#credits-and-licensing)

---

## What it does

- **Push to talk.** One global hotkey starts and stops recording.
- **Fully local.** Audio never leaves the machine. The model runs on your GPU.
- **Pastes for you.** Text is inserted at the cursor on release — no clipboard
  round-trip, no "copy" step to forget.
- **A 5-second window to grab it.** After pasting, the orb lingers. Click it and
  the text is copied to your clipboard instead, in case you want it elsewhere.
- **Cleans up speech.** Filler sounds (`um`, `uh`, `er`, …) are dropped, repeated
  words are collapsed, and a pause of 1.5 s or more becomes ` ... ` so your
  sentences keep their shape.
- **Costs nothing when idle.** The Rust process is ~14 MB and does no work until
  you press the hotkey. The GPU model is only loaded on first use, and is shut
  down again after 3 minutes of inactivity.

## How it works

Two processes, split along a hard boundary: **anything that can block runs
somewhere the UI cannot see it.**

```
  ┌─────────────────────────────────────────────┐
  │  hotkey.exe  (Rust, ~14 MB, always running)  │
  │                                              │
  │   Win+Shift+D ──► record mic (cpal)          │
  │                 ──► downsample to 16 kHz mono│
  │                 ──► orb: idle → listening    │
  │                 ──► POST to 127.0.0.1:8765   │
  │                 ──► orb: thinking → ready     │
  │                 ──► paste at cursor (SendInput)│
  │                                              │
  │   owns the mic, the hotkey, the orb, the UI │
  └───────────────┬─────────────────────────────┘
                  │ spawns on first press,
                  │ killed after 180 s idle
                  ▼
  ┌─────────────────────────────────────────────┐
  │  worker/  (Python, spawned on demand)        │
  │                                              │
  │   win_serve.py ──► Phonon-2 on CUDA/bf16    │
  │   OpenAI-compatible API on loopback :8765    │
  │   POST /v1/audio/transcriptions             │
  └─────────────────────────────────────────────┘
```

**Why the split?** Model loading takes seconds. If that happened on the thread
owning the mic and the window, the orb would freeze mid-animation and the app
would look hung. Instead the transcription runs on a worker thread, the orb
keeps animating at 60 fps, and it stays clickable the entire time — you can
cancel a slow load with <kbd>Esc</kbd> or a click.

**Why is the model a separate process?** GPU teardown in-process is unreliable:
a crashed or killed process can leave the CUDA context and ~1.2 GB of VRAM
allocated with nothing to free it. A child process can always be killed from the
outside. The parent tracks the child's PID, kills it when it has been idle for
three minutes, and adopts or kills an orphan left behind by a previous run.

### Repository layout

```
push-to-talk-dictation/
├── hotkey/                  Rust: hotkey, mic, orb, paste
│   ├── src/main.rs          dictation flow, capture, HTTP, paste
│   ├── src/orb.rs           the orb window and its animation
│   ├── manifest.xml         declares PerMonitorV2 DPI awareness
│   └── build.rs             embeds the manifest at link time
├── worker/                  Python: local transcription server
│   ├── win_serve.py         entrypoint — Phonon-2 via the CUDA fast loader
│   ├── entrypoint.py        HTTP API, model resolution/verification
│   ├── win_engine.py        bf16 safetensors fast loader
│   └── phonon2_cuda_engine.py
└── models/phonon-2/         model weights (not in git — see models/README.md)
```

## Requirements

- **Windows 10 or 11.** Win32 + GDI; there is no cross-platform path.
- **Rust toolchain** (stable, edition 2024) to build `hotkey.exe`.
- **An NVIDIA GPU + CUDA.** The worker loads the model in bfloat16 with CUDA
  graphs. CPU-only will not work — this is the whole point of the fast loader.
- **Python** in a conda env named `dictation` (or point `DICTATION_PYTHON` at any
  interpreter) with `torch`, `transformers` and CUDA available.
- **A default microphone.**
- **~1.4 GB** for the model in `models/phonon-2/`.

## Setup

```powershell
git clone https://github.com/DiyarD/push-to-talk-dictation.git
cd push-to-talk-dictation

# 1. Build the Rust side
cd hotkey
cargo build --release
cd ..

# 2. Fetch the model into models/phonon-2  (see models/README.md for the
#    Hugging Face archive and the SHA-256-verified download route)
```

Then run it:

```powershell
.\hotkey\target\release\hotkey.exe
```

There is no console window — it is a GUI subsystem binary. To watch what it is
doing, tail the log:

```powershell
Get-Content .\worker\hotkey.log -Wait
```

## Using it

| Action | Result |
| --- | --- |
| <kbd>Win</kbd>+<kbd>Shift</kbd>+<kbd>D</kbd> | Start recording. Press again to stop, transcribe and paste. |
| Click the orb | Same as the hotkey: stops recording. |
| <kbd>Esc</kbd> while recording | Cancel and discard. |
| Click the orb while it is *thinking* | Cancel — the result is discarded when it arrives. |
| Click the orb in the 5 s after pasting | Copy the text to the clipboard instead (and restart the 5 s). |
| Click the orb while red | Dismiss the error. |

The hotkey is the only thing you need. The orb is clickable but not required.

Because the app pastes at the cursor, put the cursor where you want the text
*before* you start talking.

## The orb

The entire window is the orb: a layered, always-on-top, click-through-where-
-transparent popup with no chrome, no background and no child controls. It is
drawn by rasterising an 11×11 dot matrix analytically into a premultiplied
32-bit DIB and pushing it with `UpdateLayeredWindow` — no GDI+, no Direct2D.

| Colour | Meaning |
| --- | --- |
| Light blue `#60A5FA` | Running, but the model is not loaded yet |
| Deep blue `#2563EB` | Model loaded and warm |
| Red `#F87171` | Something went wrong (mic busy, worker died, HTTP error) |

Colour tracks **model state, not recording state**, so it deepens once and stays
deep for the rest of the session.

| State | Behaviour |
| --- | --- |
| **Idle** | Waiting. Slowly breathing. |
| **Listening** | Recording. Triggered by actual voice, not by the timer: an adaptive noise-floor gate reacts to real sound and ignores a noisy room. |
| **Thinking** | Audio sent, waiting on the model. Stays clickable so you can cancel. |
| **Ready** | Text pasted. Renders like idle, with a brief flash. Fades out 5 s later over 0.5 s. |
| **Error** | Red. Stays until dismissed. |

A useful detail: while the orb is fading out you can still click it to copy the
text. Clicking resets the 5-second timer.

### DPI

The orb is sized in device pixels and rebuilt on `WM_DPICHANGED`, so it is crisp
on mixed-DPI setups instead of being bitmap-stretched by the compositor.

Getting this right required embedding a manifest that declares
`PerMonitorV2, PerMonitor` awareness — see `hotkey/build.rs` for why the linker
must be told to do this, and `hotkey/src/orb.rs` for the runtime check that
verifies the awareness was actually granted rather than assuming it.

## Configuration

Everything has a working default; these are the escape hatches.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DICTATION_PYTHON` | first of `~\.conda\envs\dictation\python.exe`, `C:\ProgramData\anaconda3\envs\dictation\python.exe`, else `python` | Which interpreter runs the worker |
| `PHONON_HOME` | `%USERPROFILE%\.cache\phonon` | Model cache root, used by the worker's own downloader |

Layout is discovered, not configured: `hotkey.exe` finds `worker/` by walking up
from `target/<profile>/`, then next to itself. `models/phonon-2` is resolved
relative to `worker/`.

Fixed by design, not exposed: loopback port **8765**, idle eviction **180 s**,
pause threshold **1.5 s**, orb grid **11×11**.

## Autostart

Press <kbd>Win</kbd>+<kbd>R</kbd>, paste

```
shell:startup
```

and drop a shortcut to `hotkey.exe` in there. The shortcut's *Start in* should be
the `hotkey` directory.

The app is single-instance: the global hotkey registration is claimed at startup,
so launching a second copy makes it exit immediately rather than fight over the
microphone.

## Troubleshooting

**Logs.** Two files, both in `worker/`:

- `hotkey.log` — startup, DPI, orb geometry, worker spawn/paste
- `dictation-worker.log` — model loading, GPU, HTTP requests

**Nothing happens when I press the hotkey.**
Another app owns <kbd>Win</kbd>+<kbd>Shift</kbd>+<kbd>D</kbd>. Check
`hotkey.log` for `RegisterHotKey failed (already running?)` — that line means a
second copy of this app is live, not a foreign app.

**The orb is blurry on my monitor.**
Startup should log `dpi awareness: per-monitor-aware (2)`. If it logs anything
else, the manifest was not embedded — rebuild so `build.rs` reruns.

**Red orb / transcription fails.**
Read `dictation-worker.log`. Almost always one of: no model in
`models/phonon-2/`, no CUDA GPU, or the `dictation` Python env missing a package.

**It is slow the first time, fast after.**
The first press pays a one-off model load (a few seconds from the bf16 fast
cache). After that the worker stays warm for three minutes.

**It uses a lot of VRAM.**
That is the model (~1.2 GB). It is released when the idle timer fires.

**I want it to not run on startup.**
Delete the shortcut from `shell:startup`.

## Development

```powershell
cd hotkey

cargo build --release     # the shipping binary
cargo fmt
cargo clippy --all-targets     # clean, no warnings
```

Two offline test modes, useful without a microphone or a hotkey:

```powershell
# transcribe a wav straight through the pipeline
.\target\release\hotkey.exe --test C:\path\to\speech.wav

# check the microphone: sample rate, peak, RMS
.\target\release\hotkey.exe --mic-test 3
```

Notes for anyone changing the code:

- **The audio callback must stay cheap.** It runs on the real-time thread and
  only accumulates one multiply-add per sample plus one relaxed atomic store per
  callback. Do not allocate or lock there.
- **The orb's frame clock is the UI thread.** Anything slow belongs on a worker
  thread, handing the result back through a mutex that the frame loop drains.
- **`UpdateLayeredWindow` DIBs are BGRA, not RGBA.** The pixels live in memory as
  `[B, G, R, A]`, so blue is the *unshifted* byte. Get this backwards and the orb
  renders orange.

## Credits and licensing

This project's own code is MIT — see [`LICENSE`](LICENSE).

**The orb animation is not mine.** `hotkey/src/orb.rs` ports the `MatrixOrb`
component from **[Rare UI](https://rareui.com)** by Swami Malode, which is
licensed **MIT + Commons Clause v1.0 + mandatory attribution**. Its verbatim
terms are in [`LICENSE-rare-ui`](LICENSE-rare-ui), and the obligations — plus an
important note about keeping this repository private — are explained in
[`NOTICE.md`](NOTICE.md).

Model weights are not redistributed here; see `models/README.md`. The Rust
crates and Python packages keep their own licences.

## License

MIT for this project's code. `hotkey/src/orb.rs` is additionally subject to
Rare UI's terms. Read [`NOTICE.md`](NOTICE.md) before you publish this.