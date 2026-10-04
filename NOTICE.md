# Third-party notices

## Rare UI — `MatrixOrb` (the orb animation in `hotkey/src/orb.rs`)

The dot-matrix orb animation in `hotkey/src/orb.rs` is a port of the **`MatrixOrb`**
React component from **Rare UI**.

- Project: **Rare UI** — <https://rareui.com>
- Upstream author: Swami Malode — Copyright (c) 2026 Swami Malode
- License: **MIT + Commons Clause v1.0 + mandatory attribution**
  (verbatim text in [`LICENSE-rare-ui`](LICENSE-rare-ui))

Rare UI is the source of the per-dot intensity model, the asymmetric envelope
follower, and the damped-spring scale used here. That design was adapted to a
hand-rolled analytic software rasteriser writing straight into a premultiplied
32-bit DIB, because the original targets the DOM/SVG.

### How the license is satisfied here

| Requirement | How it is met |
| --- | --- |
| Include the copyright + permission notice in all copies or substantial portions | [`LICENSE-rare-ui`](LICENSE-rare-ui) ships in this repository, unmodified |
| Credit Rare UI with a **visible link to https://rareui.com** where a user can find it | This file, plus the Credits section of the [README](README.md), plus a credit comment at the top of `hotkey/src/orb.rs` |
| Do not remove the credit or copyright notice from the copied source | The credit comment lives in the source file itself, not just in the README |

### The Commons Clause restriction — read this before you publish

The Commons Clause permits use, including commercially, but forbids **selling,
sublicensing or redistributing the components themselves**, whether alone, in a
bundle, or as a ported version.

Practical consequences:

- **Shipping it inside an app is fine.** Using the orb in this dictation app is
  exactly the "as part of an application" case the licence permits.
- **Publishing this repository publicly is redistribution of the component.**
  This repository is therefore kept **private**. Keep it private unless you
  remove `hotkey/src/orb.rs` or obtain Rare UI's permission.

If you ever want a public repo, the clean options are:

1. Ask Rare UI for permission to relicense/redistribute.
2. Replace `hotkey/src/orb.rs` with an independently written renderer. The orb is
   a small self-contained file — the rest of the app does not depend on it.

## Phonon / Parakeet (the speech model, in `worker/`)

The transcription worker in `worker/` serves the **Phonon-2** speech-to-text
model through NVIDIA's Parakeet-family architecture, with a CUDA/bfloat16 fast
loader in `worker/phonon2_cuda_engine.py`. The model **weights** are not part of
this repository — see `models/README.md`. Phonon and the Parakeet models carry
their own upstream licences; check the model distribution you download for the
terms that apply to it.

`worker/` contains no NVIDIA or upstream model source beyond the engine wrapper
listed above; it talks to `transformers`/`torch` as ordinary dependencies.

## Rust crates

`hotkey/` depends on `windows`, `cpal`, `hound`, and `serde_json`, each under its
own licence (MIT / Apache-2.0). `Cargo.lock` pins the exact versions used.