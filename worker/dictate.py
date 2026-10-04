"""Dictation worker: one-shot CUDA transcribe + deterministic cleanup.

Usage:
  python dictate.py fetch                       # download + verify Phonon-2 (once, ~164 MB)
  python dictate.py transcribe <wav16k>         # load model on GPU, decode, print JSON to stdout

Design: the Rust hotkey exe spawns THIS process at key-press so the ~2-4s
model load overlaps the user's speech. On key-release the exe passes the
recorded 16 kHz mono wav; by then the model is warm and decode is ~1s for
3 min of audio on an RTX 4050. Process exits after -> zero idle memory.

No LLM correction step by design: Neutrino-0.6B is a spec-decoding draft,
not a corrector, and 0.6B-Chat hallucinates. Cleanup below is rule-based,
deterministic, never invents words.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _archive
from entrypoint import CATALOG

MODEL_KEY = "phonon-2"
MODEL_DIR = HERE.parent / "models" / MODEL_KEY

# ------------------------------------------------------------- model fetch
def cmd_fetch() -> None:
    entry = CATALOG[MODEL_KEY]
    _archive.ensure_model(
        entry["repo"], entry["filename"], entry["sha256"],
        entry["download_bytes"], MODEL_DIR, log=lambda m: print(f"[fetch] {m}", file=sys.stderr, flush=True))
    print(str(MODEL_DIR), flush=True)

# ------------------------------------------------------------- cleanup
_FILLERS = re.compile(
    r"\b(um+|uh+|er+|ah+|hmm+|mm-hmm|like,? you know)\b[,\s]*", re.IGNORECASE)
_REPEAT = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?%)\]])")
_OPEN_PUNCT = re.compile(r"([(\[])\s+")

PAUSE_S = 1.5

def wall_silences(wav, sr=16000):
    """Wall-clock silences of PAUSE_S+ seconds: (start_s, end_s) list.

    Same gate as the decoder's live session (room floor + relative gate).
    Catches pauses the TDT durations smear into stretched words.
    """
    import numpy as np

    b = int(0.05 * sr)
    n = len(wav) // b
    if n == 0:
        return []
    rms = np.sqrt((wav[: n * b].reshape(n, b).astype(np.float64) ** 2).mean(1))
    peak = float(rms.max())
    gate = max(0.004, 0.18 * peak)
    out, st = [], None
    for i, r in enumerate(rms):
        if r <= gate and st is None:
            st = i
        elif r > gate and st is not None:
            if (i - st) * 0.05 >= PAUSE_S:
                out.append((st * 0.05, i * 0.05))
            st = None
    if st is not None and (n - st) * 0.05 >= PAUSE_S:
        out.append((st * 0.05, n * 0.05))
    return out

def with_pause_marks(text: str, words: list, walls=None) -> str:
    """Insert ' ... ' at gaps of PAUSE_S+ seconds.

    Union of decoder word gaps and wall-clock silences anchored to the first
    word after their midpoint. Verified against the engine text: on any
    mismatch the engine text wins unchanged, so markers can never corrupt
    a transcript.
    """
    if not words:
        return text
    mark = [False] * len(words)
    prev_end = 0.0
    for i, w in enumerate(words):
        if i > 0 and w["start"] - prev_end >= PAUSE_S:
            mark[i] = True
        prev_end = w["end"]
    for s, e in walls or []:
        mid = (s + e) / 2.0
        idx = next((i for i, w in enumerate(words) if w["start"] >= mid), None)
        if idx:
            mark[idx] = True
    parts = []
    for i, w in enumerate(words):
        if i > 0:
            parts.append(" ... " if mark[i] else " ")
        parts.append(w["text"])
    marked = "".join(parts)
    norm = lambda t: " ".join(t.split())
    if " ".join(t for t in norm(marked).split(" ") if t != "...") == norm(text):
        return marked
    return text

def cleanup(text: str) -> str:
    """Remove fillers, collapse repeated words, fix spacing. Deterministic."""
    text = text.replace(" ... ", " \0 ")  # shield pause markers from the passes below
    text = _FILLERS.sub("", text)
    # collapse "the the" / "I I I" keeping one (case of first occurrence)
    def _one(m: re.Match) -> str:
        return m.group(1)
    prev = None
    while prev != text:
        prev = text
        text = _REPEAT.sub(_one, text)
    text = _SPACE_PUNCT.sub(r"\1", text)
    text = _OPEN_PUNCT.sub(r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = text.replace("\0", "...")
    text = re.sub(r"^\.\.\. ", "", text)  # filler removal can strand a marker
    text = re.sub(r" \.\.\.$", "", text)  # ...at either edge; drop those
    if text:
        text = text[0].upper() + text[1:]
    return text

# ------------------------------------------------------------- transcribe
def cmd_transcribe(wav_path: str) -> None:
    import numpy as np
    import soundfile as sf

    t_all = time.perf_counter()
    from win_engine import CachedTranscriber

    wav, sr = sf.read(wav_path, dtype="float32")
    if sr != 16_000:
        raise SystemExit(f"need 16 kHz mono wav, got {sr} Hz (convert: ffmpeg -i in -ar 16000 -ac 1 out.wav)")
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    wav = np.ascontiguousarray(wav, dtype=np.float32)

    t_load = time.perf_counter()
    engine = CachedTranscriber(MODEL_KEY, MODEL_DIR)
    load_s = time.perf_counter() - t_load

    t_dec = time.perf_counter()
    text, segments, words = engine.transcribe_long_timed(wav)
    decode_s = time.perf_counter() - t_dec

    clean = cleanup(with_pause_marks(text, words, wall_silences(wav)))
    out = {
        "text": clean,
        "raw": text,
        "audio_s": round(len(wav) / 16_000, 2),
        "segments": len(segments),
        "words": len(words),
        "load_s": round(load_s, 2),
        "decode_s": round(decode_s, 2),
        "total_s": round(time.perf_counter() - t_all, 2),
        "gpu": engine.describe().get("gpu"),
    }
    print(json.dumps(out), flush=True)

# ------------------------------------------------------------- main
def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("fetch", "transcribe"):
        raise SystemExit("usage: dictate.py fetch | dictate.py transcribe <wav16k>")
    if sys.argv[1] == "fetch":
        cmd_fetch()
    else:
        if len(sys.argv) < 3:
            raise SystemExit("usage: dictate.py transcribe <wav16k>")
        cmd_transcribe(sys.argv[2])

if __name__ == "__main__":
    main()
