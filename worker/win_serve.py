"""Serve Phonon-2 on loopback with the Windows fast loader.

Same OpenAI-compatible endpoints as entrypoint.py (POST
/v1/audio/transcriptions, GET /v1/audio/stream, GET /health) but the model
loads from the dense-bf16 safetensors cache (~8s) instead of expanding the
five-value container (~35s). Bit-identical transcripts to the stock path.

Usage:
  python win_serve.py --model-dir <dir> --port 8765 [--api-key KEY]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import entrypoint  # noqa: E402
from win_engine import CachedTranscriber  # noqa: E402

entrypoint.Transcriber = CachedTranscriber
entrypoint.BRAND = "phonon-win"

if __name__ == "__main__":
    sys.argv[0] = "phonon-win"
    entrypoint.main()
