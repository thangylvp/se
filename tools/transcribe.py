"""ASR only (NO denoising): transcribe a wav file or directory with Whisper.

    python -m tools.transcribe out/clip_enhanced.wav
    python -m tools.transcribe out/clip_enhanced.wav --model openai/whisper-large-v3 --fp16
    python -m tools.transcribe some_dir/ --language vi
    python -m tools.transcribe clip.wav --transcript ref.txt      # also prints WER/CER
"""
import argparse
import os
from pathlib import Path

import jiwer
import torch

from afe.asr.whisper import WhisperASR
from afe.data.audio import load_wav
from afe.utils.constants import DEFAULT_NO_REPEAT_NGRAM, DEFAULT_WHISPER
from afe.utils.text import vi_norm


def main():
    ap = argparse.ArgumentParser(description="Transcribe wav(s) with Whisper — no denoising.")
    ap.add_argument("input", help="a .wav file or a directory of .wav files")
    ap.add_argument("--model", default=DEFAULT_WHISPER)
    ap.add_argument("--fp16", action="store_true", help="load ASR in float16 (large-v3 on 8GB)")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--no-repeat-ngram", type=int, default=DEFAULT_NO_REPEAT_NGRAM)
    ap.add_argument("--transcript", default=None, help="ground-truth .txt to score one file")
    ap.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    args = ap.parse_args()

    dev = f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    gk = {"no_repeat_ngram_size": args.no_repeat_ngram} if args.no_repeat_ngram > 0 else {}

    p = Path(args.input)
    files = sorted(p.glob("*.wav")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"No .wav files at {args.input}")

    asr = WhisperASR(args.model, device=dev, dtype=torch.float16 if args.fp16 else None)
    hyps = asr.transcribe([load_wav(str(f)) for f in files], language=args.language, **gk)
    for f, h in zip(files, hyps):
        print(f"{f.name}: {h.strip()}")

    if args.transcript and len(files) == 1 and os.path.exists(args.transcript):
        ref = open(args.transcript, encoding="utf-8").read().strip()
        r, h = vi_norm(ref), vi_norm(hyps[0])
        if r.strip():
            print(f"\nGT : {ref}\nWER {jiwer.wer(r, h)*100:.2f}   CER {jiwer.cer(r, h)*100:.2f}")


if __name__ == "__main__":
    main()
