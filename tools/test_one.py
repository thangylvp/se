"""Test ONE file: transcribe without vs with DeepVQE denoising, print WER/CER.

    python -m tools.test_one clip.wav
    python -m tools.test_one clip.wav --model openai/whisper-large-v3 --fp16 --save-enhanced out/e.wav
    python -m tools.test_one clip.wav --transcript ref.txt --checkpoint ../ckpt/deepvqe/model_240.tar
"""
import argparse
import os

import jiwer
import soundfile as sf
import torch

from afe.asr.whisper import WhisperASR
from afe.checkpoint.io import load_se_model
from afe.data.audio import load_wav
from afe.data.robot import autofind_transcript
from afe.utils.constants import DEFAULT_NO_REPEAT_NGRAM, DEFAULT_WHISPER, SR
from afe.utils.text import vi_norm


def _score(ref, hyp):
    r, h = vi_norm(ref), vi_norm(hyp)
    return (None, None) if not r.strip() else (jiwer.wer(r, h) * 100, jiwer.cer(r, h) * 100)


def main():
    ap = argparse.ArgumentParser(description="Transcribe one file with/without DeepVQE denoising.")
    ap.add_argument("wav")
    ap.add_argument("--checkpoint", default="checkpoints/model_D.tar")
    ap.add_argument("--model", default=DEFAULT_WHISPER)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--transcript", default=None, help="GT .txt (else auto-find; else skip WER)")
    ap.add_argument("--save-enhanced", default=None)
    ap.add_argument("--no-repeat-ngram", type=int, default=DEFAULT_NO_REPEAT_NGRAM)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}"
                          if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    gk = {"no_repeat_ngram_size": args.no_repeat_ngram} if args.no_repeat_ngram > 0 else {}

    wav = load_wav(args.wav)
    se = load_se_model(args.checkpoint, device=device)
    with torch.inference_mode():
        enh = se(torch.from_numpy(wav).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    if args.save_enhanced:
        os.makedirs(os.path.dirname(args.save_enhanced) or ".", exist_ok=True)
        sf.write(args.save_enhanced, enh, SR)

    asr = WhisperASR(args.model, device=str(device), dtype=torch.float16 if args.fp16 else None)
    hyp_noisy = asr.transcribe([wav], language=args.language, **gk)[0].strip()
    hyp_enh = asr.transcribe([enh], language=args.language, **gk)[0].strip()

    tpath = args.transcript or autofind_transcript(args.wav)
    ref = open(tpath, encoding="utf-8").read().strip() if tpath and os.path.exists(tpath) else None

    print(f"\nfile      : {args.wav}")
    print(f"ASR       : {args.model} (fp16={args.fp16})   denoiser: {args.checkpoint}")
    if ref is not None:
        print(f"GT        : {ref}   [{tpath}]")
    print(f"\nwithout model (noisy): {hyp_noisy}")
    print(f"with model (denoised): {hyp_enh}")
    if ref is not None:
        wn, cn = _score(ref, hyp_noisy)
        we, ce = _score(ref, hyp_enh)
        print(f"\n{'':22}{'WER':>8}{'CER':>8}")
        print(f"{'without model':22}{wn:8.2f}{cn:8.2f}")
        print(f"{'with model':22}{we:8.2f}{ce:8.2f}")
        print(f"{'delta (with-without)':22}{we-wn:+8.2f}{ce-cn:+8.2f}  (negative = denoise helped)")
    else:
        print("\n(no transcript found -> WER/CER skipped; pass --transcript to score)")
    if args.save_enhanced:
        print(f"\nenhanced wav saved -> {args.save_enhanced}")


if __name__ == "__main__":
    main()
