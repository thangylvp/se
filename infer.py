"""
Standalone DeepVQE denoising inference (single file or directory).

Example:
    python infer.py \
        --checkpoint "../ckpt/deepvqe/model_240 2.tar" \
        --input ../noisy_audio/audio_8c3e9f2b_1773733142.wav \
        --output-dir out

    python infer.py --checkpoint <ckpt.tar> --input <wav_dir> --output-dir out --device 0
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from se_model import load_se_model

TARGET_SR = 16000


def enhance_file(model, input_path, device, target_sr=TARGET_SR):
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono: first channel
    original_sr = sr
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    wav = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, L)
    with torch.inference_mode():
        enhanced = model(wav).squeeze(0).cpu().numpy()

    if original_sr != target_sr:
        enhanced = librosa.resample(enhanced, orig_sr=target_sr, target_sr=original_sr)
    return enhanced, original_sr


def collect_wavs(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        wavs = sorted(p.glob("*.wav")) + sorted(p.glob("*.WAV"))
        if not wavs:
            raise ValueError(f"No WAV files in {input_path}")
        return wavs
    raise FileNotFoundError(input_path)


def main():
    ap = argparse.ArgumentParser(description="DeepVQE denoising inference.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="WAV file or directory")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None, help="GPU index, e.g. 0. Omit for CPU.")
    args = ap.parse_args()

    if args.device is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_se_model(args.checkpoint, device=device)
    wavs = collect_wavs(args.input)
    print(f"Processing {len(wavs)} file(s) -> {out_dir}")

    for wav_path in tqdm(wavs):
        enhanced, sr = enhance_file(model, str(wav_path), device)
        out_path = out_dir / f"{wav_path.stem}_enhanced.wav"
        sf.write(str(out_path), enhanced, sr)


if __name__ == "__main__":
    main()
