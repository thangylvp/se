"""Standalone DeepVQE denoising (file or directory) -> writes *_enhanced.wav.

    python -m tools.infer --checkpoint checkpoints/model_D.tar --input clip.wav --output-dir out
    python -m tools.infer --checkpoint checkpoints/model_D.tar --input some_dir/ --output-dir out --device 0
"""
import argparse
from pathlib import Path

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

from afe.checkpoint.io import load_se_model
from afe.utils.constants import SR


def enhance_file(model, input_path, device):
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    orig_sr = sr
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    wav = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.inference_mode():
        enhanced = model(wav).squeeze(0).cpu().numpy()
    if orig_sr != SR:
        enhanced = librosa.resample(enhanced, orig_sr=SR, target_sr=orig_sr)
    return enhanced, orig_sr


def main():
    ap = argparse.ArgumentParser(description="DeepVQE denoising inference.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="WAV file or directory")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None, help="GPU index, e.g. 0. Omit for CPU.")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}"
                          if args.device is not None and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_se_model(args.checkpoint, device=device)
    p = Path(args.input)
    wavs = [p] if p.is_file() else (sorted(p.glob("*.wav")) + sorted(p.glob("*.WAV")))
    if not wavs:
        raise SystemExit(f"No WAV files at {args.input}")
    print(f"Processing {len(wavs)} file(s) -> {out_dir}")
    for wav_path in tqdm(wavs):
        enhanced, sr = enhance_file(model, str(wav_path), device)
        sf.write(str(out_dir / f"{wav_path.stem}_enhanced.wav"), enhanced, sr)


if __name__ == "__main__":
    main()
