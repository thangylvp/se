"""On-the-fly clean+noise mixing dataset for ASR-aware training."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from afe.data.audio import fit_noise_length, load_wav, mix_at_snr
from afe.data.librispeech import list_librispeech
from afe.utils.constants import SR

MAX_TRAIN_SAMPLES = 20 * SR  # cap training utts at 20 s for batching sanity

# SNR pool weighted toward the -5..+5 dB band where the SE hurt WER.
DEFAULT_SNR_POOL = (-10, -5, -5, 0, 0, 0, 5, 5, 10, 20)


class MixDataset(Dataset):
    def __init__(self, librispeech_dir, noise_dir, snr_choices=None, seed=0):
        self.utts = list_librispeech(librispeech_dir)
        self.noises = sorted(Path(noise_dir).rglob("*.wav"))
        self.snr_pool = list(snr_choices or DEFAULT_SNR_POOL)
        self.seed = seed

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 1_000_003 + idx)
        uid, wav_path, text = self.utts[idx]
        clean = load_wav(wav_path)
        if len(clean) > MAX_TRAIN_SAMPLES:
            start = rng.randint(0, len(clean) - MAX_TRAIN_SAMPLES)
            clean = clean[start:start + MAX_TRAIN_SAMPLES]
            text = None  # crop no longer matches transcript -> skip CE for this item
        noise = load_wav(str(self.noises[rng.randrange(len(self.noises))]))
        noise = fit_noise_length(noise, len(clean), rng)
        snr = rng.choice(self.snr_pool)
        noisy = mix_at_snr(clean, noise, snr, rng=rng)
        return {"noisy": noisy.astype(np.float32),
                "clean": clean.astype(np.float32),
                "text": text, "snr": snr}


def make_collate(processor):
    """Collate that pads waveforms and tokenizes transcripts to Whisper labels
    (-100 for padding / text-less crops, ignored by CE)."""
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="en", task="transcribe")

    def collate(batch):
        L = max(len(b["noisy"]) for b in batch)
        noisy = torch.zeros(len(batch), L)
        clean = torch.zeros(len(batch), L)
        for i, b in enumerate(batch):
            noisy[i, :len(b["noisy"])] = torch.from_numpy(b["noisy"])
            clean[i, :len(b["clean"])] = torch.from_numpy(b["clean"])
        label_ids = [tok(b["text"]).input_ids if b["text"] else [] for b in batch]
        Ll = max((len(x) for x in label_ids), default=1) or 1
        labels = torch.full((len(batch), Ll), -100, dtype=torch.long)
        for i, ids in enumerate(label_ids):
            if ids:
                labels[i, :len(ids)] = torch.tensor(ids)
        return {"noisy": noisy, "clean": clean, "labels": labels}

    return collate
