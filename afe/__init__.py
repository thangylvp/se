"""afe — Audio Front-End for ASR-aware speech enhancement.

Fine-tune a DeepVQE speech-enhancement front-end so it lowers a downstream
ASR's WER (not just denoises), using a frozen-Whisper ASR loss.
"""
__version__ = "0.1.0"
