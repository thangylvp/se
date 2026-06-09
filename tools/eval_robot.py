"""Real-data robot eval: WER/CER without vs with denoising (Vietnamese).

    python -m tools.eval_robot --device 0
    python -m tools.eval_robot --model openai/whisper-large-v3 --fp16 --detail
    python -m tools.eval_robot --per-file-csv outputs/robot_per_file.csv --folder-csv outputs/robot_folder.csv
"""
import argparse
import csv
import os

import torch

from afe.evaluation.robot import (aggregate, folder_means, per_file_records, run,
                                   win_loss)
from afe.utils.constants import DEFAULT_NO_REPEAT_NGRAM, DEFAULT_WHISPER


def _table(agg, conds, metric):
    cats = sorted(k for k in agg if k != "OVERALL")
    hdr = f"{'group':40} {'n':>4} " + " ".join(f"{c:>10}" for c in conds)
    print(f"\n=== {metric.upper()} (%) ===")
    print(hdr)
    print("-" * len(hdr))
    for cat in cats + (["OVERALL"] if "OVERALL" in agg else []):
        row = agg[cat]
        n = row[conds[0]]["n"]
        cells = " ".join(f"{row[c][metric]*100:>10.2f}" for c in conds)
        print(f"{cat:40} {n:>4} {cells}")


def main():
    ap = argparse.ArgumentParser(description="Robot real-data WER/CER (no-denoise vs denoise).")
    ap.add_argument("--data-root", default="../data_test_robot")
    ap.add_argument("--checkpoint-d", default="checkpoints/model_D.tar")
    ap.add_argument("--checkpoint-240", default="../ckpt/deepvqe/model_240.tar")
    ap.add_argument("--no-240", action="store_true", help="skip the fidelity model_240 condition")
    ap.add_argument("--model", default=DEFAULT_WHISPER)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--no-repeat-ngram", type=int, default=DEFAULT_NO_REPEAT_NGRAM)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--detail", action="store_true", help="subset/category + win-loss + examples")
    ap.add_argument("--examples", type=int, default=0)
    ap.add_argument("--per-file-csv", default=None)
    ap.add_argument("--folder-csv", default=None)
    args = ap.parse_args()

    checkpoints = [("noisy", None)]
    if not args.no_240 and os.path.exists(args.checkpoint_240):
        checkpoints.append(("model_240", args.checkpoint_240))
    checkpoints.append(("model_D", args.checkpoint_d))

    pairs, hyps = run(checkpoints, data_root=args.data_root, whisper=args.model,
                      dtype=torch.float16 if args.fp16 else None, language=args.language,
                      no_repeat_ngram=args.no_repeat_ngram, batch_size=args.batch_size,
                      device=args.device)
    conds = list(hyps)
    print(f"\n{len(pairs)} utterances | ASR={args.model} fp16={args.fp16} lang={args.language}")

    overall = aggregate(pairs, hyps, "OVERALL")
    _table(overall, conds, "wer")
    _table(overall, conds, "cer")

    if args.detail:
        for level in ("subset", "cat"):
            agg = {**aggregate(pairs, hyps, level), **overall}
            _table(agg, conds, "wer")
            _table(agg, conds, "cer")
        recs = per_file_records(pairs, hyps)
        print("\n=== PER-UTTERANCE WIN/LOSS vs noisy (by CER) ===")
        for c in conds[1:]:
            i, w, t = win_loss(recs, conds[0], c)
            print(f"  {c:10} improved {i:3}/{len(recs)}  worse {w:3}  tie {t:3}  (net {i-w:+d})")
        if args.examples:
            print(f"\n=== EXAMPLES ({args.examples}/category) ===")
            from collections import defaultdict
            seen = defaultdict(int)
            for i, p in enumerate(pairs):
                if seen[p["cat"]] >= args.examples:
                    continue
                seen[p["cat"]] += 1
                print(f"\n[{p['cat']}]\n  REF      : {p['text']}")
                for c in conds:
                    print(f"  {c:9}: {hyps[c][i].strip()}")

    if args.per_file_csv:
        os.makedirs(os.path.dirname(args.per_file_csv) or ".", exist_ok=True)
        recs = per_file_records(pairs, hyps)
        cols = ["file", "folder", "gt"] + [f"out_{c}" for c in conds] + \
               [f"{m}_{c}" for c in conds for m in ("wer", "cer")]
        with open(args.per_file_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k))
                            for k in cols})
        print(f"\nwrote {args.per_file_csv} ({len(recs)} rows)")

    if args.folder_csv:
        os.makedirs(os.path.dirname(args.folder_csv) or ".", exist_ok=True)
        recs = per_file_records(pairs, hyps)
        fm = folder_means(recs, conds)
        with open(args.folder_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            head = ["folder", "n"] + [f"{m}_{c}" for c in conds for m in ("wer", "cer")]
            w.writerow(head)
            for folder in sorted(fm):
                row = fm[folder]
                cells = [f"{row[c][m]*100:.2f}" for c in conds for m in ("wer", "cer")]
                w.writerow([folder, row["n"], *cells])
        print(f"wrote {args.folder_csv} ({len(fm)} folders)")


if __name__ == "__main__":
    main()
