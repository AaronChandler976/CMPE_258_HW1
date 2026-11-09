"""HW1 Task 4: put a number on each attention head's behaviour (the §1.6.2 analysis).

Run from the professor's llm/ folder (it imports nanolm):
    python $HW/scripts/head_scores.py --run outputs/nanolm/sweeps/char_models/gpt_modern_s1337 \
        --text $'ROMEO:\nBut soft! what light through yonder window' --out $HW/figures/heads_dialogue.csv

For every head it reports the average attention weight that position t puts on:
  prev        position t-1                       (previous-token head)
  word_start  the first letter of t's word       (word-start head; only letters >= 2 into a word,
                                                  so it never overlaps with prev)
  marker      the latest ':' or newline before t-1 (line / speaker-marker head)
  self        position t itself
  pos0        position 0                          ("attention sink"; t >= 2 only)
and the chance level for each: what a head that spreads evenly over 0..t would give (1/(t+1)),
averaged over the same positions. "x chance" = weight / chance. Also: mean look-back distance.
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())          # so `import nanolm` works when run from llm/

import numpy as np
import torch

from nanolm.generate import load_run
from nanolm.viz import attention_maps

TYPES = ["prev", "word_start", "marker", "self", "pos0"]


def is_letter(c):
    return c.isalpha() or c == "'"


def position_sets(text):
    """For each head type: list of (query t, key s) pairs to average A[t, s] over."""
    T = len(text)
    pairs = {k: [] for k in TYPES}
    for t in range(1, T):
        pairs["prev"].append((t, t - 1))
        pairs["self"].append((t, t))
        if t >= 2:
            pairs["pos0"].append((t, 0))
        if is_letter(text[t]):
            w = t
            while w > 0 and is_letter(text[w - 1]):
                w -= 1
            if w < t - 1:
                pairs["word_start"].append((t, w))
        marks = [s for s in range(t - 1) if text[s] in ":\n"]
        if marks:
            pairs["marker"].append((t, marks[-1]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir of a trained GPT (needs model.pt)")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default=None, help="CSV file to write")
    a = ap.parse_args()

    cfg, model, tok, dinfo, _ = load_run(a.run, device="cpu")
    ids = tok.encode(a.text)
    if tok.decode(ids) != a.text:
        print(f"WARNING: some characters are not in the vocabulary and became '?':\n  model sees: "
              f"{tok.decode(ids)!r}")
    with torch.no_grad():
        maps = attention_maps(model, torch.tensor([ids]))     # list over layers of [H, T, T]
    text = a.text
    T = len(text)
    pairs = position_sets(text)
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]

    rows = []
    for l, A in enumerate(maps):
        for h in range(A.shape[0]):
            M = A[h]
            r = {"head": f"L{l}.H{h}"}
            for k in TYPES:
                p = pairs[k]
                r[k] = float(np.mean([M[t, s] for t, s in p])) if p else float("nan")
                r[k + "_chance"] = float(np.mean([1 / (t + 1) for t, _ in p])) if p else float("nan")
            r["lookback"] = float((M[1:] * np.clip(dist[1:], 0, None)).sum(1).mean())
            rows.append(r)

    print(f"\ntext ({T} chars): {text!r}")
    print("n positions used: " + ", ".join(f"{k}={len(pairs[k])}" for k in TYPES))
    print("chance level:     " + ", ".join(f"{k}={rows[0][k + '_chance']:.3f}" for k in TYPES) + "\n")
    print(f"{'head':<7}" + "".join(f"{k:>12}" for k in TYPES) + f"{'lookback':>10}")
    for r in rows:
        print(f"{r['head']:<7}" + "".join(f"{r[k]:12.3f}" for k in TYPES) + f"{r['lookback']:10.1f}")

    print("\nStrongest head of each type (weight, and how many times chance):")
    for k in TYPES:
        valid = [r for r in rows if not np.isnan(r[k])]
        if not valid:
            print(f"  {k:<11} (no positions of this kind in the text)")
            continue
        b = max(valid, key=lambda r: r[k])
        print(f"  {k:<11} {b['head']:<7} {b[k]:.3f}  ({b[k] / b[k + '_chance']:.1f}x chance {b[k + '_chance']:.3f})")
    far = max(rows, key=lambda r: r["lookback"])
    print(f"  longest look-back: {far['head']} ({far['lookback']:.1f} characters on average)")

    if a.out:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] {a.out}")


if __name__ == "__main__":
    main()
