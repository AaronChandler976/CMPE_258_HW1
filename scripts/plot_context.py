"""HW1 Task 3: validation bpb vs. context length (train.seq_len), LSTM vs. modern Transformer.

    python plot_context.py outputs/nanolm/sweeps/hw1_context

Writes <sweep>/context_length.png and context_length.csv, and prints how much each doubling of
seq_len helped next to the noise threshold (2 x the larger seed std of the two lengths compared).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABEL = {"lstm": "LSTM 2x256", "gpt_modern": "Transformer, modern 4x256"}


def main(sweep: str):
    sweep = Path(sweep)
    rows = []
    for mf in sorted(sweep.glob("*/metrics.json")):
        m = json.loads(mf.read_text())
        model, T = m["name"].rsplit("_T", 1)
        rows.append({"model": model, "seq_len": int(T), "seed": m["seed"], "val_bpb": m["val_bpb"],
                     "params": m["params"], "tokens_seen": m["tokens_seen"], "train_min": m["train_time_sec"] / 60})
    if not rows:
        sys.exit(f"no metrics.json under {sweep}")
    df = pd.DataFrame(rows)
    g = (df.groupby(["model", "seq_len"])
           .agg(bpb_mean=("val_bpb", "mean"), bpb_std=("val_bpb", "std"), n_seeds=("seed", "nunique"),
                tokens_seen=("tokens_seen", "first"), params=("params", "first"), train_min=("train_min", "mean"))
           .reset_index())
    g["bpb_std"] = g["bpb_std"].fillna(0.0)
    g.to_csv(sweep / "context_length.csv", index=False)

    if g["tokens_seen"].nunique() > 1:
        print("WARNING: runs saw different numbers of tokens -- the comparison is not budget-matched!")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (model, s) in enumerate(g.groupby("model")):
        s = s.sort_values("seq_len")
        ax.errorbar(s["seq_len"], s["bpb_mean"], yerr=s["bpb_std"], marker="o", capsize=4, lw=2,
                    color=f"C{i}", label=LABEL.get(model, model))
        for _, r in s.iterrows():
            ax.annotate(f"{r.bpb_mean:.3f}", (r.seq_len, r.bpb_mean), textcoords="offset points",
                        xytext=(8, 6), ha="left", fontsize=7, color=f"C{i}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(g["seq_len"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("context length = train.seq_len (characters)")
    ax.set_ylabel("validation bits-per-byte (lower is better)")
    ax.set_title(f"bpb vs. context length at a fixed budget of {g['tokens_seen'].iloc[0] / 1e6:.1f}M tokens "
                 f"(mean ± std, {g['n_seeds'].max()} seeds)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(sweep / "context_length.png", dpi=150)
    print(f"[plot] {sweep / 'context_length.png'}\n")

    print(g[["model", "seq_len", "n_seeds", "bpb_mean", "bpb_std", "train_min"]].to_string(index=False))
    print("\nGain per doubling (positive = longer context helped):")
    for model, s in g.groupby("model"):
        s = s.sort_values("seq_len").reset_index(drop=True)
        for a, b in zip(s.index[:-1], s.index[1:]):
            gain = s.bpb_mean[a] - s.bpb_mean[b]
            noise = 2 * max(s.bpb_std[a], s.bpb_std[b])
            verdict = "helps" if gain > noise else ("hurts" if gain < -noise else "not resolved")
            print(f"  {model:<11} {s.seq_len[a]:>4} -> {s.seq_len[b]:<4} gain {gain:+.4f} bpb"
                  f"   (noise threshold {noise:.4f})  -> {verdict}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
