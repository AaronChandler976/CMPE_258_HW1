"""HW1 Task 2: training loss vs. validation loss for every model of a sweep (the overfitting figure).

    python plot_curves.py outputs/nanolm/sweeps/char_models

Writes <sweep>/train_vs_val.png: one panel per model, mean over seeds, both in nats/token.
A training curve that keeps falling while validation turns up = memorizing the training set.
(Training loss is a running average measured with dropout on, so it sits slightly high.)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(sweep: Path):
    tr, ev = [], []
    for run in sorted(p for p in sweep.iterdir() if (p / "metrics.json").exists()):
        name, seed = run.name.rsplit("_s", 1)
        e = pd.read_json(run / "eval_log.jsonl", lines=True)
        t = pd.read_json(run / "train_log.jsonl", lines=True)
        tok_per_step = (e["train_tokens"] / e["step"]).iloc[-1]
        t["train_tokens"] = t["step"] * tok_per_step
        for df, out in ((t, tr), (e, ev)):
            df["name"], df["seed"] = name, seed
            out.append(df)
    if not ev:
        sys.exit(f"no finished runs under {sweep}")
    return pd.concat(tr), pd.concat(ev)


def main(sweep: str):
    sweep = Path(sweep)
    tr, ev = load(sweep)
    names = list(dict.fromkeys(ev["name"]))
    cols = min(3, len(names))
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), sharey=True, squeeze=False)
    lo = min(tr["loss"].min(), ev["loss"].min())
    for ax, name in zip(axes.flat, names):
        t = tr[tr.name == name].groupby("train_tokens")["loss"].mean()
        e = ev[ev.name == name].groupby("train_tokens")["loss"].mean()
        ax.plot(t.index / 1e6, t.values, color="0.6", lw=1, label="train loss")
        ax.plot(e.index / 1e6, e.values, color="C0", lw=2, marker="o", ms=3, label="val loss")
        best = e.idxmin()
        ax.axvline(best / 1e6, color="C3", ls=":", lw=1)
        ax.set_title(f"{name}  (best val {e.min():.3f} @ {best / 1e6:.1f}M)", fontsize=9)
        ax.set_xlabel("training tokens (M)")
        ax.grid(alpha=0.3)
        ax.set_ylim(lo - 0.05, lo + 1.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("loss (nats/token)")
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Training vs. validation loss (mean over seeds); dotted = best validation point", fontsize=10)
    fig.tight_layout()
    out = sweep / "train_vs_val.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] {out}")

    # final train/val gap per model, for the report
    print(f"\n{'model':<16}{'final train':>12}{'final val':>11}{'gap':>8}{'best val':>10}")
    for name in names:
        t = tr[tr.name == name].groupby("train_tokens")["loss"].mean()
        e = ev[ev.name == name].groupby("train_tokens")["loss"].mean()
        print(f"{name:<16}{t.iloc[-1]:12.3f}{e.iloc[-1]:11.3f}{e.iloc[-1] - t.iloc[-1]:8.3f}{e.min():10.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
