"""HW1 Task 1: recompute loss -> perplexity -> bits/token -> bits-per-byte from metrics.json,
check them against the logged values, and compare the loss with ln(V).

    python metrics_by_hand.py <run_dir> [<run_dir> ...]

Do the arithmetic yourself first; this script is the check.
"""
import json
import math
import re
import sys
from pathlib import Path

KNOWN_V = {"char": 65, "byte": 256, "gpt2": 50257}   # char = Tiny Shakespeare's 65 characters


def vocab_size(run: Path, metrics: dict):
    log = run / "stdout.log"
    if log.exists():                                  # sweep runs: "[data] ... vocab=65 | ..."
        m = re.search(r"vocab=(\d+)", log.read_text(errors="replace"))
        if m:
            return int(m.group(1))
    ckpt = run / "model.pt"
    if ckpt.exists():                                 # single runs: the checkpoint holds the resolved config
        try:
            import torch
            c = torch.load(ckpt, map_location="cpu", weights_only=False)
            return int(c["data_meta"]["vocab_size"])
        except Exception:
            pass
    return KNOWN_V.get(str(metrics.get("tokenizer")))


def check(run: Path):
    m = json.loads((run / "metrics.json").read_text())
    loss, n_tok, n_byte = m["val_loss"], m["val_tokens"], m["val_bytes"]

    ppl = math.exp(loss)                              # perplexity = e^loss
    bits_per_token = loss / math.log(2)               # nats -> bits
    tokens_per_byte = n_tok / n_byte
    bpb = bits_per_token * tokens_per_byte            # bits per token x tokens per byte

    print(f"\n=== {run.name}  (tokenizer={m.get('tokenizer')}, arch={m.get('arch')}, steps={m.get('steps')})")
    print(f"  inputs : val_loss = {loss:.6f} nats/token | val_tokens = {n_tok:,} | val_bytes = {n_byte:,}")
    print(f"  {'quantity':<14}{'by hand':>12}{'logged':>12}{'abs diff':>12}")
    for name, mine, logged in [("perplexity", ppl, m.get("val_ppl")),
                               ("bits/token", bits_per_token, None),
                               ("bytes/token", 1 / tokens_per_byte, m.get("val_bytes_per_token")),
                               ("bpb", bpb, m.get("val_bpb"))]:
        lg = f"{logged:12.6f}" if logged is not None else f"{'-':>12}"
        df = f"{abs(mine - logged):12.2e}" if logged is not None else f"{'-':>12}"
        print(f"  {name:<14}{mine:12.6f}{lg}{df}")

    V = vocab_size(run, m)
    if V:
        lnV = math.log(V)
        print(f"  vs. uniform guess: ln V = ln {V:,} = {lnV:.4f} nats | loss / ln V = {loss / lnV:.3f} "
              f"| loss - ln V = {loss - lnV:+.4f}")
    return m


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        check(Path(d))
