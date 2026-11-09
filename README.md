# CMPE 258 HW1 — Language models from scratch (Part 1)

Report: `CMPE_258_HW1_Report.pdf`. Everything else supports it. No checkpoints or datasets are included.

| Folder | Contents |
|---|---|
| `configs/` | Every config used. Same layout as `nanolm/configs`, so sweep files' relative `base:` paths resolve. My own: `sweeps/hw1_untrained.yaml`, `sweeps/hw1_context.yaml` |
| `results/<experiment>/summary.csv`, `summary.md` | Written by `python -m nanolm.analyze`: per-run metrics; mean ± std over seeds; paired test |
| `results/<experiment>/runs/<run>/` | Each run's `metrics.json`, `meta.json`, `config.yaml` (paths shortened) |
| `results/task1_metrics_by_hand.txt`, `results/task4_attention/` | Task 1 hand check; Task 4 per-head scores |
| `figures/` | All figures in the report, plus the extra analyze plots |
| `scripts/` | My analysis and plotting scripts |
| `environment.yml`, `environment.lock.yml` | Conda environment (PyTorch installed with pip inside it) |

## Reproducibility facts (generated from the run records)

```
nanolm (DeepDataMiningLearning) commit: 8418fc1bae85657190f11749483bf2e729b9702c  (dirty=True)
uncommitted changes in that repo: ?? DeepDataMiningLearning/llm/nanolm/configs/sweeps/hw1_context.yaml; ?? DeepDataMiningLearning/llm/nanolm/configs/sweeps/hw1_untrained.yaml
device: cuda (precision=bfloat16, compile=yes) | torch 2.11.0+cu128
experiment             runs  train min  eval min
char_models              12        2.3       0.0
hw1_context              24        5.5       0.0
hw1_untrained             6        0.0       0.0
task1_trained_lstm        1        0.2       0.0
TOTAL                    43        8.0       0.0
```

## Reproduce

With the `nanolm` library at the commit above importable (its `llm/` folder on `PYTHONPATH`), from this folder:

```bash
python -m nanolm.train -c configs/laptop/L1_char_lstm.yaml                    # Task 1, trained model
python -m nanolm.sweep --sweep configs/sweeps/hw1_untrained.yaml              # Task 1, untrained models
python -m nanolm.sweep --sweep configs/sweeps/01_char_models.yaml             # Task 2
python -m nanolm.sweep --sweep configs/sweeps/hw1_context.yaml --seeds 1337 42 7   # Task 3
python -m nanolm.viz attention --run <Task 2 gpt_modern seed-1337 run> --text "..."  # Task 4 maps
python scripts/head_scores.py   --run <same run> --text "..." --out <file>.csv          # Task 4 numbers
```
