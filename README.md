# PC Layer: Polynomial Weight Preconditioning for Improving LLM Pre-Training

This repository accompanies the paper [PC Layer: Polynomial Weight Preconditioning for Improving LLM Pre-Training](https://arxiv.org/abs/2606.06470).

This project is based on an earlier version of [torchtitan](https://github.com/pytorch/torchtitan) and has since diverged significantly. It is maintained as a standalone repository rather than a fork.

## Setup

Tested with **Python 3.11** and **PyTorch 2.7.0 (CUDA 12.6)**.

```bash
# 1. Create and activate an environment (conda shown; venv works too)
conda create -n pc python=3.11 -y
conda activate pc

# 2. Install dependencies. The --extra-index-url ensures the CUDA 12.6
#    build of PyTorch is used (otherwise pip may install a CPU-only wheel).
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

The Llama-2 SentencePiece tokenizer is included at `assets/tokenizer.model`.

## Data preparation

Training reads tokenized `.bin` / `.idx` shards. See
[`scripts/data/README.md`](scripts/data/README.md) for how to turn FineWeb
parquet files into that format.

## Training

```bash
# Activate your environment first, e.g.:  conda activate pc

# Multi-GPU runs. Each .sh wraps the matching .toml under train_configs/;
# override NGPU / LOG_RANK via env (see the script header).
bash scripts/train/AdamW_baseline_271M.sh    # baseline AdamW
bash scripts/train/AdamW_pc_271M.sh          # PC layer (FFN + O preconditioning)
```

The available configs are `{AdamW,Muon}_{baseline,pc}_{271M,1B}`; each has a
matching `train_configs/<name>.toml` and `scripts/train/<name>.sh`. Before
launching, set the dataset paths in the `.toml` (`dataset_train_path` /
`dataset_val_path`) to the tokenized shards built in the previous step.

## Experiment logging

Training logs to [SwanLab](https://swanlab.cn/) by default. Set
`metrics.enable_swanlab` in the `.toml` and export your key before launching:

```bash
export SWANLAB_API_KEY=<your-key>
```

**Prefer Weights & Biases?** SwanLab's API mirrors W&B's, so switching is small:
in `train.py`, replace `import swanlab` with `import wandb as swanlab` (the
`swanlab.init(...)` / `swanlab.log(...)` calls then map directly onto W&B), and
export `WANDB_API_KEY` instead. The config flag stays `metrics.enable_swanlab`.

## Downstream evaluation

This repo does not ship a downstream-evaluation harness — pick your own (e.g.
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)).
For convenience, `torchtitan/lm_eval_wrapper.py` provides:

- `build_titan_model_and_tokenizer(config_path=..., step=...)` — load a trained
  model + tokenizer from a checkpoint (resolved from the training TOML config).
- `TitanWrapper` — an adapter exposing the loaded model through
  lm-evaluation-harness's `LM` interface (requires `lm_eval` installed).

## Visualization

`visualize/` analyzes the singular-value spectrum of trained checkpoints (the
quantity the PC layer is designed to shape). Ready-made drivers for the
baseline-vs-PC comparison live in [`scripts/visualize/`](scripts/visualize/);
see [`visualize/README.md`](visualize/README.md) for the full workflow and
plotting options.

