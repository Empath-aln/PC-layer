# CLAUDE.md

This file orients Claude Code (and new contributors) to the PC-Layer project. It is the
project-level memory loaded at the start of a session.

## What this project is

`PC-Layer-torchtitan` is a research codebase for pretraining Llama-style transformers with a
**preconditioned linear layer** ("PC layer"). It started from an early version of
[torchtitan](https://github.com/pytorch/torchtitan) and has since diverged into a standalone
repo (it is *not* a fork). The torchtitan scaffolding (config system, FSDP/TP parallelism,
checkpointing, metrics, data loading) is retained; the research contribution lives in
`torchtitan/pc_layer/`.

The core idea: wrap selected `nn.Linear` weights so that, on every forward pass, the weight
matrix is **spectrally normalized** and then passed through a polynomial **preconditioner**
(an odd-polynomial approximation of orthogonalization, à la Newton–Schulz) before the matmul.
This shapes the singular-value spectrum of the effective weight during training.

## Repo layout

```
train.py                     # main training entry point (torchrun target)
estimation.py                # FLOP/memory estimation
torchtitan/lm_eval_wrapper.py
                             # downstream-eval interface: load a checkpoint
                             # (build_titan_model_and_tokenizer) + lm-eval LM
                             # adapter (TitanWrapper). No runner script shipped.
tokenizer.model              # SentencePiece tokenizer
train_configs/               # TOML run configs
  {AdamW_baseline,AdamW_pc,Muon_baseline,Muon_pc}_{271M,1B}.toml
scripts/
  train/<config-name>.sh     # one launch script per TOML (torchrun wrapper)
  visualize/                 # SVD dump + plotting drivers (baseline vs PC)
  data/                      # tokenizer download + FineWeb tokenization
Muon/muon_fsdp.py            # Muon optimizer (FSDP-aware) — alternative to AdamW
torchtitan/
  pc_layer/pc_layer.py       # ★ the PC layer: spectral norm + preconditioner
  models/llama/model.py      # Llama Transformer + ModelArgs (research flags live here)
  models/llama/__init__.py   # flavor presets (35M … 70B) as ModelArgs instances
  models/__init__.py         # model_name_to_cls / models_config registries
  config_manager.py          # TOML+CLI → JobConfig (argparse two-level namespace)
  parallelisms/              # FSDP / TP / PP wiring (parallelize_llama.py)
  checkpoint.py, metrics.py, lr_scheduling.py, profiling.py, utils.py
parallelisms/                # ⚠ stale duplicate of torchtitan/parallelisms (UNUSED; train.py
                             #   imports torchtitan.parallelisms). Safe to ignore/delete.
test/                        # torchtitan-inherited tests; none cover pc_layer
```

