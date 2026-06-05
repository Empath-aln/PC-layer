# visualize

SVD analysis of trained checkpoints. The module has three pieces:

- **`svd_analyzer`** — loads a checkpoint, runs SVD on each weight matrix, and
  writes `singular_values_step_*.json`. Invoked by `train.py` in
  `--visualize.enable` mode.
- **`plotter`** — plots the singular-value distribution at a single step (a
  static snapshot).
- **`metrics_plotter`** — plots how SVD-derived metrics evolve over training
  steps (dynamic trends).

`plotter` and `metrics_plotter` both consume the `singular_values_step_*.json`
files produced by `svd_analyzer`.

The ready-made driver scripts under [`scripts/visualize/`](../scripts/visualize/)
wire these together for the baseline-vs-PC comparison; the workflow below uses
them. Run everything from the repo root.

---

## Quick start (baseline vs PC)

```bash
# Activate your environment first, e.g.:  conda activate pc

# 1. Dump singular values for both runs (reads train_configs/AdamW_{baseline,pc}_1B.toml,
#    loads their checkpoints, writes JSON under visualization_output/svd/).
bash scripts/visualize/svd_baseline_vs_pc.sh

# 2. GMCN trend curves (global / PC blocks / non-PC blocks).
bash scripts/visualize/plot_gmcn.sh

# 3. Singular-value distribution histograms at a given step (default: final step 61100).
bash scripts/visualize/plot_sv_hist.sh
```

Outputs land under `visualization_output/` (git-ignored). To adapt the scripts
to your own runs, edit the `CONFIG_FILES` / `BASELINE` / `PC` variables at the
top of each script.

---

## svd_analyzer — generate SV JSON from a checkpoint

`svd_analyzer.py` is the entry point for the SVD computation. For each requested
step it: loads the checkpoint weights, runs one forward pass on a val batch to
verify the checkpoint loaded correctly (printing the val loss), runs SVD on each
weight matrix, and writes `singular_values_step_{step}.json`.

It is driven through `train.py` with the same TOML config used for training, so
the model architecture / PC settings match the checkpoint:

```bash
torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 --role rank --tee 3 \
    train.py --job.config_file <config.toml> \
        --visualize.enable \
        --visualize.step "<STEPS>"
```

- `--visualize.step` takes a comma-separated list of steps (e.g.
  `1,4000,8000`); missing steps are warned about and skipped. `-1` means all
  available steps. SVD is relatively expensive, so for a long run sampling one
  point every few thousand steps (plus the first and last) is usually enough for
  the trend curves.
- The checkpoint folder is resolved from the config by the same rule `train.py`
  uses to save it: `{checkpoint.folder}/{model.name}_{model.flavor}/{optimizer.name}/{metrics.swanlab_comment}/step-{N}/`.

### Output location

The JSON is written under the current working directory at:

```
visualization_output/svd/{model.name}_{model.flavor}_{optimizer.name}_{swanlab_comment}/singular_values_step_{N}.json
```

For example, the shipped `AdamW_pc_1B` config produces
`visualization_output/svd/llama2_1B_AdamW_AdamW_pc_1B/`.

### Tips

- **Verification**: before SVD, each step runs val batches and prints
  `[Checkpoint Verification] Step N - Val loss: ...`. If a step is clearly off
  the training curve, the checkpoint was not loaded correctly — check that the
  config matches the one used for that run.
- **Crash recovery**: an occasional CUDA NVML error can kill the run midway.
  Already-written JSON files are not lost — just pass the remaining steps to
  `--visualize.step` again to resume.

---

## plotter — singular-value distribution

For a single step's JSON file, plots the singular-value distribution by weight
type (`wq`, `wk`, …) and layer. Singular values are σ/σ_max-normalized (divided
by each record's own largest singular value σ₁), so every distribution lives in
[0, 1] and shapes can be compared directly regardless of absolute scale.

### Quick start

```bash
# plot the singular-value histogram for one JSON
python3 -m visualize.plotter path/to/singular_values_step_1.json

# compare the distributions of two experiments at the same step
python3 -m visualize.plotter path/to/exp_A/...step_1.json path/to/exp_B/...step_1.json \
    --labels "baseline" "PC"
```

### Argument reference

| Argument | Type | Default | Description |
|------|------|--------|------|
| `json_files` | positional | (required) | Path(s) to one or more `singular_values_step_*.json` files |
| `--mode` | single | `hist` | Plot style: `hist` (histogram), `ecdf` (stepped cumulative curve), or `quantile` (empirical quantile function) |
| `--logy` | flag | off | Use a log-scale y-axis (histogram mode only) |
| `--labels` | multi | auto | Custom legend labels, one per JSON file |
| `--fmt` | single | `pdf` | Output format: `pdf` or `png` |
| `--output-dir` | path | auto | Output directory. A `step_{N}` subdirectory is appended when the step can be parsed from the filename. |

### Plotting modes (`--mode`)

- **`hist`** (default) — singular-value histogram. A single JSON plots both the
  original and post-PC SV (if present); multiple JSONs plot the post-PC SV of
  each for comparison. `--logy` switches the y-axis to a log scale.
- **`ecdf`** — empirical cumulative distribution F(t) = Pr(x ≤ t), a stepped
  curve. The y-axis stays in [0, 1]; `--logy` is ignored.
- **`quantile`** — empirical quantile function: SVs sorted ascending, x = the
  fraction q in [0, 1] from the smallest, y = the value at that fraction. A
  higher curve in the low-fraction region means the lower tail of the spectrum
  has been lifted. Both axes stay in [0, 1]; `--logy` is ignored.

### Output directory structure

```
step_1/
  wq/layer0.pdf
  wk/layer0.pdf
  ...
```

---

## metrics_plotter — metric trend curves

Extracts metrics from the per-step JSON files in an experiment directory and
plots metric-vs-step (or metric-vs-token) curves.

### Quick start

```bash
# plot all metrics for one experiment
python3 -m visualize.metrics_plotter path/to/exp_dir

# compare two experiments
python3 -m visualize.metrics_plotter path/to/exp_A path/to/exp_B \
    --labels "baseline" "PC"
```

Each experiment directory must contain `singular_values_step_*.json` files
produced by `svd_analyzer`.

### Argument reference

| Argument | Type | Default | Description |
|------|------|--------|------|
| `experiment_dirs` | positional | (required) | Path(s) to one or more experiment directories |
| `--metrics` | multi | all | Names of the metrics to plot (see below) |
| `--labels` | multi | auto | Custom legend labels, one per experiment directory |
| `--plot-levels` | multi | all | Plot granularity: `per_layer`, `per_block`, `by_group`, `global` |
| `--x-unit` | single | `step` | X-axis unit: `step` or `token` |
| `--total-tokens` | float | - | Total training tokens (in B); required when `--x-unit token` is used |
| `--output-dir` | path | auto | Output directory |
| `--fmt` | single | `pdf` | Output format: `pdf` or `png` |

### Available metrics (`--metrics`)

| Metric name | Meaning | Global aggregation |
|--------|------|-------------|
| `modified_condition_number` | largest singular value σ₁ / mean of the bottom 10% of singular values (a.k.a. GMCN) | geometric mean |

### Plot granularity (`--plot-levels`)

- **`per_layer`** — one plot per weight matrix per layer.
- **`per_block`** — aggregated by weight type (e.g. the `wq` of all layers in one
  plot), via the **arithmetic mean**.
- **`by_group`** — layers bundled into groups, each aggregated with the metric's
  **global aggregator** (geometric mean for GMCN), so a group curve is directly
  comparable to the `global` curve. Groups are decided per layer by whether PC
  was actually applied (recorded in the data), not by hardcoded weight-type
  names — so the split tracks whatever blocks a run preconditions. The default
  groups are `pc` (PC blocks) and `no_pc` (non-PC blocks), which are
  complementary subsets; the full-model curve is the separate `global` level.
- **`global`** — all weights of all layers aggregated into a single plot.

### Usage examples

```bash
# GMCN, PC-blocks vs non-PC-blocks (by_group) plus the full-model global curve
python3 -m visualize.metrics_plotter path/to/baseline path/to/pc \
    --metrics modified_condition_number \
    --labels "baseline" "PC" \
    --plot-levels global by_group

# use token count on the x-axis (e.g. 160 B tokens total)
python3 -m visualize.metrics_plotter path/to/exp \
    --x-unit token --total-tokens 160
```

Steps are linearly converted to token counts: `x = step / max_step * total_tokens`.

### Output directory structure

```
output_root/
  modified_condition_number/
    per_layer/
      layers.0.attention.wq.pdf
      ...
    per_block/
      wq.pdf
      wk.pdf
      ...
    pc.pdf          # by_group plots sit directly under the metric dir,
    no_pc.pdf       # alongside global.pdf
    global.pdf
```
