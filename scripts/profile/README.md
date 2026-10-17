# MLP profiling

A small, single-device benchmark for PC operator experiments. Run from the repo
root with the dependencies in `requirements.txt` (PyTorch 2.7.0 and matplotlib).
No dataset, tokenizer, distributed launch, or experiment tracking is needed.

Each MLP block is `Linear(d, h) -> GELU -> Linear(h, d)`, without bias. The PC
variant wraps both projections in the selected version's `PCLinear`. This covers both tall
and wide weight matrices. The target has shape `[N, d]`; every iteration computes
`MSE(prediction.float(), target)` and calls `loss.backward()`, including input,
weight, and gamma gradients. `--layers` can stack multiple MLP blocks.
`--grad-accum-steps` controls the number of microbatches per accumulated step
(default: 1); `--tokens` is the number of rows **per microbatch**.

## PC versions and reference

- `plain`: ordinary `nn.Linear` MLP, with no PC; not an equivalent PC implementation.
- `pc_baseline`: the frozen original PC implementation in `pc_baseline.py`.
- `pc_v1`: one PC transform and one PC backward per accumulated step; see below.
- `pc_v2`: v1's accumulation plus CUDA fusion and a smaller backward saved set.
- `pc_v3`: v2 with selectable PC checkpointing (`none`, `polynomial`, `full`).
- `pc_v4`: v3's checkpoint modes with shared PC backward inside the normal FSDP2 graph.
- `pc_v5`, etc.: future versions, each in its own `pc_*.py` file here.

`pc_baseline.py` is a byte-for-byte snapshot of
`torchtitan/pc_layer/pc_layer.py` at commit
`a97b1ecf2062065df29ba7d8fc48e117811fbb39`. Its SHA-256 is
`91204116c4204c73400c28053ca2d98ae2d800faba096da7f847156e5b94b87b`.
Keep this file unchanged; the CPU tests enforce its hash. Profiling imports this
snapshot directly, so changing the training operator will not change the reference.

Version files are discovered automatically, without editing a registry:

```bash
cp scripts/profile/pc_v4.py scripts/profile/pc_v5.py
# Edit pc_v5.py to implement the next optimization.
python scripts/profile/benchmark_mlp.py \
  --implementation pc_baseline pc_v4 pc_v5 --dim 512 1024 --tokens 256 \
  --grad-accum-steps 4 --seeds 0 1 2 \
  --output outputs/pc_profile/versions.csv
python scripts/profile/plot_mlp.py outputs/pc_profile/versions.csv \
  --x dim --output outputs/pc_profile/versions.png
```

Each file exports `PCLinear(linear, model_args, layer_id)` and
`update_model_sn_state(model)`, with compatible `state_dict`/parameter names so
the exact same weights, gamma, and SN vectors can be loaded and compared.
Internal calculations and autograd can be rewritten freely. Initialization can
be customized by exporting `initialize_model(model)`; otherwise the harness
zeros `sn_u`/`sn_v` and calls `_maybe_init_gamma()` on that version's PC layers.
Do not import implementation code from the changing training operator.

Versions may also export `accumulation_context(model, grad_accum_steps)`, a
context manager around all microbatch forwards/backwards. Entry must only set up
transient state; evaluate transforms lazily in forward so their time is measured.
Exit must finish all deferred parameter gradients and release caches, including
on exceptions. Exit time is included in the **last microbatch's backward**;
SN update runs after exit. Versions without this hook keep the original behavior.

### v1: reuse PC across microbatches

The v1 file copies the frozen PC/SN mathematics and adds a step-scoped cache.
For `grad_accum_steps > 1`, each PC layer computes `P = PC(W, gamma)` once on its
first forward. The microbatches use a detached leaf sharing P's storage, and
accumulate `g_P = sum_i d(loss_i / G)/dP`. Context exit backpropagates g_P through
the original P graph once, producing the W and gamma gradients, including the
spectral-normalization derivative. Ordinary MLP backward, including each input
gradient, still runs on every microbatch and releases that activation graph.
No microbatch uses `retain_graph=True`.

Weights, gamma, and SN vectors must remain fixed inside this context. All caches
are discarded before the SN update, and a new PC graph is built next step. At
accum=1, and outside the context, v1 uses the baseline forward/backward path.
The single-device benchmark supplies this context automatically; a caller that
only invokes `model(x)` without it gets no cross-microbatch reuse. This profiling
version does not implement distributed training or higher-order differentiation.

This moves the PC backward from every microbatch to the end of the accumulated
step. Forward/backward panels include the shared work once per step, so compare
the **total accumulated-step time** as well. The PC graph remains live until the
end, and the cached leaf needs a gradient buffer; peak memory may increase or
decrease and must be measured. BF16 changes the summation order and need not be
bitwise identical; the existing correctness tolerances are unchanged. Per-step
equivalence is checked at identical weights/state, not identical long training
trajectories after independently applying rounded gradients.

Run this comparison on the H200 slice (five seeds, including an accum=1 control):

```bash
source ./.venv-profile/bin/activate
PC_V1_OUT="outputs/pc_profile/pc_v1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PC_V1_OUT"
python scripts/profile/benchmark_mlp.py \
  --device cuda --dtype bfloat16 \
  --implementation plain pc_baseline pc_v1 \
  --dim 1024 --layers 2 --tokens 512 --grad-accum-steps 1 2 4 8 \
  --pc-level 4 --power-iter 10 --seeds 0 1 2 3 4 \
  --warmup 10 --steps 50 --label accum \
  --output "$PC_V1_OUT/accum.csv"
python scripts/profile/plot_mlp.py "$PC_V1_OUT/accum.csv" \
  --x grad_accum_steps --log-x --band std \
  --output "$PC_V1_OUT/accum.png"
```

For a separate FP32 check, repeat the benchmark with `--dtype float32` and a new
output path. On either dtype, failed correctness cases are excluded from timing.

Before timing each optimized `(shape, seed, accumulation)` case, the harness:

1. Loads identical initial weights, gamma, SN state, input, and target.
2. Checks every microbatch's MLP output, the final unscaled MSE and input
   gradient, and all accumulated parameter gradients (including gamma),
   across two accumulated steps.
3. Checks an additional forward after the last SN update, covering stateful
   behavior as well as the polynomial transform and backward.

Checks use the requested dtype: FP32 defaults to `rtol=1e-4, atol=1e-5`; BF16
defaults to `rtol=0.02, atol=0.002`. Override with `--check-rtol`/`--check-atol`.
Near-zero relative-error diagnostics use `max(abs(reference), atol, 1e-12)` as
the denominator; pass/fail uses `abs(error) <= atol + rtol*abs(reference)`.
Use FP32 runs as well as BF16 when validating a rewrite.

A failure records `status=correctness_failed`, leaves timing/memory empty,
skips that case's performance measurement, and makes the command exit nonzero.
The plot omits failed cases. Checks run separately from timing; CPU snapshots
allow reference and candidate to run sequentially on the GPU instead of holding
two models there. Performance starts from a fresh model with the checked initial
state. When an optimized version is requested, `pc_baseline` is automatically
included for performance comparison unless `--no-reference-benchmark` is set.
This option does not disable baseline correctness checks. The terminal reports paired-seed speedup
and mean CUDA peak-memory savings relative to it.

### v2: fusion and backward storage

The v2 operator has its own first-order autograd implementation; no training-loop
or accumulation-interface changes are required. It keeps v1's shared PC forward
and backward across microbatches. At accum=1 it also uses the new v2 transform/VJP.
The SN helpers and accumulation context are copied from v1 unchanged.

For level 4, the manual backward saves the original W, scalar norm/gamma, four
Gram-size matrices (the Gram matrix plus three Horner results), and the
pre-gamma output when gamma needs a gradient. It reconstructs normalized W and
the initial affine polynomial term using pointwise operations. It does **not**
recompute a GEMM. For trainable W/norm/gamma, level 4 still uses 5 forward GEMMs
and 10 backward GEMMs. Norm recovery retains its original stop-gradient, and a
zero gamma still has the correct derivative.

CUDA FP32/BF16 use Triton kernels for:

- Initial affine polynomial construction without an explicit identity matrix.
- Horner GEMM + diagonal constant addition in one kernel, keeping the
  intermediate dtype rounding between the two operations in registers.
- Norm recovery + gamma scaling, reusing the pre-gamma output buffer.
- Backward scaling/reductions and normalization gradients, avoiding full-size
  elementwise temporaries.

The Gram GEMM, final rectangular GEMM, and backward GEMMs use PyTorch. The Horner
kernel currently uses fixed 64x64x32 tiling; performance needs measurement on the
H200 slice. There is no claim that its GEMMs are faster than cuBLAS. Triton is
optional for CPU import; CUDA FP32/BF16 require it and fail explicitly if absent.
The PyTorch 2.7.0 Linux CUDA environment supplies Triton 3.3.0. CPU uses the same
manual VJP and diagonal/buffer strategy in torch, without CUDA fusion.

A CPU saved-tensor-storage check (excluding original inputs/shared storage)
reduces the level-4 saved set from two weight-size + five Gram-size tensors to
one weight-size + four Gram-size tensors. At d=1024, h=4096, BF16 this is about
26 -> 16 MiB per PCLinear, or 104 -> 64 MiB for two MLP blocks. These are saved
storage sizes, **not CUDA peak-memory measurements**; backward temporaries and
cached PC weights/gradients also contribute to the peak. CUDA kernels and actual
time/peak memory must be validated on the GPU; the local development host has
no CUDA device.

On the server, validate CUDA kernels before running the multi-seed comparison.
These tests include padded matrix sizes, multi-block reductions, tall/wide
weights, zero gamma, output and gradient checks. The benchmark also checks each
case against the frozen baseline at the existing tolerances before timing it.
Compiler/backend errors are recorded as failed correctness cases.

~~~bash
source ./.venv-profile/bin/activate
python -m unittest scripts.profile.test_pc_v2 -v

PC_V2_OUT="outputs/pc_profile/pc_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PC_V2_OUT"

# Width sweep: compare all four versions in the same plot.
python scripts/profile/benchmark_mlp.py \
  --device cuda --dtype bfloat16 \
  --implementation plain pc_baseline pc_v1 pc_v2 \
  --dim 512 1024 2048 --layers 2 --tokens 512 --grad-accum-steps 4 \
  --pc-level 4 --power-iter 10 --seeds 0 1 2 3 4 \
  --warmup 10 --steps 50 --output "$PC_V2_OUT/width.csv"
python scripts/profile/plot_mlp.py "$PC_V2_OUT/width.csv" \
  --x dim --band std --output "$PC_V2_OUT/width.png"

# Accumulation sweep: verify that v1's reuse remains effective in v2.
python scripts/profile/benchmark_mlp.py \
  --device cuda --dtype bfloat16 \
  --implementation plain pc_baseline pc_v1 pc_v2 \
  --dim 1024 --layers 2 --tokens 512 --grad-accum-steps 1 2 4 8 \
  --pc-level 4 --power-iter 10 --seeds 0 1 2 3 4 \
  --warmup 10 --steps 50 --output "$PC_V2_OUT/accum.csv"
python scripts/profile/plot_mlp.py "$PC_V2_OUT/accum.csv" \
  --x grad_accum_steps --log-x --band std --output "$PC_V2_OUT/accum.png"
~~~

### v3: PC checkpointing and the time/memory tradeoff

`pc_v3.py` is a standalone copy of v2 with operator-local recomputation inside
its manual backward. It preserves the norm's stop-gradient on recovery, gamma=0
derivatives, and the order of dtype rounding. It does not checkpoint MLP
activations. The existing accumulation context and timing boundaries apply;
recomputation runs once per PC backward in the accumulated step and is included
in the backward and total panels. No training-loop changes are needed.

For level 4, trainable W/norm/gamma, d=1024, h=4096, and BF16:

| `--pc-checkpoint` | New saved PC intermediates per PCLinear | Forward GEMMs | Backward GEMMs |
| --- | --- | --- | --- |
| `none` | R + Gram + 3 Horner results: 16 MiB, same as v2 | 5 | 10 |
| `polynomial` | R + Gram: 10 MiB | 5 | 13 |
| `full` | None; save original W/norm/gamma references | 5 | 15 |

The modes respectively avoid recomputation, rebuild the three Horner products,
or also rebuild Gram and the pre-gamma output R. Frozen gamma avoids the extra R
GEMM in full mode. Level 0 needs no GEMMs in any mode. These counts describe the
PC operator, excluding activation matmuls and SN vector operations. Saved sizes
exclude the output P, its accumulated gradient, original inputs/shared storage,
and backward temporaries. Across four PCLinear projections in two blocks, the
saved set is 64 / 40 / 0 MiB; those differences are **not promised peak savings**.

`--pc-checkpoint` accepts multiple modes and defaults to `full` for v3. Earlier
versions ignore this option and are measured only once per shape/seed; their CSV
mode is `none`. Each mode is checked independently against the frozen baseline
at the unchanged tolerances before timing. BF16 accumulation/manual-VJP rounding
can fail the existing strict gate for some shapes, including without checkpointing;
failed cases remain failed and are excluded from performance comparisons.

Start with a fixed shape on H200 to inspect the tradeoff:

```bash
set -euo pipefail
source ./.venv-profile/bin/activate
python - <<'PY'
import torch, triton
assert torch.cuda.is_available(), "CUDA is required; do not silently skip GPU checks"
print(torch.__version__, triton.__version__, torch.cuda.get_device_name())
PY
python -m unittest scripts.profile.test_pc_v3 -v

PC_CKPT_OUT="outputs/pc_profile/pc_v3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PC_CKPT_OUT"
python scripts/profile/benchmark_mlp.py \
  --device cuda --dtype bfloat16 \
  --implementation plain pc_v2 pc_v3 \
  --pc-checkpoint none polynomial full \
  --dim 1024 --layers 2 --tokens 512 --grad-accum-steps 4 \
  --pc-level 4 --power-iter 10 --seeds 0 1 2 3 4 \
  --warmup 10 --steps 50 --label checkpoint \
  --output "$PC_CKPT_OUT/checkpoint.csv"

# Original six panels, with checkpoint mode as the x axis.
python scripts/profile/plot_mlp.py "$PC_CKPT_OUT/checkpoint.csv" \
  --x pc_checkpoint --band std --output "$PC_CKPT_OUT/phases.png"
# Cross-seed mean step time vs peak CUDA memory; both axes show seed intervals.
python scripts/profile/plot_mlp.py "$PC_CKPT_OUT/checkpoint.csv" \
  --tradeoff --band std --output "$PC_CKPT_OUT/tradeoff.png"
```

The benchmark inserts pc_baseline automatically unless `--no-reference-benchmark`
is set, and `--resume-from` can reuse its previous timing rows. In the tradeoff plot, each
checkpoint mode is annotated; lower and left are better. Error bars are one
sample standard deviation across seeds by default, or seed min/max with
`--band range`. Use one fixed shape/accumulation count for the clearest comparison.
The four existing sweeps work with the same mode list: use accumulation 1/2/4/8,
tokens 128/512/2048/8192, layers 1/2/4/8, or dim 512/1024/2048 and keep the other
dimensions fixed as before. Their six-panel plots separate checkpoint modes
into distinct curves. Old CSVs without the mode column are read as `none`.

CPU checks cover FP32/BF16, tall/wide/square matrices, all PC levels, frozen
inputs, noncontiguous weights, gamma edge cases, finite differences, distinct
microbatches and changed weights, saved storage, and GEMM counts. The two v3 CUDA
checks cover the recomputation kernels and the full benchmark correctness gate;
they skip locally without CUDA. Actual H200 timing/peak memory is not measured
on the CPU development host. v3 retains v2's single-device accumulation context;
checkpointing does not resolve its known FSDP deferred-backward incompatibility.

### v4: FSDP2-compatible PC accumulation

`pc_v4.py` preserves v3's three checkpoint modes and independent operator code.
For G > 1, it caches only detached PC values and their accumulated output
gradient. Each microbatch attaches those values to its current W/norm/gamma
through a new custom autograd node. On the last microbatch, that node returns
the accumulated PC VJP during the ordinary `loss.backward()`, while FSDP has
unsharded the parameters. Early microbatches still compute their input gradients;
PC parameter gradients arrive on the last backward. The cached SN value has a
small custom derivative attached to the current weight as well. No old full
parameter, differentiable PC graph, or external backward survives between
microbatches. Context exit validates completion and clears caches.

This fixes the old context-exit backward accessing full parameter storage that
FSDP had already freed. Historical v1/v2/v3 files remain unchanged, including
that limitation. Outside the step context, or at G=1, v4 uses the ordinary
operator path. Every used PCLinear must participate once per microbatch; W,
gamma, and SN buffers stay fixed until the context exits. Entirely frozen PC
parameters still support input gradients.

The training integration is selected through the existing model configuration:

```toml
[model]
pc_implementation = "pc_v4"
pc_checkpoint = "full"  # none / polynomial / full

[training]
grad_accumulation_steps = 4
compile = false

[activation_checkpoint]
mode = "full"  # or none; selective also works with selective_ac_option = "2"
```

Keep the existing `precondition_*`, `pc_level`, normalization, gamma, optimizer,
and FSDP settings. CLI equivalents are `--model.pc_implementation pc_v4` and
`--model.pc_checkpoint full`. The model factory preserves parameter/checkpoint
names, and the training loop adds one context around the existing microbatch
loop. Clipping, optimizer updates, and SN updates run after that context. The
default `baseline` backend preserves existing training behavior. A custom loop
can use the same plugin:

```python
from torchtitan.pc_layer.runtime import accumulation_context, update_model_sn_state

optimizer.zero_grad(set_to_none=True)
with accumulation_context(model, grad_accum_steps):
    for batch in microbatches:  # exactly grad_accum_steps batches
        (compute_loss(model, batch) / grad_accum_steps).backward()
# Optional clipping here; all parameter gradients are ready.
optimizer.step()
update_model_sn_state(model)
```

Supported scope is eager, first-order, pure FSDP2 data parallelism and
non-reentrant full or layer-selective activation checkpointing. TP, PP,
`torch.compile`, and operation-selective activation checkpointing are rejected
in the training setup. DDP/higher-order gradients are not integrated. Single
device profiling needs no extra setup. To add v4 to old measurements, use the
incremental commands below with `--implementation pc_v4`; all three PC
checkpoint modes can be swept, without timing older versions again.

## Run

```bash
# Default: plain + pc_baseline, three seeds, BF16, one block, N=1024, d=1024, h=4096.
python scripts/profile/benchmark_mlp.py \
  --label reference --output outputs/pc_profile/reference.csv

# Width sweep: hidden width defaults to 4*d for each d.
python scripts/profile/benchmark_mlp.py \
  --dim 256 512 1024 2048 --tokens 512 \
  --seeds 0 1 2 3 4 --warmup 10 --steps 50 \
  --label reference --output outputs/pc_profile/width_reference.csv

# Sweep input volume instead. These are input rows, not vocabulary tokens.
python scripts/profile/benchmark_mlp.py \
  --dim 1024 --tokens 128 512 2048 --seeds 0 1 2 \
  --output outputs/pc_profile/tokens.csv

# Accumulation sweep: N=256 per microbatch; effective rows per step = 256*accum.
python scripts/profile/benchmark_mlp.py \
  --dim 1024 --tokens 256 --grad-accum-steps 1 2 4 \
  --seeds 0 1 2 --output outputs/pc_profile/accum.csv

# Small CPU smoke test (CUDA memory fields will be blank).
python scripts/profile/benchmark_mlp.py \
  --device cpu --dtype float32 --dim 16 --hidden-dim 32 --tokens 8 \
  --seeds 0 1 --warmup 2 --steps 3 --output outputs/pc_profile/smoke.csv
```

Other useful flags: `--implementation plain` or `pc_baseline`, `--layers 1 2 4`,
`--pc-level 0..4`, `--power-iter 10`, `--no-learnable-gamma`, and
`--no-recover-w-norm`. `--dtype` sets the actual parameter/input/buffer dtype:
`bfloat16` or `float32`, without autocast or FP32 master weights. Loss and targets
are always FP32. PC normalization is spectral normalization. The `pc_*`, norm
recovery, and gamma options do not affect the ordinary `plain` MLP.

Weights stay fixed; this measures forward/loss/backward, not learning progress.
Parameter gradients are cleared once at the start of each accumulated step.
Each microbatch computes `MSE / grad_accum_steps` and immediately backpropagates;
parameter gradients accumulate across microbatches. Baseline frees each graph
with its backward; v1 frees each activation graph immediately and retains one
weight-transform graph until the final backward. SN state is initialized before
warmup, then refreshed **once after all microbatches**, with its cost included in
total time.
There is no optimizer. Input/target generation, initial SN setup, warmup, and
finite-gradient/equivalence checks are outside measurement. Add rewritten PC
operators as independent version files, keeping the reference frozen.

The same fixed input/target is reused for each microbatch, as it is across
measured steps. This keeps random generation, copies, and storage for an entire
effective batch out of the benchmark. Only one microbatch's activations and
input gradient are kept at a time; input gradients are reset per microbatch.
The logged loss is unscaled MSE, so its scale is independent of accumulation.
`--warmup` and `--steps` count full accumulated steps.

Each seed independently initializes weights, input, target, and SN vectors.
Plain and PC get identical initial linear weights and data for the same seed.
PC versions also load the reference's exact initial gamma and SN state.
Implementation order alternates between seeds to reduce ordering bias. A fixed
seed reproduces data, not GPU scheduling or timing; repeated measured steps
within each seed also capture runtime noise.

## Reuse and merge results

Two workflows avoid timing the old versions again:

- `--no-reference-benchmark` measures only the explicitly requested versions.
  New PC versions still run baseline forward/backward for correctness; that
  reference work is outside performance measurement.
- `--resume-from old.csv ...` reuses matching successful cases and writes a new
  combined CSV containing the input results and newly measured cases. Select
  all desired versions to fill missing cases automatically, or only a new
  version to add it while carrying the old versions forward. A `REUSED` line
  means no correctness test, warmup, forward/backward, or measurement ran for
  that case. Failed selected cases are retried; the new attempt replaces the
  matching old failure. Failures outside the requested grid are preserved.

For example, add v3's three modes to the old width sweep:

```bash
python scripts/profile/benchmark_mlp.py \
  --device cuda --dtype bfloat16 \
  --implementation pc_v3 --no-reference-benchmark \
  --pc-checkpoint none polynomial full \
  --dim 512 1024 2048 --layers 2 --tokens 512 --grad-accum-steps 4 \
  --pc-level 4 --power-iter 10 --seeds 0 1 2 3 4 --warmup 10 --steps 50 \
  --resume-from outputs/pc_profile/previous/width.csv \
  --label width --output outputs/pc_profile/width_with_v3.csv
python scripts/profile/plot_mlp.py outputs/pc_profile/width_with_v3.csv \
  --x dim --band std --output outputs/pc_profile/width_with_v3.png
```

Replace `previous/width.csv` with the real existing CSV. If a v3 mode was already
completed, include that CSV in `--resume-from` too; only missing cases are timed.
The output must be a new path, so input files are never overwritten.

For the four existing H200 sweeps, this Bash loop adds v3 and plots each merged
result. Set `PC_PREVIOUS_OUT` to the directory containing the old four CSVs:

```bash
set -euo pipefail
source ./.venv-profile/bin/activate
PC_PREVIOUS_OUT="outputs/pc_profile/previous"
PC_INCREMENTAL_OUT="outputs/pc_profile/incremental_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PC_INCREMENTAL_OUT"
COMMON=(--device cuda --dtype bfloat16 --implementation pc_v3
  --no-reference-benchmark --pc-checkpoint none polynomial full
  --pc-level 4 --power-iter 10 --seeds 0 1 2 3 4 --warmup 10 --steps 50)
for sweep in accum batch depth width; do
  SCALE=()
  case "$sweep" in
    accum) GRID=(--dim 1024 --layers 2 --tokens 512 --grad-accum-steps 1 2 4 8)
           AXIS=grad_accum_steps; SCALE=(--log-x) ;;
    batch) GRID=(--dim 1024 --layers 2 --tokens 128 512 2048 8192 --grad-accum-steps 4)
           AXIS=tokens; SCALE=(--log-x) ;;
    depth) GRID=(--dim 1024 --layers 1 2 4 8 --tokens 512 --grad-accum-steps 4)
           AXIS=layers ;;
    width) GRID=(--dim 512 1024 2048 --layers 2 --tokens 512 --grad-accum-steps 4)
           AXIS=dim ;;
  esac
  python scripts/profile/benchmark_mlp.py "${COMMON[@]}" "${GRID[@]}" \
    --resume-from "$PC_PREVIOUS_OUT/$sweep.csv" --label "$sweep" \
    --output "$PC_INCREMENTAL_OUT/$sweep.csv"
  python scripts/profile/plot_mlp.py "$PC_INCREMENTAL_OUT/$sweep.csv" \
    --x "$AXIS" "${SCALE[@]}" --band std --output "$PC_INCREMENTAL_OUT/$sweep.png"
done
```

To merge separately measured CSVs without running torch or a GPU:

```bash
python scripts/profile/merge_mlp.py \
  outputs/pc_profile/previous/width.csv outputs/pc_profile/width_v3_only.csv \
  --label width --output outputs/pc_profile/width_combined.csv
python scripts/profile/plot_mlp.py outputs/pc_profile/width_combined.csv \
  --x dim --band std --output outputs/pc_profile/width_combined.png
```

Case identity includes implementation/reference hashes, seed, dimensions,
accumulation, dtype, PC/checkpoint settings, warmup/measurement lengths, device
and PyTorch/CUDA metadata, and correctness tolerances. Git commit, run ID, label,
and source path do not affect identity. Unknown old hashes are preserved as
unknown and do not match current hashed implementations. Missing accumulation
and checkpoint columns mean 1 and `none`, respectively. Different identities
remain separate; plotting separates configurations and hashes rather than
silently pooling measurements from different environments or implementations.

Identical copied rows are deduplicated. Successful results supersede matching
failures. Repeated **different measurements of the same case/seed** are an error
by default. Use `--on-conflict first` or `--on-conflict last` to select one
explicitly, for both merge and resume. These refer to input file/row order, not
timestamps. This is useful if two sweeps measured their shared central point
independently. Duplicate measurements are never averaged or counted as extra
seeds. All original columns are retained, with `source_csv` and `source_label`
preserving provenance; the merged display label defaults to the output filename.

## Plot CSVs

```bash
# Curves and shaded intervals as width changes.
python scripts/profile/plot_mlp.py outputs/pc_profile/width_reference.csv \
  --x dim --output outputs/pc_profile/width.png

# Compare results before and after a PC rewrite, using the same seeds/settings.
python scripts/profile/plot_mlp.py \
  outputs/pc_profile/width_reference.csv outputs/pc_profile/width_optimized.csv \
  --labels reference optimized --x dim --output outputs/pc_profile/comparison.png

# Input-volume sweep; optionally show min/max instead of standard deviation.
python scripts/profile/plot_mlp.py outputs/pc_profile/tokens.csv \
  --x tokens --band range --log-x --output outputs/pc_profile/tokens.png

# Time and memory versus accumulation count.
python scripts/profile/plot_mlp.py outputs/pc_profile/accum.csv \
  --x grad_accum_steps --output outputs/pc_profile/accum.png
```

The six panels show forward, MSE loss, backward, SN update, total time, and peak
allocated memory. Lines are **means across seeds**; the default shaded band is
**mean +/- one sample standard deviation across seeds**, not a confidence
interval. `--band range` shows the seed min/max. A single seed has no estimated
variability. Each seed contributes equally, regardless of measured step count.
Separate workloads, precision modes, PC settings, measurement lengths, and device
environments form separate curves. A width sweep groups by `h/d`; sweeps of other
axes hold the remaining dimensions fixed. PNG, PDF, and SVG outputs are supported
by selecting the output suffix. Plotting only needs CSVs and matplotlib, not torch.

## CSV and measurement details

One CSV row corresponds to one `(implementation, shape, accumulation, seed)`.
`grad_accum_steps` and `effective_tokens = tokens * grad_accum_steps` are recorded.
`forward_ms`,
`loss_ms`, `backward_ms`, `sn_update_ms`, and `total_ms` are arithmetic means over
measured accumulated steps **within that seed**. Forward, loss, and backward
times are each the **sum across all microbatches** in a step; SN update runs once
and total covers the entire accumulated step. Their `*_p95_ms` companions describe
within-seed timing variation and are not used for the cross-seed plot bands.
The terminal also reports total time and CUDA peak memory as mean +/- sample SD
across successful seeds.

CUDA phase timings use events on the current stream; total spans forward through
SN update, including gaps between microbatches. There are no extra inter-phase
synchronizations. `wall_ms` is the
synchronized wall time per iteration, including gradient clearing and measuring
overhead. CPU smoke tests use wall-clock phase timings. CUDA timing and memory
results still need validation on the actual H200 slice.

Memory counters reset after warmup and gradient clearing. `peak_allocated_mib`
is the maximum live PyTorch tensor allocation over the measured iterations,
including parameters, data, activations, gradients, and PC intermediates.
`start_allocated_mib` is the starting allocation; `extra_peak_allocated_mib` is
the peak minus this start, including newly created gradients.
`peak_reserved_mib` also includes allocator caches, including those populated
during warmup. These are PyTorch allocator figures, not the entire device's
driver-reported usage. CPU memory fields are blank, not zero.

Rows include version filename/SHA-256, reference SHA-256, correctness status,
tolerances, number of tensor checks, maximum absolute/relative errors, and
diagnostics, alongside loss, parameter/buffer sizes, seed, configuration, torch/CUDA
versions, Git commit/dirty status, and device name/visible memory/SM count.
The actual CUDA device capacity is recorded rather than assuming a full H200.
Cases run sequentially; unused caches are released between cases, not between
measured iterations. Start with small shapes on the 16GB slice and expand.

Output files are never overwritten. Rows are flushed immediately. OOM cases
are recorded with `status=oom`, and remaining cases continue. Non-finite losses
or gradients yield `status=invalid`. Any failed case produces a nonzero final
exit status. Plotting skips failed rows with a warning and reports the number
of successful seeds per point in the figure title.
The plotting script also accepts older CSVs without `grad_accum_steps`, treating
them as accumulation count 1. Older CSVs used `baseline` for ordinary Linear and
`pc` for the then-current training operator; these old labels are preserved and
do not establish equivalence to the frozen PC reference. Version hashes also
separate curves when implementation names are the same but source files differ.

Run the CPU accumulation checks from the repo root:

```bash
python -m unittest scripts.profile.test_benchmark_mlp scripts.profile.test_pc_v1 scripts.profile.test_pc_v2 scripts.profile.test_pc_v3 scripts.profile.test_pc_v4 scripts.profile.test_profile_results
```

These compare accumulated parameter/gamma/input gradients against a full batch,
verify the frozen snapshot and timing boundaries, accept identical FP32/BF16
versions, and reject incorrect outputs, backward gradients, and SN updates.
The v1 tests also use distinct microbatches, change weights/gamma between steps,
compare every input gradient, count PC forward/backward calls, verify cache
cleanup on failure, and ensure deferred backward stays inside the timed region.
The v2 tests check tall/wide/square VJPs, double-precision finite differences,
gamma edge cases, distinct microbatches and changing weights, saved storage, and
GEMM counts. Its two CUDA tests skip on hosts without CUDA/Triton.

The full local suite passes on PyTorch 2.14.0 (53 CPU tests; five CUDA tests
skipped). The v4 suite also passes on PyTorch 2.7.0. Running the historical suite
on this 2.7 CPU build reproduces an existing v1 BF16/G=3 gradient-gate failure
(max absolute error 0.00286865, above the unchanged atol=0.002); it also fails
from the committed snapshot before the FSDP changes. Historical code and
correctness tolerances are preserved.

## Two-process CPU FSDP2 checks

Run real FSDP2 with two CPU processes and Gloo, without GPUs:

```bash
python scripts/profile/check_fsdp_cpu.py \
  --implementation plain pc_baseline pc_v4 \
  --grad-accum-steps 1 3 --seeds 0 1 --with-plain-head --clip-grad 0.1
```

The checker uses FP32 master parameters and a small two-block MLP. It matches the repo's FSDP
wrapping: shard each block, reshard the first after forward, retain the last
after forward, and shard the root. Each rank gets distinct microbatches. The
unsharded reference uses ordinary Linear for `plain`, or the frozen PC baseline
for PC versions, and averages the data from both ranks. The checks compare
outputs, loss, input gradients, reconstructed full parameter/gamma gradients,
and parameters after two SGD steps. They also compare updated SN buffers with
the reference and across ranks, and verify that accumulation caches are cleared.
`--with-plain-head` includes ordinary parameters and bias alongside PC, so the
check also catches incomplete reductions in a mixed model. `--clip-grad 0.1`
checks global gradient clipping before the optimizer update.

Exercise all PC checkpoint modes together with BF16 forward parameters,
activation recomputation, and synchronization only on the final microbatch:

```bash
python scripts/profile/check_fsdp_cpu.py \
  --implementation pc_v4 --pc-checkpoint none polynomial full \
  --grad-accum-steps 3 --seeds 0 1 \
  --activation-checkpoint none full selective --mixed-precision bfloat16 \
  --gradient-sync final --no-input-grad --with-plain-head --clip-grad 0.1
```

BF16 uses FP32 master parameters and FP32 reduction by default; the unsharded
reference also casts its forward parameters to BF16. `--input-grad` is on by
default. The second command checks that FSDP's backward hooks still run when
the model inputs need no gradients (as with integer token inputs in training).

Each case runs in fresh processes with a timeout. A native worker crash is
reported as a failed case and does not stop the remaining cases. The JSON report
includes version hashes, per-rank results, the last recorded stage before a
crash, and available tracebacks. Its default path is a timestamped file under
`outputs/pc_profile/`; `--output path.json` chooses a new path. Any failure returns
exit status 1, including a reproduced incompatibility in an existing version.

On local **PyTorch 2.14.0**, the v4 checks passed with all PC checkpoint modes,
FP32/BF16, full/layer-selective activation recomputation, both synchronization
policies, clipping, ordinary parameters, and two distinct seeds. Older v1/v2
passed at G=1 but crashed during context-exit PC backward at G=3. The checker
can still select old versions to reproduce failures, without modifying them.

The same tests were also run using an isolated official **PyTorch 2.7.0** wheel.
On this macOS build, Gloo's native FP32 `reduce_scatter_tensor(..., AVG)` silently
returns the SUM: a direct collective probe returns 3.0 instead of 1.5, and even
plain MLP gradients are exactly twice the reference. The JSON records that
probe. Do not change the training loss or compensate inside the PC operator.
For this CPU backend, add `--reduce-dtype float64` to either command: it selects
FSDP's existing SUM + explicit-division path. Both commands passed with that
option, including BF16 forward/master-parameter updates; this does **not** claim
native FP32 reduction passed on this 2.7 CPU build. No FSDP internals are patched.

These are CPU correctness and parameter-lifecycle checks. Triton kernels, NCCL,
GPU overlap, peak memory, and speed still require CUDA validation on the target
machine; neither CPU reduction configuration substitutes for it. The checker
uses the older composable FSDP import when the public import is unavailable.
Local processes communicate through the loopback interface;
`GLOO_SOCKET_IFNAME` can override that selection.
