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
- `pc_v3`, etc.: future versions, each in its own `pc_*.py` file here.

`pc_baseline.py` is a byte-for-byte snapshot of
`torchtitan/pc_layer/pc_layer.py` at commit
`a97b1ecf2062065df29ba7d8fc48e117811fbb39`. Its SHA-256 is
`91204116c4204c73400c28053ca2d98ae2d800faba096da7f847156e5b94b87b`.
Keep this file unchanged; the CPU tests enforce its hash. Profiling imports this
snapshot directly, so changing the training operator will not change the reference.

Version files are discovered automatically, without editing a registry:

```bash
cp scripts/profile/pc_v2.py scripts/profile/pc_v3.py
# Edit pc_v3.py to implement the next optimization.
python scripts/profile/benchmark_mlp.py \
  --implementation pc_baseline pc_v2 pc_v3 --dim 512 1024 --tokens 256 \
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
included for performance comparison. The terminal reports paired-seed speedup
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
python -m unittest scripts.profile.test_benchmark_mlp scripts.profile.test_pc_v1 scripts.profile.test_pc_v2
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
