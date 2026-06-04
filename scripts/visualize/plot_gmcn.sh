#!/bin/bash
# Regenerate the GMCN (modified_condition_number) plots for the
# baseline-vs-PC comparison, in a single metrics_plotter call:
#   1. global.pdf   full-model GMCN (geometric mean)
#   2. pc.pdf       "GMCN (PC blocks)"      -- layers PC is applied to
#   3. no_pc.pdf    "GMCN (non-PC blocks)"  -- layers PC is not applied to
#
# Groups are decided per-layer by whether PC was actually applied (recorded in
# the data), not by hardcoded weight-type names -- so the split tracks whatever
# blocks a run preconditions.
#
# GMCN = sigma_1 / mean(bottom 10% of singular values), aggregated across a
# group with the geometric mean (the `global` and per-group levels share this
# aggregator, so the curves are directly comparable).
set -ex

# Activate your environment first (example with conda):
#   conda activate pc
# This script cd's to the repo root so the relative input/output paths resolve.
cd "$(dirname "$0")/../.." || exit 1

VIS_ROOT=${VIS_ROOT:-visualization_output}
# SVD JSON inputs live under visualization_output/svd/ (see get_output_dir).
SVD_ROOT=${VIS_ROOT}/svd
BASELINE=${SVD_ROOT}/llama2_1B_AdamW_AdamW_baseline_1B
PC=${SVD_ROOT}/llama2_1B_AdamW_AdamW_pc_1B
# All baseline-vs-PC plots share one comparison root, one subdir per plot type
# (this script -> gmcn/, plot_sv_hist.sh -> sv_hist/), so the two comparisons of
# the same runs stay together instead of in unrelated top-level dirs.
COMPARE_ROOT=${VIS_ROOT}/compare_adamw_baseline_vs_pc
OUTDIR=${COMPARE_ROOT}/gmcn

# 1B run total tokens (B): 4 * 8 * 10 * 8192 * 61100 = 160.169984 B
TOTAL_TOKENS=160.169984

# One call produces global.pdf and the per-group plots under
# OUTDIR/modified_condition_number/.
python3 -m visualize.metrics_plotter \
    "${BASELINE}" "${PC}" \
    --metrics modified_condition_number \
    --labels "baseline" "PC" \
    --plot-levels global by_group \
    --x-unit token --total-tokens "${TOTAL_TOKENS}" \
    --output-dir "${OUTDIR}"

echo "Done. Plots under: ${OUTDIR}/modified_condition_number/{global,pc,no_pc}.pdf"
