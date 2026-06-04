#!/bin/sh
# Singular-value distribution comparison between the 1B baseline and the
# baseline + PC (ffn_o) run, at a given step.
#
# Singular values are always normalized by their own sigma_1 to [0, 1] before
# plotting, so distribution shapes can be compared across experiments.
#   OUTDIR/step_<STEP>/<wt>/layer{0..17}.pdf
#
# The plot style is controlled by --mode (see the command below); pick one of:
#   hist     histogram (default)
#   ecdf     empirical CDF (stepped curve); stacks cleanly without occlusion
#   quantile empirical quantile function: x = sv percentile (small -> large),
#            y = sigma/sigma_max at that percentile
#
# Usage:
#   bash plot_sv_hist.sh                       # step 1 (default)
#   STEP=61100 bash plot_sv_hist.sh
#   VIS_ROOT=other_dir bash plot_sv_hist.sh    # change the root dir
set -ex

# Activate your environment first (example with conda):
#   conda activate pc
# This script cd's to the repo root so the relative input/output paths resolve.
cd "$(dirname "$0")/../.." || exit 1

STEP=${STEP:-1}

# Root dir holding the JSON inputs and the output; kept as a variable so a
# future rename only touches this one line (or can be overridden via env).
VIS_ROOT=${VIS_ROOT:-visualization_output}

# SVD JSON inputs live under visualization_output/svd/ (see get_output_dir).
SVD_ROOT=${VIS_ROOT}/svd
BASELINE=${SVD_ROOT}/llama2_1B_AdamW_AdamW_baseline_1B
PC=${SVD_ROOT}/llama2_1B_AdamW_AdamW_pc_1B
# All baseline-vs-PC plots share one comparison root, one subdir per plot type
# (plot_gmcn.sh -> gmcn/, this script -> sv_hist/), so the two comparisons of
# the same runs stay together. plotter appends a step_<STEP>/ subdir under this.
COMPARE_ROOT=${VIS_ROOT}/compare_adamw_baseline_vs_pc
OUTDIR=${COMPARE_ROOT}/sv_hist

BASELINE_JSON="${BASELINE}/singular_values_step_${STEP}.json"
PC_JSON="${PC}/singular_values_step_${STEP}.json"

for f in "${BASELINE_JSON}" "${PC_JSON}"; do
    if [ ! -f "${f}" ]; then
        echo "[error] expected ${f} not found" >&2
        exit 1
    fi
done

# Singular-value distribution, baseline and PC overlaid on one figure (each
# normalized by its own sigma_1 to [0, 1]). Switch --mode to ecdf / quantile
# for a different plot style.
# (plotter parses the step from the filename and appends a step_<STEP>/ subdir
#  under --output-dir.)
python3 -m visualize.plotter \
    "${BASELINE_JSON}" "${PC_JSON}" \
    --labels "baseline" "PC" \
    --mode hist \
    --output-dir "${OUTDIR}"

echo "Done."
echo "  SV distributions: ${OUTDIR}/step_${STEP}/<wt>/"
