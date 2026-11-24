#!/bin/bash
# After ./batchRuns_v2.sh v19final: rescore the measured-start sets, make all paper figures and the numbers file.
# Run from anywhere: bash analysis/paper_figs/make_v19.sh
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
export ISKYLAB_RUNS="$REPO/runs"
export ISKYLAB_INLET_DD2=$(awk '$1=="dD2"{print $2}' runs/analysis/v19_settings.txt)
echo "WELAS inlet dD2 = $ISKYLAB_INLET_DD2 um2"
# wall-sink and combined-closure rows of Table 3 (start from the measured humidity; scored with the new operator)
for t in wallsink_km3.0e-4 wallsink_km1.0e-3 wallsink_km3.0e-3 comb_ice_km3.0e-4; do
  mkdir -p runs/${t}_in/single_comparison; cp -n runs/$t/single_comparison/output-*.nc runs/${t}_in/single_comparison/ 2>/dev/null || true
done
scripts/score_existing.sh rh wallsink_km3.0e-4_in wallsink_km1.0e-3_in wallsink_km3.0e-3_in comb_ice_km3.0e-4_in
cd analysis/paper_figs
for f in replot figs_v16 new_figs fig_psd fig_inlet fig_competition_v2; do echo "== $f"; python3 $f.py "$ISKYLAB_RUNS" > "$ISKYLAB_RUNS/analysis/$f.log" 2>&1 || echo "  FAILED: see runs/analysis/$f.log"; done
python3 v19_numbers.py > "$ISKYLAB_RUNS/analysis/v19_numbers.txt" 2>&1 || echo "v19_numbers FAILED"
echo "done: figures in runs/analysis/*.png; numbers in runs/analysis/v19_numbers.txt"
