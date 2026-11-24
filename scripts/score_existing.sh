#!/bin/bash
# score_existing.sh base|rh <tag>... : recompute metrics for model output already in runs/<tag> (no model run),
# with the current WELAS inlet operator (ISKYLAB_INLET_DD2) and comparison window (ISKYLAB_WINDOW, default full).
#   base: runs started from the fitted cloud-formation time (RUNS table / v19_settings.txt if present)
#   rh  : runs started from the measured initial humidity (wallsink / combined sets)
cd "$(dirname "$0")"; mode=$1; shift
S=../runs/analysis/v19_settings.txt
for tag in "$@"; do
  while read -r e ts tau dt p rh; do
    [ -f ../runs/$tag/single_comparison/output-$e.nc ] || continue
    if [ "$mode" = rh ]; then a="--initial-rh $rh"
    else t=$( [ -f $S ] && awk -v x=$e '$1==x{print $2}' $S ); a="--cloud-formation-time-min ${t:-$ts}"; fi
    python3 dataAnalysis_new.py --experiment $e $a --tag $tag --no-run --no-plot >/dev/null 2>&1 || echo "FAIL $tag $e"
  done < <(sed -n '/^Exp005 8/,/^Exp019/p' batchRuns_v2.sh)
  echo "$tag: $(ls ../runs/$tag/single_comparison/metrics-*.csv 2>/dev/null | wc -l) metrics"
done
