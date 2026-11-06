#!/bin/bash
# =============================================================================
# iSKYLAB warm-cloud paper: extra simulation sets (Nov 2026)
#
# Each run writes to <repo>/runs/<tag>/single_comparison/ (via --tag; ignored by git), so the
# baseline is never overwritten.  Run everything with:   ./batchRuns_v2.sh all
# or a single set with e.g.                               ./batchRuns_v2.sh comp
#
# Sets:
#   baseline  : the paper baseline (identical to batchRuns.sh, for a clean rerun)
#   nobl      : no wall-conditioned mixing (closed parcel), all runs   [Fig. 6]
#   comp      : competition counterfactuals for the mixtures 006-010, 018, 019:
#               background only, NaCl only; each with and without BL mixing
#   mixtype   : homogeneous evaporation and inhomogeneous p=0 (full moving)
#   bins      : moving-centre and Chen-Lamb bin schemes
#   fixed     : one parameter set for all fan-on runs (tau=10 s, dT=-0.09 K, p=2)
#   rhmeas    : measured initial RH + finite-rate wall vapour sink (mode 2)
#   process   : no fall-out; no collisions (SCE off)
#   robust    : is competition robust? paired runs (mixture / background only)
#               for Exp006, 007, 018 (coarse NaCl) and 008 (fine NaCl) with
#               homogeneous and p=0 evaporation, Chen-Lamb bins, tau = 5 and 20 s,
#               and dT_BL +/- 0.03 K.  Tags rob_<variant>_mix / rob_<variant>_bg.
#               Summarise with analysis/paper_figs/competition_robust.py
#   refit     : wall-air temperature offset dT_BL stepped from -0.015 to +0.03 K
#               around each run's baseline value (all 14 runs), to test whether a
#               different wall effect plus welas-inlet evaporation fits number and
#               liquid water together.  Tags refit_p005 ... (offset in mK).
#               Summarise with analysis/paper_figs/inlet_refit.py
#   refitcomp : background-only runs at the refitted wall-air offsets for the
#               coarse-NaCl mixtures (006 +0.005, 007 +0.005, 018 +0.02 K), so that
#               competition can be recomputed with the inlet-refit wall settings.
# Counting-threshold sensitivity is post-processing only (no runs needed).
# =============================================================================
set -u
PY="python3 dataAnalysis_new.py"
COMMON="--no-plot"

# exp  t_sat(min)  tau(s)  dT_BL(K)  p     measured RH0 (Datasets-V2, t<0)
read -r -d '' RUNS <<'TABLE'
Exp005 8     80 -0.35  2    0.870
Exp006 12    10 -0.09  2    0.834
Exp007 12    10 -0.1   2    0.820
Exp008 12    10 -0.095 1.2  0.834
Exp009 12    10 -0.095 1.2  0.840
Exp010 11    10 -0.095 2    0.840
Exp011 13    10 -0.1   1.5  0.815
Exp013 11    10 -0.085 2    0.822
Exp014 10.5  10 -0.106 2    0.848
Exp015 10.75 10 -0.075 1.2  0.835
Exp016 10.6  10 -0.08  1.75 0.841
Exp017 1.25  10 -0.55  2    0.825
Exp018 7     10 -0.09  2    0.770
Exp019 11    10 -0.1   2    0.851
TABLE
MIXTURES="Exp006 Exp007 Exp008 Exp009 Exp010 Exp018 Exp019"

run () {   # run <exp> <tag> <extra args...>  (baseline tuned parameters + extras)
  local exp=$1 tag=$2; shift 2
  local line; line=$(echo "$RUNS" | awk -v e="$exp" '$1==e')
  read -r _ tsat tau dt p rh <<< "$line"
  echo "=== $tag  $exp ==="
  $PY --experiment "$exp" --cloud-formation-time-min "$tsat" --bl-tau-s "$tau" \
      --bl-evap-size-exp "$p" --bl-wall-water-mode 0 --bl-temp-offset-k "$dt" \
      --tag "$tag" $COMMON "$@" || echo "FAILED: $tag $exp" >> failed_runs.txt
}
allexp () { echo "$RUNS" | awk '{print $1}'; }

set_baseline () { for e in $(allexp); do run $e baseline; done; }
set_nobl ()     { for e in $(allexp); do run $e nobl --set chamber_bl_mix=0; done; }
set_comp () {
  for e in $MIXTURES; do
    run $e comp_bg   --aerosol-only 1
    run $e comp_nacl --aerosol-only 2
    run $e comp_mix_nobl  --set chamber_bl_mix=0
    run $e comp_bg_nobl   --aerosol-only 1 --set chamber_bl_mix=0
    run $e comp_nacl_nobl --aerosol-only 2 --set chamber_bl_mix=0
  done
}
set_mixtype ()  { for e in $(allexp); do
                    run $e hom --set chamber_bl_evap_mode=1
                    run $e p0  --bl-evap-size-exp 0
                  done; }
set_bins ()     { for e in $(allexp); do
                    run $e movingcentre --set bin_scheme_flag=1
                    run $e chenlamb     --set bin_scheme_flag=2
                  done; }
set_fixed ()    { for e in $(allexp); do
                    [ "$e" = Exp005 ] && continue
                    [ "$e" = Exp017 ] && continue
                    run $e fixed --bl-temp-offset-k -0.09 --bl-evap-size-exp 2
                  done; }
set_rhmeas ()   { for e in $(allexp); do
                    line=$(echo "$RUNS" | awk -v x="$e" '$1==x'); read -r _ tsat tau dt p rh <<< "$line"
                    echo "=== rhmeas $e ==="
                    $PY --experiment "$e" --initial-rh "$rh" --bl-tau-s "$tau" --bl-evap-size-exp "$p" \
                        --bl-wall-water-mode 2 --wall-vapour-transfer-velocity-ms 1.0e-3 \
                        --bl-temp-offset-k "$dt" --tag rhmeas $COMMON || echo "FAILED: rhmeas $e" >> failed_runs.txt
                  done; }
set_process ()  { for e in $(allexp); do
                    run $e nofallout --set fallout_flag=.false.
                    run $e nosce     --set sce_flag=0
                  done; }

ROBEXP="Exp006 Exp007 Exp018 Exp008"
pair () {   # pair <exp> <variant> <extra args...>: mixture and background-only run
  local e=$1 v=$2; shift 2
  run $e rob_${v}_mix "$@"
  run $e rob_${v}_bg --aerosol-only 1 "$@"
}
set_robust () {
  for e in $ROBEXP; do
    dt=$(echo "$RUNS" | awk -v x="$e" '$1==x{print $4}')
    pair $e hom      --set chamber_bl_evap_mode=1
    pair $e p0       --bl-evap-size-exp 0
    pair $e chenlamb --set bin_scheme_flag=2
    pair $e tau5     --bl-tau-s 5
    pair $e tau20    --bl-tau-s 20
    pair $e dTcold   --bl-temp-offset-k $(awk -v d="$dt" 'BEGIN{print d-0.03}')
    pair $e dTwarm   --bl-temp-offset-k $(awk -v d="$dt" 'BEGIN{print d+0.03}')
  done
}

REFIT_OFFSETS="-0.015 -0.01 -0.005 0.005 0.01 0.015 0.02 0.025 0.03"
set_refit () {
  for e in $(allexp); do
    dt=$(echo "$RUNS" | awk -v x="$e" '$1==x{print $4}')
    for off in $REFIT_OFFSETS; do
      tag=refit_$(awk -v o="$off" 'BEGIN{v=o*1000; a=(v<0?-v:v); printf "%s%03d", (v<0?"m":"p"), int(a+0.5)}')
      run $e $tag --bl-temp-offset-k $(awk -v d="$dt" -v o="$off" 'BEGIN{print d+o}')
    done
  done
}

set_refitcomp () {
  for pair_ in "Exp006 0.005 refit_p005" "Exp007 0.005 refit_p005" "Exp018 0.02 refit_p020"; do
    read -r e off tag <<< "$pair_"
    dt=$(echo "$RUNS" | awk -v x="$e" '$1==x{print $4}')
    run $e ${tag}_bg --aerosol-only 1 --bl-temp-offset-k $(awk -v d="$dt" -v o="$off" 'BEGIN{print d+o}')
  done
}

case "${1:-all}" in
  baseline) set_baseline ;; nobl) set_nobl ;; comp) set_comp ;; mixtype) set_mixtype ;;
  bins) set_bins ;; fixed) set_fixed ;; rhmeas) set_rhmeas ;; process) set_process ;; robust) set_robust ;; refit) set_refit ;; refitcomp) set_refitcomp ;;
  all) set_baseline; set_nobl; set_comp; set_mixtype; set_bins; set_fixed; set_rhmeas; set_process ;;
  *) echo "unknown set $1"; exit 1 ;;
esac
