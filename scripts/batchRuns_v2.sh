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
#   wallsink  : the mixed-phase paper's treatment of the pre-cloud wall loss,
#               applied to the warm runs: measured initial RH (no fitted
#               cloud-formation time), finite-rate wall vapour sink (mode 2) with
#               a scan of km = 3e-4 (the mixed-phase value), 1e-3, 3e-3, 1e-2 m/s,
#               current per-run offsets, tau and p (tags wallsink_km<km>); plus a
#               rerun of the baseline with the current BMM build (tag basecheck),
#               so that differences are not code changes.  Summarise with
#               analysis/paper_figs/wallsink_check.py
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

# Parallel runs: NJOBS=8 ./batchRuns_v2.sh <set>   (default 1 = one at a time; each run's log goes to logs/)
NJOBS=${NJOBS:-1}
throttle () { while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$NJOBS" ]; do sleep 1; done; }
run () {   # run <exp> <tag> <extra args...>  (baseline tuned parameters + extras)
  local exp=$1 tag=$2; shift 2
  local line; line=$(echo "$RUNS" | awk -v e="$exp" '$1==e')
  read -r _ tsat tau dt p rh <<< "$line"
  echo "=== $tag  $exp ==="
  if [ "$NJOBS" -gt 1 ]; then
    throttle; mkdir -p logs
    ( $PY --experiment "$exp" --cloud-formation-time-min "$tsat" --bl-tau-s "$tau" \
        --bl-evap-size-exp "$p" --bl-wall-water-mode 0 --bl-temp-offset-k "$dt" \
        --tag "$tag" $COMMON "$@" > "logs/$tag-$exp.log" 2>&1 || echo "FAILED: $tag $exp" >> failed_runs.txt ) &
  else
    $PY --experiment "$exp" --cloud-formation-time-min "$tsat" --bl-tau-s "$tau" \
        --bl-evap-size-exp "$p" --bl-wall-water-mode 0 --bl-temp-offset-k "$dt" \
        --tag "$tag" $COMMON "$@" || echo "FAILED: $tag $exp" >> failed_runs.txt
  fi
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

WALL_KMS="3.0e-4 1.0e-3 3.0e-3 1.0e-2"
wsrun () {   # wsrun <exp> <tag> <km> <extra args...>: measured RH0, wall mode 2, finite-rate sink km
  local exp=$1 tag=$2 km=$3; shift 3
  local line; line=$(echo "$RUNS" | awk -v e="$exp" '$1==e')
  read -r _ tsat tau dt p rh <<< "$line"
  echo "=== $tag  $exp ==="
  $PY --experiment "$exp" --initial-rh "$rh" --bl-tau-s "$tau" --bl-evap-size-exp "$p" \
      --bl-wall-water-mode 2 --wall-vapour-transfer-velocity-ms "$km" \
      --bl-temp-offset-k "$dt" --tag "$tag" $COMMON "$@" || echo "FAILED: $tag $exp" >> failed_runs.txt
}
set_wallsink () {
  for e in $(allexp); do run $e basecheck; done
  for km in $WALL_KMS; do
    for e in $(allexp); do wsrun $e wallsink_km$km $km; done
  done
}

# Combined wall closure (BMM wall-water mode 3): measured RH0, the mode-2 finite-rate wall
# flux at all times plus the mode-0 saturation cap in cloud.  Liquid wall: chamber_wall_liquid_tmin=250 K
# (wall water stays liquid when the measured wall temperature drops below 0 C); ice wall: 273.15 K.
COMB_SPECS="liq:0.0:250.0 liq:1.0e-4:250.0 liq:2.0e-4:250.0 liq:3.0e-4:250.0 liq:5.0e-4:250.0 liq:1.0e-3:250.0 ice:1.0e-4:273.15 ice:3.0e-4:273.15 ice:1.0e-3:273.15"
crun () {   # crun <exp> <tag> <km> <tmin> <extra args...>
  local exp=$1 tag=$2 km=$3 tmin=$4; shift 4
  local line; line=$(echo "$RUNS" | awk -v e="$exp" '$1==e')
  read -r _ tsat tau dt p rh <<< "$line"
  echo "=== $tag  $exp ==="
  $PY --experiment "$exp" --initial-rh "$rh" --bl-tau-s "$tau" --bl-evap-size-exp "$p" \
      --bl-wall-water-mode 3 --wall-vapour-transfer-velocity-ms "$km" \
      --set chamber_wall_liquid_tmin="$tmin" \
      --bl-temp-offset-k "$dt" --tag "$tag" $COMMON "$@" || echo "FAILED: $tag $exp" >> failed_runs.txt
}
set_combined () {   # extra args (e.g. --no-run to score outputs already in runs/<tag>) are passed on
  for sp in $COMB_SPECS; do
    IFS=: read -r ph km tmin <<< "$sp"
    for e in $(allexp); do crun $e comb_${ph}_km$km $km $tmin "$@"; done
  done
}

# =============================================================================
# v1.9 fit: (1) cloud onset, (2) wall air + inlet shrinkage over the whole cloud, (3) all paper runs.
#   ./batchRuns_v2.sh v19onset   ; python3 ../analysis/paper_figs/v19_fit.py onset
#   ./batchRuns_v2.sh v19offset  ; python3 ../analysis/paper_figs/v19_fit.py offset
#   ./batchRuns_v2.sh v19final   ; bash ../analysis/paper_figs/make_v19.sh
# Settings files (written by v19_fit.py) live in runs/analysis/.
# =============================================================================
V19_DIR="${ISKYLAB_RUNS:-$(cd .. && pwd)/runs}/analysis"
# wall-air offset from the whole-cloud fit with the 21 um2 inlet (v1.8, tag final21): the start point for v1.9
read -r -d '' V19_DT0 <<'TABLE'
Exp005 -0.30
Exp006 -0.08
Exp007 -0.09
Exp008 -0.09
Exp009 -0.09
Exp010 -0.085
Exp011 -0.09
Exp013 -0.075
Exp014 -0.091
Exp015 -0.07
Exp016 -0.07
Exp017 -0.525
Exp018 -0.075
Exp019 -0.09
TABLE
V19_ONSET_SHIFTS="0 0.5 1.0 1.5 2.0 2.5"          # minutes earlier than the original cloud-formation time
V19_OFFSETS="-0.02 -0.015 -0.01 -0.005 0 0.005 0.01 0.015 0.02"   # K, relative to the v1.9 start offset
v19tag () { awk -v x="$1" 'BEGIN{ if (x<0) printf "m%03d", -x*1000; else printf "p%03d", x*1000 }'; }
set_v19onset () {
  for e in $(allexp); do
    tsat=$(echo "$RUNS" | awk -v x="$e" '$1==x{print $2}'); dt=$(echo "$V19_DT0" | awk -v x="$e" '$1==x{print $2}')
    for s in $V19_ONSET_SHIFTS; do
      t=$(awk -v a="$tsat" -v b="$s" 'BEGIN{print a-b}')
      if awk -v t="$t" 'BEGIN{exit !(t<0.25)}'; then continue; fi
      run $e v19_on_$(v19tag $s) --cloud-formation-time-min "$t" --bl-temp-offset-k "$dt"
    done
  done
}
v19get () { awk -v x="$1" -v c="$2" '$1==x{print $c}' "$3"; }   # v19get <exp> <column> <file>
set_v19offset () {   # needs runs/analysis/v19_tsat.txt (exp tsat dT0) from v19_fit.py onset
  local f="$V19_DIR/v19_tsat.txt"; [ -f "$f" ] || { echo "run v19_fit.py onset first"; exit 1; }
  for e in $(allexp); do
    t=$(v19get $e 2 "$f"); d0=$(v19get $e 3 "$f")
    for o in $V19_OFFSETS; do
      run $e v19_off_$(v19tag $o) --cloud-formation-time-min "$t" --bl-temp-offset-k $(awk -v a="$d0" -v b="$o" 'BEGIN{print a+b}')
    done
  done
}
run19 () {   # run19 <exp> <tag> <extra args...>: the v1.9 settings (cloud-formation time, wall air) + extras
  local e=$1 tag=$2; shift 2
  local f="$V19_DIR/v19_settings.txt"
  run $e $tag --cloud-formation-time-min "$(v19get $e 2 "$f")" --bl-temp-offset-k "$(v19get $e 3 "$f")" "$@"
}
set_v19final () {   # needs runs/analysis/v19_settings.txt (dD2 line + exp tsat dT) from v19_fit.py offset
  local f="$V19_DIR/v19_settings.txt"; [ -f "$f" ] || { echo "run v19_fit.py offset first"; exit 1; }
  export ISKYLAB_INLET_DD2=$(awk '$1=="dD2"{print $2}' "$f"); echo "WELAS inlet dD2 = $ISKYLAB_INLET_DD2 um2"
  for e in $(allexp); do
    run19 $e v19_final
    run19 $e v19_nobl          --set chamber_bl_mix=0
    run19 $e v19_hom           --set chamber_bl_evap_mode=1
    run19 $e v19_p0            --bl-evap-size-exp 0
    run19 $e v19_chenlamb      --set bin_scheme_flag=2
    run19 $e v19_movingcentre  --set bin_scheme_flag=1
    run19 $e v19_nofallout     --set fallout_flag=.false.
    run19 $e v19_nosce         --set sce_flag=0
    run19 $e v19_fixed         --bl-temp-offset-k -0.09 --bl-evap-size-exp 2
  done
  for e in $MIXTURES; do
    run19 $e v19_bg      --aerosol-only 1
    run19 $e v19_nacl    --aerosol-only 2
    run19 $e v19_mix_nobl --set chamber_bl_mix=0
    run19 $e v19_bg_nobl  --aerosol-only 1 --set chamber_bl_mix=0
  done
  for e in $ROBEXP; do
    dt=$(v19get $e 3 "$V19_DIR/v19_settings.txt")
    for spec in "hom|--set chamber_bl_evap_mode=1" "p0|--bl-evap-size-exp 0" "chenlamb|--set bin_scheme_flag=2" \
                "tau5|--bl-tau-s 5" "tau20|--bl-tau-s 20" \
                "dTcold|--bl-temp-offset-k $(awk -v d="$dt" 'BEGIN{print d-0.03}')" "dTwarm|--bl-temp-offset-k $(awk -v d="$dt" 'BEGIN{print d+0.03}')"; do
      v=${spec%%|*}; a=${spec#*|}
      run19 $e v19_rob_${v}_mix $a
      run19 $e v19_rob_${v}_bg --aerosol-only 1 $a
    done
  done
}

# Competition sweep (v1.9 settings): the added NaCl scaled from 3 % to 4x of the amount used, with and without wall
# mixing.  Exp006 = AS + coarse NaCl, Exp018 = FA + coarse NaCl, Exp008 = AS + fine NaCl.  Scale 0 = runs v19_bg / v19_bg_nobl.
V19_SWEEP="0.03 0.1 0.2 0.35 0.5 0.7 1 1.5 2.5 4"
set_v19sweep () {
  local f="$V19_DIR/v19_settings.txt"; [ -f "$f" ] || { echo "run v19_fit.py offset first"; exit 1; }
  export ISKYLAB_INLET_DD2=$(awk '$1=="dD2"{print $2}' "$f")
  for e in Exp006 Exp018 Exp008; do
    for s in $V19_SWEEP; do
      t=$(awk -v x="$s" 'BEGIN{printf "%04d", x*100}')
      run19 $e v19_sw_s$t  --mode2-scale $s
      run19 $e v19_sw0_s$t --mode2-scale $s --set chamber_bl_mix=0
    done
  done
}
# Parcel check (Exp006 aerosol): pure adiabatic parcel (no chamber forcing, wall loss, wall mixing or fallout) at
# 1.3 m/s (0.76 K/min) and 0.85 m/s (0.50 K/min), NaCl scaled as in v19sweep.  Compare with v19_sw / v19_sw0.
PARCEL="--set chamber_force_pressure=.false. --set chamber_force_temperature=.false. --set chamber_bl_mix=0 --set chamber_wall_loss=0 --set fallout_flag=.false."
set_v19parcel () {
  for w in 1.3 0.85; do
    wt=$(awk -v x="$w" 'BEGIN{printf "%03d", x*100}')
    run19 Exp006 v19_pc${wt}_s0000 --winit $w --aerosol-only 1 $PARCEL
    for s in $V19_SWEEP; do
      t=$(awk -v x="$s" 'BEGIN{printf "%04d", x*100}')
      run19 Exp006 v19_pc${wt}_s$t --winit $w --mode2-scale $s $PARCEL
    done
  done
}
# Fan-off counterfactual (v1.9 settings, exchange time 80 s as in Exp005) for every fan-on experiment: total water for Fig. 9.
set_v19fanoff () {
  local f="$V19_DIR/v19_settings.txt"; [ -f "$f" ] || { echo "run v19_fit.py offset first"; exit 1; }
  export ISKYLAB_INLET_DD2=$(awk '$1=="dD2"{print $2}' "$f")
  for e in $(allexp); do [ "$e" = Exp005 ] && continue; run19 $e v19_fanoff --bl-tau-s 80; done
}
# Mixed evaporation test (BMM chamber_bl_evap_mode 3): a fraction f of the evaporated liquid removed homogeneously,
# the rest smallest first.  Exp013 first; v19mixall runs all experiments at one fraction (MIXF, default 0.5).
set_v19mix () {
  local f="$V19_DIR/v19_settings.txt"; export ISKYLAB_INLET_DD2=$(awk '$1=="dD2"{print $2}' "$f")
  for fr in 0.25 0.5 0.75; do run19 Exp013 v19_mixf$(awk -v x="$fr" 'BEGIN{printf "%03d", x*100}') --set chamber_bl_evap_mode=3 --set chamber_bl_hom_fraction=$fr; done
}
set_v19mixall () {
  local f="$V19_DIR/v19_settings.txt"; export ISKYLAB_INLET_DD2=$(awk '$1=="dD2"{print $2}' "$f"); local fr=${MIXF:-0.5}
  for e in $(allexp); do run19 $e v19_mixall$(awk -v x="$fr" 'BEGIN{printf "%03d", x*100}') --set chamber_bl_evap_mode=3 --set chamber_bl_hom_fraction=$fr; done
}
case "${1:-all}" in
  baseline) set_baseline ;; nobl) set_nobl ;; comp) set_comp ;; mixtype) set_mixtype ;;
  bins) set_bins ;; fixed) set_fixed ;; rhmeas) set_rhmeas ;; process) set_process ;; robust) set_robust ;; refit) set_refit ;; refitcomp) set_refitcomp ;; wallsink) set_wallsink ;; combined) set_combined ;; combined_score) set_combined --no-run ;; v19onset) set_v19onset ;; v19offset) set_v19offset ;; v19final) set_v19final ;; v19sweep) set_v19sweep ;; v19parcel) set_v19parcel ;; v19fanoff) set_v19fanoff ;; v19mix) set_v19mix ;; v19mixall) set_v19mixall ;;
  all) set_baseline; set_nobl; set_comp; set_mixtype; set_bins; set_fixed; set_rhmeas; set_process ;;
  *) echo "unknown set $1"; exit 1 ;;
esac
wait   # parallel runs (NJOBS>1) finish before the script exits
[ -s failed_runs.txt ] && echo "WARNING: $(wc -l < failed_runs.txt) runs listed in scripts/failed_runs.txt - see scripts/logs/"

