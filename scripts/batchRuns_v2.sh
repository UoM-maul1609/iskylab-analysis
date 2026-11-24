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
#   mp        : mixed-phase paper runs (see the block above the case statement)
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

# -----------------------------------------------------------------------------
# Mixed-phase paper (branch mixed-phase, 14 Nov 2026)
#   mp      : mp_base  - all six dust runs (021, 022, 023, 025, 026, 027)
#             mp_dust  - dust only (--aerosol-only 1) for the NaCl mixtures
#                        022, 023, 026, 027 (paired with mp_base: competition)
# Chamber settings: tau = 10 s, p = 2, wall mode 0, and the BL temperature
# T_BL = Tgas + alpha_T*(Twall - Tgas) with alpha_T = 0.25, dT_BL = 0.
# alpha_T = tau_BL/tau_h, tau_h ~ 40 s from the dry-phase gas heat budget
# (analysis/mixed_phase/wall_heat_budget.py).  The warm paper's
# alpha_T = 0, dT_BL = -0.09 K caps RH at ~99.5 % at -20 C, so no dust activates
# (tested 14 Nov; old runs kept as mp_base_cap, mp_dust_cap).  t_sat from
# experiment_metadata.py.
#   mpsens  : named sensitivity runs for Exp021 and Exp025:
#             mp_nobl  - no wall mixing
#             mp_aT1   - alpha_T = 1 (BL at the measured wall temperature)
#             mp_wall2 - prognostic wall-water reservoir (mode 2, km = 1e-3 m/s;
#                        ice phase below 0 C): frost on the wall is a sink
#             mp_wet1  - IASD wetting mode 1, X >= 0.01
#             mp_tau20 - tau = 20 s
# Ice: Koop + INAS (iskylab_config.py), SDSA01 ns = Niemand x1, ATD03 ns =
# Hiranuma et al. (2015); FHH from Kumar et al. (2011); wetting mode 0.
# Outputs go to $ISKYLAB_RUNS/mp_*  (set ISKYLAB_RUNS to the main repo's runs/).
# -----------------------------------------------------------------------------
read -r -d '' MPRUNS <<'TABLE'
Exp021 12
Exp022 14
Exp023 13
Exp025 10
Exp026 12
Exp027 14
TABLE
MPMIX="Exp022 Exp023 Exp026 Exp027"
mprun () {   # mprun <exp> <tag> <extra args...>
  local exp=$1 tag=$2; shift 2
  local tsat; tsat=$(echo "$MPRUNS" | awk -v e="$exp" '$1==e{print $2}')
  echo "=== $tag  $exp ==="
  $PY --experiment "$exp" --cloud-formation-time-min "$tsat" --bl-tau-s 10 \
      --bl-evap-size-exp 2 --bl-wall-water-mode 0 --bl-temp-offset-k 0 \
      --set chamber_bl_alpha_t=0.25 \
      --tag "$tag" $COMMON "$@" || echo "FAILED: $tag $exp" >> failed_runs.txt
}
set_mp () {
  for e in $(echo "$MPRUNS" | awk '{print $1}'); do mprun $e mp_base; done
  for e in $MPMIX; do mprun $e mp_dust --aerosol-only 1; done
}
# One closure for both papers?  T_BL = Tgas + alpha_T*(Twall - Tgas), no offset.
# Warm runs: Twall - Tgas ~ -0.27 K, so alpha_T = 0.33 ~ the fitted dT_BL = -0.09 K.
# Heat budget gives alpha_T ~ 0.25 for the mixed-phase runs; test both values.
# Tags: alpha025, alpha033 (14 warm runs).
set_alpha () {
  for e in $(allexp); do
    run $e alpha025 --bl-temp-offset-k 0 --set chamber_bl_alpha_t=0.25
    run $e alpha033 --bl-temp-offset-k 0 --set chamber_bl_alpha_t=0.33
  done
}
# Round 3 (14 Nov): alpha_T = 0.33 (best single value for the warm runs) and
# wetting mode 1 for the NaCl mixtures, where mode 0 gives no dust activation.
set_mp3 () {
  for e in $(echo "$MPRUNS" | awk '{print $1}'); do mprun $e mp_a033 --set chamber_bl_alpha_t=0.33; done
  for e in $MPMIX; do
    mprun $e mp_wet1     --iasd-xthresh 0.01
    mprun $e mp_wet1dust --iasd-xthresh 0.01 --aerosol-only 1
  done
}
# Round 4: why does NaCl INCREASE ice with wetting mode 1?  ATD mixtures only.
#   mp_wet1nf / mp_wet1nfdust : mode 1 without fall-out (is it INP loss by sedimentation?)
#   mp_x001 / mp_x001dust     : Xthresh = 0.001
#   mp_x1   / mp_x1dust       : Xthresh = 0.1
set_mp4 () {
  for e in Exp026 Exp027; do
    mprun $e mp_wet1nf     --iasd-xthresh 0.01 --set fallout_flag=.false.
    mprun $e mp_wet1nfdust --iasd-xthresh 0.01 --set fallout_flag=.false. --aerosol-only 1
    mprun $e mp_x001       --iasd-xthresh 0.001
    mprun $e mp_x001dust   --iasd-xthresh 0.001 --aerosol-only 1
    mprun $e mp_x1         --iasd-xthresh 0.1
    mprun $e mp_x1dust     --iasd-xthresh 0.1 --aerosol-only 1
  done
}
# Warm paper check: is the simulated competition robust to the alpha_T closure?
# Background-only partners for the mixtures at alpha_T = 0.33 (mixtures are in alpha033).
set_alphacomp () {
  for e in $MIXTURES; do run $e alpha033_bg --aerosol-only 1 --bl-temp-offset-k 0 --set chamber_bl_alpha_t=0.33; done
}
# Round 5: is the ice deficit a growth/fall-out problem?  Dust alone (021, 025).
# Fall-out removes ice at Vt/H (H = 2.15 m), so crystals of ~150 um live seconds.
#   mp_nf     : no fall-out (upper bound on airborne ice)
set_mp5 () {
  for e in Exp021 Exp025; do
    mprun $e mp_nf  --set fallout_flag=.false.
  done
}
# Round 6 (14 Nov): per-run wall OFFSET for the cold runs, as in the warm paper
# (alpha_T = 0, T_BL = T_gas + dT_BL).  dT_BL <= -0.09 K stops cloud forming, so
# scan from slightly negative to positive (BL warmer than the gas: no wall vapour
# sink, droplets evaporate in the BL).  Wetting mode 0.  Tags mp_off_<m|p><mK>.
MPOFFSETS="-0.03 -0.01 0.0 0.05 0.1 0.2 0.3 0.5"
set_mpoff () {
  for e in $(echo "$MPRUNS" | awk '{print $1}'); do
    for off in $MPOFFSETS; do
      tag=mp_off_$(awk -v o="$off" 'BEGIN{v=o*1000; a=(v<0?-v:v); printf "%s%03d", (v<0?"m":"p"), int(a+0.5)}')
      mprun $e $tag --set chamber_bl_alpha_t=0.0 --bl-temp-offset-k $off
    done
  done
}
# Round 7 (14 Nov): finite-rate wall vapour sink (wall mode 2, ice phase below
# 0 C, vapour flux = km*(e_air - e_eq(Twall))/(Rv*T)), alpha_T = 0, no offset.
# km = 1e-3 m/s (mp_wall2) stopped the cloud; scan smaller km and fit to the MBW
# total water.  Tags mp_km<1e4*km>: mp_km01 = 1e-4 m/s ... mp_km10 = 1e-3 m/s.
set_mpkm () {
  for e in $(echo "$MPRUNS" | awk '{print $1}'); do
    for k in 01 02 05 10; do
      mprun $e mp_km$k --set chamber_bl_alpha_t=0.0 --bl-temp-offset-k 0 \
        --bl-wall-water-mode 2 --wall-vapour-transfer-velocity-ms $(awk -v k=$k 'BEGIN{print k*1e-4}')
    done
  done
}
# Round 8: km between 2e-4 (cloud, too much water) and 5e-4 (water right, no
# droplets), wetting mode 0 and mode 1.  Tags mp_km03, mp_km04, mp_km03w1, mp_km04w1.
set_mpkm2 () {
  for e in $(echo "$MPRUNS" | awk '{print $1}'); do
    for k in 03 04; do
      km=$(awk -v k=$k 'BEGIN{print k*1e-4}')
      W="--set chamber_bl_alpha_t=0.0 --bl-temp-offset-k 0 --bl-wall-water-mode 2 --wall-vapour-transfer-velocity-ms $km"
      mprun $e mp_km$k $W
      mprun $e mp_km${k}w1 $W --iasd-xthresh 0.01
    done
  done
}
# Does the cold-run wall (finite-rate ice wall, mode 2) also work for the warm
# runs?  alpha_T = 0, no offset, cloud-onset initial RH as in the baseline.
# Tags warm_km01, warm_km03, warm_km10 (km = 1e-4, 3e-4, 1e-3 m/s).
set_warmkm () {
  for e in $(allexp); do
    for k in 01 03 10; do
      run $e warm_km$k --bl-temp-offset-k 0 --set chamber_bl_alpha_t=0.0 \
        --bl-wall-water-mode 2 --wall-vapour-transfer-velocity-ms $(awk -v k=$k 'BEGIN{print k*1e-4}')
    done
  done
}
# Combined wall (PJC's mechanism, 14 Nov): wall mode 2 already does both
#  (i) evaporation of cloud carried into a BL at T_sens = T_gas + alpha_T*(T_wall-T_gas)
#  (ii) finite-rate vapour transfer to the wall (ice below 0 C) at km.
# A warmer wall (cold runs) evaporates cloud in the BL and the vapour goes to the
# ice wall; a colder wall (warm runs) condenses.  No offset.  All 20 runs.
# Tags combo_a<100*alpha>_k<1e4*km>: e.g. combo_a033_k03.
set_combo () {
  for a in 0.33 1.0; do
    for k in 01 03; do
      km=$(awk -v k=$k 'BEGIN{print k*1e-4}')
      at=$(awk -v a=$a 'BEGIN{printf "%03d", a*100}')
      W="--bl-temp-offset-k 0 --set chamber_bl_alpha_t=$a --bl-wall-water-mode 2 --wall-vapour-transfer-velocity-ms $km"
      for e in $(allexp); do run $e combo_a${at}_k$k $W; done
      for e in $(echo "$MPRUNS" | awk '{print $1}'); do mprun $e combo_a${at}_k$k $W; done
    done
  done
}
# =============================================================================
# FINAL mixed-phase set (choices fixed 14 Nov 2026; do not reopen):
#  wall   : warm-paper mixing closure, alpha_T = 0, wall mode 0, tau = 10 s, p = 2;
#           dT_BL = 0.35 * <Twall - Tgas> (mean over t_sat .. t_sat + 15 min) where
#           |<Twall - Tgas>| > 0.15 K (sensor uncertainty), else 0.
#           Measured: 021 +0.36, 022 +0.09, 023 +0.02, 025 +0.14, 026 +0.09,
#           027 +0.08 K  ->  021 +0.13 K, all others 0.
#  freeze : mode 0 (activation required) and mode 1 (X >= 0.01)
#  ATD n_s: Niemand ATD fit extrapolated to -31 C (BMM edit 14 Nov; recompile)
# Tags: fin_m0, fin_m1 (all six); fin_m0dust, fin_m1dust (dust only, mixtures);
#       sensitivity fin_m{0,1}_dTp03 / _dTm03 (dT_BL +/- 0.03 K), fin_x1 (X >= 0.1)
# =============================================================================
read -r -d '' FINDT <<'TABLE'
Exp021 0.13
Exp022 0.0
Exp023 0.0
Exp025 0.0
Exp026 0.0
Exp027 0.0
TABLE
finrun () {   # finrun <exp> <tag> <dT shift> <extra args...>
  local e=$1 tag=$2 sh=$3; shift 3
  local dt; dt=$(echo "$FINDT" | awk -v x="$e" -v s="$sh" '$1==x{print $2+s}')
  mprun $e $tag --set chamber_bl_alpha_t=0.0 --bl-temp-offset-k $dt "$@"
}
set_final () {
  for e in $(echo "$MPRUNS" | awk '{print $1}'); do
    finrun $e fin_m0 0
    finrun $e fin_m1 0 --iasd-xthresh 0.01
    finrun $e fin_x1 0 --iasd-xthresh 0.1
    for sh in 0.03 -0.03; do
      lab=$(awk -v s=$sh 'BEGIN{print (s<0?"m03":"p03")}')
      finrun $e fin_m0_dT$lab $sh
      finrun $e fin_m1_dT$lab $sh --iasd-xthresh 0.01
    done
  done
  for e in $MPMIX; do
    finrun $e fin_m0dust 0 --aerosol-only 1
    finrun $e fin_m1dust 0 --aerosol-only 1 --iasd-xthresh 0.01
  done
}
# Warm runs with the same predicted-offset rule as the final cold set:
# dT_BL = 0.35 * <Twall - Tgas> (t_sat .. t_sat + 15 min); all warm runs exceed
# the 0.15 K sensor threshold.  Tag warm_rule.
read -r -d '' WARMRULE <<'TABLE'
Exp005 -0.196
Exp006 -0.0875
Exp007 -0.091
Exp008 -0.0945
Exp009 -0.0945
Exp010 -0.0945
Exp011 -0.1085
Exp013 -0.105
Exp014 -0.1015
Exp015 -0.0945
Exp016 -0.098
Exp017 -0.385
Exp018 -0.112
Exp019 -0.0945
TABLE
set_warmrule () {
  for e in $(allexp); do
    dt=$(echo "$WARMRULE" | awk -v x="$e" '$1==x{print $2}')
    run $e warm_rule --bl-temp-offset-k $dt
  done
}
set_mpsens () {
  for e in Exp021 Exp025; do
    mprun $e mp_nobl  --set chamber_bl_mix=0
    mprun $e mp_aT1   --set chamber_bl_alpha_t=1.0
    mprun $e mp_wall2 --bl-wall-water-mode 2 --wall-vapour-transfer-velocity-ms 1.0e-3
    mprun $e mp_wet1  --iasd-xthresh 0.01
    mprun $e mp_tau20 --bl-tau-s 20
  done
}

case "${1:-all}" in
  baseline) set_baseline ;; nobl) set_nobl ;; comp) set_comp ;; mixtype) set_mixtype ;;
  bins) set_bins ;; fixed) set_fixed ;; rhmeas) set_rhmeas ;; process) set_process ;; robust) set_robust ;; refit) set_refit ;; refitcomp) set_refitcomp ;; mp) set_mp ;; mpsens) set_mpsens ;; alpha) set_alpha ;; mp3) set_mp3 ;; mp4) set_mp4 ;; alphacomp) set_alphacomp ;; mp5) set_mp5 ;; mpoff) set_mpoff ;; mpkm) set_mpkm ;; mpkm2) set_mpkm2 ;; warmkm) set_warmkm ;; combo) set_combo ;; final) set_final ;; warmrule) set_warmrule ;;
  all) set_baseline; set_nobl; set_comp; set_mixtype; set_bins; set_fixed; set_rhmeas; set_process ;;
  *) echo "unknown set $1"; exit 1 ;;
esac
