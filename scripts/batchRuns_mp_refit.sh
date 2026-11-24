#!/usr/bin/env bash
# =============================================================================
# Mixed-phase paper: refit of the wall-conditioned air offset (dT_BL) on the
# LIQUID phase, separately for each freezing mode (16 Nov 2026).
#
# Why: the `final` set fixed dT_BL from droplet number alone (021 +0.13 K,
# others 0 K).  Liquid water is then poor (Exp025 mean WELAS LWC 0.03x obs).
# Existing mp_off runs show the dust-alone runs fit number AND LWC together at
# 021 +0.2..+0.3 K and 025 +0.05..+0.1 K.  In the mixtures the liquid depends on
# the freezing mode (ice removes liquid), so each mode is fitted separately.
#
# Kept separate from batchRuns_v2.sh so that a batch running from that file is
# not disturbed.  Settings are identical to `finrun` in batchRuns_v2.sh
# (alpha_T = 0, tau = 10 s, p = 2, wall mode 0); only the offset changes.
#
# Tags: rf_m<0|1>_<p|m><mK>, e.g. rf_m1_p050 = mode 1, dT_BL = +0.050 K.
#
# Usage (from scripts/, with ISKYLAB_RUNS and BMM_MODEL_FOLDER exported as usual):
#   ./batchRuns_mp_refit.sh fit          # 44 runs: offset scan, both modes
#   python3 ../analysis/mixed_phase/refit_score.py      # picks the offsets
#   ./batchRuns_mp_refit.sh fallout      # 13 runs: ice fall depth x2 / x4 (tags ..._H2, ..._H4)
#   ./batchRuns_mp_refit.sh pair         # NaCl-removed runs at the chosen offsets
#                                          (reads ../analysis/mixed_phase/paper/refit_choice.txt)
# =============================================================================
set -u
PY="python3 dataAnalysis_new.py"
COMMON="--no-plot"
read -r -d '' MPRUNS <<'TABLE'
Exp021 12
Exp022 14
Exp023 13
Exp025 10
Exp026 12
Exp027 14
TABLE

tagof () {  # tagof <mode> <offset>
  awk -v p="${PREFIX:-rf}" -v m="$1" -v o="$2" 'BEGIN{v=o*1000; a=(v<0?-v:v); printf "%s_m%s_%s%03d", p, m, (v<0?"m":"p"), int(a+0.5)}'
}

rfrun () {  # rfrun <exp> <mode> <offset> [extra args]
  local e=$1 mode=$2 off=$3; shift 3
  local tsat tag
  tsat=$(echo "$MPRUNS" | awk -v x="$e" '$1==x{print $2}')
  tag=$(tagof "$mode" "$off")"${SUFFIX:-}"
  local wet=()
  [ "$mode" = "1" ] && wet=(--iasd-xthresh 0.01)
  echo "=== $tag  $e ==="
  $PY --experiment "$e" --cloud-formation-time-min "$tsat" --bl-tau-s 10 \
      --bl-evap-size-exp 2 --bl-wall-water-mode 0 \
      --set chamber_bl_alpha_t=0.0 --bl-temp-offset-k "$off" \
      ${wet[@]+"${wet[@]}"} --tag "$tag" $COMMON "$@" || echo "FAILED: $tag $e" >> failed_runs.txt
}

set_fit () {
  for mode in 0 1; do
    for off in 0.15 0.2 0.25 0.3 0.35; do rfrun Exp021 $mode $off; done
    for off in 0.03 0.05 0.08 0.1 0.15; do rfrun Exp025 $mode $off; done
    # mixtures: -0.03 and 0 already exist (fin_m*_dTm03, fin_m*); add warmer offsets
    for e in Exp022 Exp023 Exp026 Exp027; do
      for off in 0.05 0.1 0.2; do rfrun $e $mode $off; done
    done
  done
}

set_pair () {   # NaCl-removed runs at the offsets chosen by refit_score.py
  local f=${CHOICE:-../analysis/mixed_phase/paper/refit_choice.txt}
  [ -f "$f" ] || { echo "run refit_score.py first ($f missing)"; exit 1; }
  while read -r e mode off; do
    case "$e" in Exp022|Exp023|Exp026|Exp027) SUFFIX=dust rfrun "$e" "$mode" "$off" --aerosol-only 1 ;; esac
  done < <(grep -v '^#' "$f")
}

set_fallout () {  # slower ice fall-out: fall depth chamber_height x2 (and x4), offsets from refit_score.py
  # NB chamber_height also sets the wall area/volume for particle wall deposition
  # (about 13 % less at x2); the fall-out effect dominates for crystals.
  for H in 4.3 8.6; do
    lab=$(awk -v h="$H" 'BEGIN{printf "_H%d", h/2.15+0.5}')
    SUFFIX=$lab rfrun Exp025 1 0.05 --set chamber_height=$H
    SUFFIX=$lab rfrun Exp026 1 0.2  --set chamber_height=$H
    SUFFIX=$lab rfrun Exp027 1 0.2  --set chamber_height=$H
  done
  SUFFIX=_H2 rfrun Exp025 1 0.1  --set chamber_height=4.3
  SUFFIX=_H2 rfrun Exp025 0 0.05 --set chamber_height=4.3
  SUFFIX=_H2 rfrun Exp026 1 0.05 --set chamber_height=4.3
  SUFFIX=_H2 rfrun Exp027 1 0.05 --set chamber_height=4.3
  SUFFIX=_H2 rfrun Exp021 1 0.35 --set chamber_height=4.3
  SUFFIX=_H2 rfrun Exp022 1 -0.03 --set chamber_height=4.3
  SUFFIX=_H2 rfrun Exp023 1 0.05 --set chamber_height=4.3
}

# ---- ns(T) fitted to the dust-alone runs (BMM categories *_settable) ----------
# Tags ns_m<k>_<offset>.  Needs the BMM build of 16 Nov 2026.
set_nsfit () {
  export ISKYLAB_DUST_NS=settable PREFIX=ns
  for mode in 0 1; do
    for off in 0.2 0.25 0.3 0.35; do rfrun Exp021 $mode $off; done
    for off in 0.0 0.03 0.05 0.08 0.1; do rfrun Exp025 $mode $off; done
    for e in Exp022 Exp023 Exp026 Exp027; do
      for off in -0.03 0.0 0.05 0.1 0.2; do rfrun $e $mode $off; done
    done
  done
}
set_nsext () {  # Exp025 and Exp026 best fits were at the edge of the nsfit grid
  export ISKYLAB_DUST_NS=settable PREFIX=ns
  for mode in 0 1; do
    for off in 0.15 0.2 0.3; do rfrun Exp025 $mode $off; done
    for off in 0.3 0.4; do rfrun Exp026 $mode $off; done
  done
}
set_nspair () {
  export ISKYLAB_DUST_NS=settable PREFIX=ns CHOICE=../analysis/mixed_phase/paper/refit_choice_ns.txt
  set_pair
}
set_nsflat () {  # ns held constant colder than the fitted range, mode 1, at the chosen offsets
  export ISKYLAB_DUST_NS=settable_flat PREFIX=nsf
  local f=../analysis/mixed_phase/paper/refit_choice_ns.txt
  [ -f "$f" ] || { echo "run refit_score.py --prefix ns first"; exit 1; }
  while read -r e mode off; do
    [ "$mode" = "1" ] && rfrun "$e" 1 "$off"
  done < <(grep -v '^#' "$f")
}

# ---- Measured start + wall vapour sink (16 Nov 2026, step 1 of the redo) ------
# Start at the measured humidity (MBW dew point at t=0), aerosol normalised at t=0,
# and a finite-rate vapour exchange with the (frost-covered) wall at the measured
# wall temperature (BMM wall mode 2), transfer velocity km.  No cloud-formation
# time is imposed: cloud onset becomes a prediction.  ns fitted to the dust-alone
# runs (as before); offsets as chosen in the constrained refit (mode 1).
# Tags: wk_m1_k<km in 1e-4 m/s, two digits>, e.g. wk_m1_k03 = 3e-4 m/s.
wkrun () {  # wkrun <exp> <mode> <offset> <km> <tag> [extra]
  local e=$1 mode=$2 off=$3 km=$4 tag=$5; shift 5
  local wet=()
  [ "$mode" = "1" ] && wet=(--iasd-xthresh 0.01)
  echo "=== $tag  $e ==="
  ISKYLAB_INITIAL_RH=dewpoint ISKYLAB_AEROSOL_INIT=t0 ISKYLAB_DUST_NS=settable \
  $PY --experiment "$e" --bl-tau-s 10 --bl-evap-size-exp 2 \
      --bl-wall-water-mode 2 --wall-vapour-transfer-velocity-ms "$km" \
      --set chamber_bl_alpha_t=0.0 --bl-temp-offset-k "$off" \
      ${wet[@]+"${wet[@]}"} --tag "$tag" $COMMON "$@" || echo "FAILED: $tag $e" >> failed_runs.txt
}
read -r -d '' WKOFF <<'TABLE'
Exp021 0.35
Exp022 -0.03
Exp023 0.1
Exp025 0.2
Exp026 0.2
Exp027 0.2
TABLE
set_wallfit () {
  for e in Exp021 Exp022 Exp023 Exp025 Exp026 Exp027; do
    off=$(echo "$WKOFF" | awk -v x="$e" '$1==x{print $2}')
    for k in 01 02 03 05 08; do
      wkrun $e 1 "$off" "$(awk -v k=$k 'BEGIN{print k*1e-4}')" wk_m1_k$k
    done
  done
}

# Step 2 (16 Nov 2026): wall sink km = 3e-4 m/s (fitted to the hygrometer before
# cloud onset, all six runs), measured start, aerosol at t=0.  Offset scan for the
# dust-alone runs, both modes, within [-0.15 K, (Tw-Tg) + 0.15 K].  Tags wk3_m<k>_<offset>.
KM=3e-4
set_wkalone () {
  for mode in 0 1; do
    for off in -0.05 0.05 0.15 0.25 0.35 0.45; do
      wkrun Exp021 $mode $off $KM "$(PREFIX=wk3 tagof $mode $off)"
    done
    for off in -0.05 0.0 0.05 0.1 0.15 0.2 0.25; do
      wkrun Exp025 $mode $off $KM "$(PREFIX=wk3 tagof $mode $off)"
    done
  done
}

# Step 3-4: ns recalibrated inside the model (ns_calibrate.py -> paper/ns_coef_wk3.txt,
# one set of coefficients per freezing mode and dust, passed as --set ns_fit_*).  Tags wk3c_m<k>_<offset>.
nscoef () {  # nscoef <mode> <exp> -> --set ns_fit_* for that experiment's dust
  local d; case "$2" in Exp021|Exp022|Exp023) d=SDSA01 ;; *) d=ATD03 ;; esac
  awk -v m="$1" -v d="$d" '$1==m && $2==d{printf "--set ns_fit_b=%s --set ns_fit_a=%s --set ns_fit_tc=%s --set ns_fit_cold=%s ", $3,$4,$5,$6}' \
      ../analysis/mixed_phase/paper/ns_coef_wk3.txt
}
choice_wk3 () {  # choice_wk3 <exp> <mode> -> chosen offset from the step-2 scoring
  awk -v e="$1" -v m="$2" '$1==e && $2==m{print $3}' ../analysis/mixed_phase/paper/refit_choice_wk3_constrained.txt
}
set_wkcal () {   # dust-alone runs with the recalibrated ns, chosen offset and +/-0.05 K
  [ -f ../analysis/mixed_phase/paper/ns_coef_wk3.txt ] || { echo "run ns_calibrate.py first"; exit 1; }
  for mode in 0 1; do
    for e in Exp021 Exp025; do
      c=$(choice_wk3 $e $mode)
      for off in $(awk -v c="$c" 'BEGIN{printf "%.2f %.2f %.2f", c-0.05, c, c+0.05}'); do
        wkrun $e $mode $off $KM "$(PREFIX=wk3c tagof $mode $off)" $(nscoef $mode $e)
      done
    done
  done
}
set_wkcal2 () {  # Exp021 best fit was at the low edge of wkcal (+0.10 K): extend down
  for mode in 0 1; do
    for off in 0.0 0.05; do
      wkrun Exp021 $mode $off $KM "$(PREFIX=wk3c tagof $mode $off)" $(nscoef $mode Exp021)
    done
  done
}
set_wkmix () {   # mixtures with the recalibrated ns, offset scan, both modes
  [ -f ../analysis/mixed_phase/paper/ns_coef_wk3.txt ] || { echo "run ns_calibrate.py first"; exit 1; }
  for mode in 0 1; do
    for e in Exp022 Exp023 Exp026 Exp027; do
      for off in -0.05 0.0 0.05 0.1 0.15 0.2; do
        wkrun $e $mode $off $KM "$(PREFIX=wk3c tagof $mode $off)" $(nscoef $mode $e)
      done
    done
  done
}
set_wkpair () {  # NaCl removed, at the offsets chosen for the wk3c runs
  local f=../analysis/mixed_phase/paper/refit_choice_wk3c_constrained.txt
  [ -f "$f" ] || { echo "run refit_score.py --prefix wk3c --constrain first"; exit 1; }
  while read -r e mode off; do
    case "$e" in Exp022|Exp023|Exp026|Exp027)
      wkrun "$e" "$mode" "$off" $KM "$(PREFIX=wk3c tagof $mode $off)dust" $(nscoef $mode $e) --aerosol-only 1 ;;
    esac
  done < <(grep -v '^#' "$f")
}

case "${1:-}" in
  fit) set_fit ;;
  pair) set_pair ;;
  fallout) set_fallout ;;
  nsfit) set_nsfit ;;
  nsext) set_nsext ;;
  nspair) set_nspair ;;
  nsflat) set_nsflat ;;
  wallfit) set_wallfit ;;
  wkalone) set_wkalone ;;
  wkcal) set_wkcal ;;
  wkcal2) set_wkcal2 ;;
  wkmix) set_wkmix ;;
  wkpair) set_wkpair ;;
  *) echo "usage: $0 fit|pair|fallout|nsfit|nsext|nspair|nsflat|wallfit|wkalone|wkcal|wkcal2|wkmix|wkpair"; exit 1 ;;
esac
