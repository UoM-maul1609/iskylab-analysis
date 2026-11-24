"""Run sets (tags under runs/) used by the paper figures and numbers.  v1.9: batchRuns_v2.sh v19final."""
FINAL, NOBL, HOM, P0 = "v19_final", "v19_nobl", "v19_hom", "v19_p0"
CHENLAMB, MOVINGCENTRE, FIXED = "v19_chenlamb", "v19_movingcentre", "v19_fixed"
NOFALLOUT, NOSCE = "v19_nofallout", "v19_nosce"
BG, NACL, MIX_NOBL, BG_NOBL = "v19_bg", "v19_nacl", "v19_mix_nobl", "v19_bg_nobl"   # competition pairs
ROB = "v19_rob_{v}_{k}"                      # k = mix or bg; v = hom p0 chenlamb tau5 tau20 dTcold dTwarm
ROB_VARIANTS = ["hom", "p0", "chenlamb", "tau5", "tau20", "dTcold", "dTwarm"]
REFIT_CSV = "inlet_refit_all_v19.csv"        # from v19_fit.py offset
