# Big-5 soccer DNB rebuild v1

Data: football-data.co.uk, 12,459 matches 2019-20 .. 2025-26 (results + shots/SoT).
FBref and Understat both block datacenter pulls; shots-based form stands in for xG and
carries real weight (shot-volume diff is the #2 coefficient behind Elo).

Target: home win | no draw (draws ~25% push). Tuned on 2024-25 only. Early-season shrink
fitted on 849 walk-forward OOS games chose a0=1.0 (identity): promoted flags + state
shrinkage already cover the cold start. 2025-26 touched once.

Frozen 2025-26 holdout (n=1,306 decided):
  base rate 0.6773 | raw Elo 0.5895 (hit 68.2%) | REBUILD v1 LL 0.5753, Brier 0.1964, hit 69.8%

Spec: K=16, HA=50, carry=0.85, promoted Elo init -200, hl 6/19, ridge C=0.05.
Files: matches.csv, eval_2526.csv, spec.json, soccer_dnb_rebuild_v1.py
