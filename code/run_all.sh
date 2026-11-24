#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper and supplement.
# Usage: bash run_all.sh [output_dir]      (default ../results)
# Runtime on 2 CPU cores: about 2 hours (nested CV ~25 min, cross-fitting ~30 min, simulation ~60 min).
set -euo pipefail
cd "$(dirname "$0")"
DATA=../data/WA_Fn-UseC_-HR-Employee-Attrition.csv
OUT=${1:-../results}
mkdir -p "$OUT/figures"
python attrition_shap_analysis.py  --data "$DATA" --out "$OUT"  # descriptives, nested CV, corrected t-tests, robustness
python explain_crossfit.py         --data "$DATA" --out "$OUT"  # cross-fitting, calibration, SHAP, tuned screen, fairness, roles
python explain_crossfit_addons.py  --data "$DATA" --out "$OUT"  # risk-based RERI, SHAP by decile
python screen_fixed.py             --data "$DATA" --out "$OUT"  # fixed-configuration screen, refit intervals, 70/80/90 cut-offs
python robustness_v3.py            --data "$DATA" --out "$OUT"  # specification count, adjusted RERI, Holm, threshold test
python fairness_intersectional.py  --data "$DATA" --out "$OUT"  # gender x age audit
python monte_carlo_cost.py         --data "$DATA" --out "$OUT"  # 10,000-draw cost analysis and variants
python policy_simulation.py        --data "$DATA" --out "$OUT"  # model-implied scenarios (not causal)
python simulation_study.py         --data "$DATA" --out "$OUT"  # known-truth simulation, four scenarios (resumable)
python make_figure1.py             --out "$OUT"
python make_figure_constructs.py   --out "$OUT"
# Study 2 (US federal workforce). Needs the OPM parquet files in ../data/opm (see README, "Study 2 data").
if [ -d ../data/opm ]; then
  python study2_opm.py --raw ../data/opm --out "$OUT/study2" --base-frac 0.10   # case-base analysis, about 10 minutes, 7 GB RAM
  python make_figure_study2.py "$OUT/study2"
  python study2_opm.py --raw ../data/opm --out "$OUT/study2_12m" --base-frac 0.10 --end 202409 --quick   # 12-month sensitivity
  python simulation_study2.py --x "$OUT/study2/s2_X_base.parquet" --out "$OUT/study2_sim" --reps 6 --refits 20   # about 90 minutes
else
  echo "Skipping Study 2: ../data/opm not found"
fi
echo "Done. Outputs in $OUT"
