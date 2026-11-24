"""
Version-3 robustness analyses (fast, < 5 minutes):
  1. Specification count for the screen: 2 configurations (tuned in explain_crossfit.py; fixed in
     screen_fixed.py) x 3 cut-offs (70%, 80%, 90%) -> number of specifications (0-6) each feature passes.
  2. Adjusted RERI for overtime x lowest-pay-quintile from the covariate-adjusted logistic model
     (RERI_OR = OR11 - OR10 - OR01 + 1; VanderWeele & Knol, 2014), 1,000 bootstrap refits.
  3. Holm adjustment of the three corrected resampled t-tests (LR vs each ensemble, ROC-AUC).
  4. Threshold test: smooth cubic baseline (log scale) vs baseline + one step (BIC) for pay and age SHAP, bootstrap CI.
     Reported for completeness only: the simulation shows this test flags steps under smooth effects.
Usage: python robustness_v3.py --data ../data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out ../results
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from common import SEED, linstep_fit

warnings.filterwarnings("ignore")
CTRL = ["Age", "TotalWorkingYears", "NumCompaniesWorked", "DistanceFromHome", "EnvironmentSatisfaction",
        "JobSatisfaction", "StockOptionLevel", "YearsWithCurrManager", "JobLevel"]


def design(raw, idx=None):
    r = raw if idx is None else raw.iloc[idx].reset_index(drop=True)
    q20 = raw.MonthlyIncome.quantile(0.2)
    ot = (r.OverTime == "Yes").astype(int)
    low = (r.MonthlyIncome <= q20).astype(int)
    D = pd.DataFrame({"OverTime": ot, "LowPayQ1": low, "OT_x_LowPay": ot * low,
                      "TravelFrequent": (r.BusinessTravel == "Travel_Frequently").astype(int)})
    D = pd.concat([D, (r[CTRL] - raw[CTRL].mean()) / raw[CTRL].std()], axis=1)
    return sm.add_constant(D), (r.Attrition == "Yes").astype(int).values


def reri_adj(raw, idx=None):
    D, y = design(raw, idx)
    p = sm.Logit(y, D).fit(disp=0).params
    return float(np.exp(p.OverTime + p.LowPayQ1 + p.OT_x_LowPay) - np.exp(p.OverTime) - np.exp(p.LowPayQ1) + 1)


def main(data, out):
    raw = pd.read_csv(data, encoding="utf-8-sig")
    res = {}
    # 1 specification count
    t = pd.read_csv(os.path.join(out, "crossfit_stability_shadow.csv")).set_index("feature")
    f = pd.read_csv(os.path.join(out, "screen_fixed_features.csv")).set_index("feature")
    spec = pd.DataFrame(index=f.index)
    for name, df, top, hit in [("tuned", t, "top10_share", "beats_max_shadow_share"),
                               ("fixed", f, "top10_share", "beats_shadow_share")]:
        for c in (0.7, 0.8, 0.9):
            spec[f"{name}_{int(c * 100)}"] = (df.loc[spec.index, top] >= c) & (df.loc[spec.index, hit] >= c)
    spec["n_specs_passed"] = spec.sum(1)
    spec["fixed_median_rank"] = f.median_rank
    spec = spec.sort_values(["n_specs_passed", "fixed_median_rank"], ascending=[False, True])
    spec.to_csv(os.path.join(out, "screen_specification_count.csv"))
    res["spec_count"] = spec.n_specs_passed[spec.n_specs_passed > 0].to_dict()
    # 2 adjusted RERI
    est = reri_adj(raw)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(1000):
        try:
            bs.append(reri_adj(raw, rng.integers(0, len(raw), len(raw))))
        except Exception:
            pass
    res["RERI_adjusted"] = {"estimate": est, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                            "n_boot": len(bs), "share_boot_positive": float((np.array(bs) > 0).mean())}
    # 3 Holm
    tt = pd.read_csv(os.path.join(out, "corrected_ttests.csv"))
    a = tt[tt.metric == "ROC_AUC"].copy()
    a["p_holm"] = multipletests(a.p, method="holm")[1]
    a.to_csv(os.path.join(out, "corrected_ttests_holm.csv"), index=False)
    res["holm"] = dict(zip(a.comparison, a.p_holm.round(3)))
    # 4 threshold test against a smooth baseline (see common.linstep_fit), with bootstrap CI for the break
    sv = pd.read_csv(os.path.join(out, "oof_shap_xgb.csv"))
    for feat, grid in [("MonthlyIncome", np.arange(1500, 8001, 50)), ("Age", np.arange(22, 50.5, 0.5))]:
        x, s_ = raw[feat].values.astype(float), sv[feat].values
        c, d, dbic = linstep_fit(x, s_, grid)
        bs = [linstep_fit(x[i], s_[i], grid)[0] for i in [rng.integers(0, len(x), len(x)) for _ in range(300)]]
        res[f"threshold_{feat}"] = {"break": c, "step_size_logodds": d, "delta_bic_vs_smooth_baseline": dbic,
                                    "break_lo": float(np.percentile(bs, 2.5)), "break_hi": float(np.percentile(bs, 97.5)),
                                    "share_below": float((x < c).mean())}
    json.dump(res, open(os.path.join(out, "robustness_v3.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    main(a.data, a.out)
