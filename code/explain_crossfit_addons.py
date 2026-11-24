"""
Add-on analyses run after explain_crossfit.py (fast):
  * additive-scale interaction (RERI) for overtime x lowest-pay-quintile, with bootstrap CI
  * SHAP of Age and MonthlyIncome by decile (cross-fitted)
Usage: python explain_crossfit_addons.py --data ../data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out ../results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import SEED


def reri(y, ot, low):
    r = lambda m: y[m].mean()
    r00, r10, r01, r11 = r((ot == 0) & (low == 0)), r((ot == 1) & (low == 0)), r((ot == 0) & (low == 1)), r((ot == 1) & (low == 1))
    return (r11 - r10 - r01 + r00) / r00, r11 - r10 - r01 + r00


def main(data, out):
    raw = pd.read_csv(data, encoding="utf-8-sig")
    y = (raw.Attrition == "Yes").astype(int).values
    ot = (raw.OverTime == "Yes").astype(int).values
    low = (raw.MonthlyIncome <= raw.MonthlyIncome.quantile(0.2)).astype(int).values
    est, diff = reri(y, ot, low)
    rng = np.random.default_rng(SEED)
    bs = np.array([reri(y[i], ot[i], low[i]) for i in [rng.integers(0, len(y), len(y)) for _ in range(2000)]])
    res = {"RERI": est, "RERI_lo": float(np.percentile(bs[:, 0], 2.5)), "RERI_hi": float(np.percentile(bs[:, 0], 97.5)),
           "risk_difference_interaction": diff, "rd_lo": float(np.percentile(bs[:, 1], 2.5)),
           "rd_hi": float(np.percentile(bs[:, 1], 97.5))}
    sv = pd.read_csv(os.path.join(out, "oof_shap_xgb.csv"))
    dec = {}
    for f in ["Age", "MonthlyIncome"]:
        dec[f] = pd.DataFrame({"bin": pd.qcut(raw[f], 10, duplicates="drop").astype(str), "s": sv[f]}) \
            .groupby("bin", sort=False)["s"].mean().to_dict()
    res["deciles"] = dec
    json.dump(res, open(os.path.join(out, "addons_summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    main(a.data, a.out)
