"""Intersectional audit (gender x age band) on cross-fitted calibrated XGBoost scores, with
1,000-resample bootstrap intervals for flag rate and calibration gap (mean predicted - observed).
Usage: python fairness_intersectional.py --data ../data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out ../results"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import SEED


def main(data, out):
    raw = pd.read_csv(data, encoding="utf-8-sig")
    oof = pd.read_csv(os.path.join(out, "oof_predictions.csv"))
    y, p = oof.y.values, oof.XGB_cal.values
    thr = np.quantile(p, 1 - y.mean())
    flag = (p >= thr).astype(int)
    grp = (raw.Gender + ", " + np.where(raw.Age < 30, "under 30", "30 and over")).values
    rng = np.random.default_rng(SEED)
    rows = []
    for g in sorted(set(grp)):
        m = np.where(grp == g)[0]
        bs = [rng.choice(m, len(m)) for _ in range(1000)]
        fr = [flag[i].mean() for i in bs]
        cg = [p[i].mean() - y[i].mean() for i in bs]
        rows.append({"group": g, "n": len(m), "observed": y[m].mean(), "predicted": p[m].mean(),
                     "flag_rate": flag[m].mean(), "flag_lo": np.percentile(fr, 2.5), "flag_hi": np.percentile(fr, 97.5),
                     "TPR": flag[m][y[m] == 1].mean(), "calib_gap": p[m].mean() - y[m].mean(),
                     "calib_lo": np.percentile(cg, 2.5), "calib_hi": np.percentile(cg, 97.5)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "fairness_intersectional.csv"), index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    main(a.data, a.out)
