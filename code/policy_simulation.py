"""
Model-implied policy scenarios (reviewer point 3.2). NOT causal estimates.

A calibrated model is fitted to all employees; selected inputs are then changed and the
model re-scores everyone. The output shows what the fitted model expects, which is useful
for deciding which interventions deserve a field test. Correlated inputs that a real policy
would also move (for example JobLevel when pay rises) are left unchanged, so effects can be
under- or over-stated. Results from XGBoost and logistic regression are shown side by side
to expose model dependence.

Scenarios (target group = employees in the lowest pay quintile)
  S1  remove overtime
  S2  raise monthly pay by 25% (no threshold is assumed; SHAP curves cannot identify one reliably)
  S3  grant stock option level 1 to those holding none
  S4  S1 + S2 + S3

Usage: python policy_simulation.py --data ../data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out ../results
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import SEED, encode, load

warnings.filterwarnings("ignore")


def main(data, out):
    raw, y = load(data)
    X = encode(raw)
    spw = (y == 0).sum() / (y == 1).sum()
    cfg = json.load(open(os.path.join(out, "xgb_explain_config.json")))
    lr_C = float(pd.read_csv(os.path.join(out, "crossfit_fold_params.csv")).query("model == 'LR'")["C"].median())
    models = {
        "XGB": CalibratedClassifierCV(XGBClassifier(**cfg, eval_metric="logloss", tree_method="hist",
                                                    scale_pos_weight=spw, random_state=SEED, n_jobs=-1),
                                      method="sigmoid", cv=5),
        "LR": CalibratedClassifierCV(Pipeline([("sc", StandardScaler()),
                                               ("m", LogisticRegression(C=lr_C, penalty="l2", max_iter=5000,
                                                                        class_weight="balanced"))]),
                                     method="sigmoid", cv=5)}
    q20 = raw.MonthlyIncome.quantile(0.2)
    tgt = (raw.MonthlyIncome <= q20).values

    def scen(name):
        Z = X.copy().astype(float)
        cost = 0.0
        if name in ("S1", "S4"):
            Z.loc[tgt, "OverTime_Yes"] = 0
        if name in ("S2", "S4"):
            cost += (Z.loc[tgt, "MonthlyIncome"] * 0.25 * 12).sum()
            Z.loc[tgt, "MonthlyIncome"] = Z.loc[tgt, "MonthlyIncome"] * 1.25
        if name in ("S3", "S4"):
            Z.loc[tgt & (Z.StockOptionLevel.values == 0), "StockOptionLevel"] = 1
        return Z, cost

    labels = {"S1": "Remove overtime (pay Q1)", "S2": "Raise pay by 25% (pay Q1)",
              "S3": "Grant stock option level 1 (pay Q1, none held)", "S4": "All three combined"}
    rows = []
    for mname, mdl in models.items():
        f = clone(mdl).fit(X, y)
        base = f.predict_proba(X)[:, 1]
        for sname in ["S1", "S2", "S3", "S4"]:
            Z, cost = scen(sname)
            changed = (Z.values != X.values).any(1)
            new = f.predict_proba(Z)[:, 1]
            rows.append({"model": mname, "scenario": labels[sname], "employees_changed": int(changed.sum()),
                         "mean_risk_changed_before": base[changed].mean(),
                         "mean_risk_changed_after": new[changed].mean(),
                         "expected_leavers_averted": (base - new).sum(),
                         "workforce_attrition_before": base.mean(), "workforce_attrition_after": new.mean(),
                         "annual_pay_cost_usd": cost})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "policy_scenarios.csv"), index=False)
    print(res.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    main(a.data, a.out)
