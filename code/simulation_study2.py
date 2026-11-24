"""Known-truth simulation for the Study 2 (large-n, variable-level) screen.

Covariates are resampled from the real federal base sample (s2_X_base.parquet written by study2_opm.py), so all
correlations are real. Outcomes are simulated from a known model in which four variables matter (length of service,
age, occupational category and low pay) and the rest, including correlated decoys (grade, step, career-conditional
status, pay plan, supervisory status), have no effect. Each replication applies the Study 2 variable-level screen
(fixed XGBoost, shadow dummy blocks permuted jointly, top 4 of 17 variables, 70/80/90% cut-offs) and the naive
single-fit top-4 ranking.

Usage: python simulation_study2.py --x ../results/study2/s2_X_base.parquet --out ../results/study2_sim [--reps 6 --refits 20]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from common import SEED, XGB_FIXED, shap_matrix

CAT = ["occ_type", "occ_category", "bargaining", "payplan", "agency", "state"]
TRUE = ["los_years", "age_mid", "occ_category", "log_salary"]
DECOYS = ["grade_num", "step", "career_conditional", "payplan", "supervisory"]
K_VAR = 4


def variables(cols):
    num = [c for c in cols if not any(c.startswith(k + "_") for k in CAT)]
    vars_ = num + CAT
    return vars_, {v: [i for i, c in enumerate(cols) if c == v or c.startswith(v + "_")] for v in vars_}


def simulate_y(X, rng):
    los = X["los_years"].values
    age = X["age_mid"].values
    pay = X["log_salary"].values
    low = np.where(np.isfinite(pay), pay < np.nanquantile(pay, 0.2), False)
    occ = np.zeros(len(X))
    for k, b in {"CLERICAL": 0.5, "TECHNICAL": 0.35, "BLUE COLLAR": 0.3, "OTHER WHITE COLLAR": 0.2, "ADMINISTRATIVE": 0.15}.items():
        c = "occ_category_" + k
        if c in X:
            occ += b * X[c].values
    eta = 1.4 * np.exp(-los / 1.5) - 0.025 * (age - 40) + occ + 0.35 * low
    lo, hi = -10, 5
    for _ in range(60):  # intercept for 15% prevalence, about the case share in the Study 2 analysis sample
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + eta)))).mean() > 0.15:
            hi = mid
        else:
            lo = mid
    return (rng.random(len(X)) < 1 / (1 + np.exp(-(mid + eta)))).astype(int)


def var_importance(s_, vars_, vidx, offset=None):
    return np.array([np.abs(s_[:, vidx[v]].sum(1)).mean() for v in vars_])


def run_rep(Xall, r, n, refits, nsub):
    rng = np.random.default_rng(SEED + 7000 + r)
    X = Xall.iloc[rng.choice(len(Xall), n, replace=False)].reset_index(drop=True)
    y = simulate_y(X, rng)
    cols = list(X.columns)
    vars_, vidx = variables(cols)
    # naive: one fit, explain held-out 30%
    idx = rng.permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):][:20000]
    m = XGBClassifier(**XGB_FIXED, random_state=SEED + r, n_jobs=-1).fit(X.iloc[tr], y[tr])
    s0 = shap_matrix(shap.TreeExplainer(m).shap_values(X.iloc[te]))
    vi = var_importance(s0, vars_, vidx)
    naive_vars = [vars_[i] for i in np.argsort(-vi)[:K_VAR]]
    col_imp = np.abs(s0).mean(0)
    top10_cols = [cols[i] for i in np.argsort(-col_imp)[:10]]
    # screen
    beat, top = np.zeros(len(vars_)), np.zeros(len(vars_))
    for b in range(refits):
        rb = np.random.default_rng(SEED + 8000 + 100 * r + b)
        ii = rb.choice(n, nsub, replace=False)
        trb, teb = ii[: int(0.7 * nsub)], ii[int(0.7 * nsub):]

        def shv(Z):
            parts = [Z]
            for v in vars_:
                blk = Z.iloc[:, vidx[v]]
                parts.append(blk.iloc[rb.permutation(len(blk))].reset_index(drop=True).add_prefix("shadow__"))
            return pd.concat(parts, axis=1)

        Xb, Xo = shv(X.iloc[trb].reset_index(drop=True)), shv(X.iloc[teb].reset_index(drop=True))
        mb = XGBClassifier(**XGB_FIXED, random_state=SEED + 9000 + b, n_jobs=-1).fit(Xb, y[trb])
        sb = shap_matrix(shap.TreeExplainer(mb).shap_values(Xo))
        real = var_importance(sb, vars_, vidx)
        shad, o = [], len(cols)
        for v in vars_:
            k_ = len(vidx[v])
            shad.append(np.abs(sb[:, o:o + k_].sum(1)).mean())
            o += k_
        beat += real > max(shad)
        top += pd.Series(real).rank(ascending=False).values <= K_VAR
    res = {"rep": r, "prevalence": float(y.mean()), "naive_top4": naive_vars,
           "naive_top10_columns_occupation": int(sum(c.startswith("occ_category_") for c in top10_cols))}
    for c in (0.7, 0.8, 0.9):
        sel = [v for i, v in enumerate(vars_) if beat[i] / refits >= c and top[i] / refits >= c]
        res[f"selected_{int(c * 100)}"] = sel
    res["beat_share"] = dict(zip(vars_, (beat / refits).round(3).tolist()))
    res["top4_share"] = dict(zip(vars_, (top / refits).round(3).tolist()))
    return res


def summarise(reps):
    def rates(sel):
        tp = len(set(sel) & set(TRUE))
        return tp / len(TRUE), (len(sel) - tp) / len(sel) if sel else 0.0, len(set(sel) & set(DECOYS)) > 0
    out = {}
    for key in ["naive_top4", "selected_70", "selected_80", "selected_90"]:
        t = np.array([rates(r[key]) for r in reps], dtype=float)
        out[key] = {"TPR": float(t[:, 0].mean()), "FDR": float(t[:, 1].mean()), "any_decoy_share": float(t[:, 2].mean())}
    out["per_variable_selection_80"] = {v: float(np.mean([v in r["selected_80"] for r in reps]))
                                        for v in sorted({v for r in reps for v in r["beat_share"]})}
    out["naive_top10_columns_occupation_mean"] = float(np.mean([r["naive_top10_columns_occupation"] for r in reps]))
    out["n_reps"] = len(reps)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--refits", type=int, default=20)
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--nsub", type=int, default=60000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Xall = pd.read_parquet(a.x)
    ck = os.path.join(a.out, "sim2_checkpoint.jsonl")
    done = [json.loads(l) for l in open(ck)] if os.path.exists(ck) else []
    for r in range(len(done), a.reps):
        res = run_rep(Xall, r, a.n, a.refits, a.nsub)
        with open(ck, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        done.append(res)
        print(f"rep {r + 1}/{a.reps}: screen80={res['selected_80']} naive={res['naive_top4']}", flush=True)
    summ = summarise(done)
    json.dump(summ, open(os.path.join(a.out, "sim2_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))
