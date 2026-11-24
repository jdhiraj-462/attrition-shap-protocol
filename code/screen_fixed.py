"""
Screening step (Step 3) with a fixed, data-independent XGBoost configuration.

For B = 50 bootstrap samples: append a permuted "shadow" copy of every predictor, fit XGBoost
(common.XGB_FIXED, class-weighted), compute SHAP on out-of-bag employees, and record each real
feature's mean |SHAP|, its rank among real features, and whether it beats the largest shadow.
Outputs
  screen_fixed_features.csv   per feature: refit-based 95% interval of mean |SHAP|, median rank,
                              top-10 share, beats-shadow share, pass/fail at 70%, 80%, 90% cut-offs
  screen_fixed_constructs.csv same at construct level
  screen_fixed_refits.csv     raw per-refit importances (for audit)
  screen_threshold_sensitivity.csv  number and identity of passing features by cut-off
Also compares with the tuned-configuration screen from explain_crossfit.py if present.

Usage: python screen_fixed.py --data ../data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out ../results
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from common import CONSTRUCTS, SEED, XGB_FIXED, construct_map, encode, load, shap_matrix

warnings.filterwarnings("ignore")
B = 50
CUTS = (0.7, 0.8, 0.9)


def main(data, out):
    raw, y = load(data)
    X = encode(raw)
    yv = y.values
    cols = list(X.columns)
    cmap = construct_map(cols)
    groups = list(CONSTRUCTS)
    spw = (yv == 0).sum() / yv.sum()
    imp_r, rank_r, hit_r, gimp_r, grank_r, ghit_r = [], [], [], [], [], []
    for b in range(B):
        r = np.random.default_rng(SEED + 5000 + b)
        idx = r.integers(0, len(yv), len(yv))
        oob = np.setdiff1d(np.arange(len(yv)), idx)

        def with_shadows(Z):
            sh = Z.apply(lambda c: r.permutation(c.values)).add_prefix("shadow__")
            return pd.concat([Z, sh], axis=1)

        Xb = with_shadows(X.iloc[idx].reset_index(drop=True))
        Xo = with_shadows(X.iloc[oob].reset_index(drop=True))
        m = XGBClassifier(**XGB_FIXED, scale_pos_weight=spw, random_state=SEED + b, n_jobs=-1).fit(Xb, yv[idx])
        sv = shap_matrix(shap.TreeExplainer(m).shap_values(Xo))
        a = np.abs(sv).mean(0)
        real, shad = a[:len(cols)], a[len(cols):]
        imp_r.append(real)
        rank_r.append(pd.Series(real).rank(ascending=False).values)
        hit_r.append(real > shad.max())
        gr = np.column_stack([sv[:, [i for i, c in enumerate(cols) if cmap[c] == g]].sum(1) for g in groups])
        gs = np.column_stack([sv[:, [len(cols) + i for i, c in enumerate(cols) if cmap[c] == g]].sum(1) for g in groups])
        ga, gsa = np.abs(gr).mean(0), np.abs(gs).mean(0)
        gimp_r.append(ga)
        grank_r.append(pd.Series(ga).rank(ascending=False).values)
        ghit_r.append(ga > gsa.max())
        if (b + 1) % 10 == 0:
            print(f"  refit {b + 1}/{B}", flush=True)
    I, R, H = np.array(imp_r), np.array(rank_r), np.array(hit_r)
    f = pd.DataFrame({"feature": cols, "construct": [cmap[c] for c in cols],
                      "mean_abs_shap_median": np.median(I, 0), "ci_lo": np.percentile(I, 2.5, 0),
                      "ci_hi": np.percentile(I, 97.5, 0), "median_rank": np.median(R, 0),
                      "rank_q05": np.percentile(R, 5, 0), "rank_q95": np.percentile(R, 95, 0),
                      "top10_share": (R <= 10).mean(0), "beats_shadow_share": H.mean(0)})
    for c in CUTS:
        f[f"passes_{int(c * 100)}"] = (f.top10_share >= c) & (f.beats_shadow_share >= c)
    f = f.sort_values("median_rank")
    f.to_csv(os.path.join(out, "screen_fixed_features.csv"), index=False)
    pd.DataFrame(I, columns=cols).to_csv(os.path.join(out, "screen_fixed_refits.csv"), index=False)
    GI, GR, GH = np.array(gimp_r), np.array(grank_r), np.array(ghit_r)
    g = pd.DataFrame({"construct": groups, "mean_abs_shap_median": np.median(GI, 0), "ci_lo": np.percentile(GI, 2.5, 0),
                      "ci_hi": np.percentile(GI, 97.5, 0), "share_median": np.median(GI / GI.sum(1, keepdims=True), 0),
                      "share_lo": np.percentile(GI / GI.sum(1, keepdims=True), 2.5, 0),
                      "share_hi": np.percentile(GI / GI.sum(1, keepdims=True), 97.5, 0),
                      "median_rank": np.median(GR, 0), "beats_shadow_share": GH.mean(0)})
    g.to_csv(os.path.join(out, "screen_fixed_constructs.csv"), index=False)
    sens = [{"cut": c, "n_pass": int(f[f"passes_{int(c * 100)}"].sum()),
             "features": "; ".join(f[f[f"passes_{int(c * 100)}"]].feature)} for c in CUTS]
    pd.DataFrame(sens).to_csv(os.path.join(out, "screen_threshold_sensitivity.csv"), index=False)
    res = {"B": B, "config": XGB_FIXED, "threshold_sensitivity": sens}
    tuned = os.path.join(out, "crossfit_stability_shadow.csv")
    if os.path.exists(tuned):
        t = pd.read_csv(tuned).set_index("feature")
        fx = f.set_index("feature")
        res["agreement_with_tuned_config"] = {
            "same_pass_set_80": sorted(fx[fx.passes_80].index) == sorted(t[t.passes_screen].index),
            "tuned_pass": sorted(t[t.passes_screen].index), "fixed_pass": sorted(fx[fx.passes_80].index),
            "spearman_median_rank": float(pd.concat([fx.median_rank, t.median_rank], axis=1).corr("spearman").iloc[0, 1])}
    json.dump(res, open(os.path.join(out, "screen_fixed_summary.json"), "w"), indent=2, default=str)
    print(json.dumps(res, indent=1, default=str))
    print(f.head(16).round(3).to_string())
    print(g.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    main(a.data, a.out)
