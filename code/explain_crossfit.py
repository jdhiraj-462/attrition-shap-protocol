"""
Cross-fitted prediction and explanation for all 1,470 employees (revision analyses).

Addresses reviewer points 2.1-2.8, 1.3-1.4, 3.3-3.4:
  * 5-fold cross-fitting with inner tuning -> out-of-fold (OOF) scores for every employee
  * Platt calibration inside each training fold; calibration slope/intercept; top-k metrics
  * OOF SHAP: TreeSHAP (path-dependent and interventional) for XGBoost, linear SHAP for LR
  * construct-level (grouped) SHAP for theory constructs and correlated clusters
  * shadow-feature null screen inside 50 bootstrap refits (Kursa & Rudnicki, 2010 logic)
  * interaction values plus a pre-specified parametric test of overtime x low pay
  * segmented-regression breakpoints with bootstrap intervals
  * fairness (gender, age) with bootstrap intervals and within-group calibration
  * controllability under three classification schemes

Usage: python explain_crossfit.py --data ../data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out ../results
Runtime: about 25-35 minutes on 2 cores.
"""
import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from statsmodels.stats.proportion import proportion_confint
from xgboost import XGBClassifier

from attrition_shap_analysis import model_specs
from common import (SEED, CONSTRUCTS, CTRL_SCHEMES, calib_slope_intercept, construct_map, ctrl_members,
                    encode, load, segmented_breakpoint, shap_matrix, top_k_metrics)

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300})
rng = np.random.default_rng(SEED)
STY = {"LR": "-", "RF": "--", "XGB": "-.", "LGBM": ":"}


def boot_ci(f, n, B=1000, seed=SEED):
    r = np.random.default_rng(seed)
    vals = [f(r.integers(0, n, n)) for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def main(data, out):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    raw, y = load(data)
    X = encode(raw)
    yv = y.values
    cols = list(X.columns)
    pos_weight = (y == 0).sum() / (y == 1).sum()
    specs = model_specs(pos_weight)
    S = {"n": len(y)}

    # ------------------------------------------------------------ 1. cross-fitting
    print("Cross-fitting ...", flush=True)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = np.zeros(len(y), int)
    oof_raw = {m: np.zeros(len(y)) for m in specs}
    oof_cal = {m: np.zeros(len(y)) for m in specs}
    sv_xgb = np.zeros((len(y), len(cols)))
    sv_xgb_int = np.zeros((len(y), len(cols)))
    sv_lr = np.zeros((len(y), len(cols)))
    inter = np.zeros((len(cols), len(cols)))
    inter_row_ot_inc = np.zeros(len(y))
    j_ot, j_inc = cols.index("OverTime_Yes"), cols.index("MonthlyIncome")
    fold_params = []
    for f, (tr, te) in enumerate(skf.split(X, y)):
        folds[te] = f
        Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y.iloc[tr]
        for m, (pipe, grid) in specs.items():
            rs = RandomizedSearchCV(clone(pipe), grid, n_iter=int(os.environ.get("NITER", 20)), scoring="roc_auc",
                                    cv=StratifiedKFold(3, shuffle=True, random_state=SEED + f),
                                    random_state=SEED + f, n_jobs=-1).fit(Xtr, ytr)
            best = rs.best_estimator_
            oof_raw[m][te] = best.predict_proba(Xte)[:, 1]
            cal = CalibratedClassifierCV(clone(best), method="sigmoid", cv=3).fit(Xtr, ytr)
            oof_cal[m][te] = cal.predict_proba(Xte)[:, 1]
            fold_params.append({"fold": f, "model": m, **{k.replace("m__", ""): (v.item() if hasattr(v, "item") else v)
                                                          for k, v in rs.best_params_.items()}})
            if m == "XGB":
                mdl = best.named_steps["m"]
                ex = shap.TreeExplainer(mdl)
                sv_xgb[te] = shap_matrix(ex.shap_values(Xte))
                bg = Xtr.sample(200, random_state=SEED)
                exi = shap.TreeExplainer(mdl, data=bg, feature_perturbation="interventional",
                                         model_output="raw")
                sv_xgb_int[te] = shap_matrix(exi.shap_values(Xte, check_additivity=False))
                iv = np.asarray(ex.shap_interaction_values(Xte))
                if iv.ndim == 4:
                    iv = iv[:, :, :, 1]
                inter += np.abs(iv).sum(0)
                inter_row_ot_inc[te] = iv[:, j_inc, j_ot] * 2  # symmetric split -> total pair effect
            if m == "LR":
                sc, lm = best.named_steps["sc"], best.named_steps["m"]
                Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=cols)
                Xte_s = pd.DataFrame(sc.transform(Xte), columns=cols)
                sv_lr[te] = np.asarray(shap.LinearExplainer(lm, shap.maskers.Independent(Xtr_s, 100)).shap_values(Xte_s))
        print(f"  fold {f + 1}/5", flush=True)
    pd.DataFrame(fold_params).to_csv(os.path.join(out, "crossfit_fold_params.csv"), index=False)
    pd.DataFrame({**{f"{m}_raw": v for m, v in oof_raw.items()}, **{f"{m}_cal": v for m, v in oof_cal.items()},
                  "y": yv, "fold": folds}).to_csv(os.path.join(out, "oof_predictions.csv"), index=False)
    pd.DataFrame(sv_xgb, columns=cols).to_csv(os.path.join(out, "oof_shap_xgb.csv"), index=False)
    pd.DataFrame(sv_lr, columns=cols).to_csv(os.path.join(out, "oof_shap_lr.csv"), index=False)

    # ------------------------------------------------------------ 2. OOF performance
    rows = []
    for m in specs:
        r = {"model": m, "ROC_AUC": roc_auc_score(yv, oof_cal[m]),
             "PR_AUC": average_precision_score(yv, oof_cal[m]),
             "Brier_raw": brier_score_loss(yv, oof_raw[m]), "Brier_cal": brier_score_loss(yv, oof_cal[m])}
        r["cal_slope"], r["cal_intercept"] = calib_slope_intercept(yv, oof_cal[m])
        for k in (0.1, 0.2):
            r[f"precision_top{int(k * 100)}"], r[f"recall_top{int(k * 100)}"] = top_k_metrics(yv, oof_cal[m], k)
        lo, hi = boot_ci(lambda i: roc_auc_score(yv[i], oof_cal[m][i]), len(yv))
        r["ROC_AUC_lo"], r["ROC_AUC_hi"] = lo, hi
        rows.append(r)
    perf = pd.DataFrame(rows)
    for m in ["RF", "XGB", "LGBM"]:
        d = boot_ci(lambda i: roc_auc_score(yv[i], oof_cal["LR"][i]) - roc_auc_score(yv[i], oof_cal[m][i]), len(yv))
        perf.loc[perf.model == m, "AUC_diff_LR_minus_model_lo"] = d[0]
        perf.loc[perf.model == m, "AUC_diff_LR_minus_model_hi"] = d[1]
    perf.to_csv(os.path.join(out, "crossfit_performance.csv"), index=False)
    print(perf.round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for m in specs:
        fpr, tpr, _ = roc_curve(yv, oof_cal[m])
        ax[0].plot(fpr, tpr, STY[m], color="black", label=f"{m} (AUC = {roc_auc_score(yv, oof_cal[m]):.3f})")
        fr, mp = calibration_curve(yv, oof_cal[m], n_bins=10, strategy="quantile")
        ax[1].plot(mp, fr, STY[m], marker="o", ms=3, color="black", label=m)
    for a, t, xl, yl in [(ax[0], "(a) ROC curves", "False positive rate", "True positive rate"),
                         (ax[1], "(b) Calibration after Platt scaling", "Mean predicted probability",
                          "Observed attrition rate")]:
        a.plot([0, 1], [0, 1], color="grey", lw=0.6)
        a.set(title=t, xlabel=xl, ylabel=yl)
        a.legend(frameon=False, fontsize=8)
    ax[1].set(xlim=(0, 0.8), ylim=(0, 0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "Fig_ROC_calibration_crossfit.png"))
    plt.close()

    # ------------------------------------------------------------ 3. global and construct importance
    cmap = construct_map(cols)
    groups = list(CONSTRUCTS.keys())

    def grouped(sv):
        return np.column_stack([sv[:, [i for i, c in enumerate(cols) if cmap[c] == g]].sum(1) for g in groups])

    imp = pd.DataFrame({"feature": cols, "construct": [cmap[c] for c in cols],
                        "XGB": np.abs(sv_xgb).mean(0), "XGB_interventional": np.abs(sv_xgb_int).mean(0),
                        "LR": np.abs(sv_lr).mean(0)})
    ci = np.array([np.percentile([np.abs(sv_xgb[i]).mean(0) for i in
                                  [rng.integers(0, len(yv), len(yv)) for _ in range(1000)]], q, axis=0)
                   for q in (2.5, 97.5)])
    imp["XGB_lo"], imp["XGB_hi"] = ci[0], ci[1]
    for c in ["XGB", "XGB_interventional", "LR"]:
        imp[f"rank_{c}"] = imp[c].rank(ascending=False).astype(int)
    imp = imp.sort_values("XGB", ascending=False)
    imp.to_csv(os.path.join(out, "crossfit_shap_importance.csv"), index=False)
    S["rank_agreement"] = {
        "spearman_xgb_lr": float(stats.spearmanr(imp.XGB, imp.LR).statistic),
        "spearman_path_vs_interventional": float(stats.spearmanr(imp.XGB, imp.XGB_interventional).statistic),
        "top10_overlap_xgb_lr": len(set(imp.nlargest(10, "XGB").feature) & set(imp.nlargest(10, "LR").feature)),
        "top10_overlap_path_int": len(set(imp.nlargest(10, "XGB").feature) &
                                      set(imp.nlargest(10, "XGB_interventional").feature))}

    gx, gl, gi = grouped(sv_xgb), grouped(sv_lr), grouped(sv_xgb_int)
    gimp = pd.DataFrame({"construct": groups, "n_features": [sum(cmap[c] == g for c in cols) for g in groups],
                         "XGB": np.abs(gx).mean(0), "XGB_interventional": np.abs(gi).mean(0), "LR": np.abs(gl).mean(0)})
    gci = np.array([np.percentile([np.abs(gx[i]).mean(0) for i in
                                   [rng.integers(0, len(yv), len(yv)) for _ in range(1000)]], q, axis=0)
                    for q in (2.5, 97.5)])
    gimp["XGB_lo"], gimp["XGB_hi"] = gci[0], gci[1]
    for c in ["XGB", "XGB_interventional", "LR"]:
        gimp[f"share_{c}"] = gimp[c] / gimp[c].sum()

    # ------------------------------------------------------------ 4. explanation config + shadow-feature bootstrap
    print("Tuning explanation config and running shadow bootstrap ...", flush=True)
    xgb_pipe, xgb_grid = specs["XGB"]
    rs = RandomizedSearchCV(clone(xgb_pipe), xgb_grid, n_iter=30, scoring="roc_auc",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), random_state=SEED,
                            n_jobs=-1).fit(X, y)
    cfg = {k.replace("m__", ""): (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
    json.dump(cfg, open(os.path.join(out, "xgb_explain_config.json"), "w"), indent=2)
    S["xgb_explain_config"] = cfg

    B = int(os.environ.get("BOOT", 50))
    ranks, hits, ghits, granks = [], [], [], []
    for b in range(B):
        r = np.random.default_rng(SEED + 1000 + b)
        idx = r.integers(0, len(yv), len(yv))
        oob = np.setdiff1d(np.arange(len(yv)), idx)

        def with_shadows(Z, rr):
            sh = Z.apply(lambda c: rr.permutation(c.values)).add_prefix("shadow__")
            sh.index = Z.index
            return pd.concat([Z, sh], axis=1)

        Xb = with_shadows(X.iloc[idx].reset_index(drop=True), r)
        Xo = with_shadows(X.iloc[oob].reset_index(drop=True), r)
        mdl = XGBClassifier(**cfg, eval_metric="logloss", tree_method="hist", scale_pos_weight=pos_weight,
                            random_state=SEED + b, n_jobs=-1).fit(Xb, yv[idx])
        sv = shap_matrix(shap.TreeExplainer(mdl).shap_values(Xo))
        a = np.abs(sv).mean(0)
        real, shad = a[:len(cols)], a[len(cols):]
        hits.append(real > shad.max())
        ranks.append(pd.Series(real, index=cols).rank(ascending=False).values)
        gr = np.column_stack([sv[:, [i for i, c in enumerate(cols) if cmap[c] == g]].sum(1) for g in groups])
        gs = np.column_stack([sv[:, [len(cols) + i for i, c in enumerate(cols) if cmap[c] == g]].sum(1) for g in groups])
        ga, gsa = np.abs(gr).mean(0), np.abs(gs).mean(0)
        ghits.append(ga > gsa.max())
        granks.append(pd.Series(ga).rank(ascending=False).values)
    R, H = np.array(ranks), np.array(hits)
    stab = pd.DataFrame({"feature": cols, "median_rank": np.median(R, 0), "q05": np.percentile(R, 5, 0),
                         "q95": np.percentile(R, 95, 0), "top10_share": (R <= 10).mean(0),
                         "beats_max_shadow_share": H.mean(0)}).sort_values("median_rank")
    stab["passes_screen"] = (stab.beats_max_shadow_share >= 0.8) & (stab.top10_share >= 0.8)
    stab.to_csv(os.path.join(out, "crossfit_stability_shadow.csv"), index=False)
    gimp["beats_max_shadow_share"] = np.array(ghits).mean(0)
    gimp["median_rank_boot"] = np.median(np.array(granks), 0)
    gimp.to_csv(os.path.join(out, "crossfit_construct_importance.csv"), index=False)
    S["shadow_B"] = B
    S["passing_features"] = stab[stab.passes_screen].feature.tolist()
    print(gimp.round(3).to_string())
    print(stab.head(15).round(2).to_string())

    # ------------------------------------------------------------ 5. interactions and parametric test
    inter /= len(yv)
    np.fill_diagonal(inter, 0)
    I = pd.DataFrame(inter, index=cols, columns=cols)
    pairs = I.where(np.triu(np.ones(I.shape), 1).astype(bool)).stack().sort_values(ascending=False).head(10) * 2
    pairs.rename("mean_abs_pair_interaction").to_csv(os.path.join(out, "crossfit_top_interactions.csv"))
    S["top_interactions"] = {f"{a} x {b}": float(v) for (a, b), v in pairs.items()}

    q20 = float(raw.MonthlyIncome.quantile(0.2))
    low = (raw.MonthlyIncome <= q20).astype(int)
    ot = (raw.OverTime == "Yes").astype(int)
    ctrl = ["Age", "TotalWorkingYears", "NumCompaniesWorked", "DistanceFromHome", "EnvironmentSatisfaction",
            "JobSatisfaction", "StockOptionLevel", "YearsWithCurrManager", "JobLevel"]
    D = pd.DataFrame({"OverTime": ot, "LowPayQ1": low, "OT_x_LowPay": ot * low,
                      "TravelFrequent": (raw.BusinessTravel == "Travel_Frequently").astype(int)})
    D = pd.concat([D, (raw[ctrl] - raw[ctrl].mean()) / raw[ctrl].std()], axis=1)
    glm = sm.Logit(yv, sm.add_constant(D)).fit(disp=0, cov_type="HC1")
    tab = pd.DataFrame({"coef": glm.params, "OR": np.exp(glm.params), "se": glm.bse, "p": glm.pvalues,
                        "OR_lo": np.exp(glm.conf_int()[0]), "OR_hi": np.exp(glm.conf_int()[1])})
    tab.to_csv(os.path.join(out, "interaction_logit.csv"))
    cells = []
    for o in (0, 1):
        for l in (0, 1):
            m_ = (ot == o) & (low == l)
            k, n = int(y[m_].sum()), int(m_.sum())
            lo_, hi_ = proportion_confint(k, n, method="wilson")
            cells.append({"OverTime": o, "LowPayQ1": l, "n": n, "leavers": k, "rate": k / n, "lo": lo_, "hi": hi_})
    cells = pd.DataFrame(cells)
    cells.to_csv(os.path.join(out, "interaction_cells.csv"), index=False)
    # relative excess risk due to interaction (additive scale), from the fitted logit
    S["interaction_test"] = {"low_pay_cutoff_Q1": q20, "OR_interaction": float(tab.loc["OT_x_LowPay", "OR"]),
                             "OR_lo": float(tab.loc["OT_x_LowPay", "OR_lo"]),
                             "OR_hi": float(tab.loc["OT_x_LowPay", "OR_hi"]),
                             "p": float(tab.loc["OT_x_LowPay", "p"]),
                             "OR_overtime": float(tab.loc["OverTime", "OR"]),
                             "OR_lowpay": float(tab.loc["LowPayQ1", "OR"])}

    # ------------------------------------------------------------ 6. breakpoints
    inc, age = raw.MonthlyIncome.values.astype(float), raw.Age.values.astype(float)
    s_inc, s_age = sv_xgb[:, j_inc], sv_xgb[:, cols.index("Age")]
    g_inc, g_age = np.arange(1500, 8001, 50), np.arange(22, 50.5, 0.5)
    bp_inc, bp_age = segmented_breakpoint(inc, s_inc, g_inc), segmented_breakpoint(age, s_age, g_age)
    bi = [segmented_breakpoint(inc[i], s_inc[i], g_inc) for i in [rng.integers(0, len(yv), len(yv)) for _ in range(500)]]
    ba = [segmented_breakpoint(age[i], s_age[i], g_age) for i in [rng.integers(0, len(yv), len(yv)) for _ in range(500)]]
    S["breakpoints"] = {"income": float(bp_inc), "income_ci": list(np.percentile(bi, [2.5, 97.5])),
                        "age": float(bp_age), "age_ci": list(np.percentile(ba, [2.5, 97.5])),
                        "share_below_income_bp": float((inc < bp_inc).mean())}
    dec = pd.DataFrame({"decile": pd.qcut(inc, 10), "shap": s_inc}).groupby("decile", observed=True)["shap"].mean()
    dec.to_csv(os.path.join(out, "crossfit_income_shap_by_decile.csv"))

    # ------------------------------------------------------------ 7. fairness on calibrated XGB OOF
    p = oof_cal["XGB"]
    thr = np.quantile(p, 1 - yv.mean())
    flag = (p >= thr).astype(int)
    grpdefs = {"Gender": (raw.Gender == "Female").values, "Age<30": (raw.Age < 30).values,
               "Age>=40": (raw.Age >= 40).values}
    fr = []
    for gname, gm in grpdefs.items():
        for lab, m_ in [(gname if gname != "Gender" else "Female", gm),
                        ({"Gender": "Male", "Age<30": "Age>=30", "Age>=40": "Age<40"}[gname], ~gm)]:
            fr.append({"group": lab, "n": int(m_.sum()), "observed_rate": yv[m_].mean(),
                       "mean_predicted": p[m_].mean(), "flag_rate": flag[m_].mean(),
                       "TPR": flag[m_][yv[m_] == 1].mean(), "FPR": flag[m_][yv[m_] == 0].mean(),
                       "AUC": roc_auc_score(yv[m_], p[m_])})
        a_m, b_m = gm, ~gm

        def gap(i, key):
            ia, ib = i[a_m[i]], i[b_m[i]]
            if key == "flag":
                return flag[ia].mean() - flag[ib].mean()
            if key == "TPR":
                return flag[ia][yv[ia] == 1].mean() - flag[ib][yv[ib] == 1].mean()
            if key == "calib":
                return (p[ia].mean() - yv[ia].mean()) - (p[ib].mean() - yv[ib].mean())
        for key in ["flag", "TPR", "calib"]:
            est = gap(np.arange(len(yv)), key)
            lo, hi = boot_ci(lambda i: gap(i, key), len(yv))
            fr.append({"group": f"GAP {gname} {key}", "estimate": est, "lo": lo, "hi": hi})
    fair = pd.DataFrame(fr)
    fair.to_csv(os.path.join(out, "crossfit_fairness.csv"), index=False)
    S["fairness_threshold"] = float(thr)

    # ------------------------------------------------------------ 8. roles, controllability, local cases
    role = raw.JobRole.values
    rt = pd.DataFrame({"role": role, "p": p, "y": yv}).groupby("role").agg(n=("y", "size"), observed=("y", "mean"),
                                                                          predicted=("p", "mean"))
    seg = pd.DataFrame(sv_xgb, columns=cols)
    seg["role"] = role
    m_ = seg.groupby("role").mean()
    nonrole = [c for c in cols if not c.startswith("JobRole_")]
    rt["largest_positive_nonrole_driver"] = m_[nonrole].idxmax(1)
    rt["driver_mean_shap"] = m_[nonrole].max(1)
    rt.sort_values("predicted", ascending=False).to_csv(os.path.join(out, "crossfit_roles.csv"))

    absm = pd.Series(np.abs(sv_xgb).mean(0), index=cols)
    absl = pd.Series(np.abs(sv_lr).mean(0), index=cols)
    ctrl_rows = []
    for sch in CTRL_SCHEMES:
        mem = ctrl_members(sch, cols)
        ctrl_rows.append({"scheme": sch, "n_features": len(mem), "share_XGB": absm[mem].sum() / absm.sum(),
                          "share_LR": absl[mem].sum() / absl.sum()})
    pd.DataFrame(ctrl_rows).to_csv(os.path.join(out, "crossfit_controllability.csv"), index=False)
    S["controllability"] = ctrl_rows

    lv = np.where(yv == 1)[0]
    st = np.where(yv == 0)[0]
    cases = {"A": lv[np.argmax(p[lv])], "B": lv[np.argmin(p[lv])], "C": st[np.argmax(p[st])]}
    local = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    titles = {"A": "highest-risk leaver", "B": "lowest-risk leaver (missed)", "C": "highest-risk stayer"}
    for axx, (lab, i) in zip(axes, cases.items()):
        c = pd.Series(sv_xgb[i], index=cols)
        top = c.reindex(c.abs().sort_values(ascending=False).index).head(8)
        local[lab] = {"row": int(i), "prob": float(p[i]), "left": int(yv[i]),
                      "contrib": {k: [float(X.iloc[i][k]), float(v)] for k, v in top.items()}}
        t = top.iloc[::-1]
        axx.barh(range(len(t)), t.values, color=["black" if v > 0 else "#a8a8a8" for v in t.values])
        axx.set_yticks(range(len(t)))
        axx.set_yticklabels([f"{k} = {X.iloc[i][k]:g}" for k in t.index], fontsize=8)
        axx.axvline(0, color="grey", lw=0.6)
        axx.set_title(f"Case {lab}, {titles[lab]}: p = {p[i]:.2f}", fontsize=9)
        axx.set_xlabel("SHAP contribution (log-odds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "Fig_local_explanations_crossfit.png"))
    plt.close()
    S["local_cases"] = local

    # ------------------------------------------------------------ 9. figures: beeswarm, constructs, dependence, interaction
    plt.figure()
    shap.summary_plot(sv_xgb, X, max_display=15, show=False, plot_size=(8, 6), cmap="coolwarm")
    plt.title("Cross-fitted SHAP values, XGBoost (all employees, n = 1,470)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "Fig_beeswarm_crossfit.png"), bbox_inches="tight")
    plt.close()

    gplot = gimp.sort_values("XGB")
    fig, ax = plt.subplots(figsize=(8, 4))
    yy = np.arange(len(gplot))
    ax.barh(yy + 0.2, gplot.share_XGB, 0.4, color="#3a3a3a", label="XGBoost")
    ax.barh(yy - 0.2, gplot.share_LR, 0.4, color="#b0b0b0", label="Logistic regression")
    ax.errorbar(gplot.share_XGB, yy + 0.2,
                xerr=[(gplot.XGB - gplot.XGB_lo) / gplot.XGB.sum(), (gplot.XGB_hi - gplot.XGB) / gplot.XGB.sum()],
                fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{g} ({k})" for g, k in zip(gplot.construct, gplot.n_features)])
    ax.set_xlabel("Share of total construct-level mean |SHAP|")
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "Fig_construct_importance.png"))
    plt.close()

    dep = ["OverTime_Yes", "MonthlyIncome", "StockOptionLevel", "Age", "NumCompaniesWorked",
           "DistanceFromHome", "EnvironmentSatisfaction", "YearsWithCurrManager"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for axx, fe in zip(axes.ravel(), dep):
        j = cols.index(fe)
        xv = X[fe].values
        jit = rng.uniform(-0.15, 0.15, len(xv)) if X[fe].nunique() <= 5 else 0
        axx.scatter(xv + jit, sv_xgb[:, j], s=5, alpha=0.4, color="black")
        axx.axhline(0, color="grey", lw=0.6)
        if fe == "MonthlyIncome":
            axx.axvline(bp_inc, color="grey", ls="--", lw=0.8)
        if fe == "Age":
            axx.axvline(bp_age, color="grey", ls="--", lw=0.8)
        axx.set_title(fe, fontsize=9)
        axx.set_ylabel("SHAP (log-odds)", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "Fig_dependence_crossfit.png"))
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for o, mk, col, lab in [(1, "o", "black", "Overtime = Yes"), (0, "^", "#9a9a9a", "Overtime = No")]:
        msk = ot.values == o
        ax[0].scatter(inc[msk], inter_row_ot_inc[msk], s=8, marker=mk, color=col, alpha=0.6, label=lab)
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set(xlabel="Monthly income (USD)", ylabel="SHAP interaction value, income x overtime",
              title="(a) Model: interaction contribution")
    ax[0].legend(frameon=False, fontsize=8)
    xs = np.arange(2)
    for l, off, col, lab in [(0, -0.18, "#b0b0b0", f"Income above Q1 (> USD {q20:,.0f})"),
                             (1, 0.18, "#3a3a3a", f"Income in Q1 (<= USD {q20:,.0f})")]:
        cc = cells[cells.LowPayQ1 == l].sort_values("OverTime")
        ax[1].bar(xs + off, cc.rate * 100, 0.36, color=col, label=lab,
                  yerr=[(cc.rate - cc.lo) * 100, (cc.hi - cc.rate) * 100], capsize=3)
        for x_, h_, n_ in zip(xs + off, cc.hi, cc.n):
            ax[1].text(x_, h_ * 100 + 1.5, f"n = {n_}", ha="center", fontsize=7)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(["No overtime", "Overtime"])
    ax[1].set(ylabel="Observed attrition (%), 95% Wilson CI", title="(b) Data: attrition by overtime and pay")
    ax[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "Fig_interaction_crossfit.png"))
    plt.close()

    json.dump(S, open(os.path.join(out, "summary_crossfit.json"), "w"), indent=2, default=str)
    print(json.dumps({k: S[k] for k in ["rank_agreement", "passing_features", "interaction_test", "breakpoints",
                                        "controllability"]}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    main(a.data, a.out)
