"""
Explainable Machine Learning for Employee Attrition Prediction:
Managerial Insights Using SHAP  -- full replication script.

Usage:
    python attrition_shap_analysis.py --data ../data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out ../results

All random operations are seeded (SEED = 42). Running time on a 4-core
laptop is roughly 20-30 minutes on 2 cores, dominated by nested cross-validation.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "figure.dpi": 100})

# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
DROP = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "Gender",
               "JobRole", "MaritalStatus", "OverTime"]
PROTECTED = ["Gender", "MaritalStatus"]  # removed in the main specification


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.drop(columns=DROP)
    y = (df.pop("Attrition") == "Yes").astype(int)
    return df, y


def encode(df, drop_protected=True):
    """One-hot encode categoricals. Binary OverTime becomes OverTime_Yes."""
    X = df.copy()
    if drop_protected:
        X = X.drop(columns=PROTECTED)
    cats = [c for c in CATEGORICAL if c in X.columns]
    X = pd.get_dummies(X, columns=cats, drop_first=False, dtype=float)
    # binary variables: keep a single indicator
    for b in ["OverTime_No", "Gender_Female"]:
        if b in X.columns:
            X = X.drop(columns=b)
    return X


# --------------------------------------------------------------------------
# 2. Models and search spaces
# --------------------------------------------------------------------------
def model_specs(pos_weight):
    return {
        "LR": (Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=5000, class_weight="balanced",
                                                  solver="liblinear", random_state=SEED))]),
               {"m__C": np.logspace(-3, 1, 30), "m__penalty": ["l1", "l2"]}),
        "RF": (Pipeline([("m", RandomForestClassifier(class_weight="balanced_subsample",
                                                      random_state=SEED, n_jobs=1))]),
               {"m__n_estimators": [200, 400], "m__max_depth": [4, 6, 8, 12, None],
                "m__min_samples_leaf": [1, 3, 5, 10], "m__max_features": ["sqrt", 0.3, 0.5]}),
        "XGB": (Pipeline([("m", XGBClassifier(eval_metric="logloss", random_state=SEED,
                                              n_jobs=1, scale_pos_weight=pos_weight,
                                              tree_method="hist"))]),
                {"m__n_estimators": [200, 400, 600], "m__max_depth": [2, 3, 4, 5],
                 "m__learning_rate": [0.01, 0.03, 0.05, 0.1], "m__subsample": [0.7, 0.85, 1.0],
                 "m__colsample_bytree": [0.5, 0.7, 0.9], "m__min_child_weight": [1, 3, 5],
                 "m__reg_lambda": [1, 5, 10]}),
        "LGBM": (Pipeline([("m", LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1,
                                                class_weight="balanced"))]),
                 {"m__n_estimators": [200, 400, 600], "m__num_leaves": [4, 8, 15, 31],
                  "m__learning_rate": [0.01, 0.03, 0.05, 0.1], "m__subsample": [0.7, 0.85, 1.0],
                  "m__subsample_freq": [1], "m__colsample_bytree": [0.5, 0.7, 0.9],
                  "m__min_child_samples": [10, 20, 40], "m__reg_lambda": [0, 1, 5]}),
    }


def metrics_at(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "Recall": recall_score(y, yhat),
            "Precision": precision_score(y, yhat, zero_division=0),
            "F1": f1_score(y, yhat)}


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return float(t), float(p)


# --------------------------------------------------------------------------
# 3. Nested cross-validation
# --------------------------------------------------------------------------
def nested_cv(X, y, specs, n_repeats=3, n_iter=15):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    rows, fold_preds = [], {}
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        for name, (pipe, grid) in specs.items():
            inner = StratifiedKFold(3, shuffle=True, random_state=SEED + fold)
            rs = RandomizedSearchCV(clone(pipe), grid, n_iter=n_iter, scoring="roc_auc",
                                    cv=inner, random_state=SEED + fold, n_jobs=-1)
            rs.fit(Xtr, ytr)
            p = rs.predict_proba(Xte)[:, 1]
            r = metrics_at(yte, p)
            r.update({"model": name, "fold": fold})
            rows.append(r)
        print(f"  outer fold {fold + 1}/{5 * n_repeats} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main(data, out):
    os.makedirs(out, exist_ok=True)
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    raw, y = load(data)
    X = encode(raw, drop_protected=True)
    X_full = encode(raw, drop_protected=False)
    pos_weight = (y == 0).sum() / (y == 1).sum()
    summary = {"n": int(len(y)), "n_attrition": int(y.sum()),
               "attrition_rate": float(y.mean()), "n_features_main": X.shape[1],
               "n_features_full": X_full.shape[1], "scale_pos_weight": float(pos_weight)}

    # descriptive table
    desc = raw.copy()
    desc["Attrition"] = y
    num = desc.select_dtypes("number").columns.drop("Attrition")
    dtab = []
    for c in num:
        a, b = desc.loc[y == 1, c], desc.loc[y == 0, c]
        u = stats.mannwhitneyu(a, b)
        dtab.append({"variable": c, "mean_leavers": a.mean(), "sd_leavers": a.std(),
                     "mean_stayers": b.mean(), "sd_stayers": b.std(), "MWU_p": u.pvalue})
    pd.DataFrame(dtab).to_csv(os.path.join(out, "descriptives_numeric.csv"), index=False)
    ctab = []
    for c in CATEGORICAL:
        ct = pd.crosstab(desc[c], desc["Attrition"])
        chi = stats.chi2_contingency(ct)
        for lvl in ct.index:
            ctab.append({"variable": c, "level": lvl, "n": int(ct.loc[lvl].sum()),
                         "attrition_rate": ct.loc[lvl, 1] / ct.loc[lvl].sum(),
                         "chi2_p": chi.pvalue})
    pd.DataFrame(ctab).to_csv(os.path.join(out, "descriptives_categorical.csv"), index=False)

    # ---- 4a nested CV (main specification)
    specs = model_specs(pos_weight)
    print("Nested CV, main specification ...")
    cv = nested_cv(X, y, specs)
    cv.to_csv(os.path.join(out, "nested_cv_folds.csv"), index=False)
    agg = cv.groupby("model").agg(["mean", "std"]).drop(columns="fold")
    agg.to_csv(os.path.join(out, "nested_cv_summary.csv"))
    print(agg.round(3))

    n_tr, n_te = int(len(y) * 0.8), int(len(y) * 0.2)
    tests = []
    best = cv.groupby("model")["ROC_AUC"].mean().idxmax()
    for m in specs:
        if m == best:
            continue
        for met in ["ROC_AUC", "PR_AUC", "Brier"]:
            a = cv[cv.model == best].sort_values("fold")[met].values
            b = cv[cv.model == m].sort_values("fold")[met].values
            t, p = corrected_ttest(a, b, n_tr, n_te)
            tests.append({"comparison": f"{best} vs {m}", "metric": met,
                          "mean_diff": float(a.mean() - b.mean()), "t": t, "p": p})
    pd.DataFrame(tests).to_csv(os.path.join(out, "corrected_ttests.csv"), index=False)
    summary["best_cv_model"] = best

    # ---- 4b robustness: SMOTE within folds and full feature set (single 5x2 run, LR+best tree)
    print("Robustness checks ...")
    rob_rows = []
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    lr_smote = ImbPipeline([("sc", StandardScaler()), ("sm", SMOTE(random_state=SEED)),
                            ("m", LogisticRegression(max_iter=5000, C=0.1, solver="liblinear"))])
    xgb_smote = ImbPipeline([("sm", SMOTE(random_state=SEED)),
                             ("m", XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03,
                                                 subsample=0.85, colsample_bytree=0.7,
                                                 eval_metric="logloss", random_state=SEED,
                                                 n_jobs=-1))])
    lr_full = Pipeline([("sc", StandardScaler()),
                        ("m", LogisticRegression(max_iter=5000, C=0.1, solver="liblinear",
                                                 class_weight="balanced"))])
    for name, est, XX in [("LR+SMOTE", lr_smote, X), ("XGB+SMOTE", xgb_smote, X),
                          ("LR incl. Gender/MaritalStatus", lr_full, X_full),
                          ("LR main (fixed C=0.1)", lr_full, X)]:
        for f, (tr, te) in enumerate(skf.split(XX, y)):
            e = clone(est).fit(XX.iloc[tr], y.iloc[tr])
            p = e.predict_proba(XX.iloc[te])[:, 1]
            r = metrics_at(y.iloc[te], p)
            r.update({"spec": name, "fold": f})
            rob_rows.append(r)
    rob = pd.DataFrame(rob_rows).groupby("spec").mean().drop(columns="fold")
    rob.to_csv(os.path.join(out, "robustness.csv"))
    print(rob.round(3))

    # ---- 4c hold-out split for explanation, calibration and cost analysis
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    fitted, hold = {}, {}
    for name, (pipe, grid) in specs.items():
        rs = RandomizedSearchCV(clone(pipe), grid, n_iter=30, scoring="roc_auc",
                                cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        fitted[name] = rs.best_estimator_
        p = rs.predict_proba(Xte)[:, 1]
        hold[name] = p
        r = metrics_at(yte, p)
        r["best_params"] = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                            for k, v in rs.best_params_.items()}
        summary[f"holdout_{name}"] = r
    pd.DataFrame({k: v for k, v in hold.items()}).assign(y=yte.values, row_index=Xte.index).to_csv(
        os.path.join(out, "holdout_predictions.csv"), index=False)

    # ROC and calibration figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    styles = {"LR": "-", "RF": "--", "XGB": "-.", "LGBM": ":"}
    for name, p in hold.items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0].plot(fpr, tpr, styles[name], color="black",
                   label=f"{name} (AUC = {roc_auc_score(yte, p):.3f})")
        fr, mp = calibration_curve(yte, p, n_bins=8, strategy="quantile")
        ax[1].plot(mp, fr, styles[name], marker="o", ms=3, color="black", label=name)
    ax[0].plot([0, 1], [0, 1], color="grey", lw=0.6)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC curves")
    ax[1].plot([0, 1], [0, 1], color="grey", lw=0.6)
    ax[1].set(xlabel="Mean predicted probability", ylabel="Observed attrition rate",
              title="(b) Calibration (quantile bins)")
    for a in ax:
        a.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure2_ROC_calibration.png"))
    plt.close()

    # ---- 4d SHAP
    print("SHAP ...")
    # tree model: best of RF/XGB/LGBM by CV
    tree_best = cv[cv.model.isin(["RF", "XGB", "LGBM"])].groupby("model")["ROC_AUC"].mean().idxmax()
    summary["tree_model_explained"] = tree_best
    tree_model = fitted[tree_best].named_steps["m"]
    expl_t = shap.TreeExplainer(tree_model)
    sv_t = expl_t.shap_values(Xte)
    if isinstance(sv_t, list):
        sv_t = sv_t[1]
    sv_t = np.asarray(sv_t)
    if sv_t.ndim == 3:
        sv_t = sv_t[:, :, 1]
    base_t = expl_t.expected_value
    base_t = float(np.ravel(base_t)[-1])

    lr = fitted["LR"]
    Xtr_s = pd.DataFrame(lr.named_steps["sc"].transform(Xtr), columns=X.columns)
    Xte_s = pd.DataFrame(lr.named_steps["sc"].transform(Xte), columns=X.columns)
    expl_l = shap.LinearExplainer(lr.named_steps["m"], Xtr_s)
    sv_l = np.asarray(expl_l.shap_values(Xte_s))

    imp = pd.DataFrame({"feature": X.columns,
                        f"meanabs_{tree_best}": np.abs(sv_t).mean(0),
                        "meanabs_LR": np.abs(sv_l).mean(0)})
    imp["rank_tree"] = imp[f"meanabs_{tree_best}"].rank(ascending=False)
    imp["rank_LR"] = imp["meanabs_LR"].rank(ascending=False)
    imp = imp.sort_values(f"meanabs_{tree_best}", ascending=False)
    imp.to_csv(os.path.join(out, "shap_global_importance.csv"), index=False)
    rho = stats.spearmanr(imp["rank_tree"], imp["rank_LR"])
    tau = stats.kendalltau(imp["rank_tree"].head(15), imp["rank_LR"].head(15))
    top10_t = set(imp.head(10).feature)
    top10_l = set(imp.sort_values("meanabs_LR", ascending=False).head(10).feature)
    summary["shap_rank_agreement"] = {"spearman_all": float(rho.statistic),
                                      "spearman_p": float(rho.pvalue),
                                      "top10_overlap": len(top10_t & top10_l),
                                      "top10_tree": sorted(top10_t),
                                      "top10_LR": sorted(top10_l)}

    # SHAP beeswarm
    plt.figure()
    shap.summary_plot(sv_t, Xte, max_display=15, show=False, plot_size=(8, 6), cmap="coolwarm")
    plt.title(f"SHAP values, {tree_best} model (hold-out set, n = {len(Xte)})", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure3_SHAP_beeswarm.png"), bbox_inches="tight")
    plt.close()

    # global bar comparing two models
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    yy = np.arange(len(top))
    a = top[f"meanabs_{tree_best}"] / top[f"meanabs_{tree_best}"].max()
    b = top["meanabs_LR"] / imp["meanabs_LR"].max()
    ax.barh(yy + 0.2, a, 0.4, color="#3a3a3a", label=tree_best)
    ax.barh(yy - 0.2, b, 0.4, color="#b0b0b0", label="LR")
    ax.set_yticks(yy)
    ax.set_yticklabels(top.feature)
    ax.set_xlabel("Mean |SHAP| (scaled to the model's largest value)")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure4_SHAP_importance_two_models.png"))
    plt.close()

    # explanation stability across bootstrap refits (tree model, fixed hyper-parameters)
    print("SHAP stability (bootstrap) ...")
    B = 30
    rng = np.random.default_rng(SEED)
    ranks = []
    for bidx in range(B):
        idx = rng.choice(len(Xtr), len(Xtr), replace=True)
        mb = clone(tree_model).fit(Xtr.iloc[idx], ytr.iloc[idx])
        s = shap.TreeExplainer(mb).shap_values(Xte)
        if isinstance(s, list):
            s = s[1]
        s = np.asarray(s)
        if s.ndim == 3:
            s = s[:, :, 1]
        ranks.append(pd.Series(np.abs(s).mean(0), index=X.columns).rank(ascending=False))
    R = pd.concat(ranks, axis=1)
    stab = pd.DataFrame({"median_rank": R.median(1), "q05": R.quantile(0.05, axis=1),
                         "q95": R.quantile(0.95, axis=1),
                         "share_top10": (R <= 10).mean(1)}).sort_values("median_rank")
    stab.to_csv(os.path.join(out, "shap_bootstrap_stability.csv"))
    summary["bootstrap_B"] = B

    # dependence plots for managerially relevant drivers
    dep_feats = ["OverTime_Yes", "MonthlyIncome", "StockOptionLevel", "TotalWorkingYears",
                 "Age", "EnvironmentSatisfaction", "JobSatisfaction", "YearsWithCurrManager"]
    dep_feats = [f for f in dep_feats if f in X.columns]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for axx, f in zip(axes.ravel(), dep_feats):
        j = list(X.columns).index(f)
        xv = Xte[f].values
        jitter = (rng.uniform(-0.15, 0.15, len(xv))
                  if Xte[f].nunique() <= 5 else 0)
        axx.scatter(xv + jitter, sv_t[:, j], s=8, alpha=0.6, color="black")
        axx.axhline(0, color="grey", lw=0.6)
        axx.set_title(f, fontsize=9)
        axx.set_ylabel("SHAP (log-odds)", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure5_SHAP_dependence.png"))
    plt.close()

    # interaction: overtime x income and overtime x stock options
    inter = {}
    if hasattr(expl_t, "shap_interaction_values") and tree_best in ("XGB", "LGBM", "RF"):
        try:
            iv = expl_t.shap_interaction_values(Xte)
            if isinstance(iv, list):
                iv = iv[1]
            iv = np.asarray(iv)
            if iv.ndim == 4:
                iv = iv[:, :, :, 1]
            M = np.abs(iv).mean(0)
            np.fill_diagonal(M, 0)
            Mi = pd.DataFrame(M, index=X.columns, columns=X.columns)
            pairs = (Mi.where(np.triu(np.ones(M.shape), 1).astype(bool)).stack()
                     .sort_values(ascending=False).head(10))
            pairs.to_csv(os.path.join(out, "shap_top_interactions.csv"), header=["mean_abs"])
            inter = {f"{a} x {b}": float(v) for (a, b), v in pairs.items()}
        except Exception as e:  # noqa
            inter = {"error": str(e)}
    summary["top_interactions"] = inter

    # overtime x income descriptive, on the hold-out SHAP
    j_inc = list(X.columns).index("MonthlyIncome")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for ot, mk, col in [(1, "o", "black"), (0, "^", "#9a9a9a")]:
        msk = Xte["OverTime_Yes"].values == ot
        ax.scatter(Xte["MonthlyIncome"].values[msk], sv_t[msk, j_inc], s=12, marker=mk,
                   color=col, alpha=0.7, label="Overtime = " + ("Yes" if ot else "No"))
    ax.axhline(0, color="grey", lw=0.6)
    ax.set(xlabel="Monthly income (USD)", ylabel="SHAP value of MonthlyIncome")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure6_income_overtime_interaction.png"))
    plt.close()

    # threshold where income SHAP crosses zero (lowess-free: binned means)
    bins = pd.qcut(Xte["MonthlyIncome"], 10, duplicates="drop")
    bm = pd.DataFrame({"bin": bins, "shap": sv_t[:, j_inc]}).groupby("bin", observed=True)["shap"].mean()
    bm.to_csv(os.path.join(out, "income_shap_by_decile.csv"))

    # ---- 4e controllability grouping
    ctrl = {
        "Directly controllable (work design and rewards)": [
            "OverTime_Yes", "StockOptionLevel", "MonthlyIncome", "PercentSalaryHike",
            "TrainingTimesLastYear", "BusinessTravel_Travel_Frequently",
            "BusinessTravel_Travel_Rarely", "BusinessTravel_Non-Travel", "JobLevel"],
        "Indirectly controllable (climate and relationships)": [
            "EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
            "WorkLifeBalance", "JobInvolvement", "YearsWithCurrManager",
            "YearsSinceLastPromotion", "PerformanceRating"],
        "Largely non-controllable (personal and history)": [
            "Age", "TotalWorkingYears", "NumCompaniesWorked", "DistanceFromHome",
            "Education", "YearsAtCompany", "YearsInCurrentRole"],
    }
    absmean = pd.Series(np.abs(sv_t).mean(0), index=X.columns)
    total = absmean.sum()
    grp = {}
    assigned = set()
    for g, fs in ctrl.items():
        fs = [f for f in fs if f in absmean.index]
        assigned |= set(fs)
        grp[g] = float(absmean[fs].sum() / total)
    rest = [f for f in absmean.index if f not in assigned]
    grp["Structural (role, department, field, pay-rate fields)"] = float(absmean[rest].sum() / total)
    summary["controllability_share"] = grp
    pd.Series(grp).to_csv(os.path.join(out, "shap_controllability_share.csv"), header=["share"])

    # ---- 4f job-role segment explanation
    role_cols = [c for c in X.columns if c.startswith("JobRole_")]
    role = Xte[role_cols].idxmax(1).str.replace("JobRole_", "")
    seg = pd.DataFrame(sv_t, columns=X.columns, index=Xte.index)
    seg["role"] = role.values
    seg_out = []
    for r, g in seg.groupby("role"):
        m = g.drop(columns="role").mean()
        seg_out.append({"role": r, "n": len(g),
                        "mean_pred": float(hold[tree_best][Xte.index.get_indexer(g.index)].mean()),
                        "top_positive_driver": m.idxmax(), "value": float(m.max())})
    pd.DataFrame(seg_out).sort_values("mean_pred", ascending=False).to_csv(
        os.path.join(out, "shap_by_jobrole.csv"), index=False)

    # ---- 4g local explanations: three archetypes from hold-out leavers / stayers
    p_t = hold[tree_best]
    order = np.argsort(-p_t)
    yv = yte.values
    tp = [i for i in order if yv[i] == 1][0]                       # highest-risk true leaver
    fn = [i for i in order[::-1] if yv[i] == 1][0]                 # leaver the model missed
    fp = [i for i in order if yv[i] == 0][0]                       # stayer flagged as high risk
    cases = {"A_high_risk_leaver": tp, "B_missed_leaver": fn, "C_flagged_stayer": fp}
    local = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axx, (lab, i) in zip(axes, cases.items()):
        contrib = pd.Series(sv_t[i], index=X.columns)
        topc = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        local[lab] = {"pred_prob": float(p_t[i]), "actual": int(yv[i]),
                      "top_contributions": {k: [float(Xte.iloc[i][k]), float(v)]
                                            for k, v in topc.items()}}
        tc = topc.iloc[::-1]
        axx.barh(range(len(tc)), tc.values,
                 color=["black" if v > 0 else "#a8a8a8" for v in tc.values])
        axx.set_yticks(range(len(tc)))
        axx.set_yticklabels([f"{k} = {Xte.iloc[i][k]:g}" for k in tc.index], fontsize=8)
        axx.axvline(0, color="grey", lw=0.6)
        axx.set_title(f"Case {lab[0]}: p = {p_t[i]:.2f}, left = {'Yes' if yv[i] else 'No'}",
                      fontsize=9)
        axx.set_xlabel("SHAP contribution (log-odds)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure7_local_explanations.png"))
    plt.close()
    summary["local_cases"] = local
    summary["shap_base_value_logodds"] = base_t

    # ---- 4h fairness check on the hold-out (main model excludes gender/marital status)
    g = raw.loc[Xte.index, "Gender"].values
    fair = {}
    thr = np.quantile(p_t, 1 - yte.mean())  # flag as many as the base rate
    yhat = (p_t >= thr).astype(int)
    for grp_ in ["Male", "Female"]:
        m = g == grp_
        fair[grp_] = {"n": int(m.sum()), "flag_rate": float(yhat[m].mean()),
                      "TPR": float(recall_score(yv[m], yhat[m])),
                      "AUC": float(roc_auc_score(yv[m], p_t[m]))}
    age_band = np.where(raw.loc[Xte.index, "Age"].values < 30, "<30", ">=30")
    for grp_ in ["<30", ">=30"]:
        m = age_band == grp_
        fair[f"Age{grp_}"] = {"n": int(m.sum()), "flag_rate": float(yhat[m].mean()),
                              "TPR": float(recall_score(yv[m], yhat[m])),
                              "AUC": float(roc_auc_score(yv[m], p_t[m]))}
    summary["fairness_threshold"] = float(thr)
    summary["fairness"] = fair

    # cost-sensitive targeting is in cost_targeting.py (run after this script)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    main(a.data, a.out)
