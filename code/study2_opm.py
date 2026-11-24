"""
Study 2: replication of the protocol on real federal workforce records (U.S. Office of Personnel Management,
Federal Workforce Data, record-level Employment and Separations files; no personally identifiable information).

Design (case-base): the public files carry no employee identifier, so separations cannot be linked to an
employee's earlier record. We therefore use a case-base design. The base is everyone in the Employment
snapshot of 30 September 2023 (a random sample of it, fraction f); cases are all voluntary quits effective
October 2023 to September 2024 by employees already employed at the snapshot (length of service at separation
minus elapsed time >= 0). A classifier separating cases from base estimates odds_cb(x) = cases(x) / (f * N(x)),
so the one-year quit risk is risk(x) = f * odds_cb(x). Rankings, SHAP explanations and the screen need no
correction; absolute risks use the formula above.

Inputs (Parquet, downloaded from https://data.opm.gov/get-data/data-downloads or the API
https://data.opm.gov/api/v1/files/{dataset}/{year}/{month}/{version}/download):
    employment_202309_4.parquet
    separations_202310_4.parquet ... separations_202409_4.parquet  (12 files)

Usage: python study2_opm.py --raw ../data/opm --out ../results/study2 [--base-frac 0.10]
"""
import argparse
import glob
import json
import os
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common import SEED, XGB_FIXED, shap_matrix

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300})
SNAP = (2023, 9)

CONSTRUCTS2 = {
    "Rewards (pay, grade, step)": ["log_salary", "pay_redacted", "grade_num", "step", "flsa_exempt"],
    "Career stage and tenure": ["age_mid", "los_years", "career_conditional"],
    "Human capital": ["edu_level", "veteran"],
    "Job content": ["supervisory", "occ_type", "occ_category"],
    "Employment terms": ["bargaining", "payplan"],
    "Organisation and location": ["agency", "state"],
}
EDU = {"HIGH SCHOOL OR LESS": 1, "SOME COLLEGE OR ASSOCIATES DEGREE": 2, "BACHELORS DEGREE": 3,
       "MASTERS OR PROFESSIONAL DEGREE": 4, "DOCTORATE DEGREE": 5}


USECOLS = ["age_bracket", "length_of_service_years", "annualized_adjusted_basic_pay", "grade", "step_or_rate_type",
           "flsa_category", "tenure", "education_level_bracket", "veteran_indicator", "supervisory_status",
           "stem_occupation", "occupational_category", "bargaining_unit_status", "pay_plan_code", "agency_code",
           "duty_station_state_abbreviation", "work_schedule"]
SEPCOLS = ["separation_category", "personnel_action_effective_date_yyyymm"]


def read_parquets(pattern, extra=()):
    """Read only the columns the analysis uses (the full employment file has 64 text columns)."""
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise FileNotFoundError(pattern)
    cols = USECOLS + list(extra)
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    for c in cols:
        df[c] = df[c].astype("category")
    return df, [os.path.basename(f) for f in fs]


def age_mid(s):
    def f(v):
        m = re.findall(r"\d+", str(v))
        if "LESS" in str(v).upper():
            return 18.0
        if "MORE" in str(v).upper() or "+" in str(v):
            return 67.0
        return (float(m[0]) + float(m[1])) / 2 if len(m) >= 2 else np.nan
    return s.astype(str).map(f).astype(float)


def population(df):
    return df[(df["work_schedule"].astype(str) == "FULL-TIME") &
              df["tenure"].astype(str).str.startswith(("TENURE GROUP 1", "TENURE GROUP 2"))]


def features(df, top_agency, top_state):
    out = pd.DataFrame(index=df.index)
    out["age_mid"] = age_mid(df["age_bracket"])
    out["los_years"] = pd.to_numeric(df["length_of_service_years"], errors="coerce").clip(upper=50)  # a few records exceed 100
    sal = pd.to_numeric(df["annualized_adjusted_basic_pay"], errors="coerce")
    out["log_salary"] = np.log(sal.where(sal > 10_000))
    out["pay_redacted"] = sal.isna().astype(float)
    out["grade_num"] = pd.to_numeric(df["grade"], errors="coerce")
    out["step"] = pd.to_numeric(df["step_or_rate_type"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    out["flsa_exempt"] = (df["flsa_category"].astype(str) == "EXEMPT EMPLOYEE").astype(float)
    out["career_conditional"] = df["tenure"].astype(str).str.startswith("TENURE GROUP 2").astype(float)
    out["edu_level"] = df["education_level_bracket"].astype(str).map(EDU)
    out["veteran"] = (df["veteran_indicator"].astype(str) == "Y").astype(float)
    out["supervisory"] = (~df["supervisory_status"].astype(str).isin(["ALL OTHER POSITIONS", "INVALID"])).astype(float)
    out["occ_type"] = df["stem_occupation"].astype(str).replace({"UNSPECIFIED": "ALL OTHER OCCUPATIONS"})
    out["occ_category"] = df["occupational_category"].astype(str).replace({"INVALID": "OTHER WHITE COLLAR"})
    bu = df["bargaining_unit_status"].astype(str)
    out["bargaining"] = bu.where(bu.isin(["ELIGIBLE_IN_BU", "NOT_ELIGIBLE", "ELIGIBLE_NOT_IN_BU", "REDACTED"]), "REDACTED")
    pp = df["pay_plan_code"].astype(str)
    out["payplan"] = np.where(pp.isin(["GS", "GL", "GM"]), "GS", np.where(pp.str.startswith("W"), "Wage grade",
                              np.where(pp.isin(["VN", "VM", "VP"]), "VA medical", "Other")))
    ag = df["agency_code"].astype(str)
    out["agency"] = ag.where(ag.isin(top_agency), "Other")
    st = df["duty_station_state_abbreviation"].astype(str)
    out["state"] = st.where(st.isin(top_state), "Other")
    return out


def main(raw, out, base_frac, quick=False, end=202405):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    emp, ef = read_parquets(os.path.join(raw, "employment_202309_*.parquet"))
    sep, sf = read_parquets(os.path.join(raw, "separations_*.parquet"), extra=SEPCOLS)
    log = {"employment_files": ef, "separation_files": sf, "n_employment_rows": len(emp), "n_separation_rows": len(sep)}
    ef_date = pd.to_numeric(sep["personnel_action_effective_date_yyyymm"], errors="coerce")
    # Window: October 2023 to `end`. The default ends in May 2024 because occupational-category coding on separation
    # records changes from June 2024 (share coded 'no data' jumps from 0% to 10-28%, and the category mix shifts;
    # total variation distance from October 2023 rises from <= 0.06 to 0.15-0.30). The 12-month window is run as a
    # sensitivity analysis (--end 202409).
    sep = sep[(ef_date >= 202310) & (ef_date <= end)].copy()
    log["window_end"] = int(end)
    log["separation_categories_in_window"] = sep["separation_category"].astype(str).value_counts().to_dict()
    emp = population(emp)
    q = population(sep[sep["separation_category"].astype(str) == "QUIT"]).copy()
    ed = pd.to_numeric(q["personnel_action_effective_date_yyyymm"], errors="coerce")
    months = (ed // 100 - SNAP[0]) * 12 + (ed % 100 - SNAP[1])
    # rounded to the 0.1-year precision of the source field, so cases and base share the same value grid
    q["length_of_service_years"] = (pd.to_numeric(q["length_of_service_years"], errors="coerce") - months / 12.0).round(1)
    # Cases are measured at separation, the base at the snapshot. Two further alignments (deviations from the
    # pre-specification, reported as such): pay effective from January 2024 is deflated by the 5.2% average 2024
    # federal pay adjustment (rounded to whole dollars). Age is left in its five-year brackets: shifting case ages by
    # elapsed months would put cases off the bracket-midpoint grid and let the model separate them trivially.
    pay = pd.to_numeric(q["annualized_adjusted_basic_pay"], errors="coerce")
    # The 5.2% figure is the General Schedule average, so it is applied only to GS-type pay plans (GS, GL, GM);
    # other pay systems are left undeflated (sensitivity: deflating all plans gave the same screen and ROC-AUC 0.836).
    gs = q["pay_plan_code"].astype(str).isin(["GS", "GL", "GM"]).values
    q["annualized_adjusted_basic_pay"] = np.where((ed >= 202401) & gs, (pay / 1.052).round(0), pay)
    log["n_quits_population_window"] = len(q)
    q = q[q["length_of_service_years"] >= 0]
    log["n_quits_employed_at_snapshot"] = len(q)
    log["N_population"] = len(emp)
    log["crude_one_year_quit_rate"] = len(q) / len(emp)
    rng = np.random.default_rng(SEED)
    base = emp.sample(frac=base_frac, random_state=SEED) if base_frac < 1 else emp
    log["base_fraction"], log["n_base"] = base_frac, len(base)
    top_agency = emp["agency_code"].astype(str).value_counts().head(15).index
    top_state = emp["duty_station_state_abbreviation"].astype(str).value_counts().head(12).index
    Xq = features(q, top_agency, top_state)
    X = pd.concat([features(base, top_agency, top_state), Xq], ignore_index=True)
    y = np.r_[np.zeros(len(base)), np.ones(len(q))].astype(int)
    cmonth = np.r_[np.zeros(len(base)), pd.to_numeric(q["personnel_action_effective_date_yyyymm"], errors="coerce").values]
    req = ["age_mid", "los_years", "edu_level", "grade_num"]
    # "NO DATA REPORTED" occupation codes occur almost only on separation records (1.4% of quits vs <0.1% of the
    # snapshot), so they would act as a recording-artefact label; such rows are dropped and counted.
    nodata = (X["occ_category"] == "NO DATA REPORTED").values
    log["dropped_occ_no_data"] = {"base": int((nodata & (y == 0)).sum()), "cases": int((nodata & (y == 1)).sum())}
    keep = X[req].notna().all(1).values & ~nodata
    miss = ~X[req].notna().all(1).values
    log["dropped_missing_core"] = {"base": int((miss & (y == 0)).sum()), "cases": int((miss & (y == 1)).sum())}
    X, y, cmonth = X[keep].reset_index(drop=True), y[keep], cmonth[keep]
    log["n_analysis"], log["n_cases_analysis"] = len(y), int(y.sum())

    num = ["age_mid", "los_years", "log_salary", "pay_redacted", "grade_num", "step", "flsa_exempt", "career_conditional",
           "edu_level", "veteran", "supervisory"]
    cat = ["occ_type", "occ_category", "bargaining", "payplan", "agency", "state"]
    Xd = pd.get_dummies(X, columns=cat, dtype=float)
    cols = list(Xd.columns)
    # LR uses reference coding (first level of each categorical dropped) so its coefficients and SHAP values are
    # identified; reference columns get LR SHAP = 0 and construct totals are unaffected.
    ref = [c for k in cat for c in [sorted([c for c in cols if c.startswith(k + "_")])[0]]]
    lr_cols = [c for c in cols if c not in ref]
    lr_idx = [cols.index(c) for c in lr_cols]
    log["lr_reference_levels"] = ref
    cmap = {}
    for c in cols:
        root = next((k for k in num + cat if c == k or c.startswith(k + "_")), c)
        cmap[c] = next(g for g, fs in CONSTRUCTS2.items() if root in fs)

    # ---- leakage probe: can the grid of a value alone (its decimal part) separate cases from base?
    probe = {}
    for c_ in ["los_years", "log_salary", "age_mid"]:
        v = X[c_].values
        frac = np.where(np.isfinite(v), np.round((np.exp(v) if c_ == "log_salary" else v * 10) % 1, 6), -1)
        probe[c_] = float(roc_auc_score(y, frac)) if len(np.unique(frac)) > 1 else 0.5
    log["value_grid_probe_AUC"] = probe

    # ---- cross-fitted LR and XGBoost
    spw = (y == 0).sum() / y.sum()
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = {"LR": np.zeros(len(y)), "LR_spline": np.zeros(len(y)), "XGB": np.zeros(len(y))}
    sv = np.zeros((len(y), len(cols)))
    sv_lr = np.zeros((len(y), len(cols)))
    for f, (tr, te) in enumerate(skf.split(Xd, y)):
        med = Xd.iloc[tr].median()
        XL = Xd[lr_cols].fillna(med[lr_cols])
        lr = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(C=0.1, max_iter=3000))]).fit(XL.iloc[tr], y[tr])
        oof["LR"][te] = lr.predict_proba(XL.iloc[te])[:, 1]
        # LR with natural-cubic-type spline bases for the four continuous predictors: tests whether the gap to
        # boosting is due to non-linearity alone
        spl_cols = ["los_years", "age_mid", "grade_num", "log_salary"]
        ct = ColumnTransformer([("spl", SplineTransformer(n_knots=6, degree=3), spl_cols)], remainder="passthrough")
        lrs = Pipeline([("ct", ct), ("sc", StandardScaler()), ("m", LogisticRegression(C=0.1, max_iter=3000))]).fit(XL.iloc[tr], y[tr])
        oof["LR_spline"][te] = lrs.predict_proba(XL.iloc[te])[:, 1]
        # XGBoost is fitted on 80% of the training fold and Platt-recalibrated on the other 20%. A logistic
        # recalibration with an intercept restores the case-base identity sum(odds over base) = n_cases, so that
        # risk = f x odds is on the population scale (King and Zeng, 2001).
        rc = np.random.default_rng(SEED + f)
        calm = rc.random(len(tr)) < 0.2
        xg = XGBClassifier(**XGB_FIXED, random_state=SEED, n_jobs=-1).fit(Xd.iloc[tr[~calm]], y[tr[~calm]])
        lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(lg(xg.predict_proba(Xd.iloc[tr[calm]])[:, 1]), y[tr[calm]])
        oof["XGB"][te] = cal.predict_proba(lg(xg.predict_proba(Xd.iloc[te])[:, 1]))[:, 1]
        sub = te if len(te) <= 20000 else rng.choice(te, 20000, replace=False)
        sv[sub] = shap_matrix(shap.TreeExplainer(xg).shap_values(Xd.iloc[sub]))
        bg = pd.DataFrame(lr.named_steps["sc"].transform(XL.iloc[tr[:2000]]), columns=lr_cols)
        sv_lr[np.ix_(sub, lr_idx)] = np.asarray(shap.LinearExplainer(lr.named_steps["m"], shap.maskers.Independent(bg, 100))
                                .shap_values(pd.DataFrame(lr.named_steps["sc"].transform(XL.iloc[sub]), columns=lr_cols)))
        print(f"  fold {f + 1}/5", flush=True)
    has = np.abs(sv).sum(1) > 0
    perf = {m: {"case_base_ROC_AUC": roc_auc_score(y, p), "case_base_PR_AUC": average_precision_score(y, p)}
            for m, p in oof.items()}
    log["performance"] = perf

    # absolute one-year risk among the base sample: risk = f * odds
    for m, p in oof.items():
        odds = p / (1 - p)
        risk = np.clip(base_frac * odds, 0, 1)
        log[f"{m}_mean_risk_base"] = float(risk[y == 0].mean())

    # ---- global and construct importance
    imp = pd.DataFrame({"feature": cols, "construct": [cmap[c] for c in cols],
                        "XGB": np.abs(sv[has]).mean(0), "LR": np.abs(sv_lr[has]).mean(0)}).sort_values("XGB", ascending=False)
    imp.to_csv(os.path.join(out, "s2_importance.csv"), index=False)
    groups = list(CONSTRUCTS2)
    G = lambda S_: np.column_stack([S_[:, [i for i, c in enumerate(cols) if cmap[c] == g]].sum(1) for g in groups])
    gx, gl = np.abs(G(sv[has])).mean(0), np.abs(G(sv_lr[has])).mean(0)
    gtab = pd.DataFrame({"construct": groups, "share_XGB": gx / gx.sum(), "share_LR": gl / gl.sum()})

    # ---- shadow screen: 2 configs not needed here; fixed config, 3 cut-offs, B refits on subsamples
    B = 6 if quick else 30
    nsub = min(len(y), 6000 if quick else 60000)
    hits, top10, gh = np.zeros(len(cols)), np.zeros(len(cols)), np.zeros(len(groups))
    for b in range(B):
        r = np.random.default_rng(SEED + 9000 + b)
        idx = r.choice(len(y), nsub, replace=False)
        tr, te = idx[: int(0.7 * nsub)], idx[int(0.7 * nsub):]
        sh = lambda Z: pd.concat([Z, Z.apply(lambda c: r.permutation(c.values)).add_prefix("shadow__")], axis=1)
        Xb, Xo = sh(Xd.iloc[tr].reset_index(drop=True)), sh(Xd.iloc[te].reset_index(drop=True))
        m = XGBClassifier(**XGB_FIXED, random_state=SEED + b, n_jobs=-1).fit(Xb, y[tr])
        s_ = shap_matrix(shap.TreeExplainer(m).shap_values(Xo))
        a = np.abs(s_).mean(0)
        real, shad = a[:len(cols)], a[len(cols):]
        hits += real > shad.max()
        top10 += pd.Series(real).rank(ascending=False).values <= 10
        gr = np.abs(np.column_stack([s_[:, [i for i, c in enumerate(cols) if cmap[c] == g]].sum(1) for g in groups])).mean(0)
        gs = np.abs(np.column_stack([s_[:, [len(cols) + i for i, c in enumerate(cols) if cmap[c] == g]].sum(1) for g in groups])).mean(0)
        gh += gr > gs.max()
    scr = pd.DataFrame({"feature": cols, "top10_share": top10 / B, "beats_shadow_share": hits / B})
    for c in (0.7, 0.8, 0.9):
        scr[f"passes_{int(c * 100)}"] = (scr.top10_share >= c) & (scr.beats_shadow_share >= c)
    scr["n_cutoffs_passed"] = scr[[f"passes_{k}" for k in (70, 80, 90)]].sum(1)
    scr = scr.merge(imp[["feature", "XGB", "LR", "construct"]], on="feature").sort_values("XGB", ascending=False)
    scr.to_csv(os.path.join(out, "s2_screen.csv"), index=False)
    gtab["beats_shadow_share"] = gh / B
    gtab.to_csv(os.path.join(out, "s2_constructs.csv"), index=False)
    log["screen_B"], log["screen_subsample"] = B, nsub

    # ---- variable-level screen: each original variable is one unit, so a categorical variable is not split into
    # many dummies competing for top-k places. Shadows permute whole dummy blocks (rows jointly). Importance =
    # mean |sum of SHAP over the variable's columns|. Stability = rank within the top K_VAR of the 17 variables
    # (K_VAR = 4, about the same share as top 10 of 45 in Study 1).
    K_VAR = 4
    vars_ = num + cat
    vidx = {v: [i for i, c in enumerate(cols) if c == v or c.startswith(v + "_")] for v in vars_}
    vh, vt = np.zeros(len(vars_)), {k: np.zeros(len(vars_)) for k in (3, 4, 5)}
    for b in range(B):
        r = np.random.default_rng(SEED + 9500 + b)
        idx = r.choice(len(y), nsub, replace=False)
        tr, te = idx[: int(0.7 * nsub)], idx[int(0.7 * nsub):]
        def shv(Z):
            parts = [Z]
            for v in vars_:
                blk = Z.iloc[:, vidx[v]]
                parts.append(blk.iloc[r.permutation(len(blk))].reset_index(drop=True).add_prefix("shadow__"))
            return pd.concat(parts, axis=1)
        Xb, Xo = shv(Xd.iloc[tr].reset_index(drop=True)), shv(Xd.iloc[te].reset_index(drop=True))
        m = XGBClassifier(**XGB_FIXED, random_state=SEED + 500 + b, n_jobs=-1).fit(Xb, y[tr])
        s_ = shap_matrix(shap.TreeExplainer(m).shap_values(Xo))
        off = len(cols)
        real = np.array([np.abs(s_[:, vidx[v]].sum(1)).mean() for v in vars_])
        shad, o = [], off
        for v in vars_:
            k_ = len(vidx[v]); shad.append(np.abs(s_[:, o:o + k_].sum(1)).mean()); o += k_
        vh += real > max(shad)
        rk_ = pd.Series(real).rank(ascending=False).values
        for k in vt:
            vt[k] += rk_ <= k
    vfull = np.array([np.abs(sv[has][:, vidx[v]].sum(1)).mean() for v in vars_])
    vscr = pd.DataFrame({"variable": vars_, "mean_abs_shap_XGB": vfull, "beats_shadow_share": vh / B,
                         **{f"top{k}_share": vt[k] / B for k in vt}})
    for c in (0.7, 0.8, 0.9):
        vscr[f"passes_{int(c * 100)}"] = (vscr[f"top{K_VAR}_share"] >= c) & (vscr.beats_shadow_share >= c)
    vscr["n_cutoffs_passed"] = vscr[[f"passes_{k}" for k in (70, 80, 90)]].sum(1)
    vscr["construct"] = [next(g for g, fs in CONSTRUCTS2.items() if v in fs) for v in vars_]
    vscr = vscr.sort_values("mean_abs_shap_XGB", ascending=False)
    vscr.to_csv(os.path.join(out, "s2_screen_variables.csv"), index=False)
    log["variable_screen_K"] = K_VAR

    # ---- split by quit date (not a forward validation: all predictors come from the same snapshot): train on quits effective October 2023 to March 2024 plus a random half of the base;
    # test on quits effective April to September 2024 plus the other half. Precision and lift on the test half are
    # population-scale estimates (base half sampled at base_frac / 2) relative to the six-month quit rate.
    rh = np.random.default_rng(SEED + 31)
    half = rh.random(len(y)) < 0.5
    split_m = 202401 if end == 202405 else 202403
    trm = ((y == 0) & half) | ((y == 1) & (cmonth <= split_m))
    tem = ((y == 0) & ~half) | ((y == 1) & (cmonth > split_m))
    log["date_split_train_end"] = split_m
    mt = XGBClassifier(**XGB_FIXED, random_state=SEED, n_jobs=-1).fit(Xd[trm], y[trm])
    pt = mt.predict_proba(Xd[tem])[:, 1]
    medt = Xd[lr_cols][trm].median()
    lrt = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(C=0.1, max_iter=3000))]).fit(
        Xd[lr_cols][trm].fillna(medt), y[trm])
    plt_ = lrt.predict_proba(Xd[lr_cols][tem].fillna(medt))[:, 1]
    yt = y[tem]
    ft = base_frac / 2
    rate6 = yt.sum() / ((yt == 0).sum() / ft)
    th = {"n_train": int(trm.sum()), "n_test": int(tem.sum()), "test_cases": int(yt.sum()),
          "test_period_quit_rate": float(rate6),
          "ROC_AUC_XGB": float(roc_auc_score(yt, pt)), "ROC_AUC_LR": float(roc_auc_score(yt, plt_))}
    for share in (0.05, 0.10):
        thr_ = np.quantile(pt[yt == 0], 1 - share)
        nl = (pt[yt == 0] >= thr_).sum() / ft
        cg = ((pt >= thr_) & (yt == 1)).sum()
        th[f"top_{int(share * 100)}pct"] = {"precision": float(cg / nl), "recall": float(cg / yt.sum()),
                                           "lift": float(cg / nl / rate6)}
    log["temporal_holdout"] = th

    # ---- direction of binary predictors named in the expectations: population-scale quit rate by group and the
    # mean XGBoost SHAP difference (group = 1 minus group = 0) among explained rows
    dirs = {}
    for c_ in ["career_conditional", "supervisory", "veteran", "flsa_exempt", "pay_redacted",
               "occ_type_STEM OCCUPATIONS", "occ_type_HEALTH OCCUPATIONS"]:
        g1 = Xd[c_].values == 1
        rate = lambda m: float((m & (y == 1)).sum() / ((m & (y == 0)).sum() / base_frac))
        j = cols.index(c_)
        dirs[c_] = {"share_of_base": float(g1[y == 0].mean()), "quit_rate_group1": rate(g1), "quit_rate_group0": rate(~g1),
                    "mean_shap_diff_XGB": float(sv[has & g1, j].mean() - sv[has & ~g1, j].mean()),
                    "mean_shap_diff_LR": float(sv_lr[has & g1, j].mean() - sv_lr[has & ~g1, j].mean())}
    log["binary_directions"] = dirs

    # ---- dependence by decile for pay, age, tenure (on explained rows)
    dep = {}
    for f_ in ["log_salary", "age_mid", "los_years", "grade_num"]:
        j = cols.index(f_)
        d = pd.DataFrame({"x": X.loc[has, f_].values, "s": sv[has, j]}).dropna()
        d["bin"] = pd.qcut(d.x.rank(method="first"), 10, labels=False)
        dep[f_] = d.groupby("bin").agg(x_min=("x", "min"), x_max=("x", "max"), shap=("s", "mean")).round(4).to_dict("records")
    log["dependence_deciles"] = dep

    # ---- age-band audit (calibration within band, absolute risk scale)
    p = oof["XGB"]
    risk = np.clip(base_frac * p / (1 - p), 0, 1)
    band = np.where(X["age_mid"] < 30, "<30", np.where(X["age_mid"] >= 40, "40+", "30-39"))
    aud = []
    for bnd in ["<30", "30-39", "40+"]:
        mb = (band == bnd) & (y == 0)
        mc = (band == bnd) & (y == 1)
        aud.append({"band": bnd, "n_base": int(mb.sum()), "observed_rate": mc.sum() / (mb.sum() / base_frac),
                    "mean_predicted_risk": float(risk[mb].mean()),
                    "ROC_AUC_within_band": float(roc_auc_score(y[(band == bnd)], p[(band == bnd)]))})
    log["age_audit"] = aud
    # calibration by decile of predicted risk (base sample), population scale
    dq = pd.qcut(risk[y == 0], 10, labels=False, duplicates="drop")
    thr_d = np.quantile(risk[y == 0], np.linspace(0, 1, 11))
    cald = []
    for k in range(10):
        lo_, hi_ = thr_d[k], thr_d[k + 1]
        mb = (y == 0) & (risk >= lo_) & ((risk < hi_) if k < 9 else (risk <= hi_))
        mc = (y == 1) & (risk >= lo_) & ((risk < hi_) if k < 9 else (risk <= hi_))
        cald.append({"decile": k + 1, "mean_predicted": float(risk[mb].mean()),
                     "observed": float(mc.sum() / (mb.sum() / base_frac))})
    log["calibration_deciles"] = cald
    # measurement-timing check: if features recorded at separation drove the ranking, cases measured long after the
    # snapshot would be easier to separate. ROC-AUC of each quarter's cases against the whole base:
    q_auc = {}
    periods = ({"Oct-Nov 2023": (202310, 202311), "Dec 2023-Jan 2024": (202312, 202401),
                "Feb-Mar 2024": (202402, 202403), "Apr-May 2024": (202404, 202405)} if end == 202405 else
               {"Q1 Oct-Dec 2023": (202310, 202312), "Q2 Jan-Mar 2024": (202401, 202403),
                "Q3 Apr-Jun 2024": (202404, 202406), "Q4 Jul-Sep 2024": (202407, 202409)})
    for qn, (a_, b_) in periods.items():
        m_ = (y == 0) | ((cmonth >= a_) & (cmonth <= b_))
        q_auc[qn] = {"n_cases": int(((y == 1) & m_).sum()), "ROC_AUC_XGB": float(roc_auc_score(y[m_], p[m_])),
                     "ROC_AUC_LR": float(roc_auc_score(y[m_], oof["LR"][m_]))}
    log["auc_by_quit_quarter"] = q_auc
    # base sample of the encoded predictors, for the Study 2 simulation (simulation_study2.py)
    Xd[y == 0].reset_index(drop=True).to_parquet(os.path.join(out, "s2_X_base.parquet"))

    # ---- top-k lists on the population scale, estimated without relying on the model's calibration:
    # the base is a simple random sample of the snapshot population, so (base members above a score threshold)/f
    # estimates how many employees the list would contain, and the cases above it are counted directly.
    sal = np.exp(X["log_salary"].values)
    lists = {}
    for share in (0.05, 0.10, 0.20):
        thr = np.quantile(p[y == 0], 1 - share)
        n_list = (p[y == 0] >= thr).sum() / base_frac
        caught = ((p >= thr) & (y == 1)).sum()
        lists[f"top_{int(share * 100)}pct"] = {"precision": float(caught / n_list), "recall": float(caught / y.sum()),
                                               "lift": float(caught / n_list / (y.sum() / ((y == 0).sum() / base_frac)))}
    log["top_k_population"] = lists
    log["crude_rate_analysis"] = float(y.sum() / ((y == 0).sum() / base_frac))

    # ---- Monte Carlo cost check (same priors as Study 1). Value of a prevented quit = replacement-cost share x the
    # salary of caught quitters with unredacted pay; contact cost per listed employee as in Study 1.
    thr = np.quantile(p[y == 0], 0.95)
    caught_pay = sal[(p >= thr) & (y == 1) & np.isfinite(sal)]
    prec = lists["top_5pct"]["precision"]
    r2 = np.random.default_rng(SEED + 77)
    D = 10000
    cost_share, success, unit = r2.triangular(0.10, 0.21, 0.50, D), r2.beta(6, 14, D), r2.uniform(500, 3000, D)
    pay_draw = r2.choice(caught_pay, D)
    net = success * cost_share * prec * pay_draw - unit
    net2 = 0.5 * success * cost_share * prec * pay_draw - unit
    log["mc_cost"] = {"precision_top5": prec, "median_salary_caught_quitters": float(np.median(caught_pay)),
                      "share_caught_with_pay": float(np.isfinite(sal[(p >= thr) & (y == 1)]).mean()),
                      "P_net_positive": float((net > 0).mean()), "P_net_positive_half_response": float((net2 > 0).mean()),
                      "net_median_per_listed_USD": float(np.median(net)),
                      "net_p05_per_listed_USD": float(np.quantile(net, .05)), "net_p95_per_listed_USD": float(np.quantile(net, .95)),
                      "breakeven_success_at_prior_medians": float(1750 / (0.21 * prec * np.median(caught_pay))),
                      "breakeven_unit_cost_at_prior_medians": float(0.30 * 0.21 * prec * np.median(caught_pay))}

    # ---- figure: construct shares
    g2 = gtab.sort_values("share_XGB")
    yy = np.arange(len(g2))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(yy + 0.2, g2.share_XGB, 0.4, color="#3a3a3a", label="XGBoost")
    ax.barh(yy - 0.2, g2.share_LR, 0.4, color="#b0b0b0", label="Logistic regression")
    ax.set_yticks(yy)
    ax.set_yticklabels(g2.construct)
    ax.set_xlabel("Share of construct-level mean |SHAP|")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "Fig_study2_constructs.png"), dpi=300)
    plt.close()

    json.dump(log, open(os.path.join(out, "s2_summary.json"), "w"), indent=2, default=str)
    print(json.dumps({k: log[k] for k in ["n_employment_rows", "n_quits_employed_at_snapshot", "crude_one_year_quit_rate",
                                         "performance"]}, indent=1, default=str))
    print(scr.head(15).round(3).to_string())
    print(gtab.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/opm")
    ap.add_argument("--out", default="../results/study2")
    ap.add_argument("--base-frac", type=float, default=0.10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--end", type=int, default=202405)
    a = ap.parse_args()
    main(a.raw, a.out, a.base_frac, a.quick, a.end)
