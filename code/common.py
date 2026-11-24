"""Shared data handling, feature groups and helpers for all scripts."""
import numpy as np
import pandas as pd

SEED = 42
DROP = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "Gender",
               "JobRole", "MaritalStatus", "OverTime"]
PROTECTED = ["Gender", "MaritalStatus"]

# Theory-based constructs, specified before the explanation analysis (Section 2.4 of the paper).
# Correlated clusters (pay and level, r = .95; tenure block, r = .71-.77) sit inside one construct.
CONSTRUCTS = {
    "Job demands": ["OverTime_Yes", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely",
                    "BusinessTravel_Non-Travel", "DistanceFromHome"],
    "Rewards (pay, level, equity)": ["MonthlyIncome", "JobLevel", "StockOptionLevel", "PercentSalaryHike"],
    "Work attitudes and climate": ["EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
                                   "JobInvolvement", "WorkLifeBalance"],
    "Embeddedness and career progress": ["YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager",
                                         "YearsSinceLastPromotion", "TrainingTimesLastYear", "PerformanceRating"],
    "Ease of movement": ["Age", "TotalWorkingYears", "NumCompaniesWorked", "Education"],
    "Role and structure": None,  # everything else: JobRole_*, Department_*, EducationField_*, rate fields
}

# Three controllability schemes (sensitivity analysis for reviewer point 1.3)
CTRL_SCHEMES = {
    "Baseline": {
        "Controllable": ["OverTime_Yes", "StockOptionLevel", "MonthlyIncome", "PercentSalaryHike",
                         "TrainingTimesLastYear", "BusinessTravel_Travel_Frequently",
                         "BusinessTravel_Travel_Rarely", "BusinessTravel_Non-Travel", "JobLevel",
                         "EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
                         "WorkLifeBalance", "JobInvolvement", "YearsWithCurrManager",
                         "YearsSinceLastPromotion", "PerformanceRating"]},
    "Conservative (pay, level and climate treated as not short-term controllable)": {
        "Controllable": ["OverTime_Yes", "StockOptionLevel", "TrainingTimesLastYear",
                         "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely",
                         "BusinessTravel_Non-Travel", "YearsWithCurrManager", "YearsSinceLastPromotion"]},
    "Liberal (adds role assignment and distance via relocation or remote work)": {
        "Controllable": ["OverTime_Yes", "StockOptionLevel", "MonthlyIncome", "PercentSalaryHike",
                         "TrainingTimesLastYear", "BusinessTravel_Travel_Frequently",
                         "BusinessTravel_Travel_Rarely", "BusinessTravel_Non-Travel", "JobLevel",
                         "EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
                         "WorkLifeBalance", "JobInvolvement", "YearsWithCurrManager",
                         "YearsSinceLastPromotion", "PerformanceRating", "DistanceFromHome", "JOBROLE*"]},
}

# Fixed XGBoost configuration for the screening step and the simulation study. It is declared
# a priori (a conventional shallow, slow-learning boosted model) and is NOT tuned on the data,
# so the screen cannot inherit information from the full sample through hyperparameter choice.
XGB_FIXED = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.85,
                 colsample_bytree=0.7, min_child_weight=3, reg_lambda=5,
                 eval_metric="logloss", tree_method="hist")


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig").drop(columns=DROP)
    y = (df.pop("Attrition") == "Yes").astype(int)
    return df, y


def encode(df, drop_protected=True):
    X = df.copy()
    if drop_protected:
        X = X.drop(columns=PROTECTED)
    cats = [c for c in CATEGORICAL if c in X.columns]
    X = pd.get_dummies(X, columns=cats, drop_first=False, dtype=float)
    for b in ["OverTime_No", "Gender_Female"]:
        if b in X.columns:
            X = X.drop(columns=b)
    return X


def construct_map(columns):
    m = {}
    for c in columns:
        for g, fs in CONSTRUCTS.items():
            if fs is not None and c in fs:
                m[c] = g
        m.setdefault(c, "Role and structure")
    return m


def ctrl_members(scheme, columns):
    fs = CTRL_SCHEMES[scheme]["Controllable"]
    out = [c for c in columns if c in fs]
    if "JOBROLE*" in fs:
        out += [c for c in columns if c.startswith("JobRole_")]
    return out


def shap_matrix(sv):
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def top_k_metrics(y, p, k):
    n = int(round(k * len(p)))
    idx = np.argsort(-p)[:n]
    return y[idx].mean(), y[idx].sum() / y.sum()


def calib_slope_intercept(y, p):
    from sklearn.linear_model import LogisticRegression
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0, 0]
    # calibration-in-the-large: intercept with slope fixed at 1 (offset model)
    import statsmodels.api as sm
    inter = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp.ravel()).fit().params[0]
    return float(slope), float(inter)


def segmented_breakpoint(x, s, grid):
    """Step change-point: the split b that minimises within-segment squared error of s
    (two means, below and at-or-above b). Matches the step-like shape of tree SHAP curves."""
    order = np.argsort(x)
    xs, ss = x[order], s[order]
    cs, cs2 = np.cumsum(ss), np.cumsum(ss ** 2)
    n = len(ss)
    best, bk = np.inf, None
    for b in grid:
        k = np.searchsorted(xs, b)  # rows with x < b
        if k < 20 or n - k < 20:
            continue
        sse = (cs2[k - 1] - cs[k - 1] ** 2 / k) + ((cs2[-1] - cs2[k - 1]) - (cs[-1] - cs[k - 1]) ** 2 / (n - k))
        if sse < best:
            best, bk = sse, b
    return bk


def step_vs_linear_bic(x, s, grid):
    """BIC(linear) - BIC(step) for SHAP values s on feature x. Positive values favour a single
    step (threshold) over a straight line; > 10 is conventionally strong evidence."""
    n = len(s)
    Z = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(Z, s, rcond=None)
    sse_lin = ((s - Z @ coef) ** 2).sum()
    b = segmented_breakpoint(x, s, grid)
    lo = x < b
    sse_step = ((s[lo] - s[lo].mean()) ** 2).sum() + ((s[~lo] - s[~lo].mean()) ** 2).sum()
    bic = lambda sse, k: n * np.log(sse / n) + k * np.log(n)
    return float(bic(sse_lin, 2) - bic(sse_step, 3)), b


def linstep_fit(x, s, grid):
    """Threshold test against a flexible smooth baseline. Baseline: cubic polynomial in standardised
    log(x) (any smooth monotone or curved effect). Alternative: baseline + one step 1[x < c].
    Returns (c, step_size, delta_bic) with delta_bic = BIC(smooth) - BIC(smooth + step); > 10 is
    strong evidence of a threshold that a smooth curve cannot reproduce."""
    n = len(s)
    lx = np.log(np.maximum(x, 1e-9)) if x.min() > 0 else x
    z = (lx - lx.mean()) / lx.std()
    base = np.column_stack([np.ones(n), z, z ** 2, z ** 3])

    def sse_coef(Z):
        c_, *_ = np.linalg.lstsq(Z, s, rcond=None)
        return ((s - Z @ c_) ** 2).sum(), c_
    sse0, _ = sse_coef(base)
    best = (np.inf, None, None)
    for c in grid:
        k = (x < c).sum()
        if k < 20 or n - k < 20:
            continue
        sse, coef = sse_coef(np.column_stack([base, (x < c).astype(float)]))
        if sse < best[0]:
            best = (sse, c, coef[-1])
    bic = lambda sse, k: n * np.log(sse / n) + k * np.log(n)
    return float(best[1]), float(best[2]), float(bic(sse0, 4) - bic(best[0], 6))
