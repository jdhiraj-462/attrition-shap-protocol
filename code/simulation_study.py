"""
Known-truth simulation study (reviewer point 2.4).

Question: does the four-step protocol (cross-fitted SHAP, construct grouping, shadow-feature
null screen, bootstrap stability) recover true attrition drivers and reject spurious ones
when predictors are as correlated as real HR data?

Design
  * Covariates: rows resampled with replacement from the real IBM file, so every correlation
    (income-job level r = .95, tenure block, age-experience) is preserved.
  * Outcome: Bernoulli draw from a known logit built from turnover theory:
        eta = b0 + 1.0*OverTime + 1.0*LowPay + 0.9*OverTime*LowPay + 0.8*NoStock
              + 0.6*Under30 + 0.5*FiveOrMorePriorEmployers + 0.4*TravelFrequent
              - 0.3*(EnvironmentSatisfaction - 2.5)
    LowPay = MonthlyIncome < 2,700. b0 is solved so that expected attrition is 16.1%.
  * True drivers (7 features): OverTime_Yes, MonthlyIncome, StockOptionLevel, Age,
    NumCompaniesWorked, BusinessTravel_Travel_Frequently, EnvironmentSatisfaction.
    All other 38 features are null, including JobLevel (r = .95 with income),
    TotalWorkingYears (r = .68 with age) and DailyRate.
  * Scenarios (version 3):
      base     the model above (step at 2,700, interaction)            n = 1,470 (R = 30) and 5,000 (R = 12)
      smooth   pay enters as -0.9*log(income / median); no step, no interaction    n = 1,470 (R = 30)
      weak     every coefficient halved                                 n = 1,470 (R = 30)
      shifted  step and interaction at USD 4,000 instead of 2,700       n = 1,470 (R = 30)
  * Per replication: 5-fold cross-fitted XGBoost and LR SHAP; B = 20 bootstrap refits with shadow
    features using the same fixed configuration as the empirical screen (common.XGB_FIXED);
    interaction values; linear-plus-step threshold model for income with a BIC test against linear.

Selection rules compared
  naive_top7_XGB / naive_top10_XGB / naive_top10_LR : rank by mean |SHAP|
  screen_XGB_70 / _80 / _90 : beats the largest shadow feature AND ranks in the top 10 in at least
                              70%, 80% or 90% of refits

Usage: python simulation_study.py --data ../data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out ../results
Runtime: roughly 25-40 minutes on 2 cores (use --quick for a 2-minute check).
"""
import argparse
import json
import os
import warnings
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import SEED, XGB_FIXED, encode, linstep_fit, load, shap_matrix

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300})
TRUE = ["OverTime_Yes", "MonthlyIncome", "StockOptionLevel", "Age", "NumCompaniesWorked",
        "BusinessTravel_Travel_Frequently", "EnvironmentSatisfaction"]
DECOYS = ["JobLevel", "TotalWorkingYears", "DailyRate"]
CFG = XGB_FIXED
SCEN = {"base": dict(scale=1.0, thr=2700, smooth=False), "smooth": dict(scale=1.0, thr=None, smooth=True),
        "weak": dict(scale=0.5, thr=2700, smooth=False), "shifted": dict(scale=1.0, thr=4000, smooth=False)}
MED_INC = 4919.0
X_BASE = None


def linpred(X, b0=0.0, scen="base"):
    sc = SCEN[scen]
    k = sc["scale"]
    ot = X["OverTime_Yes"].values
    inc = X["MonthlyIncome"].values
    if sc["smooth"]:
        pay = -0.9 * np.log(inc / MED_INC)
        inter = 0.0
    else:
        low = (inc < sc["thr"]).astype(float)
        pay = 1.0 * low
        inter = 0.9 * ot * low
    return (b0 + k * (1.0 * ot + pay + inter + 0.8 * (X["StockOptionLevel"].values == 0)
                      + 0.6 * (X["Age"].values < 30) + 0.5 * (X["NumCompaniesWorked"].values >= 5)
                      + 0.4 * X["BusinessTravel_Travel_Frequently"].values
                      - 0.3 * (X["EnvironmentSatisfaction"].values - 2.5)))


def solve_b0(X, scen, target=0.161):
    return brentq(lambda b: (1 / (1 + np.exp(-linpred(X, b, scen)))).mean() - target, -15, 10)


def one_rep(args):
    rep, n, B, b0, scen = args
    r = np.random.default_rng(SEED + 10_000 * n + rep + 1_000_000 * list(SCEN).index(scen))
    X = X_BASE.iloc[r.integers(0, len(X_BASE), n)].reset_index(drop=True)
    p_true = 1 / (1 + np.exp(-linpred(X, b0, scen)))
    y = r.binomial(1, p_true)
    cols = list(X.columns)
    spw = (y == 0).sum() / max(y.sum(), 1)

    # cross-fitted SHAP
    sv_x, sv_l = np.zeros(X.shape), np.zeros(X.shape)
    oof = np.zeros(n)
    inter = np.zeros((len(cols), len(cols)))
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y)):
        m = XGBClassifier(**CFG, scale_pos_weight=spw, random_state=rep, n_jobs=1).fit(X.iloc[tr], y[tr])
        ex = shap.TreeExplainer(m)
        sv_x[te] = shap_matrix(ex.shap_values(X.iloc[te]))
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        sub = te[:300]
        iv = np.asarray(ex.shap_interaction_values(X.iloc[sub]))
        inter += np.abs(iv).sum(0)
        sc = StandardScaler().fit(X.iloc[tr])
        lr = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(sc.transform(X.iloc[tr]), y[tr])
        Xs_tr = pd.DataFrame(sc.transform(X.iloc[tr]), columns=cols)
        sv_l[te] = np.asarray(shap.LinearExplainer(lr, shap.maskers.Independent(Xs_tr, 100))
                              .shap_values(pd.DataFrame(sc.transform(X.iloc[te]), columns=cols)))
    imp_x = pd.Series(np.abs(sv_x).mean(0), index=cols)
    imp_l = pd.Series(np.abs(sv_l).mean(0), index=cols)

    # shadow bootstrap
    hits, top10 = np.zeros(len(cols)), np.zeros(len(cols))
    for b in range(B):
        rb = np.random.default_rng(SEED + 7 * rep + 1000 * b + n)
        idx = rb.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        sh = lambda Z: pd.concat([Z, Z.apply(lambda c: rb.permutation(c.values)).add_prefix("shadow__")], axis=1)
        Xb, Xo = sh(X.iloc[idx].reset_index(drop=True)), sh(X.iloc[oob].reset_index(drop=True))
        m = XGBClassifier(**CFG, scale_pos_weight=spw, random_state=b, n_jobs=1).fit(Xb, y[idx])
        a = np.abs(shap_matrix(shap.TreeExplainer(m).shap_values(Xo))).mean(0)
        real, shad = a[:len(cols)], a[len(cols):]
        hits += real > shad.max()
        top10 += pd.Series(real).rank(ascending=False).values <= 10
    hit_share, top_share = hits / B, top10 / B
    sel = {"naive_top7_XGB": list(imp_x.nlargest(7).index), "naive_top10_XGB": list(imp_x.nlargest(10).index),
           "naive_top10_LR": list(imp_l.nlargest(10).index)}
    for cut in (0.7, 0.8, 0.9):
        sel[f"screen_XGB_{int(cut * 100)}"] = [c for c, h, t in zip(cols, hit_share, top_share) if h >= cut and t >= cut]
    res = {"scenario": scen, "rep": rep, "n": n, "prevalence": y.mean(), "auc_oof": roc_auc_score(y, oof),
           "auc_true": roc_auc_score(y, p_true)}
    for k, s in sel.items():
        tp = len(set(s) & set(TRUE))
        res[f"{k}_TPR"] = tp / len(TRUE)
        res[f"{k}_FDR"] = (len(s) - tp) / max(len(s), 1)
        res[f"{k}_size"] = len(s)
        for d in DECOYS:
            res[f"{k}_selects_{d}"] = int(d in s)
    # credit leakage: share of the pay-and-level pair credited to JobLevel
    res["jobLevel_share_of_pay_pair"] = imp_x["JobLevel"] / (imp_x["JobLevel"] + imp_x["MonthlyIncome"])
    # interaction recovery
    np.fill_diagonal(inter, 0)
    I = pd.DataFrame(inter, index=cols, columns=cols)
    pairs = I.where(np.triu(np.ones(I.shape), 1).astype(bool)).stack().sort_values(ascending=False)
    keys = [tuple(sorted(k)) for k in pairs.index]
    target = tuple(sorted(("OverTime_Yes", "MonthlyIncome")))
    res["interaction_rank_OTxIncome"] = keys.index(target) + 1
    # threshold recovery
    j = cols.index("MonthlyIncome")
    c_, d_, dbic = linstep_fit(X["MonthlyIncome"].values.astype(float), sv_x[:, j], np.arange(1500, 8001, 50))
    res["income_breakpoint"], res["step_size"], res["step_dbic"] = c_, d_, dbic
    res["step_detected"] = int(dbic > 10 and d_ > 0)
    res["hit_share"] = dict(zip(cols, hit_share.round(3)))
    return res


def init(base):
    global X_BASE
    X_BASE = base


def main(data, out, quick=False):
    raw, _ = load(data)
    X = encode(raw)
    conds = [("base", 1470, 30), ("base", 5000, 12), ("smooth", 1470, 30), ("weak", 1470, 30), ("shifted", 1470, 30)]
    if quick:
        conds = [(sc, n, 2) for sc, n, _ in conds]
    B = 3 if quick else 20
    b0 = {sc: solve_b0(X, sc) for sc in SCEN}
    jobs = [(rep, n, B, b0[sc], sc) for sc, n, R in conds for rep in range(R)]
    ckpt = os.path.join(out, "simulation_checkpoint.jsonl")
    results = []
    if os.path.exists(ckpt) and not quick:   # resume after interruption
        results = [json.loads(l) for l in open(ckpt)]
        done = {(r["scenario"], r["n"], r["rep"]) for r in results}
        jobs = [j for j in jobs if (j[4], j[1], j[0]) not in done]
    with Pool(processes=max(1, os.cpu_count() or 1), initializer=init, initargs=(X,)) as pool, open(ckpt, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(one_rep, jobs)):
            r = json.loads(json.dumps(r, default=float))
            results.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(jobs)} replications", flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "hit_share"} for r in results]).sort_values(["scenario", "n", "rep"])
    df.to_csv(os.path.join(out, "simulation_replications.csv"), index=False)
    hs = pd.DataFrame([{"scenario": r["scenario"], "n": r["n"], **r["hit_share"]} for r in results]) \
        .groupby(["scenario", "n"]).mean().T
    hs.to_csv(os.path.join(out, "simulation_shadow_hit_share_by_feature.csv"))

    rules = ["naive_top7_XGB", "naive_top10_XGB", "naive_top10_LR", "screen_XGB_70", "screen_XGB_80", "screen_XGB_90"]
    rows = []
    for (sc, n), g in df.groupby(["scenario", "n"]):
        for rl in rules:
            rows.append({"scenario": sc, "n": n, "rule": rl, "replications": len(g), "TPR": g[f"{rl}_TPR"].mean(),
                         "FDR": g[f"{rl}_FDR"].mean(), "mean_size": g[f"{rl}_size"].mean(),
                         **{f"selects_{d}": g[f"{rl}_selects_{d}"].mean() for d in DECOYS}})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out, "simulation_summary.csv"), index=False)
    extra = df.groupby(["scenario", "n"]).agg(
        replications=("rep", "size"), prevalence=("prevalence", "mean"), auc_oof=("auc_oof", "mean"),
        auc_true=("auc_true", "mean"), jobLevel_share=("jobLevel_share_of_pay_pair", "mean"),
        inter_rank_median=("interaction_rank_OTxIncome", "median"),
        inter_rank_top1=("interaction_rank_OTxIncome", lambda s: (s == 1).mean()),
        inter_rank_top3=("interaction_rank_OTxIncome", lambda s: (s <= 3).mean()),
        step_detected=("step_detected", "mean"), bp_median=("income_breakpoint", "median"),
        bp_q05=("income_breakpoint", lambda s: s.quantile(0.05)),
        bp_q95=("income_breakpoint", lambda s: s.quantile(0.95)))
    extra.to_csv(os.path.join(out, "simulation_recovery.csv"))
    json.dump({"b0": b0, "B": B, "conditions": conds, "true_drivers": TRUE, "decoys": DECOYS, "config": CFG},
              open(os.path.join(out, "simulation_design.json"), "w"), indent=2)
    print(summ.round(3).to_string())
    print(extra.round(3).to_string())

    plot_sim(summ, out)


def plot_sim(summ, out):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    order = [("base", 1470), ("base", 5000), ("weak", 1470), ("smooth", 1470), ("shifted", 1470)]
    labs = {"naive_top10_XGB": "Top 10, XGB", "naive_top10_LR": "Top 10, LR", "screen_XGB_70": "Screen 70%",
            "screen_XGB_80": "Screen 80%", "screen_XGB_90": "Screen 90%"}
    cols_ = ["#e0e0e0", "#b0b0b0", "#808080", "#505050", "#202020"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    xs = np.arange(len(labs))
    w = 0.16
    for k, ((sc, n), col) in enumerate(zip(order, cols_)):
        s_ = summ[(summ.scenario == sc) & (summ.n == n)].set_index("rule").loc[list(labs)]
        lab = f"{sc}, n = {n:,}"
        ax[0].bar(xs + (k - 2) * w, s_.TPR, w, color=col, edgecolor="black", lw=0.3, label=lab)
        ax[1].bar(xs + (k - 2) * w, s_.FDR, w, color=col, edgecolor="black", lw=0.3, label=lab)
    for a, t in zip(ax, ["(a) True drivers recovered (TPR)", "(b) Share of selected features that are null (FDR)"]):
        a.set_xticks(xs)
        a.set_xticklabels(list(labs.values()), fontsize=8)
        a.set_ylim(0, 1)
        a.set_title(t, fontsize=10)
    ax[1].legend(frameon=False, fontsize=7.5, ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "Fig_simulation_recovery.png"), dpi=300)
    plt.close()




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw Figure 7 from simulation_summary.csv")
    a = ap.parse_args()
    if a.plot_only:
        plot_sim(pd.read_csv(os.path.join(a.out, "simulation_summary.csv")), a.out)
    else:
        main(a.data, a.out, a.quick)
