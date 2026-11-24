"""
Monte Carlo (probabilistic sensitivity) analysis of risk-ranked retention targeting (reviewer point 3.1).

Inputs: cross-fitted calibrated scores (oof_predictions.csv) for all 1,470 employees.
Each of M = 10,000 draws samples
    replacement cost share  c ~ Triangular(0.10, 0.21, 0.50)   mode = median in Boushey & Glynn (2012)
    intervention success    s ~ Beta(6, 14)                    mean 0.30, SD about 0.10
    cost per intervention   m ~ Uniform(500, 3,000) USD
and a bootstrap resample of employees (sampling uncertainty). Net saving for the top k% is
    sum_{i in top k} y_i * pay_i * c * s  -  m * |top k|,   reported per 100 employees.
Break-even success rate at fixed c = 0.21, m = 1,500 is also reported.

Version-3 variants (mc_variants.csv):
  base                 as above
  harder_high_risk     success falls with risk: s_i = s * (1 - 0.5 * p_i)
  senior_costlier      replacement cost multiplied by 0.75 (job levels 1-2), 1.0 (level 3), 2.0 (levels 4-5)
  both                 the two together
Results are also expressed in months of median monthly pay per 100 employees (currency-free).

Usage: python monte_carlo_cost.py --data ../data/WA_Fn-UseC_-HR-Employee-Attrition.csv --out ../results
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SEED

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300})
M = 10_000
KS = np.round(np.arange(0, 0.51, 0.01), 2)


def main(data, out):
    raw = pd.read_csv(data, encoding="utf-8-sig")
    pay = raw.MonthlyIncome.values * 12.0
    oof = pd.read_csv(os.path.join(out, "oof_predictions.csv"))
    y = oof.y.values
    assert np.array_equal(y, (raw.Attrition == "Yes").astype(int).values)
    n = len(y)
    rng = np.random.default_rng(SEED)
    c = rng.triangular(0.10, 0.21, 0.50, M)
    s = rng.beta(6, 14, M)
    m = rng.uniform(500, 3000, M)
    kpos = np.round(KS * n).astype(int)
    models = {"XGB": oof.XGB_cal.values, "LR": oof.LR_cal.values}
    nets = {k: np.zeros((M, len(KS))) for k in list(models) + ["Random"]}
    for d in range(M):
        idx = rng.integers(0, n, n)
        yp = y[idx] * pay[idx]
        for name, sc in models.items():
            o = np.argsort(-sc[idx], kind="stable")
            cum = np.concatenate([[0], np.cumsum(yp[o])])
            nets[name][d] = (cum[kpos] * c[d] * s[d] - m[d] * kpos) / n * 100
        nets["Random"][d] = (kpos * yp.mean() * c[d] * s[d] - m[d] * kpos) / n * 100

    rows = []
    for name, N in nets.items():
        for j, k in enumerate(KS):
            v = N[:, j]
            rows.append({"model": name, "share_targeted": k, "P_net_positive": (v > 0).mean() if k > 0 else np.nan,
                         "median": np.median(v), "q05": np.percentile(v, 5), "q95": np.percentile(v, 95),
                         "mean": v.mean()})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "mc_cost_by_share.csv"), index=False)

    opt = {name: KS[np.argmax(N, axis=1)] for name, N in nets.items() if name != "Random"}
    best = {name: N.max(1) for name, N in nets.items() if name != "Random"}
    diff = nets["XGB"] - nets["LR"]
    summ = {"M": M}
    for name in ["XGB", "LR"]:
        summ[name] = {"optimal_share_median": float(np.median(opt[name])),
                      "optimal_share_q05": float(np.percentile(opt[name], 5)),
                      "optimal_share_q95": float(np.percentile(opt[name], 95)),
                      "P_some_list_profitable": float((best[name] > 0).mean()),
                      "best_net_median": float(np.median(best[name])),
                      "best_net_q05": float(np.percentile(best[name], 5)),
                      "best_net_q95": float(np.percentile(best[name], 95))}
        for k in (0.05, 0.10, 0.20, 0.50):
            r = res[(res.model == name) & (res.share_targeted == k)].iloc[0]
            summ[name][f"k{int(k * 100)}"] = {"P_pos": float(r.P_net_positive), "median": float(r["median"]),
                                              "q05": float(r.q05), "q95": float(r.q95)}
    for k in (0.05, 0.10):
        j = list(KS).index(k)
        summ[f"XGB_minus_LR_k{int(k * 100)}"] = {"median": float(np.median(diff[:, j])),
                                                 "q05": float(np.percentile(diff[:, j], 5)),
                                                 "q95": float(np.percentile(diff[:, j], 95)),
                                                 "P_XGB_better": float((diff[:, j] > 0).mean())}
    r = res[(res.model == "Random")]
    summ["Random_P_pos_any_share"] = float(r.P_net_positive.max())
    full = (y * pay).sum()
    summ["target_all_net_per100_basecase"] = float((full * 0.21 * 0.30 - 1500 * n) / n * 100)

    # break-even success rate, base-case c and m
    be = []
    for name, sc in models.items():
        o = np.argsort(-sc, kind="stable")
        cum = np.cumsum((y * pay)[o])
        for k in KS[1:]:
            kk = int(round(k * n))
            be.append({"model": name, "share_targeted": k, "leavers_captured": y[o[:kk]].sum() / y.sum(),
                       "precision": y[o[:kk]].mean(),
                       "breakeven_success_rate": 1500 * kk / (0.21 * cum[kk - 1])})
    be.append({"model": "Random", "share_targeted": np.nan, "leavers_captured": np.nan, "precision": y.mean(),
               "breakeven_success_rate": 1500 / (0.21 * (y * pay).mean())})
    be = pd.DataFrame(be)
    be.to_csv(os.path.join(out, "mc_breakeven.csv"), index=False)
    for name in models:
        for k in (0.05, 0.10, 0.20):
            r = be[(be.model == name) & (np.isclose(be.share_targeted, k))].iloc[0]
            summ[name][f"breakeven_k{int(k * 100)}"] = float(r.breakeven_success_rate)
            summ[name][f"captured_k{int(k * 100)}"] = float(r.leavers_captured)
            summ[name][f"precision_k{int(k * 100)}"] = float(r.precision)
    summ["Random_breakeven"] = float(be[be.model == "Random"].breakeven_success_rate.iloc[0])
    # ---- version-3 variants and currency-free units
    med_month = float(np.median(raw.MonthlyIncome.values))
    lvl = raw.JobLevel.values
    cmult = np.where(lvl <= 2, 0.75, np.where(lvl == 3, 1.0, 2.0))
    rng2 = np.random.default_rng(SEED + 1)
    c2, s2, m2 = rng2.triangular(0.10, 0.21, 0.50, M), rng2.beta(6, 14, M), rng2.uniform(500, 3000, M)
    vrows = []
    for vname in ["base", "harder_high_risk", "senior_costlier", "both"]:
        for mname, sc in models.items():
            smult = (1 - 0.5 * sc) if vname in ("harder_high_risk", "both") else np.ones(n)
            cm = cmult if vname in ("senior_costlier", "both") else np.ones(n)
            V = np.zeros((M, len(KS)))
            for d in range(M):
                idx = rng2.integers(0, n, n)
                o = np.argsort(-sc[idx], kind="stable")
                w = (y * pay * cm * smult)[idx][o]
                cum = np.concatenate([[0], np.cumsum(w)])
                V[d] = (cum[kpos] * c2[d] * s2[d] - m2[d] * kpos) / n * 100
            for k in (0.05, 0.10, 0.20, 0.50):
                j = list(KS).index(k)
                vrows.append({"variant": vname, "model": mname, "share_targeted": k, "P_net_positive": (V[:, j] > 0).mean(),
                              "median_usd": np.median(V[:, j]), "median_months_pay": np.median(V[:, j]) / med_month,
                              "q05_months_pay": np.percentile(V[:, j], 5) / med_month,
                              "q95_months_pay": np.percentile(V[:, j], 95) / med_month,
                              "optimal_share_median": float(np.median(KS[np.argmax(V, 1)]))})
    var = pd.DataFrame(vrows)
    var.to_csv(os.path.join(out, "mc_variants.csv"), index=False)
    summ["median_monthly_pay"] = med_month
    summ["variants_P_pos_k5"] = {f"{r.variant}|{r.model}": float(r.P_net_positive)
                                 for r in var[var.share_targeted == 0.05].itertuples()}
    print(var.round(3).to_string())
    json.dump(summ, open(os.path.join(out, "mc_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    sty = {"XGB": ("-", "black"), "LR": ("--", "black"), "Random": (":", "grey")}
    for name in ["XGB", "LR", "Random"]:
        r = res[(res.model == name) & (res.share_targeted > 0)]
        ax[0].plot(r.share_targeted * 100, r.P_net_positive, sty[name][0], color=sty[name][1], label=name)
    ax[0].axhline(0.5, color="grey", lw=0.5)
    ax[0].set(xlabel="Share of employees targeted (%)", ylabel="Probability that net saving > 0",
              title="(a) Probability the programme pays off", ylim=(0, 1))
    ax[0].legend(frameon=False, fontsize=8)
    for name in ["XGB", "LR"]:
        r = res[res.model == name]
        ax[1].plot(r.share_targeted * 100, r["median"] / 1000, sty[name][0], color="black", label=f"{name} median")
    r = res[res.model == "XGB"]
    ax[1].fill_between(r.share_targeted * 100, r.q05 / 1000, r.q95 / 1000, color="#cfcfcf", label="XGB 90% interval")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set(xlabel="Share of employees targeted (%)", ylabel="Net saving per 100 employees (USD thousand)",
              title="(b) Net saving under joint uncertainty")
    ax[1].legend(frameon=False, fontsize=8)
    for name in ["XGB", "LR"]:
        r = be[be.model == name]
        ax[2].plot(r.share_targeted * 100, r.breakeven_success_rate * 100, sty[name][0], color="black", label=name)
    ax[2].axhline(summ["Random_breakeven"] * 100, color="grey", ls=":", label="Random targeting")
    ax[2].axhline(30, color="grey", lw=0.5)
    ax[2].set(xlabel="Share of employees targeted (%)", ylabel="Break-even intervention success rate (%)",
              title="(c) Success rate needed to break even", ylim=(0, 100))
    ax[2].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    plt.savefig(os.path.join(out, "figures", "Fig_monte_carlo_cost.png"))
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    main(a.data, a.out)
