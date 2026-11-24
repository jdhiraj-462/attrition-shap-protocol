"""Figure 9: Study 2 (US federal workforce) summary.
(a) construct shares of mean |SHAP|; (b) SHAP by decile for service, age and pay; (c) precision of risk-ranked lists.
Usage: python make_figure_study2.py ../results_v3/study2"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "../results/study2"
d = json.load(open(os.path.join(R, "s2_summary.json")))
g = pd.read_csv(os.path.join(R, "s2_constructs.csv")).sort_values("share_XGB")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.25, 1.1, 0.9]})

yy = np.arange(len(g))
ax[0].barh(yy + 0.2, g.share_XGB, 0.4, color="#3a3a3a", label="XGBoost")
ax[0].barh(yy - 0.2, g.share_LR, 0.4, color="#b0b0b0", label="Logistic regression")
ax[0].set_yticks(yy)
ax[0].set_yticklabels([c.replace(" (pay, grade, step)", "") for c in g.construct])
ax[0].set_xlabel("Share of construct-level mean |SHAP|")
ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
ax[0].set_title("(a) Constructs", loc="left", fontsize=9)

dep = d["dependence_deciles"]
for f, lab, st in [("los_years", "Length of service", "-"), ("age_mid", "Age", "--"), ("log_salary", "Pay (unredacted)", ":")]:
    v = pd.DataFrame(dep[f])
    ax[1].plot(np.arange(1, 11), v.shap, st, marker="o", ms=3, color="#222222", label=lab)
ax[1].axhline(0, color="#999999", lw=0.6)
ax[1].set_xticks(range(1, 11))
ax[1].set_xlabel("Decile of predictor (1 = lowest)")
ax[1].set_ylabel("Mean SHAP, log-odds")
ax[1].legend(frameon=False, fontsize=7.5)
ax[1].set_title("(b) Dependence by decile (XGBoost)", loc="left", fontsize=9)

tk = d["top_k_population"]
ks = ["top_5pct", "top_10pct", "top_20pct"]
prec = [tk[k]["precision"] * 100 for k in ks]
ax[2].bar([5, 10, 20], prec, width=3, color="#3a3a3a")
ax[2].axhline(d["crude_rate_analysis"] * 100, color="#777777", ls="--", lw=0.9)
ax[2].text(15, d["crude_rate_analysis"] * 100 + 0.4, "base rate", fontsize=7, ha="center", color="#555555")
for x, p_, k in zip([5, 10, 20], prec, ks):
    ax[2].text(x, p_ + 0.4, f"{p_:.1f}%\nrecall {tk[k]['recall'] * 100:.0f}%", ha="center", fontsize=7)
ax[2].set_xticks([5, 10, 20])
ax[2].set_xlabel("Share of workforce listed (%)")
ax[2].set_ylabel("Quitters among listed (%)")
ax[2].set_ylim(0, max(prec) * 1.35)
ax[2].set_title("(c) Risk-ranked lists", loc="left", fontsize=9)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
os.makedirs(os.path.join(R, "figures"), exist_ok=True)
plt.savefig(os.path.join(R, "figures", "Fig_study2_summary.png"), dpi=300)
print("saved")
