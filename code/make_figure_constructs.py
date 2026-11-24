"""Figure 4 (version 3): construct-level shares with 95% refit intervals from the fixed-configuration
screen (screen_fixed_constructs.csv), plus LR shares from cross-fitted linear SHAP.
Usage: python make_figure_constructs.py --out ../results"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300})


def main(out):
    g = pd.read_csv(os.path.join(out, "screen_fixed_constructs.csv"))
    lr = pd.read_csv(os.path.join(out, "crossfit_construct_importance.csv")).set_index("construct")
    g["share_LR"] = g.construct.map(lr.share_LR)
    g["n"] = g.construct.map(lr.n_features)
    g = g.sort_values("share_median")
    yy = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(yy + 0.2, g.share_median, 0.4, color="#3a3a3a", label="XGBoost (median of 50 refits)")
    ax.errorbar(g.share_median, yy + 0.2, xerr=[g.share_median - g.share_lo, g.share_hi - g.share_median],
                fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.barh(yy - 0.2, g.share_LR, 0.4, color="#b0b0b0", label="Logistic regression (cross-fitted)")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{c} ({int(k)})" for c, k in zip(g.construct, g.n)])
    ax.set_xlabel("Share of construct-level mean |SHAP|")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "Fig_construct_importance.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results")
    main(ap.parse_args().out)
