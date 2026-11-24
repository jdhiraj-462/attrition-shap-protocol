"""Figure 1: the four-step protocol and its validation."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 300})

STEPS = [
    ("Step 1  Predict",
     "LR, RF, XGBoost, LightGBM\nNested CV + corrected t-tests\nCross-fitted, Platt-calibrated\nscores for every employee"),
    ("Step 2  Explain",
     "Cross-fitted SHAP (tree, linear)\nPath-dependent vs interventional\nTheory-based construct grouping\nfor correlated clusters"),
    ("Step 3  Screen",
     "50 bootstrap refits with\nshadow (permuted) features\nKeep: beats all shadows AND\nin top 10, each in >= 80% of refits"),
    ("Step 4  Decide",
     "Interaction tests (OR, RERI)\nFairness: flag rate, TPR, calibration\nMonte Carlo cost, 10,000 draws\nBreak-even success rate"),
]


def main(out):
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.38, 1)
    ax.axis("off")
    for i, (h, txt) in enumerate(STEPS):
        ax.add_patch(FancyBboxPatch((i + 0.03, 0.06), 0.84, 0.88, boxstyle="round,pad=0.02",
                                    fc="#f2f2f2" if i % 2 == 0 else "white", ec="black", lw=0.9))
        ax.text(i + 0.45, 0.80, h, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(i + 0.45, 0.42, txt, ha="center", va="center", fontsize=7.6, linespacing=1.45)
        if i < 3:
            ax.annotate("", xy=(i + 1.03, 0.5), xytext=(i + 0.89, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1))
    ax.add_patch(FancyBboxPatch((1.03, -0.34), 1.84, 0.26, boxstyle="round,pad=0.02", fc="white",
                                ec="black", lw=0.8, ls="--"))
    ax.text(1.95, -0.21, "Validation: known-truth simulation (resampled real covariates, 7 true drivers,\n"
                         "38 null incl. correlated decoys; n = 1,470 and 5,000) tests Steps 2-3",
            ha="center", va="center", fontsize=7.6)
    ax.text(0.45, -0.21, "Input: IBM HR Analytics\n1,470 employees, 45 predictors", ha="center", va="center", fontsize=7.6)
    ax.text(3.45, -0.21, "Output: screened, costed\nretention list + audit", ha="center", va="center", fontsize=7.6)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    plt.savefig(os.path.join(out, "figures", "Figure1_study_design.png"), bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results")
    main(ap.parse_args().out)
