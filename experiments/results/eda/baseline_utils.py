import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches, rcParams
import seaborn as sns
from scipy.stats import wilcoxon
import scikit_posthocs as sp


rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
})

C_MRSQM   = "#E69F00"   # amber
C_HYDRA   = "#0072B2"   # blue
C_POS     = "#009E73"   # green  (our result > reported)
C_NEG     = "#D55E00"   # red    (our result < reported)
C_NEUTRAL = "#999999"   # grey

C_LR  = "#CC79A7"  # purple (distinct, readable)
C_RF  = "#56B4E9"  # light blue (different from Hydra blue)
C_DTW = "#F0E442"  # yellow (stands out but still soft)

colour_map = {
    "MRSQM": C_MRSQM,
    "Hydra": C_HYDRA,
    "LR": C_LR,
    "RF": C_RF,
    "DTW": C_DTW,
}

legend_labels ={
    "dtw_1nn": "DTW",
    "lr_classifier": "LR",
    "rf_classifier": "RF"
}

OUTPUT_DIR = "../figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath: str = "baseline_summary.csv") -> pd.DataFrame:
    """
    Reads baseline results from a CSV file and returns a DataFrame.
    Expected columns: "model", "accuracy", "precision", "recall", "f1_score"
    """
    df = pd.read_csv(filepath)
    df.drop_duplicates(subset=["dataset", "model"], keep="last", inplace=True)
    print(f"[read_baseline_results] Loaded {len(df)} rows from {filepath}")
    return df


def plot_baseline_distributions(df: pd.DataFrame,
                               metric_cols: dict,
                               title: str,
                               xlabel: str,
                               filename: str, 
                               clip_range=None,
                               use_log_scale=False) -> str:
    """
    Overlapping KDE plots for baseline models.

    metric_cols: dict like {
        "LR": "lr_accuracy",
        "RF": "rf_accuracy",
        "DTW": "dtw_accuracy",
    }
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for label, col in metric_cols.items():
        sns.kdeplot(
            df[col],
            ax=ax,
            label=label,
            color=colour_map.get(label),
            clip=clip_range,
            bw_adjust=0.5,
            linewidth=1.8,
            fill=True,
            alpha=0.25
        )
    if use_log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=True)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.show()
    plt.close(fig)

    print(f"[plot_baseline_distributions] Saved -> {path}")
    return path


def plot_accuracy_latency_scatter(df: pd.DataFrame,
                                  filename: str = "baseline_accuracy_vs_latency.png"):
    """
    Single scatter plot:
      X-axis: Latency
      Y-axis: Accuracy
      Colour: Model (LR, RF, DTW, HYDRA, MrSQM, etc.)

    Assumes dataframe is in long format with columns:
      - 'accuracy'
      - 'total_time'
      - 'model'
    """

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Lists to store custom legend handles and labels
    custom_handles = []
    custom_labels = []

    # Plot each model separately for colour control
    for label, group in df.groupby("model"):
        legend_label = legend_labels.get(label, label)
        c = colour_map.get(legend_label)

        ax.scatter(
            group["accuracy"],
            group["total_time"],
            s=12,
            alpha=0.85,
            edgecolors="none",
            color=c,
            linewidth=1.8,
        )

        # Create a rectangular patch for the legend to match the KDE plot
        patch = patches.Patch(color=c, alpha=0.85, label=legend_label)
        custom_handles.append(patch)
        custom_labels.append(legend_label)

    ax.set_title("Baseline Model Accuracy vs Latency")

    # Labels
    ax.set_ylabel("Latency (seconds)")
    ax.set_xlabel("Accuracy")

    # Log scale for latency (VERY important)
    ax.set_yscale("log")

    # Optional: tighten accuracy bounds
    ax.set_xlim(0, 1.05)

    # Custom Legend: passing the handles and labels, and setting frameon=True for the box
    ax.legend(
        handles=custom_handles,
        labels=custom_labels,
        frameon=True,  # Turns the bounding box around the legend ON
        loc="upper left"
    )

    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.show()
    plt.close(fig)

    print(f"[plot_accuracy_latency_scatter] Saved -> {path}")
    return path