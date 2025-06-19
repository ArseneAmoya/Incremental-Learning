"""
cil_plots.py  –  Utility functions for class-incremental-learning dashboards
Five methods:
    1. Baseline-Finetune
    2. iCaRL
    3. Baseline-NoFinetune
    4. Corr-Dist
    5. Custom-CorrDist
The CSV layouts follow the ones you shared (70 epochs × 10 tasks).
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# 0.  Configuration – edit here if your paths are different
# -------------------------------------------------------------------------
ROOT = Path("output")        # directory where all CSVs live

FILES = {
    # "Baseline-Finetune": {
    #     "metrics": ROOT / "metrics_baseline.csv",
    #     #"losses":  ROOT / "losses_base_line.csv",
    #     "time":    ROOT / "time_baseline.csv",
    # },
    "iCaRL": {
        "metrics": ROOT / "icarl_metrics_cub.csv",
        #"losses":  ROOT / "losses_icarl.csv",
        "time":    ROOT / "icarl_metrics_cub.csv",
    },
    "Baseline-NoFinetune": {
        "metrics": ROOT / "baseline_metrics_cub.csv",
        #"losses":  ROOT / "losses_baseline_not_finetuning.csv",
        "time":    ROOT / "baseline_time_cub.csv",
    },
    # "Corr-Dist": {
    #     "metrics": ROOT / "metrics_correlation_dist.csv",
    #     #"losses":  ROOT / "losses_correlation_dist.csv",
    #     "time":    ROOT / "time_correlation_dist.csv",
    # },
    "Custom-CorrDist": {
        "metrics": ROOT / "custom_corrdist_metrics_cub.csv",
        #"losses":  ROOT / "losses_custom_correlation_dist.csv",
        "time":    ROOT / "custom_corrdist_times_cub.csv",
    },
}


# -------------------------------------------------------------------------
# 1.  Generic helpers
# -------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _to_percent(s: pd.Series) -> pd.Series:
    """Scale 0–1 floats up to 0–100 %; leave % values untouched."""
    return s * 100 if s.max() <= 1.0 else s


def _add_num_classes(df: pd.DataFrame) -> None:
    """Add a `num_classes` column (10, 20, …, 100)."""
    df["num_classes"] = [(i + 1) * 10 for i in range(len(df))]


# -------------------------------------------------------------------------
# 2.  Plotting functions
# -------------------------------------------------------------------------
def plot_map(methods=None, last10=True, save=None, ax=None):
    """
    Plot mAP vs #classes for the chosen methods.
    last10=True  ⇒  mAP on last 10 classes  (current)
    last10=False ⇒  cumulative mAP          (all classes)
    """
    if methods is None:
        methods = FILES.keys()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))

    ylabel = "mAP (%)"
    title = "mAP – Last 10 Classes" if last10 else "mAP – All Classes"

    for name in methods:
        mdf = _read_csv(FILES[name]["metrics"])
        _add_num_classes(mdf)

        # auto-detect column names
        cur_col = next(c for c in mdf.columns if "map" in c.lower() and "current" in c.lower())
        cum_col = next(c for c in mdf.columns if "map" in c.lower() and ("cumul" in c.lower() or "cumulative" in c.lower()))

        y = mdf[cur_col] if last10 else mdf[cum_col]
        y = _to_percent(y)
        ax.plot(mdf["num_classes"], y, marker="o", label=name)

    ax.set_xlabel("Number of Classes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    if save:
        plt.savefig(save, bbox_inches="tight")
    return ax


def plot_top1_accuracy(methods=None, last10=True, save=None, ax=None):
    """
    Same pattern but for Top-1 accuracy.
    last10=True  →  `top1_*_current`
    last10=False →  `top1_*_cumul`
    """
    if methods is None:
        methods = FILES.keys()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))

    title = "Top-1 Accuracy – Last 10 Classes" if last10 else "Top-1 Accuracy – All Classes"

    for name in methods:
        mdf = _read_csv(FILES[name]["metrics"])
        _add_num_classes(mdf)

        cur_col = next(c for c in mdf.columns if "top1" in c.lower() and "current" in c.lower() and "test" in c.lower())
        cum_col = next(c for c in mdf.columns if "top1" in c.lower() and "cumul" in c.lower() and "test" in c.lower())

        y = mdf[cur_col] if last10 else mdf[cum_col]
        y = _to_percent(y)
        ax.plot(mdf["num_classes"], y, marker="o", label=name)

    ax.set_xlabel("Number of Classes")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    if save:
        plt.savefig(save, bbox_inches="tight")
    return ax


def plot_training_time(methods=None, save=None, ax=None):
    """Total training time per session for each method."""
    if methods is None:
        methods = FILES.keys()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))

    for name in methods:
        tdf = _read_csv(FILES[name]["time"])
        _add_num_classes(tdf)
        # first 70 columns = per-epoch seconds
        tdf["total_time"] = tdf.iloc[:, :70].sum(axis=1)
        ax.plot(tdf["num_classes"], tdf["total_time"], marker="o", label=name)

    ax.set_xlabel("Number of Classes")
    ax.set_ylabel("Total Training Time (s)")
    ax.set_title("Training Time per Session")
    ax.grid(True)
    ax.legend()
    if save:
        plt.savefig(save, bbox_inches="tight")
    return ax


def plot_avg_val_loss(methods=None, save=None, ax=None):
    """Average validation loss per session."""
    if methods is None:
        methods = FILES.keys()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))

    for name in methods:
        ldf = _read_csv(FILES[name]["losses"])
        ldf["epoch"] = range(1, 701)
        ldf["session"] = (ldf["epoch"] - 1) // 70 + 1
        ldf["num_classes"] = ldf["session"] * 10
        avg = ldf.groupby("num_classes")["val_loss"].mean()
        ax.plot(avg.index, avg.values, marker="o", label=name)

    ax.set_xlabel("Number of Classes")
    ax.set_ylabel("Average Val-Loss over 70 Epochs")
    ax.set_title("Average Validation Loss per Session")
    ax.grid(True)
    ax.legend()
    if save:
        plt.savefig(save, bbox_inches="tight")
    return ax


# -------------------------------------------------------------------------
# 3.  Quick demo
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: generate and save all plots to PNGs
    plot_training_time(save= ROOT /"train_time_methods.png")
    plot_top1_accuracy(last10=True,  save= ROOT /"top1_last10_methods.png")
    plot_top1_accuracy(last10=False, save= ROOT /"top1_all_methods.png")
    plot_map(last10=True,  save= ROOT /"map_last10_methods.png")
    plot_map(last10=False, save= ROOT /"map_all_methods.png")
    #plot_avg_val_loss(save="avg_val_loss_methods.png")
    print("Plots saved.")
