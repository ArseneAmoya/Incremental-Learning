#!/usr/bin/env python3
# -------------------------------------------------------------
# requirements : pandas, matplotlib
# usage        : python make_curves.py --out ./figs
# -------------------------------------------------------------
import argparse, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# CONFIGURATION – renseigner ici tes couples <méthode : préfixe-fichier>
# -------------------------------------------------------------
FILES = {
    "Custom Corr-Dist":  ("corrdist_time_cub.csv", "custom_corrdist_metrics_cub.csv"),
    "iCaRL":             ("icarl_time_cub.csv",     "icarl_metrics_cub.csv"),
    "Scratch":           ("baseline_time_cub (2).csv",    "baseline_metric_cub.csv"),  # optionnel
}

TASKS_PER_BATCH  = 1     # si tes CSV sont déjà par batch 1…20
CLASSES_PER_TASK = 10    # sert uniquement pour l’axe X « nombre de classes »

# -------------------------------------------------------------
def total_time_per_batch(time_csv: Path) -> pd.DataFrame:
    """
    Agrège la durée totale d’entraînement par batch.
    Attendu : colonnes [batch, epoch, time_sec].
    """
    df = pd.read_csv(time_csv)
    if "time_sec" not in df.columns:
        df = df.reset_index().melt(id_vars=["index"], var_name="epoch", value_name="time_sec")
        df = df.rename(columns={"index": "batch"})
    return (df.groupby("batch")["time_sec"].sum()
            .reset_index()
            .assign(num_classes=lambda d: d["batch"] * CLASSES_PER_TASK))

def load_metrics(metric_csv: Path) -> pd.DataFrame:
    """
    Lit le CSV de métriques batch-wise et ajoute num_classes.
    Colonnes attendues :
        current_test_acc, cumul_test_acc,
        current_train_acc, cumul_train_acc,
        map_current, map_cumul
    """
    df = pd.read_csv(metric_csv)
    df = df.dropna(axis=0, how='any').reset_index(drop=True)  # Réinitialiser l'index
    try:
        df = df.drop("batch", axis = 1) # Supprimer les lignes avec des NaN
    except KeyError:
        df = df.rename(columns={"top1_train_current": "current_train_acc",
                                "top1_train_cumul":   "cumul_train_acc",
                                "top1_test_current":  "current_test_acc",
                                "top1_test_cumul":    "cumul_test_acc",
                                "map_cumulative":          "map_cumul",
                                "map_current":        "map_current"})
        pass
    for col in df.columns:
        if df[col].max() <= 1:
            # Mettez à l'échelle en % si nécessaire
            df[col] *= 100
    return df.assign(num_classes=lambda d: d.index * CLASSES_PER_TASK)

# -------------------------------------------------------------
def line_plot(ax, x, y, label):
    ax.plot(x, y, marker="o", label=label)

def make_plots(all_times, all_metrics, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- TIME ----------
    fig, ax = plt.subplots()
    for meth, df in all_times.items():
        line_plot(ax, list(range(1, len(df["num_classes"])+1))[:6], df["time_sec"][:6], meth)
    ax.set_xlabel("Nombre de classes")
    ax.set_ylabel("Durée totale par tâche (s)")
    ax.set_title("Training time per task")
    ax.grid(True, alpha=.3); ax.legend()
    (out_dir / "time_per_task.png").with_suffix(".png")
    plt.tight_layout(); plt.savefig(out_dir / "time_per_task.png", dpi=150); plt.close()

    # ---------- ACCURACIES & MAP ----------
    curve_specs = [
        ("current_test_acc",  "Top-1 accuracy – dernières 10 classes"),
        ("cumul_test_acc",    "Top-1 accuracy – toutes classes"),
        ("current_train_acc", "Train accuracy – dernières 10 classes"),
        ("cumul_train_acc",   "Train accuracy – toutes classes"),
        ("map_current",       "mAP – dernières 10 classes"),
        ("map_cumul",         "mAP – toutes classes"),
    ]

    for col, title in curve_specs:
        fig, ax = plt.subplots()
        for meth, df in all_metrics.items():
            line_plot(ax, list(range(1, len(df["num_classes"]) +1))[:6], df[col][:6], meth)
        ax.set_xlabel("Nombre de classes")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=.3); ax.legend()
        fname = re.sub(r"[^\w]+", "_", col.lower()) + ".png"
        plt.tight_layout(); plt.savefig(out_dir / fname, dpi=150); plt.close()

# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default="output", help="répertoire où se trouvent les CSV")
    parser.add_argument("--out", default="output", help="dossier de sortie pour les PNG")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out)

    all_times, all_metrics = {}, {}
    for method, (time_csv, metric_csv) in FILES.items():
        time_path   = csv_dir / time_csv
        metric_path = csv_dir / metric_csv
        if not time_path.exists() or not metric_path.exists():
            print(f"[WARN] fichiers manquants pour {method} – ignoré.")
            continue
        all_times[method]   = total_time_per_batch(time_path)
        all_metrics[method] = load_metrics(metric_path)

    make_plots(all_times, all_metrics, out_dir)
    print(f"Figures enregistrées dans {out_dir.resolve()}")

if __name__ == "__main__":
    main()
