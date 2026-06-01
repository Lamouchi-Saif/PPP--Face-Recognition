"""
Evaluation utilities: metrics, confusion matrix, comparison plots.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BASE_DIR

RESULTS_DIR = os.path.join(BASE_DIR+"\static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, method: str = "", labels=None) -> dict:
    """
    Compute accuracy, precision, recall and F1-score.

    Parameters
    ----------
    y_true  : ground-truth labels.
    y_pred  : predicted labels.
    method  : optional method name, used in printed output.

    Returns
    -------
    dict with keys: accuracy, report (str), confusion_matrix (np.ndarray).
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    header = f"=== {method} ===" if method else "=== Evaluation ==="
    print(header)
    print(f"Accuracy : {acc:.4f}")
    print(report)

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels,
    method: str = "model",
    save: bool = True,
) -> str:
    """
    Plot and optionally save a confusion matrix heatmap.

    Returns
    -------
    str – path to the saved figure (or empty string when save=False).
    """
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(6, len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, xticks_rotation="vertical")
    ax.set_title(f"Confusion Matrix – {method}")
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(RESULTS_DIR, f"confusion_matrix_{method}.png")
        fig.savefig(path, dpi=150)
        print(f"Confusion matrix saved → {path}")
    plt.close(fig)
    return path


def plot_method_comparison(results: dict, save: bool = True) -> str:
    """
    Bar chart comparing accuracy across different methods.

    Parameters
    ----------
    results : dict mapping method_name → metrics_dict (from compute_metrics).
    save    : whether to save the plot to disk.

    Returns
    -------
    str – path to saved figure or empty string.
    """
    methods = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, accuracies, color=plt.cm.Set2.colors[: len(methods)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Method Comparison – Accuracy")
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(RESULTS_DIR, "method_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"Comparison plot saved → {path}")
    plt.close(fig)
    return path
