#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
)


from pathlib import Path
import json

def load_run(run_path: Path):
    metrics_path = run_path / "metrics.json"
    history_path = run_path / "history.json"
    config_path = run_path / "config.json"

    # ----------------------------
    # REQUIRED FILES
    # ----------------------------
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.json in {run_path}")

    # ----------------------------
    # OPTIONAL FILE
    # ----------------------------
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        print(f"[WARNING] config.json missing in {run_path}")
        config = None
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        print(f"[WARNING] metrics.json missing in {run_path}")
        metrics = None

    # ----------------------------
    # REQUIRED LOADS
    # ----------------------------
    with open(history_path, "r") as f:
        history = json.load(f)

    return metrics, history, config


# =========================================================
# Training-history diagnostics
# =========================================================
def plot_training_diagnostics(history, title_prefix="Model", save_path=None):
    print(history)

    def normalize_history(history):
        # CASE 1: wrapped dict -> list of epoch dicts
        if isinstance(history, dict) and "history" in history:
            history = history["history"]

        # CASE 2: new format = list of epoch dicts
        if isinstance(history, list):
            epochs = []
            train_loss, val_loss = [], []
            train_acc, val_acc = [], []
            train_f1, val_f1 = [], []

            for i, h in enumerate(history, start=1):
                train = h.get("train", {})
                val = h.get("val", {})

                epochs.append(h.get("epoch", i))
                train_loss.append(train.get("loss", np.nan))
                val_loss.append(val.get("loss", np.nan))
                train_acc.append(train.get("bal_acc", np.nan))
                val_acc.append(val.get("bal_acc", np.nan))
                train_f1.append(train.get("macro_f1", np.nan))
                val_f1.append(val.get("macro_f1", np.nan))

            return {
                "epochs": np.array(epochs, dtype=float),
                "train_loss": np.array(train_loss, dtype=float),
                "val_loss": np.array(val_loss, dtype=float),
                "train_acc": np.array(train_acc, dtype=float),
                "val_acc": np.array(val_acc, dtype=float),
                # "train_f1": np.array(train_f1, dtype=float),
                "val_f1": np.array(val_f1, dtype=float),
            }

        # CASE 3: old flat dict
        if isinstance(history, dict) and "train_loss" in history:
            n = len(history.get("train_loss", []))
            return {
                "epochs": np.arange(1, n + 1),
                "train_loss": np.array(history.get("train_loss", [np.nan] * n), dtype=float),
                "val_loss": np.array(history.get("val_loss", [np.nan] * n), dtype=float),
                "train_acc": np.array(history.get("train_acc", [np.nan] * n), dtype=float),
                "val_acc": np.array(history.get("val_acc", [np.nan] * n), dtype=float),
                "val_f1_macro": np.array(
                    history.get("val_f1_macro", history.get("val_f1_macro", [np.nan] * n)),
                    dtype=float
                ),
                "val_recall_macro": np.array(
                    history.get("val_recall_macro", [np.nan] * n),
                    dtype=float
                ),
            }
        raise ValueError(f"Unknown history format: {type(history)}")
    
    hist = normalize_history(history)
    acc_gap = hist["train_acc"] - hist["val_acc"]
    best_epoch = np.nanargmax(hist["val_acc"]) + 1  # +1 for 1-based epoch indexing

    fig = plt.figure(figsize=(16, 10))

    # Top: loss
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax1.plot(hist["epochs"], hist["train_loss"], marker="o", label="Train loss")
    ax1.plot(hist["epochs"], hist["val_loss"], marker="o", label="Val loss")
    ax1.axvline(best_epoch, color="red", linestyle="--", label=f"Best val acc epoch {best_epoch}")
    ax1.set_title(f"{title_prefix} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if isinstance(hist["train_acc"], np.ndarray) and isinstance(hist["val_acc"], np.ndarray):
        # Bottom left: accuracy
        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax2.plot(hist["epochs"], hist["train_acc"], marker="o", label="Train acc")
        ax2.plot(hist["epochs"], hist["val_acc"], marker="o", label="Val acc")
        ax2.fill_between(hist["epochs"], hist["train_acc"], hist["val_acc"], alpha=0.12)
        ax2.axvline(best_epoch, color="red", linestyle="--")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax2.text(
            0.03, 0.05,
            f"Mean accuracy gap = {acc_gap.mean()*100:.2f}%",
            transform=ax2.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    if  isinstance(hist["val_f1_macro"], np.ndarray):
        # Bottom middle: F1
        ax3 = plt.subplot2grid((2, 3), (1, 1))
        ax3.plot(hist["epochs"], hist["val_f1_macro"], marker="o")
        ax3.axvline(best_epoch, color="red", linestyle="--")
        ax3.set_title("Validation Macro F1")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1-score")
        ax3.grid(True, alpha=0.3)

    if isinstance(hist.get("val_recall_macro"), np.ndarray):
        # Bottom right: recall
        ax4 = plt.subplot2grid((2, 3), (1, 2))
        ax4.plot(hist["epochs"], hist["val_recall_macro"], marker="o")
        ax4.axvline(best_epoch, color="red", linestyle="--")
        ax4.set_title("Validation Macro Recall")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Recall")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


# =========================================================
# Evaluation diagnostics
# =========================================================
def plot_evaluation_diagnostics(
    y_true,
    y_pred,
    class_names,
    title_prefix="Model",
    normalize_cm=True,
    save_path=None,
    report_path=None,
):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    report_txt = []
    report_txt.append(f"Accuracy:      {acc:.4f}")
    report_txt.append(f"Macro F1:      {f1_macro:.4f}")
    report_txt.append(f"Macro Recall:  {recall_macro:.4f}")
    report_txt.append("")
    report_txt.append("Classification report:")
    report_txt.append(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    report_str = "\n".join(report_txt)

    print(report_str)

    if report_path is not None:
        with open(report_path, "w") as f:
            f.write(report_str)

    cm = confusion_matrix(y_true, y_pred)

    if normalize_cm:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cbar_label = "Proportion of true class"
        annot = np.array([
            [f"{cm[i,j]}\n{cm_display[i,j]:.2f}" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])
        fmt = ""
    else:
        cm_display = cm
        cbar_label = "Count"
        annot = cm
        fmt = "d"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix
    sns.heatmap(
        cm_display,
        annot=annot,
        fmt=fmt,
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": cbar_label},
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_title(f"{title_prefix} — Confusion Matrix")

    # Per-class recall
    per_class_recall = np.diag(cm) / cm.sum(axis=1)
    axes[1].bar(class_names, per_class_recall)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Recall")
    axes[1].set_title(f"{title_prefix} — Per-class Recall")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Plot diagnostics for a saved tree model run.")
    parser.add_argument(
        "--run",
        required=True,
        help="Path to the run folder, e.g. /path/to/tree_models/resnet50_rgbd_20260312_163522"
    )
    args = parser.parse_args()

    run_path = Path(args.run)
    if not run_path.exists():
        raise FileNotFoundError(f"Run path not found: {run_path}")

    plots_dir = run_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics, history, config = load_run(run_path)

    run_name = run_path.name
    print(f"Loaded run: {run_name}")
    if metrics is not None:
        print(f"Best epoch: {metrics.get('best_epoch')}")
        print(f"Best validation accuracy: {metrics.get('best_val_accuracy')}")
    if config is not None:
        print(f"Config: {config}")  

    # Always save training history diagnostics
    plot_training_diagnostics(
        history,
        title_prefix=run_name,
        save_path=plots_dir / "training_diagnostics.png"
    )

    # Optional evaluation diagnostics if predictions were saved
    val_labels_path = run_path / "val_labels.npy"
    val_preds_path = run_path / "val_preds.npy"

    if val_labels_path.exists() and val_preds_path.exists():
        y_true = np.load(val_labels_path)
        y_pred = np.load(val_preds_path)

        class_names = config.get("class_names", ["<5", "5-10", "10-15", "15-20", "20+"])

        plot_evaluation_diagnostics(
            y_true,
            y_pred,
            class_names=class_names,
            title_prefix=run_name,
            normalize_cm=True,
            save_path=plots_dir / "evaluation_diagnostics.png",
            report_path=plots_dir / "classification_report.txt",
        )
        print("Saved evaluation diagnostics.")
    else:
        print("No val_labels.npy / val_preds.npy found, so only training diagnostics were generated.")

    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()

