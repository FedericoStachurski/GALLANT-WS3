#!/usr/bin/env python3
"""
plot_run_metrics.py

Given a run folder (the same folder where you saved best.pt/metrics.csv),
this script will:
- load metrics.csv (or metrics.json fallback)
- plot train_loss vs val_loss
- plot height_acc and trunk_acc (if present)
- save plots into the SAME run folder

Usage:
python plot_run_metrics.py --run_dir /home/fss6k/models/tree_size_runs/jan26_boxed_run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "metrics.csv"
    json_path = run_dir / "metrics.json"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df

    if json_path.exists():
        with open(json_path, "r") as f:
            j = json.load(f)
        hist = j.get("history", [])
        if not hist:
            raise RuntimeError(f"metrics.json exists but has no 'history' in {json_path}")
        df = pd.DataFrame(hist)
        return df

    raise RuntimeError(f"No metrics.csv or metrics.json found in {run_dir}")


def plot_loss(df: pd.DataFrame, out_path: Path):
    if "epoch" not in df.columns:
        raise RuntimeError("Missing 'epoch' column in metrics")

    if "train_loss" not in df.columns or "val_loss" not in df.columns:
        raise RuntimeError("Missing train_loss/val_loss columns in metrics")

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.plot(df["epoch"], df["val_loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Validation Loss")
    plt.legend(["train_loss", "val_loss"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_acc(df: pd.DataFrame, out_path: Path):
    if "epoch" not in df.columns:
        raise RuntimeError("Missing 'epoch' column in metrics")

    # Not all runs may have both
    has_height = "height_acc" in df.columns
    has_trunk = "trunk_acc" in df.columns

    if not (has_height or has_trunk):
        # nothing to plot
        return False

    plt.figure()
    if has_height:
        plt.plot(df["epoch"], df["height_acc"])
    if has_trunk:
        plt.plot(df["epoch"], df["trunk_acc"])

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Validation Accuracies")
    labels = []
    if has_height:
        labels.append("height_acc")
    if has_trunk:
        labels.append("trunk_acc")
    plt.legend(labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str, help="Folder containing metrics.csv/metrics.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run folder not found: {run_dir}")

    df = load_metrics(run_dir)

    # Save plots in same folder
    loss_out = run_dir / "plot_loss_train_vs_val.png"
    acc_out = run_dir / "plot_val_accuracies.png"

    plot_loss(df, loss_out)
    did_acc = plot_acc(df, acc_out)

    print("Saved:", loss_out)
    if did_acc:
        print("Saved:", acc_out)
    else:
        print("No accuracy columns found (height_acc/trunk_acc). Skipped accuracy plot.")


if __name__ == "__main__":
    main()
