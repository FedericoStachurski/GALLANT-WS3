#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


# ============================================================
# Model loading
# ============================================================

def build_resnet_model(
    backbone="resnet50",
    num_classes=5,
    in_channels=3,
    dropout=0.0,
    device="cpu",
):
    backbone = backbone.lower()

    if backbone == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone == "resnet34":
        model = models.resnet34(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    elif backbone == "resnet101":
        model = models.resnet101(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

    if dropout and dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


def load_tree_model_run(run_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_path = Path(run_path)

    with open(run_path / "metrics.json", "r") as f:
        metrics = json.load(f)

    with open(run_path / "history.json", "r") as f:
        history = json.load(f)

    with open(run_path / "config.json", "r") as f:
        config = json.load(f)

    checkpoint = torch.load(run_path / "best_model.pth", map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    dropout = config.get("dropout", 0.0)
    if dropout == 0.0 and any(k.startswith("fc.1.") for k in state_dict.keys()):
        dropout = 0.3

    model = build_resnet_model(
        backbone=config.get("backbone", "resnet50"),
        num_classes=config.get("num_classes", 5),
        in_channels=config.get("in_channels", 3),
        dropout=dropout,
        device=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, metrics, history, config


# ============================================================
# Dataset
# ============================================================

class TreeHeightInferenceDataset(Dataset):
    def __init__(self, df, image_size=224, use_depth=True):
        self.df = df.reset_index(drop=True)
        self.use_depth = use_depth
        self.image_size = image_size

        self.rgb_tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.depth_resize = transforms.Resize((image_size, image_size))
        self.depth_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rgb = Image.open(row["RGB_CROP_PATH"]).convert("RGB")
        rgb_tensor = self.rgb_tfms(rgb)

        if self.use_depth:
            depth = np.load(row["DEPTH_PATH"]).astype(np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_img = Image.fromarray((depth * 255).astype(np.uint8)).convert("L")
            depth_img = self.depth_resize(depth_img)
            depth_tensor = self.depth_to_tensor(depth_img)
            x = torch.cat([rgb_tensor, depth_tensor], dim=0)
        else:
            x = rgb_tensor

        return x, idx


# ============================================================
# Inference
# ============================================================

def safe_class_name(cls):
    return (
        cls.replace(">", "gt")
        .replace("<", "lt")
        .replace("+", "plus")
        .replace("-", "_")
        .replace(" ", "")
    )


def predict_missing_heights(
    dataset_dir,
    run_path,
    batch_size=16,
    num_workers=4,
    conf_threshold=None,
    output_name="tree_dataset_manifest_with_height_predictions.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = Path(dataset_dir)
    run_path = Path(run_path)

    manifest_path = dataset_dir / "manifests" / "tree_dataset_manifest.csv"
    out_path = dataset_dir / "manifests" / output_name

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    model, metrics, history, config = load_tree_model_run(run_path, device=device)

    use_depth = bool(config.get("use_depth", False))
    image_size = int(config.get("image_size", 224))
    class_names = config.get("class_names", ["<5", "5-10", "10-15", "15-20", "20+"])
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    missing_mask = df["HEIGHT_CLASS_IDX"].isna()

    if use_depth:
        predict_df = df[
            missing_mask
            & df["RGB_CROP_PATH"].notna()
            & df["DEPTH_PATH"].notna()
        ].copy()
    else:
        predict_df = df[
            missing_mask
            & df["RGB_CROP_PATH"].notna()
        ].copy()

    # only keep rows whose files actually exist
    predict_df = predict_df[
        predict_df["RGB_CROP_PATH"].map(lambda p: Path(p).exists() if pd.notna(p) else False)
    ].copy()

    if use_depth:
        predict_df = predict_df[
            predict_df["DEPTH_PATH"].map(lambda p: Path(p).exists() if pd.notna(p) else False)
        ].copy()

    predict_df = predict_df.reset_index().rename(columns={"index": "orig_index"})

    print("Rows with missing height:", int(missing_mask.sum()))
    print("Rows eligible for prediction:", len(predict_df))

    if len(predict_df) == 0:
        print("No rows to predict.")
        df.to_csv(out_path, index=False)
        return df, out_path, class_names

    infer_ds = TreeHeightInferenceDataset(
        predict_df,
        image_size=image_size,
        use_depth=use_depth,
    )

    infer_loader = DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_pred_idx = []
    all_conf = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for x, _ in infer_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            pred_idx = probs.argmax(dim=1).cpu().numpy()
            conf = probs.max(dim=1).values.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_pred_idx.extend(pred_idx.tolist())
            all_conf.extend(conf.tolist())
            all_probs.extend(probs_np.tolist())

    pred_table = predict_df.copy()
    pred_table["PROBS_TREE_HEIGHT"] = [json.dumps(p) for p in all_probs]
    pred_table["HEIGHT_CLASS_PRED_IDX"] = all_pred_idx
    pred_table["HEIGHT_CLASS_PRED_STR"] = pred_table["HEIGHT_CLASS_PRED_IDX"].map(idx_to_class)
    pred_table["HEIGHT_CLASS_PRED_CONF"] = all_conf

    probs_arr = np.array(all_probs)
    for i, cls in enumerate(class_names):
        pred_table[f"HEIGHT_PROB_{safe_class_name(cls)}"] = probs_arr[:, i]

    # create columns if missing
    if "PROBS_TREE_HEIGHT" not in df.columns:
        df["PROBS_TREE_HEIGHT"] = np.nan
    if "HEIGHT_CLASS_PRED_IDX" not in df.columns:
        df["HEIGHT_CLASS_PRED_IDX"] = np.nan
    if "HEIGHT_CLASS_PRED_STR" not in df.columns:
        df["HEIGHT_CLASS_PRED_STR"] = np.nan
    if "HEIGHT_CLASS_PRED_CONF" not in df.columns:
        df["HEIGHT_CLASS_PRED_CONF"] = np.nan

    for cls in class_names:
        col = f"HEIGHT_PROB_{safe_class_name(cls)}"
        if col not in df.columns:
            df[col] = np.nan

    # only write predictions into rows that were inferred
    for _, row in pred_table.iterrows():
        i = int(row["orig_index"])
        df.loc[i, "PROBS_TREE_HEIGHT"] = row["PROBS_TREE_HEIGHT"]
        df.loc[i, "HEIGHT_CLASS_PRED_IDX"] = row["HEIGHT_CLASS_PRED_IDX"]
        df.loc[i, "HEIGHT_CLASS_PRED_STR"] = row["HEIGHT_CLASS_PRED_STR"]
        df.loc[i, "HEIGHT_CLASS_PRED_CONF"] = row["HEIGHT_CLASS_PRED_CONF"]

        for cls in class_names:
            col = f"HEIGHT_PROB_{safe_class_name(cls)}"
            df.loc[i, col] = row[col]

    # final filled columns
    df["HEIGHT_CLASS_FINAL_IDX"] = df["HEIGHT_CLASS_IDX"]
    df["HEIGHT_CLASS_FINAL_STR"] = df["HEIGHT_CLASS_STR"]

    if conf_threshold is None:
        fill_mask = df["HEIGHT_CLASS_IDX"].isna() & df["HEIGHT_CLASS_PRED_IDX"].notna()
    else:
        fill_mask = (
            df["HEIGHT_CLASS_IDX"].isna()
            & df["HEIGHT_CLASS_PRED_IDX"].notna()
            & (df["HEIGHT_CLASS_PRED_CONF"] >= conf_threshold)
        )

    df.loc[fill_mask, "HEIGHT_CLASS_FINAL_IDX"] = df.loc[fill_mask, "HEIGHT_CLASS_PRED_IDX"]
    df.loc[fill_mask, "HEIGHT_CLASS_FINAL_STR"] = df.loc[fill_mask, "HEIGHT_CLASS_PRED_STR"]

    df.to_csv(out_path, index=False)

    print(f"Saved updated manifest to: {out_path}")
    print("Filled rows:", int(fill_mask.sum()))
    if conf_threshold is not None:
        print(f"Confidence threshold used: {conf_threshold}")

    return df, out_path, class_names


# ============================================================
# Aggregation
# ============================================================

def aggregate_tree_predictions(df_pred, class_names):
    grouped = []

    for tree_id, g in df_pred.groupby("ID"):
        g = g[g["HEIGHT_CLASS_PRED_IDX"].notna()].copy()
        if len(g) == 0:
            continue

        probs_list = []
        for p in g["PROBS_TREE_HEIGHT"]:
            if pd.isna(p):
                continue
            if isinstance(p, str):
                probs_list.append(np.array(json.loads(p), dtype=float))
            else:
                probs_list.append(np.array(p, dtype=float))

        if len(probs_list) == 0:
            continue

        probs_arr = np.vstack(probs_list)
        mean_probs = probs_arr.mean(axis=0)

        pred_idx = int(np.argmax(mean_probs))
        avg_conf = float(g["HEIGHT_CLASS_PRED_CONF"].mean())

        out = {
            "ID": tree_id,
            "N_IMAGES": len(g),
            "HEIGHT_CLASS_PRED_IDX": pred_idx,
            "HEIGHT_CLASS_PRED_STR": class_names[pred_idx],
            "HEIGHT_CLASS_CONF_MEAN": avg_conf,
            "PROBS_TREE_HEIGHT": json.dumps(mean_probs.tolist()),
        }

        if "TREE_BOX" in g.columns:
            out["TREE_BOX"] = g["TREE_BOX"].iloc[0]

        grouped.append(out)

    return pd.DataFrame(grouped)


# ============================================================
# Plot helpers
# ============================================================

def _parse_tree_box(tree_box):
    if tree_box is None:
        return None, None

    if isinstance(tree_box, float) and np.isnan(tree_box):
        return None, None

    if isinstance(tree_box, str):
        tree_box = tree_box.strip()
        if not tree_box:
            return None, None
        try:
            tree_box = json.loads(tree_box)
        except Exception:
            return None, None

    arr = np.array(tree_box, dtype=float)

    if arr.ndim == 1 and arr.size == 4:
        x1, y1, x2, y2 = arr
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        return "rect", (x, y, w, h)

    if arr.ndim == 2 and arr.shape == (2, 2):
        (x1, y1), (x2, y2) = arr
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        return "rect", (x, y, w, h)

    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 4:
        return "poly", arr

    return None, None


def save_distribution_plot(df_pred, save_path, class_names):
    before_counts = (
        df_pred["HEIGHT_CLASS_STR"]
        .value_counts()
        .reindex(class_names, fill_value=0)
    )

    after_counts = (
        df_pred["HEIGHT_CLASS_FINAL_STR"]
        .value_counts()
        .reindex(class_names, fill_value=0)
    )

    x = np.arange(len(class_names))
    width = 0.4

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        before_counts.values,
        width=width,
        label="Original labelled",
    )

    plt.bar(
        x + width / 2,
        after_counts.values,
        width=width,
        label="After model filling",
    )

    plt.xticks(x, class_names)
    plt.ylabel("Number of trees")
    plt.xlabel("Height class (meters)")
    plt.title("Tree Height Distribution: Before vs After Model Filling")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved distribution plot to: {save_path}")


def save_inferred_examples_plot(
    df_pred,
    save_path,
    n=9,
    random_state=42,
    conf_threshold=0.5,
    class_names=None,
):
    if class_names is None:
        class_names = ["<5", "5-10", "10-15", "15-20", "20+"]

    x_pos = np.arange(len(class_names))

    # prefer aggregated columns if available, otherwise fall back
    conf_col = "HEIGHT_CLASS_CONF_MEAN" if "HEIGHT_CLASS_CONF_MEAN" in df_pred.columns else "HEIGHT_CLASS_PRED_CONF"

    mask = (
        df_pred["HEIGHT_CLASS_STR"].isna()
        & df_pred["HEIGHT_CLASS_PRED_STR"].notna()
        & df_pred[conf_col].notna()
        & (df_pred[conf_col] > conf_threshold)
        & df_pred["PROBS_TREE_HEIGHT"].notna()
        & df_pred["RGB_CROP_PATH"].notna()
    )

    examples = df_pred[mask].copy()
    if len(examples) == 0:
        print("No examples found for inferred examples plot.")
        return

    examples = examples.sample(min(n, len(examples)), random_state=random_state)

    n_show = len(examples)
    ncols = min(3, n_show)
    nrows = math.ceil(n_show / ncols)

    fig = plt.figure(figsize=(6 * ncols, 5.8 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.25, wspace=0.15)

    for k, (_, row) in enumerate(examples.iterrows()):
        r = k // ncols
        c = k % ncols

        inner = outer[r, c].subgridspec(2, 1, height_ratios=[4, 1.7], hspace=0.08)

        ax = fig.add_subplot(inner[0])

        img = Image.open(row["RGB_CROP_PATH"]).convert("RGB")
        ax.imshow(img)

        pred_height = row["HEIGHT_CLASS_PRED_STR"]
        conf = float(row[conf_col])

        tree_box = row["TREE_BOX"] if "TREE_BOX" in row.index else None
        box_kind, box_data = _parse_tree_box(tree_box)

        if box_kind == "rect":
            x, y, w, h = box_data
            ax.add_patch(
                Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=2.5,
                    edgecolor="red",
                    facecolor="none",
                )
            )
        elif box_kind == "poly":
            ax.add_patch(
                Polygon(
                    box_data,
                    closed=True,
                    linewidth=2.5,
                    edgecolor="red",
                    facecolor="none",
                )
            )
            ax.scatter(box_data[:, 0], box_data[:, 1], s=20, c="red")

        ax.set_title(f"Predicted: {pred_height} ({conf*100:.1f}%)", fontsize=12)
        ax.axis("off")

        ax2 = fig.add_subplot(inner[1])

        probs = row["PROBS_TREE_HEIGHT"]
        if isinstance(probs, str):
            probs = json.loads(probs)
        probs = np.array(probs, dtype=float)

        bars = ax2.bar(x_pos, probs, width=0.7)
        pred_idx = int(np.argmax(probs))

        for i, bar in enumerate(bars):
            bar.set_alpha(0.85 if i == pred_idx else 0.45)
            if i == pred_idx:
                bar.set_linewidth(2)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(class_names, fontsize=10)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Prob.", fontsize=10)
        ax2.grid(axis="y", alpha=0.25)

        for i, p in enumerate(probs):
            ax2.text(
                i,
                min(p + 0.03, 0.98),
                f"{p:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved inferred examples plot to: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Infer missing tree heights and save updated manifest + plots.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset cache folder")
    parser.add_argument("--run_path", type=str, required=True, help="Path to trained model run folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--conf_threshold", type=float, default=None)
    parser.add_argument(
        "--output_name",
        type=str,
        default="tree_dataset_manifest_with_height_predictions.csv",
        help="Output CSV name to save inside dataset_dir/manifests/"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=9,
        help="How many inferred examples to include in the saved panel plot"
    )
    parser.add_argument(
        "--example_conf_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for the saved inferred examples plot"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for choosing example panels"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    run_path = Path(args.run_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_pred, out_csv_path, class_names = predict_missing_heights(
        dataset_dir=dataset_dir,
        run_path=run_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_threshold=args.conf_threshold,
        output_name=args.output_name,
    )

    # tree-level aggregation for example plot
    tree_predictions = aggregate_tree_predictions(df_pred, class_names=class_names)

    if len(tree_predictions) > 0:
        merge_cols = [
            "ID",
            "HEIGHT_CLASS_PRED_STR",
            "HEIGHT_CLASS_CONF_MEAN",
            "PROBS_TREE_HEIGHT",
        ]
        if "TREE_BOX" in tree_predictions.columns:
            merge_cols.append("TREE_BOX")

        df_pred_viz = df_pred.merge(
            tree_predictions[merge_cols],
            on="ID",
            how="left",
            suffixes=("", "_TREE"),
        )
    else:
        df_pred_viz = df_pred.copy()

    dist_plot_path = run_path / f"plots/height_distribution_before_after_{timestamp}.png"
    examples_plot_path = run_path / f"plots/inferred_height_examples_{timestamp}.png"

    save_distribution_plot(
        df_pred=df_pred,
        save_path=dist_plot_path,
        class_names=class_names,
    )

    save_inferred_examples_plot(
        df_pred=df_pred_viz,
        save_path=examples_plot_path,
        n=args.n_examples,
        random_state=args.seed,
        conf_threshold=args.example_conf_threshold,
        class_names=class_names,
    )

    print("\nDone.")
    print(f"Updated CSV: {out_csv_path}")
    print(f"Distribution plot: {dist_plot_path}")
    print(f"Examples plot: {examples_plot_path}")


if __name__ == "__main__":
    main()