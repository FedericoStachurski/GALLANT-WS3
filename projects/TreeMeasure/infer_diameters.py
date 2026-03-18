#!/usr/bin/env python3

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


# ============================================================
# Model definitions
# ============================================================

class ResNetDiameterWithHeight(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        in_channels: int,
        n_tabular_features: int,
        device,
        dropout_rate: float = 0.1,
        tab_hidden_dim: int = 16,
    ):
        super().__init__()

        backbone = backbone.lower()

        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "resnet101": models.ResNet101_Weights.DEFAULT,
        }

        if backbone not in weights_map:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = getattr(models, backbone)(weights=weights_map[backbone])

        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        self.backbone.conv1.weight[:, c:c+1] = mean_w

        image_feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.tabular_head = nn.Sequential(
            nn.Linear(n_tabular_features, tab_hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(image_feat_dim + tab_hidden_dim, num_classes),
        )

        self.to(device)

    def forward(self, x_img, x_tab):
        img_feat = self.backbone(x_img)
        tab_feat = self.tabular_head(x_tab)
        feat = torch.cat([img_feat, tab_feat], dim=1)
        out = self.classifier(feat)
        return out


def load_diameter_model_run(run_path, device=None):
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

    dropout = config.get("dropout_rate", config.get("dropout", 0.0))
    n_tabular_features = len(config.get("height_feature_cols", ["HEIGHT_CLASS_FINAL_IDX"]))

    model = ResNetDiameterWithHeight(
        backbone=config.get("backbone", "resnet50"),
        num_classes=config.get("num_classes", 5),
        in_channels=config.get("in_channels", 3),
        n_tabular_features=n_tabular_features,
        device=device,
        dropout_rate=dropout,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, metrics, history, config


# ============================================================
# Feature helpers
# ============================================================

def safe_class_name(cls):
    return (
        cls.replace(">", "gt")
        .replace("<", "lt")
        .replace("+", "plus")
        .replace("-", "_")
        .replace(" ", "")
    )


def build_height_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a consistent 5-dim height feature vector for every row.

    Priority:
    1) use HEIGHT_PROB_* if all are present
    2) else use HEIGHT_CLASS_FINAL_IDX one-hot
    3) else use HEIGHT_CLASS_IDX one-hot
    4) else leave all zeros
    """
    df = df.copy()

    feat_cols = [
        "HEIGHT_FEAT_lt5",
        "HEIGHT_FEAT_5_10",
        "HEIGHT_FEAT_10_15",
        "HEIGHT_FEAT_15_20",
        "HEIGHT_FEAT_20plus",
    ]

    for c in feat_cols:
        df[c] = 0.0

    prob_map = {
        "HEIGHT_FEAT_lt5": "HEIGHT_PROB_lt5",
        "HEIGHT_FEAT_5_10": "HEIGHT_PROB_5_10",
        "HEIGHT_FEAT_10_15": "HEIGHT_PROB_10_15",
        "HEIGHT_FEAT_15_20": "HEIGHT_PROB_15_20",
        "HEIGHT_FEAT_20plus": "HEIGHT_PROB_20plus",
    }

    prob_cols_present = all(col in df.columns for col in prob_map.values())

    if prob_cols_present:
        has_probs = df[list(prob_map.values())].notna().all(axis=1)
    else:
        has_probs = pd.Series(False, index=df.index)

    if prob_cols_present:
        for feat_col, prob_col in prob_map.items():
            df.loc[has_probs, feat_col] = df.loc[has_probs, prob_col].astype(float)

    if "HEIGHT_CLASS_FINAL_IDX" in df.columns:
        hard_idx = df["HEIGHT_CLASS_FINAL_IDX"]
    elif "HEIGHT_CLASS_IDX" in df.columns:
        hard_idx = df["HEIGHT_CLASS_IDX"]
    elif "HEIGHT_CLASS_PRED_IDX" in df.columns:
        hard_idx = df["HEIGHT_CLASS_PRED_IDX"]
    else:
        hard_idx = pd.Series(np.nan, index=df.index)

    needs_fallback = ~has_probs & hard_idx.notna()

    for idx, feat_col in enumerate(feat_cols):
        df.loc[needs_fallback & (hard_idx == idx), feat_col] = 1.0

    return df, feat_cols


# ============================================================
# Dataset
# ============================================================

class TreeDiameterInferenceDataset(Dataset):
    def __init__(self, df, image_size=224, use_depth=True, height_feature_cols=None):
        self.df = df.reset_index(drop=True)
        self.use_depth = use_depth
        self.image_size = image_size
        self.height_feature_cols = height_feature_cols or ["HEIGHT_CLASS_FINAL_IDX"]

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

    def _load_depth_tensor(self, depth_path: str, dtype: torch.dtype) -> torch.Tensor:
        try:
            depth = np.load(depth_path).astype(np.float32)
            if depth.ndim != 2:
                raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")

            dmin, dmax = depth.min(), depth.max()
            depth = (depth - dmin) / (dmax - dmin + 1e-8)

            depth_img = Image.fromarray((depth * 255).astype(np.uint8)).convert("L")
            depth_img = self.depth_resize(depth_img)
            depth_tensor = self.depth_to_tensor(depth_img).to(dtype=dtype)
        except Exception:
            depth_tensor = torch.zeros(1, self.image_size, self.image_size, dtype=dtype)

        return depth_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rgb = Image.open(row["RGB_CROP_PATH"]).convert("RGB")
        rgb_tensor = self.rgb_tfms(rgb)

        if self.use_depth:
            depth_tensor = self._load_depth_tensor(
                row["DEPTH_PATH"],
                dtype=rgb_tensor.dtype
            )
            x_img = torch.cat([rgb_tensor, depth_tensor], dim=0)
        else:
            x_img = rgb_tensor

        x_tab = row[self.height_feature_cols].astype(np.float32).values
        x_tab = torch.tensor(x_tab, dtype=torch.float32)

        return x_img, x_tab, idx


# ============================================================
# Inference
# ============================================================

def predict_missing_diameters(
    dataset_dir,
    run_path,
    batch_size=16,
    num_workers=4,
    conf_threshold=None,
    output_name="tree_dataset_manifest_with_diameter_predictions.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = Path(dataset_dir)
    run_path = Path(run_path)

    manifest_candidates = [
        dataset_dir / "manifests" / "tree_dataset_manifest_with_height_predictions.csv",
        dataset_dir / "manifests" / "tree_dataset_manifest.csv",
    ]

    manifest_path = None
    for p in manifest_candidates:
        if p.exists():
            manifest_path = p
            break

    if manifest_path is None:
        raise FileNotFoundError(
            f"No manifest found. Checked: {[str(p) for p in manifest_candidates]}"
        )

    out_path = dataset_dir / "manifests" / output_name

    df = pd.read_csv(manifest_path)

    model, metrics, history, config = load_diameter_model_run(run_path, device=device)

    use_depth = bool(config.get("use_depth", False))
    image_size = int(config.get("image_size", 224))
    configured_height_cols = config.get("height_feature_cols", ["HEIGHT_CLASS_FINAL_IDX"])
    class_names = config.get("class_names", ["<2", "2-20", "20-40", "40-60", "60+"])
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    # Build height features if the run expects probability-style features
    if any(c.startswith("HEIGHT_FEAT_") for c in configured_height_cols):
        df, height_feature_cols = build_height_feature_columns(df)
    else:
        height_feature_cols = configured_height_cols
        for c in height_feature_cols:
            if c not in df.columns:
                raise ValueError(f"Required height feature column missing from manifest: {c}")

    missing_mask = df["DIAMETER_CLASS_IDX"].isna() if "DIAMETER_CLASS_IDX" in df.columns else pd.Series(True, index=df.index)

    needed = ["RGB_CROP_PATH"] + height_feature_cols
    if use_depth:
        needed.append("DEPTH_PATH")

    predict_df = df[missing_mask].copy()
    predict_df = predict_df.dropna(subset=needed).copy()

    predict_df = predict_df[
        predict_df["RGB_CROP_PATH"].map(lambda p: Path(p).exists() if pd.notna(p) else False)
    ].copy()

    if use_depth:
        predict_df = predict_df[
            predict_df["DEPTH_PATH"].map(lambda p: Path(p).exists() if pd.notna(p) else False)
        ].copy()

    predict_df = predict_df.reset_index().rename(columns={"index": "orig_index"})

    print("Rows with missing diameter:", int(missing_mask.sum()))
    print("Rows eligible for prediction:", len(predict_df))

    if len(predict_df) == 0:
        print("No rows to predict.")
        df.to_csv(out_path, index=False)
        return df, out_path, class_names

    infer_ds = TreeDiameterInferenceDataset(
        predict_df,
        image_size=image_size,
        use_depth=use_depth,
        height_feature_cols=height_feature_cols,
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
        for x_img, x_tab, _ in infer_loader:
            x_img = x_img.to(device, non_blocking=True)
            x_tab = x_tab.to(device, non_blocking=True)

            logits = model(x_img, x_tab)
            probs = torch.softmax(logits, dim=1)

            pred_idx = probs.argmax(dim=1).cpu().numpy()
            conf = probs.max(dim=1).values.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_pred_idx.extend(pred_idx.tolist())
            all_conf.extend(conf.tolist())
            all_probs.extend(probs_np.tolist())

    pred_table = predict_df.copy()
    pred_table["PROBS_TREE_DIAMETER"] = [json.dumps(p) for p in all_probs]
    pred_table["DIAMETER_CLASS_PRED_IDX"] = all_pred_idx
    pred_table["DIAMETER_CLASS_PRED_STR"] = pred_table["DIAMETER_CLASS_PRED_IDX"].map(idx_to_class)
    pred_table["DIAMETER_CLASS_PRED_CONF"] = all_conf

    probs_arr = np.array(all_probs)
    for i, cls in enumerate(class_names):
        pred_table[f"DIAMETER_PROB_{safe_class_name(cls)}"] = probs_arr[:, i]

    # create columns if missing
    if "PROBS_TREE_DIAMETER" not in df.columns:
        df["PROBS_TREE_DIAMETER"] = np.nan
    if "DIAMETER_CLASS_PRED_IDX" not in df.columns:
        df["DIAMETER_CLASS_PRED_IDX"] = np.nan
    if "DIAMETER_CLASS_PRED_STR" not in df.columns:
        df["DIAMETER_CLASS_PRED_STR"] = np.nan
    if "DIAMETER_CLASS_PRED_CONF" not in df.columns:
        df["DIAMETER_CLASS_PRED_CONF"] = np.nan

    for cls in class_names:
        col = f"DIAMETER_PROB_{safe_class_name(cls)}"
        if col not in df.columns:
            df[col] = np.nan

    # write predictions into original dataframe
    for _, row in pred_table.iterrows():
        i = int(row["orig_index"])
        df.loc[i, "PROBS_TREE_DIAMETER"] = row["PROBS_TREE_DIAMETER"]
        df.loc[i, "DIAMETER_CLASS_PRED_IDX"] = row["DIAMETER_CLASS_PRED_IDX"]
        df.loc[i, "DIAMETER_CLASS_PRED_STR"] = row["DIAMETER_CLASS_PRED_STR"]
        df.loc[i, "DIAMETER_CLASS_PRED_CONF"] = row["DIAMETER_CLASS_PRED_CONF"]

        for cls in class_names:
            col = f"DIAMETER_PROB_{safe_class_name(cls)}"
            df.loc[i, col] = row[col]

    # final filled columns
    if "DIAMETER_CLASS_FINAL_IDX" not in df.columns:
        df["DIAMETER_CLASS_FINAL_IDX"] = df["DIAMETER_CLASS_IDX"] if "DIAMETER_CLASS_IDX" in df.columns else np.nan
    else:
        df["DIAMETER_CLASS_FINAL_IDX"] = df["DIAMETER_CLASS_FINAL_IDX"].fillna(df.get("DIAMETER_CLASS_IDX"))

    if "DIAMETER_CLASS_FINAL_STR" not in df.columns:
        df["DIAMETER_CLASS_FINAL_STR"] = df["DIAMETER_CLASS_STR"] if "DIAMETER_CLASS_STR" in df.columns else np.nan
    else:
        df["DIAMETER_CLASS_FINAL_STR"] = df["DIAMETER_CLASS_FINAL_STR"].fillna(df.get("DIAMETER_CLASS_STR"))

    if "DIAMETER_CLASS_IDX" not in df.columns:
        if conf_threshold is None:
            fill_mask = df["DIAMETER_CLASS_PRED_IDX"].notna()
        else:
            fill_mask = (
                df["DIAMETER_CLASS_PRED_IDX"].notna()
                & (df["DIAMETER_CLASS_PRED_CONF"] >= conf_threshold)
            )
    else:
        if conf_threshold is None:
            fill_mask = df["DIAMETER_CLASS_IDX"].isna() & df["DIAMETER_CLASS_PRED_IDX"].notna()
        else:
            fill_mask = (
                df["DIAMETER_CLASS_IDX"].isna()
                & df["DIAMETER_CLASS_PRED_IDX"].notna()
                & (df["DIAMETER_CLASS_PRED_CONF"] >= conf_threshold)
            )

    df.loc[fill_mask, "DIAMETER_CLASS_FINAL_IDX"] = df.loc[fill_mask, "DIAMETER_CLASS_PRED_IDX"]
    df.loc[fill_mask, "DIAMETER_CLASS_FINAL_STR"] = df.loc[fill_mask, "DIAMETER_CLASS_PRED_STR"]

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
        g = g[g["DIAMETER_CLASS_PRED_IDX"].notna()].copy()
        if len(g) == 0:
            continue

        probs_list = []
        for p in g["PROBS_TREE_DIAMETER"]:
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
        avg_conf = float(g["DIAMETER_CLASS_PRED_CONF"].mean())

        out = {
            "ID": tree_id,
            "N_IMAGES": len(g),
            "DIAMETER_CLASS_PRED_IDX": pred_idx,
            "DIAMETER_CLASS_PRED_STR": class_names[pred_idx],
            "DIAMETER_CLASS_CONF_MEAN": avg_conf,
            "PROBS_TREE_DIAMETER": json.dumps(mean_probs.tolist()),
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
        df_pred["DIAMETER_CLASS_STR"]
        .value_counts()
        .reindex(class_names, fill_value=0)
    )

    after_counts = (
        df_pred["DIAMETER_CLASS_FINAL_STR"]
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
    plt.xlabel("Diameter class (cm)")
    plt.title("Tree Diameter Distribution: Before vs After Model Filling")
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
        class_names = ["<2", "2-20", "20-40", "40-60", "60+"]

    x_pos = np.arange(len(class_names))

    conf_col = "DIAMETER_CLASS_CONF_MEAN" if "DIAMETER_CLASS_CONF_MEAN" in df_pred.columns else "DIAMETER_CLASS_PRED_CONF"

    mask = (
        df_pred["DIAMETER_CLASS_STR"].isna()
        & df_pred["DIAMETER_CLASS_PRED_STR"].notna()
        & df_pred[conf_col].notna()
        & (df_pred[conf_col] > conf_threshold)
        & df_pred["PROBS_TREE_DIAMETER"].notna()
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

        pred_label = row["DIAMETER_CLASS_PRED_STR"]
        conf = float(row[conf_col])

        tree_box = row["TREE_BOX"] if "TREE_BOX" in row.index else None
        box_kind, box_data = _parse_tree_box(tree_box)

        ax.set_title(f"Predicted: {pred_label} ({conf*100:.1f}%)", fontsize=12)
        ax.axis("off")

        ax2 = fig.add_subplot(inner[1])

        probs = row["PROBS_TREE_DIAMETER"]
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
    parser = argparse.ArgumentParser(description="Infer missing tree diameters and save updated manifest + plots.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset cache folder")
    parser.add_argument("--run_path", type=str, required=True, help="Path to trained diameter model run folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--conf_threshold", type=float, default=None)
    parser.add_argument(
        "--output_name",
        type=str,
        default="tree_dataset_manifest_with_diameter_predictions.csv",
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

    df_pred, out_csv_path, class_names = predict_missing_diameters(
        dataset_dir=dataset_dir,
        run_path=run_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_threshold=args.conf_threshold,
        output_name=args.output_name,
    )

    tree_predictions = aggregate_tree_predictions(df_pred, class_names=class_names)

    if len(tree_predictions) > 0:
        merge_cols = [
            "ID",
            "DIAMETER_CLASS_PRED_STR",
            "DIAMETER_CLASS_CONF_MEAN",
            "PROBS_TREE_DIAMETER",
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

    plots_dir = run_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dist_plot_path = plots_dir / f"diameter_distribution_before_after_{timestamp}.png"
    examples_plot_path = plots_dir / f"inferred_diameter_examples_{timestamp}.png"

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