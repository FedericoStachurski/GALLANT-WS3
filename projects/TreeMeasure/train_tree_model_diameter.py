from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, recall_score


# =========================================================
# Dataset
# =========================================================

class TreeDiameterDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        use_depth: bool = True,
        train: bool = True,
        height_feature_cols: list[str] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.use_depth = use_depth
        self.image_size = image_size
        self.train = train
        self.height_feature_cols = height_feature_cols or ["HEIGHT_CLASS_FINAL_IDX"]

        # -------------------------
        # RGB transforms
        # -------------------------
        if self.train:
            self.rgb_tfms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(8),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self.rgb_erasing = transforms.RandomErasing(
                p=0.20,
                scale=(0.02, 0.10),
                ratio=(0.3, 3.3),
                value="random",
            )
        else:
            self.rgb_tfms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            self.rgb_erasing = None

        # -------------------------
        # Depth transforms
        # -------------------------
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

        # -------------------------
        # RGB
        # -------------------------
        rgb = Image.open(row["RGB_CROP_PATH"]).convert("RGB")
        rgb_tensor = self.rgb_tfms(rgb)

        if self.train and self.rgb_erasing is not None:
            rgb_tensor = self.rgb_erasing(rgb_tensor)

        # -------------------------
        # Depth
        # -------------------------
        if self.use_depth:
            depth_tensor = self._load_depth_tensor(
                row["DEPTH_PATH"],
                dtype=rgb_tensor.dtype
            )
            x_img = torch.cat([rgb_tensor, depth_tensor], dim=0)
        else:
            x_img = rgb_tensor

        # -------------------------
        # Height features
        # -------------------------
        x_tab = row[self.height_feature_cols].astype(np.float32).values
        x_tab = torch.tensor(x_tab, dtype=torch.float32)

        # -------------------------
        # Target = trunk diameter
        # -------------------------
        y = int(row["DIAMETER_CLASS_IDX"])

        return x_img, x_tab, y


# =========================================================
# Models
# =========================================================

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


# =========================================================
# Train / eval
# =========================================================

def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds, all_labels = [], []

    for x_img, x_tab, y in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            out = model(x_img, x_tab)
            loss = criterion(out, y)

            if train_mode:
                loss.backward()
                optimizer.step()

        preds = out.argmax(dim=1)

        total_loss += loss.item() * x_img.size(0)
        total_correct += (preds == y).sum().item()
        total_samples += x_img.size(0)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        np.array(all_preds),
        np.array(all_labels),
    )


# =========================================================
# Utilities
# =========================================================

def find_dataset_dir(out_root: Path, dataset_name: str | None, full_dataset_path: str | None) -> Path:
    if full_dataset_path is not None:
        p = Path(full_dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset path not found: {p}")
        return p

    if dataset_name is None:
        raise ValueError("Provide either --dataset_path or --dataset_name")

    matches = sorted([p for p in out_root.glob(f"{dataset_name}_*") if p.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No dataset folders found for pattern: {dataset_name}_* under {out_root}")

    return matches[-1]


def load_manifest(dataset_dir: Path, use_height_probs: bool = False) -> tuple[pd.DataFrame, list[str]]:
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
            f"Manifest not found. Checked: {[str(p) for p in manifest_candidates]}"
        )

    df = pd.read_csv(manifest_path)

    # ---- required columns for diameter training
    required = ["RGB_CROP_PATH", "DIAMETER_CLASS_IDX", "DIAMETER_CLASS_STR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns for diameter training: {missing}")

    if "DEPTH_PATH" not in df.columns:
        df["DEPTH_PATH"] = np.nan

    # ---- choose height features
    if use_height_probs:
        height_feature_cols = sorted([c for c in df.columns if c.startswith("HEIGHT_PROB_")])
        if len(height_feature_cols) == 0:
            raise ValueError(
                "Requested --use_height_probs but no HEIGHT_PROB_* columns were found."
            )
    else:
        if "HEIGHT_CLASS_FINAL_IDX" in df.columns:
            height_feature_cols = ["HEIGHT_CLASS_FINAL_IDX"]
        elif "HEIGHT_CLASS_PRED_IDX" in df.columns:
            height_feature_cols = ["HEIGHT_CLASS_PRED_IDX"]
        elif "HEIGHT_CLASS_IDX" in df.columns:
            height_feature_cols = ["HEIGHT_CLASS_IDX"]
        else:
            raise ValueError(
                "No usable height feature column found. Expected one of "
                "HEIGHT_CLASS_FINAL_IDX, HEIGHT_CLASS_PRED_IDX, or HEIGHT_CLASS_IDX."
            )

    needed_for_training = ["RGB_CROP_PATH", "DIAMETER_CLASS_IDX"] + height_feature_cols
    df = df.dropna(subset=needed_for_training).copy()

    return df.reset_index(drop=True), height_feature_cols


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()

    # dataset selection
    ap.add_argument("--out_root", type=str, default="/home/fss6k/datasets")
    ap.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Base dataset name, e.g. communimap_trees. Script will load newest matching folder.",
    )
    ap.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Full dataset folder path. Overrides --dataset_name.",
    )

    # model / data options
    ap.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    )
    ap.add_argument("--use_depth", action="store_true")
    ap.add_argument("--use_height_probs", action="store_true")
    ap.add_argument("--image_size", type=int, default=224)

    # training hyperparameters
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dropout_rate", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.out_root)
    dataset_dir = find_dataset_dir(out_root, args.dataset_name, args.dataset_path)
    df, height_feature_cols = load_manifest(
        dataset_dir=dataset_dir,
        use_height_probs=args.use_height_probs,
    )

    if args.use_depth:
        df = df[df["DEPTH_PATH"].notna()].copy()

    df = df.drop_duplicates(subset=["RGB_CROP_PATH"]).reset_index(drop=True)

    print(f"Using dataset: {dataset_dir}")
    print(f"Total rows: {len(df)}")
    print("\nDiameter class counts:")
    print(df["DIAMETER_CLASS_STR"].value_counts().sort_index())
    print(f"\nUsing height features: {height_feature_cols}")

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=df["DIAMETER_CLASS_IDX"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("\nTrain class counts:")
    print(train_df["DIAMETER_CLASS_STR"].value_counts().sort_index())
    print("\nVal class counts:")
    print(val_df["DIAMETER_CLASS_STR"].value_counts().sort_index())

    # class weights
    classes = np.sort(train_df["DIAMETER_CLASS_IDX"].unique())
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["DIAMETER_CLASS_IDX"],
    )
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # datasets
    train_ds = TreeDiameterDataset(
        train_df,
        image_size=args.image_size,
        use_depth=args.use_depth,
        train=True,
        height_feature_cols=height_feature_cols,
    )
    val_ds = TreeDiameterDataset(
        val_df,
        image_size=args.image_size,
        use_depth=args.use_depth,
        train=False,
        height_feature_cols=height_feature_cols,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # model
    in_channels = 4 if args.use_depth else 3
    num_classes = int(df["DIAMETER_CLASS_IDX"].nunique())

    model = ResNetDiameterWithHeight(
        backbone=args.backbone,
        num_classes=num_classes,
        in_channels=in_channels,
        n_tabular_features=len(height_feature_cols),
        device=device,
        dropout_rate=args.dropout_rate,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    # save directory
    models_root = Path(args.out_root) / "tree_diameter_models"
    models_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"diameter_{args.backbone}_{'rgbd' if args.use_depth else 'rgb'}_{timestamp}"
    run_dir = models_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.pth"
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"

    config = {
        "dataset_dir": str(dataset_dir),
        "manifest_height_source": "HEIGHT_PROB_*" if args.use_height_probs else height_feature_cols,
        "backbone": args.backbone,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "use_depth": args.use_depth,
        "use_height_probs": args.use_height_probs,
        "height_feature_cols": height_feature_cols,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "num_workers": args.num_workers,
        "device": str(device),
        "target_column": "DIAMETER_CLASS_IDX",
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_recall_macro": [],
    }

    best_val_acc = 0.0
    best_epoch = -1

    print(f"\nSaving training run to: {run_dir}")

    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_preds, val_labels = run_epoch(
            model, val_loader, criterion, device
        )

        val_f1_macro = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_recall_macro = recall_score(val_labels, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)
        history["val_recall_macro"].append(val_recall_macro)

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"val_f1={val_f1_macro:.4f} val_recall={val_recall_macro:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "val_accuracy": best_val_acc,
                },
                best_model_path,
            )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
        "final_val_accuracy": float(history["val_acc"][-1]),
        "final_val_f1_macro": float(history["val_f1_macro"][-1]),
        "final_val_recall_macro": float(history["val_recall_macro"][-1]),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved model to: {best_model_path}")


if __name__ == "__main__":
    main()