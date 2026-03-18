#!/usr/bin/env python3
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
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class TreeHeightDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        use_depth: bool = True,
        train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.use_depth = use_depth
        self.image_size = image_size
        self.train = train

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

            # Apply after normalization, RGB only
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

            # guard against bad arrays
            if depth.ndim != 2:
                raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")

            dmin, dmax = depth.min(), depth.max()
            depth = (depth - dmin) / (dmax - dmin + 1e-8)

            depth_img = Image.fromarray((depth * 255).astype(np.uint8)).convert("L")
            depth_img = self.depth_resize(depth_img)
            depth_tensor = self.depth_to_tensor(depth_img).to(dtype=dtype)  # (1,H,W)

        except Exception:
            depth_tensor = torch.zeros(1, self.image_size, self.image_size, dtype=dtype)

        return depth_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------------------------
        # RGB
        # -------------------------
        rgb = Image.open(row["RGB_CROP_PATH"]).convert("RGB")
        rgb_tensor = self.rgb_tfms(rgb)  # (3,H,W)

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
            x = torch.cat([rgb_tensor, depth_tensor], dim=0)  # (4,H,W)
        else:
            x = rgb_tensor  # (3,H,W)

        y = int(row["HEIGHT_CLASS_IDX"])
        return x, y


# =========================================================
# Models
# =========================================================
def build_resnet(backbone: str, num_classes: int, in_channels: int, device, dropout_rate: float = 0.1) -> nn.Module:
    backbone = backbone.lower()

    weights_map = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
    }

    if backbone not in weights_map:
        raise ValueError(f"Unsupported backbone: {backbone}")

    model = getattr(models, backbone)(weights=weights_map[backbone])

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
        with torch.no_grad():
            model.conv1.weight[:, :3] = old_conv.weight
            if in_channels > 3:
                for c in range(3, in_channels):
                    model.conv1.weight[:, c:c+1] = old_conv.weight.mean(dim=1, keepdim=True)

    model.fc = nn.Sequential(
        nn.Dropout(p= dropout_rate),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model.to(device)


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

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            out = model(x)
            loss = criterion(out, y)

            if train_mode:
                loss.backward()
                optimizer.step()

        preds = out.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

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

    return matches[-1]  # newest lexicographically because timestamp suffix


def load_manifest(dataset_dir: Path) -> pd.DataFrame:
    manifest_path = dataset_dir / "manifests" / "tree_dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    needed = ["RGB_CROP_PATH", "HEIGHT_CLASS_STR", "HEIGHT_CLASS_IDX"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    df = df[df["RGB_CROP_PATH"].notna() & df["HEIGHT_CLASS_IDX"].notna()].copy()

    # if depth is available it can be used later
    if "DEPTH_PATH" not in df.columns:
        df["DEPTH_PATH"] = np.nan

    return df.reset_index(drop=True)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    # dataset selection
    ap.add_argument("--out_root", type=str, default="/home/fss6k/datasets")
    ap.add_argument("--dataset_name", type=str, default=None,
                    help="Base dataset name, e.g. communimap_trees. Script will load newest matching folder.")
    ap.add_argument("--dataset_path", type=str, default=None,
                    help="Full dataset folder path. Overrides --dataset_name.")

    # model / data options
    ap.add_argument("--backbone", type=str, default="resnet50",
                    choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    ap.add_argument("--use_depth", action="store_true")
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
    df = load_manifest(dataset_dir)

    if args.use_depth:
        df = df[df["DEPTH_PATH"].notna()].copy()

    df = df.drop_duplicates(subset=["RGB_CROP_PATH"]).reset_index(drop=True)

    print(f"Using dataset: {dataset_dir}")
    print(f"Total rows: {len(df)}")
    print(df["HEIGHT_CLASS_STR"].value_counts().sort_index())

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=df["HEIGHT_CLASS_IDX"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("\nTrain class counts:")
    print(train_df["HEIGHT_CLASS_STR"].value_counts().sort_index())
    print("\nVal class counts:")
    print(val_df["HEIGHT_CLASS_STR"].value_counts().sort_index())

    # class weights
    classes = np.sort(train_df["HEIGHT_CLASS_IDX"].unique())
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["HEIGHT_CLASS_IDX"],
    )
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # datasets
    train_ds = TreeHeightDataset(
        train_df,
        image_size=args.image_size,
        use_depth=args.use_depth,
        train=True,
    )
    val_ds = TreeHeightDataset(
        val_df,
        image_size=args.image_size,
        use_depth=args.use_depth,
        train=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model
    in_channels = 4 if args.use_depth else 3
    num_classes = int(df["HEIGHT_CLASS_IDX"].nunique())

    model = build_resnet(
        backbone=args.backbone,
        num_classes=num_classes,
        in_channels=in_channels,
        device=device,
        dropout_rate=args.dropout_rate
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    # save directory
    models_root = Path(args.out_root) / "tree_models"
    models_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.backbone}_{'rgbd' if args.use_depth else 'rgb'}_{timestamp}"
    run_dir = models_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.pth"
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"

    config = {
        "dataset_dir": str(dataset_dir),
        "backbone": args.backbone,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "use_depth": args.use_depth,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "num_workers": args.num_workers,
        "device": str(device),
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