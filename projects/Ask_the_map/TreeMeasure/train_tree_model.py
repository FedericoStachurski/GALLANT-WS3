#!/usr/bin/env python3
"""
train_tree_size_boxed_flex.py

Multitask training (HEIGHT_Y, TRUNK_Y) using:
- RGB (jpg)
- Depth (.npy)
- BOX_JSON (GroundingDINO) to crop RGB+Depth

Configurable:
- train/val split (val_frac)
- backbone depth: resnet18/resnet34/resnet50
- head MLP size/layers, optional BatchNorm, dropout
- freeze backbone

Saves to --out_dir:
- best.pt, last.pt
- metrics.csv, metrics.json
- summary.json

Example:
python train_tree_size_boxed_flex.py \
  --manifest /home/fss6k/embedded_data_CM/tree_manifest_jan26.csv \
  --out_dir /home/fss6k/models/tree_size_runs/jan26_flex_run1 \
  --backbone resnet18 \
  --head_dims 512 256 \
  --head_bn \
  --dropout 0.2 \
  --val_frac 0.2 \
  --epochs 20 --batch_size 16 --lr 3e-4 --weight_decay 1e-4 --img_size 384
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms


# -------------------------
# Box + Depth utils
# -------------------------
def load_box_xyxy(box_json_path: str) -> Optional[Tuple[int, int, int, int]]:
    try:
        with open(box_json_path, "r") as f:
            j = json.load(f)
        box = j.get("box_xyxy_padded") or j.get("box_xyxy")
        if box is None or len(box) != 4:
            return None
        x1, y1, x2, y2 = box
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None


def clamp_box(x1, y1, x2, y2, W, H) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(1, min(int(x2), W))
    y2 = max(1, min(int(y2), H))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    lo, hi = np.percentile(d, [2, 98])
    if hi - lo < 1e-6:
        dn = np.zeros_like(d, dtype=np.float32)
    else:
        dn = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    return (dn * 255.0).astype(np.uint8)


# -------------------------
# Dataset: crop by box, stack RGB+Depth
# -------------------------
class TreeSizeBoxedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, augment: bool):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

        if augment:
            # augment only on TRAIN split
            self.tf_rgb = transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size, scale=(0.80, 1.00)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf_rgb = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ]
            )

        self.tf_depth = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # (1,H,W) from uint8
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        img_path = str(r["IMAGE_PATH"])
        depth_path = str(r["DEPTH_PATH"])
        box_json = str(r["BOX_JSON"])

        rgb = Image.open(img_path).convert("RGB")
        W, H = rgb.size

        box = load_box_xyxy(box_json)
        if box is None:
            x1, y1, x2, y2 = 0, 0, W, H
        else:
            x1, y1, x2, y2 = clamp_box(*box, W=W, H=H)

        rgb_crop = rgb.crop((x1, y1, x2, y2))
        rgb_t = self.tf_rgb(rgb_crop)  # (3,S,S)

        d = np.load(depth_path).astype(np.float32)
        d = np.squeeze(d)
        if d.ndim != 2:
            raise ValueError(f"Depth array has unexpected shape {d.shape} for {depth_path}")

        Hd, Wd = d.shape
        x1d, y1d, x2d, y2d = clamp_box(x1, y1, x2, y2, W=Wd, H=Hd)
        d_crop = d[y1d:y2d, x1d:x2d]
        d_u8 = depth_to_uint8(d_crop)
        d_img = Image.fromarray(d_u8)
        d_t = self.tf_depth(d_img) / 255.0  # (1,S,S)

        x = torch.cat([rgb_t, d_t], dim=0)  # (4,S,S)
        y_h = int(r["HEIGHT_Y"])
        y_t = int(r["TRUNK_Y"])
        return x, torch.tensor(y_h, dtype=torch.long), torch.tensor(y_t, dtype=torch.long)


# -------------------------
# Model builders
# -------------------------
def build_backbone(name: str, in_ch: int = 4, pretrained: bool = True) -> Tuple[nn.Module, int]:
    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    # adapt first conv to 4ch
    old = net.conv1
    if in_ch != old.in_channels:
        new = nn.Conv2d(in_ch, old.out_channels,
                        kernel_size=old.kernel_size,
                        stride=old.stride,
                        padding=old.padding,
                        bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            # depth channel init = mean RGB filters
            mean_w = old.weight.mean(dim=1, keepdim=True)
            if in_ch > 3:
                new.weight[:, 3:4] = mean_w
        net.conv1 = new

    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


def build_mlp_head(in_dim: int, out_dim: int, head_dims: List[int], dropout: float, use_bn: bool) -> nn.Module:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in head_dims:
        layers.append(nn.Linear(prev, h))
        if use_bn:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU(inplace=True))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MultiHeadModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        head_dims: List[int],
        dropout: float,
        head_bn: bool,
        pretrained: bool = True,
        num_height: int = 4,
        num_trunk: int = 4,
    ):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, in_ch=4, pretrained=pretrained)
        self.head_h = build_mlp_head(feat_dim, num_height, head_dims, dropout, head_bn)
        self.head_t = build_mlp_head(feat_dim, num_trunk, head_dims, dropout, head_bn)

    def forward(self, x):
        f = self.backbone(x)
        return self.head_h(f), self.head_t(f)


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    n = 0
    h_correct = h_total = 0
    t_correct = t_total = 0

    for x, y_h, y_t in loader:
        x, y_h, y_t = x.to(device), y_h.to(device), y_t.to(device)
        logits_h, logits_t = model(x)

        loss_h = F.cross_entropy(logits_h, y_h, ignore_index=-1)
        loss_t = F.cross_entropy(logits_t, y_t, ignore_index=-1)
        loss = loss_h + loss_t

        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)

        mask_h = (y_h != -1)
        if mask_h.any():
            pred_h = logits_h.argmax(1)
            h_correct += int((pred_h[mask_h] == y_h[mask_h]).sum().item())
            h_total += int(mask_h.sum().item())

        mask_t = (y_t != -1)
        if mask_t.any():
            pred_t = logits_t.argmax(1)
            t_correct += int((pred_t[mask_t] == y_t[mask_t]).sum().item())
            t_total += int(mask_t.sum().item())

    return {
        "val_loss": loss_sum / max(1, n),
        "height_acc": (h_correct / h_total) if h_total > 0 else None,
        "trunk_acc": (t_correct / t_total) if t_total > 0 else None,
        "height_n": int(h_total),
        "trunk_n": int(t_total),
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    # Split control
    ap.add_argument("--val_frac", type=float, default=0.2, help="Fraction of data used for validation (0-1).")
    ap.add_argument("--seed", type=int, default=42)

    # Model control
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained backbone.")
    ap.add_argument("--head_dims", type=int, nargs="*", default=[512], help="MLP hidden dims for each head.")
    ap.add_argument("--head_bn", action="store_true", help="Use BatchNorm in head MLP.")
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights (train heads only).")

    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--augment", action="store_true", help="Enable RGB augmentations on training set only.")

    ap.add_argument("--w_height", type=float, default=1.0)
    ap.add_argument("--w_trunk", type=float, default=0.5)


    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Output files
    metrics_csv = out_dir / "metrics.csv"
    metrics_json = out_dir / "metrics.json"
    summary_json = out_dir / "summary.json"
    best_ckpt_path = out_dir / "best.pt"
    last_ckpt_path = out_dir / "last.pt"

    # init CSV
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "height_acc", "height_n", "trunk_acc", "trunk_n", "lr"])

    # Load manifest
    mpath = Path(args.manifest)
    df = pd.read_parquet(mpath) if mpath.suffix.lower() == ".parquet" else pd.read_csv(mpath)

    required = ["IMAGE_PATH", "DEPTH_PATH", "BOX_JSON", "HEIGHT_Y", "TRUNK_Y"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Manifest missing required column: {c}")

    df["HEIGHT_Y"] = df["HEIGHT_Y"].fillna(-1).astype(int)
    df["TRUNK_Y"] = df["TRUNK_Y"].fillna(-1).astype(int)

    # Must have files
    df = df[df["IMAGE_PATH"].apply(lambda p: Path(str(p)).exists())].copy()
    df = df[df["DEPTH_PATH"].apply(lambda p: Path(str(p)).exists())].copy()
    df = df[df["BOX_JSON"].apply(lambda p: Path(str(p)).exists())].copy()

    # Keep rows with at least one label
    df = df[(df["HEIGHT_Y"] != -1) | (df["TRUNK_Y"] != -1)].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No rows after filtering. Check manifest paths/boxes/labels.")

    print("Rows:", len(df))
    print("Height labeled:", int((df["HEIGHT_Y"] != -1).sum()))
    print("Trunk labeled:", int((df["TRUNK_Y"] != -1).sum()))
    print("val_frac:", args.val_frac)

    # split
    torch.manual_seed(args.seed)
    n_val = max(1, int(len(df) * args.val_frac))
    n_train = len(df) - n_val
    if n_train <= 0:
        raise RuntimeError("val_frac too large: no training samples left.")

    # build datasets with different augmentation settings
    full_ds_train = TreeSizeBoxedDataset(df, img_size=args.img_size, augment=args.augment)
    full_ds_val = TreeSizeBoxedDataset(df, img_size=args.img_size, augment=False)

    # same indices split for both
    train_ds, val_ds = random_split(
        range(len(df)),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # wrap subsets
    train_idx = list(train_ds)
    val_idx = list(val_ds)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = TreeSizeBoxedDataset(train_df, img_size=args.img_size, augment=args.augment)
    val_dataset = TreeSizeBoxedDataset(val_df, img_size=args.img_size, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # model
    model = MultiHeadModel(
        backbone_name=args.backbone,
        head_dims=list(args.head_dims) if args.head_dims else [],
        dropout=args.dropout,
        head_bn=args.head_bn,
        pretrained=args.pretrained,
    ).to(device)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen: training heads only.")

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for x, y_h, y_t in train_loader:
            x, y_h, y_t = x.to(device), y_h.to(device), y_t.to(device)
            opt.zero_grad(set_to_none=True)

            logits_h, logits_t = model(x)
            loss_h = F.cross_entropy(logits_h, y_h, ignore_index=-1)
            loss_t = F.cross_entropy(logits_t, y_t, ignore_index=-1)
            loss = args.w_height * loss_h + args.w_trunk * loss_t

            loss.backward()
            opt.step()

            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        train_loss = running / max(1, n_seen)
        val_m = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_m["val_loss"]),
            "height_acc": val_m["height_acc"],
            "height_n": int(val_m["height_n"]),
            "trunk_acc": val_m["trunk_acc"],
            "trunk_n": int(val_m["trunk_n"]),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | val_loss={row['val_loss']:.4f} | "
            f"height_acc={row['height_acc']} (n={row['height_n']}) | trunk_acc={row['trunk_acc']} (n={row['trunk_n']})"
        )

        with open(metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([row["epoch"], row["train_loss"], row["val_loss"], row["height_acc"], row["height_n"],
                        row["trunk_acc"], row["trunk_n"], row["lr"]])

        with open(metrics_json, "w") as f:
            json.dump({"history": history, "best_val_loss": best_val, "best_epoch": best_epoch, "args": vars(args)},
                      f, indent=2)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "history": history,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print("  saved best:", best_ckpt_path)

    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "history": history,
            "args": vars(args),
        },
        last_ckpt_path,
    )

    summary = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "height_labeled_total": int((df["HEIGHT_Y"] != -1).sum()),
        "trunk_labeled_total": int((df["TRUNK_Y"] != -1).sum()),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "final_train_loss": float(history[-1]["train_loss"]) if history else None,
        "final_val_loss": float(history[-1]["val_loss"]) if history else None,
        "final_height_acc": history[-1]["height_acc"] if history else None,
        "final_trunk_acc": history[-1]["trunk_acc"] if history else None,
        "args": vars(args),
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", best_ckpt_path)
    print("Saved:", last_ckpt_path)
    print("Saved:", metrics_csv)
    print("Saved:", metrics_json)
    print("Saved:", summary_json)


if __name__ == "__main__":
    main()
