#!/usr/bin/env python3
"""
train_bayesian_height_cnn.py

Train/Validation only (no test set)

Features:
- Automatic 80/20 stratified train/val split from a single manifest CSV
- Bayesian CNN via MC Dropout
- ResNet18 or ResNet50 backbone
- Optional RGB + depth input
- Saves:
    - best_model.pt                  (full checkpoint)
    - best_model_weights_only.pt    (weights only)
    - final_model.pt                (last epoch checkpoint)
    - history.json
    - best_val_metrics.json
    - best_val_confusion_matrix.png
    - best_val_posteriors.csv
    - best_val_example_posteriors.png

Posterior figure:
- 8 subplots total
- 4 rows x 2 columns
- for 4 validation samples:
    left  = tree image
    right = posterior over classes
"""

from __future__ import annotations

import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table import Table
from rich.live import Live

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ===================== SEED =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===================== JSON =====================
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(make_json_safe(obj), f, indent=2)


# ===================== DATA =====================
def load_and_split_manifest(manifest_path: str | Path, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    df = df[df["HEIGHT_CLASS_IDX"].notna()].copy()
    df["HEIGHT_CLASS_IDX"] = df["HEIGHT_CLASS_IDX"].astype(int)

    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["HEIGHT_CLASS_IDX"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"

    full_df = pd.concat([train_df, val_df], ignore_index=True)

    print("\n=== Split Summary ===")
    print(full_df["split"].value_counts())
    print("\nClass distribution:")
    print(full_df.groupby(["split", "HEIGHT_CLASS_STR"]).size())

    return full_df


# ===================== DATASET =====================
class TreeHeightDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        image_size: int = 224,
        use_depth: bool = True,
        augment: bool = False,
    ):
        self.df = df.copy()
        self.df = self.df[
            self.df["HEIGHT_CLASS_IDX"].notna() &
            (self.df["split"] == split)
        ].copy()

        self.df["HEIGHT_CLASS_IDX"] = self.df["HEIGHT_CLASS_IDX"].astype(int)

        self.df = self.df[
            self.df["RGB_CROP_PATH"].apply(lambda p: Path(p).exists()) &
            self.df["DEPTH_PATH"].apply(lambda p: Path(p).exists())
        ].reset_index(drop=True)

        self.use_depth = use_depth
        self.image_size = image_size

        if augment:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
                transforms.ToTensor(),
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def _load_depth(self, path: str) -> torch.Tensor:
        d = np.load(path).astype(np.float32)

        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        elif d.ndim == 3 and d.shape[0] == 1:
            d = d[0]

        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        dmin, dmax = d.min(), d.max()
        if dmax > dmin:
            d = (d - dmin) / (dmax - dmin)
        else:
            d = np.zeros_like(d, dtype=np.float32)

        d = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
        d = F.interpolate(
            d.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return d

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rgb = Image.open(row["RGB_CROP_PATH"]).convert("RGB")
        rgb = self.rgb_transform(rgb)

        if self.use_depth:
            depth = self._load_depth(row["DEPTH_PATH"])
            x = torch.cat([rgb, depth], dim=0)
        else:
            x = rgb

        return {
            "image": x,
            "label": int(row["HEIGHT_CLASS_IDX"]),
            "id": int(row["ID"]),
            "rgb_path": str(row["RGB_CROP_PATH"]),
            "height_class_str": str(row["HEIGHT_CLASS_STR"]),
        }


# ===================== MODEL =====================
class MCDropoutHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, p: float = 0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BayesianResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 4,
        pretrained: bool = False,
        dropout_p: float = 0.35,
        resnet_variant: str = "18",
    ):
        super().__init__()

        if resnet_variant == "18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif resnet_variant == "50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported resnet variant: {resnet_variant}. Use '18' or '50'.")

        old_conv = backbone.conv1
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            with torch.no_grad():
                if in_channels == 4:
                    backbone.conv1.weight[:, :3] = old_conv.weight
                    backbone.conv1.weight[:, 3:4] = old_conv.weight[:, :1]
                else:
                    nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = MCDropoutHead(in_feats, num_classes, dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


# ===================== MC DROPOUT =====================
def enable_mc_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model: nn.Module, x: torch.Tensor, mc_passes: int = 30):
    model.eval()
    enable_mc_dropout(model)

    probs = []
    for _ in range(mc_passes):
        probs.append(torch.softmax(model(x), dim=1).unsqueeze(0))

    probs = torch.cat(probs, dim=0)
    mean = probs.mean(0)
    entropy = -(mean * mean.clamp_min(1e-8).log()).sum(1)

    return mean, entropy


# ===================== TRAIN =====================
def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    loss_total = 0.0
    y_true, y_pred = [], []

    for b in loader:
        x = b["image"].to(device)
        y = b["label"].to(device)

        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        loss_total += loss.item() * x.size(0)
        y_true += y.detach().cpu().tolist()
        y_pred += logits.argmax(1).detach().cpu().tolist()

    return {
        "loss": loss_total / len(loader.dataset),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, crit, device, mc_passes: int = 30):
    model.eval()

    y_true, y_pred = [], []
    loss_total = 0.0

    for b in loader:
        x = b["image"].to(device)
        y = b["label"].to(device)

        logits = model(x)
        loss_total += crit(logits, y).item() * x.size(0)

        mean, _ = mc_predict(model, x, mc_passes=mc_passes)
        preds = mean.argmax(1)

        y_true += y.detach().cpu().tolist()
        y_pred += preds.detach().cpu().tolist()

    return {
        "loss": loss_total / len(loader.dataset),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ===================== SAVING FIGURES =====================
def save_confusion_matrix_figure(cm, class_names, out_path: Path, title: str = "Confusion Matrix"):
    cm = np.asarray(cm)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def collect_val_posteriors(model, loader, device, mc_passes: int = 30):
    model.eval()

    all_ids = []
    all_y_true = []
    all_y_pred = []
    all_entropy = []
    all_mean_probs = []
    all_rgb_paths = []

    for b in loader:
        x = b["image"].to(device)
        y = b["label"].to(device)
        ids = b["id"]
        rgb_paths = b["rgb_path"]

        mean_probs, entropy = mc_predict(model, x, mc_passes=mc_passes)
        preds = mean_probs.argmax(dim=1)

        all_ids.extend([int(v) for v in ids])
        all_y_true.extend(y.detach().cpu().tolist())
        all_y_pred.extend(preds.detach().cpu().tolist())
        all_entropy.extend(entropy.detach().cpu().tolist())
        all_mean_probs.extend(mean_probs.detach().cpu().tolist())
        all_rgb_paths.extend(list(rgb_paths))

    return pd.DataFrame({
        "ID": all_ids,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "entropy": all_entropy,
        "mean_probs": all_mean_probs,
        "rgb_path": all_rgb_paths,
    })


def save_example_posteriors_figure(
    posterior_df: pd.DataFrame,
    class_names,
    out_path: Path,
    n_examples: int = 4,
):
    if len(posterior_df) == 0:
        return

    df_plot = posterior_df.sort_values("entropy", ascending=False).head(n_examples).copy()

    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 4 * n_examples))

    if n_examples == 1:
        axes = np.array([axes])

    for row_idx, (_, row) in enumerate(df_plot.iterrows()):
        # left: image
        ax_img = axes[row_idx, 0]
        img = Image.open(row["rgb_path"]).convert("RGB")
        ax_img.imshow(img)
        ax_img.set_title(
            f"ID={int(row['ID'])}\ntrue={class_names[int(row['y_true'])]} | "
            f"pred={class_names[int(row['y_pred'])]}"
        )
        ax_img.axis("off")

        # right: posterior bar plot
        ax_bar = axes[row_idx, 1]
        probs = np.asarray(row["mean_probs"], dtype=float)
        ax_bar.bar(class_names, probs)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("Prob")
        ax_bar.set_title(f"Posterior | entropy={float(row['entropy']):.3f}")
        ax_bar.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mc_passes", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resnet", type=str, default="18", help="ResNet variant: 18 or 50")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["<5", "5-10", "10-15", "15-20", "20+"]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_and_split_manifest(args.manifest, seed=args.seed)

    train_ds = TreeHeightDataset(
        df=df,
        split="train",
        image_size=args.image_size,
        augment=True,
        use_depth=args.use_depth,
    )
    val_ds = TreeHeightDataset(
        df=df,
        split="val",
        image_size=args.image_size,
        use_depth=args.use_depth,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = BayesianResNet(
        in_channels=4 if args.use_depth else 3,
        pretrained=args.pretrained,
        resnet_variant=args.resnet,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    console = Console()
    table = Table(title=f"Training Progress / {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for c in ["Epoch", "Train Loss", "Val Loss", "Val Bal Acc", "Val F1"]:
        table.add_column(c)

    history = []
    best = -1.0
    best_epoch = -1

    best_ckpt_path = out_dir / "best_model.pt"
    best_weights_path = out_dir / "best_model_weights_only.pt"
    final_ckpt_path = out_dir / "final_model.pt"

    with Live(table, refresh_per_second=1):
        for epoch in range(1, args.epochs + 1):
            train_m = train_one_epoch(model, train_loader, opt, crit, device)

            # faster MC during training-time validation
            train_mc_passes = min(args.mc_passes, 5)
            val_m = evaluate(model, val_loader, crit, device, mc_passes=train_mc_passes)

            history.append({
                "epoch": epoch,
                "train": train_m,
                "val": {
                    "loss": val_m["loss"],
                    "bal_acc": val_m["bal_acc"],
                    "macro_f1": val_m["macro_f1"],
                }
            })

            table.add_row(
                str(epoch),
                f"{train_m['loss']:.3f}",
                f"{val_m['loss']:.3f}",
                f"{val_m['bal_acc']:.3f}",
                f"{val_m['macro_f1']:.3f}",
            )

            save_json({"history": history}, out_dir / "history.json")

            if val_m["bal_acc"] > best:
                best = val_m["bal_acc"]
                best_epoch = epoch

                # save full checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "args": vars(args),
                        "best_val_bal_acc": best,
                    },
                    best_ckpt_path,
                )

                # weights only
                torch.save(model.state_dict(), best_weights_path)

                # save quick metrics
                save_json(
                    {
                        "epoch": epoch,
                        "loss": val_m["loss"],
                        "bal_acc": val_m["bal_acc"],
                        "macro_f1": val_m["macro_f1"],
                        "cm": val_m["cm"],
                        "y_true": val_m["y_true"],
                        "y_pred": val_m["y_pred"],
                    },
                    out_dir / "best_val_metrics.json",
                )

                # save confusion matrix png
                save_confusion_matrix_figure(
                    cm=val_m["cm"],
                    class_names=class_names,
                    out_path=out_dir / "best_val_confusion_matrix.png",
                    title=f"Best Val Confusion Matrix (epoch {epoch})",
                )

                # save full posteriors with full MC
                posterior_df = collect_val_posteriors(
                    model=model,
                    loader=val_loader,
                    device=device,
                    mc_passes=args.mc_passes,
                )
                posterior_df.to_csv(out_dir / "best_val_posteriors.csv", index=False)

                # save 4 examples -> 8 subplots total
                save_example_posteriors_figure(
                    posterior_df=posterior_df,
                    class_names=class_names,
                    out_path=out_dir / "best_val_example_posteriors.png",
                    n_examples=4,
                )

    # save final model checkpoint
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "args": vars(args),
        },
        final_ckpt_path,
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation balanced accuracy: {best:.4f}")
    print(f"Best model saved at: {best_ckpt_path}")
    print(f"Best weights-only model saved at: {best_weights_path}")
    print(f"Final model saved at: {final_ckpt_path}")
    print(f"History saved at: {out_dir / 'history.json'}")
    print(f"Best validation metrics saved at: {out_dir / 'best_val_metrics.json'}")
    print(f"Best validation confusion matrix saved at: {out_dir / 'best_val_confusion_matrix.png'}")
    print(f"Best validation posteriors saved at: {out_dir / 'best_val_posteriors.csv'}")
    print(f"Example posterior figure saved at: {out_dir / 'best_val_example_posteriors.png'}")


if __name__ == "__main__":
    main()