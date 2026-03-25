#!/usr/bin/env python3
"""
build_tree_dataset.py

Build a local tree training dataset from a CommuniMap export.

Pipeline:
1. Read input table (.xlsx or .csv)
2. Keep TREE == yes
3. Explode MEDIA_2635_* columns -> one row per image
4. Derive height and diameter labels
5. Download image in memory
6. CLIP filter -> keep likely tree images
7. GroundingDINO -> detect best tree box
8. Crop padded box
9. Depth Anything -> save depth map of crop
10. Save final manifest CSV

Example:
python build_tree_dataset.py \
  --input "/path/to/communiMap data March 26.xlsx" \
  --out_root "/home/fss6k/datasets/tree_dataset" \
  --dataset_name "communimap_trees_march26"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
from faker import Faker

fake = Faker()

# --------------------------------------------------
# Shared utilities
# --------------------------------------------------
sys.path.append("/home/fss6k/GALLANT-WS3/projects/utilities_scr")

import CLIP_filtering_system
import groundingdino_box_cropping
import depth_anything


# --------------------------------------------------
# Constants
# --------------------------------------------------
MEDIA_COLS = [f"MEDIA_2635_{i}" for i in range(6)]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def clean_missing(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x.lower() in {"", "nan", "none", "null", "n/a", "na"}:
            return np.nan
    return x


def extract_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    m = re.search(r"[-+]?\d*\.?\d+", str(x).replace(",", "."))
    return float(m.group()) if m else np.nan


def explode_media(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "ID",
        "TREE",
        "TREE_HEIGHT_METHOD",
        "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT",
        "TREE_CIRCUMFERENCE_METHOD",
        "CIRCUMFERENCE_IN_CM",
        "TREE_TRUNK_SIZE",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    base = df[keep_cols + [c for c in MEDIA_COLS if c in df.columns]].copy()

    rows = []
    for _, row in base.iterrows():
        for media_col in MEDIA_COLS:
            if media_col not in row.index:
                continue
            val = row.get(media_col)
            if pd.isna(val) or str(val).strip() == "":
                continue

            out = {k: row.get(k) for k in keep_cols}
            out["MEDIA_COL"] = media_col
            out["MEDIA_SRC"] = str(val)
            rows.append(out)

    return pd.DataFrame(rows)


def add_height_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fixed height classes.

    Height bins:
      0-5, 5-10, 10-15, 15+
    """
    df = df.copy()

    df["TREE_HEIGHT_IN_METERS_CLEAN"] = df["TREE_HEIGHT_IN_METERS"].map(extract_numeric)
    df["ESTIMATED_TREE_HEIGHT_CLEAN"] = df["ESTIMATED_TREE_HEIGHT"].map(extract_numeric)
    df["HEIGHT_VALUE_M"] = df["TREE_HEIGHT_IN_METERS_CLEAN"].fillna(df["ESTIMATED_TREE_HEIGHT_CLEAN"])

    bins = [-np.inf, 5, 10, 15, np.inf]
    labels = ["0-5", "5-10", "10-15", "15+"]

    df["HEIGHT_CLASS_STR"] = pd.cut(
        df["HEIGHT_VALUE_M"],
        bins=bins,
        labels=labels,
        right=False
    ).astype("object")

    label_map = {k: i for i, k in enumerate(labels)}
    df["HEIGHT_CLASS_IDX"] = df["HEIGHT_CLASS_STR"].map(label_map)

    return df


def add_diameter_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fixed trunk diameter (DBH) classes from circumference in cm.

    Diameter bins:
      0-20, 20-40, 40-60, 60+
    """
    df = df.copy()

    df["CIRCUMFERENCE_IN_CM_CLEAN"] = df["CIRCUMFERENCE_IN_CM"].map(extract_numeric)
    df["DBH_CM"] = df["CIRCUMFERENCE_IN_CM_CLEAN"] / np.pi

    bins = [-np.inf, 20, 40, 60, np.inf]
    labels = ["0-20", "20-40", "40-60", "60+"]

    # [0,20), [20,40), [40,60), [60,inf)
    df["DIAMETER_CLASS_STR"] = pd.cut(
        df["DBH_CM"],
        bins=bins,
        labels=labels,
        right=False
    ).astype("object")

    label_map = {k: i for i, k in enumerate(labels)}
    df["DIAMETER_CLASS_IDX"] = df["DIAMETER_CLASS_STR"].map(label_map)

    return df


def make_local_filename(row) -> str:
    return f"{row['ID']}_{row['MEDIA_COL']}.jpg"


def load_remote_image(url: str, timeout: int = 20):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def expand_box_for_context(box, image_size, scale_x=1.25, scale_y=1.45):
    return groundingdino_box_cropping.expand_box(
        box,
        image_size=image_size,
        scale_x=scale_x,
        scale_y=scale_y,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--out_root", required=True, type=str)
    ap.add_argument("--dataset_name", required=True, type=str)

    ap.add_argument("--clip_model_path", default="/home/fss6k/models/clip-vit-base-patch32", type=str)
    ap.add_argument("--clip_threshold", default=0.5, type=float)

    ap.add_argument("--dino_model_path", default="/home/fss6k/models/grounding_dino_tiny", type=str)
    ap.add_argument("--dino_threshold", default=0.2, type=float)
    ap.add_argument("--dino_text_threshold", default=0.25, type=float)
    ap.add_argument("--dino_score_min", default=0.2, type=float)

    ap.add_argument("--depth_ckpt", default="/home/fss6k/models/depth_anything_v2_vitb.pth", type=str)

    args = ap.parse_args()

    input_path = Path(args.input)
    out_root = Path(args.out_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = out_root / f"{args.dataset_name}_{fake.name().replace(' ', '')}_{timestamp}"

    crop_dir = dataset_dir / "rgb_crops"
    depth_dir = dataset_dir / "depth_maps"
    manifest_dir = dataset_dir / "manifests"

    for d in [crop_dir, depth_dir, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Building dataset in: {dataset_dir}")

    # ------------------------------------------
    # 1. Read and clean
    # ------------------------------------------
    df = load_table(input_path)
    df = df.applymap(clean_missing)

    if "TREE" in df.columns:
        df = df[df["TREE"].astype(str).str.lower().str.strip().eq("yes")].copy()

    df_img = explode_media(df)
    if df_img.empty:
        raise RuntimeError("No media rows found.")

    # Derive labels here, before image processing
    df_img = add_height_labels(df_img)
    df_img = add_diameter_labels(df_img)

    # ------------------------------------------
    # 2. Load CLIP, DINO, Depth once
    # ------------------------------------------
    clip_model, clip_processor, clip_device = CLIP_filtering_system.load_clip_model(
        model_path=args.clip_model_path
    )

    dino_processor, dino_model, dino_device = groundingdino_box_cropping.load_grounding_dino(
        model_path=args.dino_model_path
    )

    depth_model, depth_device = depth_anything.load_depth_anything_v2(
        ckpt_path=args.depth_ckpt
    )

    # ------------------------------------------
    # 3. Process image-by-image in memory
    # ------------------------------------------
    tree_scores = []
    top_prompts = []
    keep_flags = []
    crop_paths = []
    depth_paths = []
    box_jsons = []

    for _, row in tqdm(df_img.iterrows(), total=len(df_img), desc="Filter + crop + depth"):
        img = load_remote_image(row["MEDIA_SRC"])

        if img is None:
            tree_scores.append(None)
            top_prompts.append("LOAD_ERROR")
            keep_flags.append(False)
            crop_paths.append(None)
            depth_paths.append(None)
            box_jsons.append(None)
            continue

        # CLIP filter
        clip_res = CLIP_filtering_system.score_images_batch(
            image_paths=[img],
            model=clip_model,
            processor=clip_processor,
            device=clip_device,
            batch_size=1,
            threshold=args.clip_threshold,
        )[0]

        tree_scores.append(clip_res["tree_score"])
        top_prompts.append(clip_res["top_prompt"])
        keep_flags.append(bool(clip_res["is_tree"]))

        if not clip_res["is_tree"]:
            crop_paths.append(None)
            depth_paths.append(None)
            box_jsons.append(None)
            continue

        # GroundingDINO crop + depth
        try:
            result = groundingdino_box_cropping.detect_tree_box(
                img,
                processor=dino_processor,
                model=dino_model,
                device=dino_device,
                threshold=args.dino_threshold,
                text_threshold=args.dino_text_threshold,
                score_min=args.dino_score_min,
            )

            box = result["best_box"]

            if box is None:
                crop = img
                box_data = None
            else:
                box = expand_box_for_context(box, img.size)
                crop = img.crop(box)
                box_data = box

            crop_name = make_local_filename(row)
            crop_path = crop_dir / crop_name
            crop.save(crop_path, format="JPEG", quality=95)

            depth = depth_anything.infer_depth(crop, depth_model, depth_device)
            depth_path = depth_dir / crop_name.replace(".jpg", "_depth.npy")
            np.save(depth_path, depth.astype(np.float32))

            crop_paths.append(str(crop_path))
            depth_paths.append(str(depth_path))
            box_jsons.append(json.dumps(box_data) if box_data is not None else None)

        except Exception:
            crop_paths.append(None)
            depth_paths.append(None)
            box_jsons.append(None)

    df_img["tree_score"] = tree_scores
    df_img["top_prompt"] = top_prompts
    df_img["is_tree"] = keep_flags
    df_img["RGB_CROP_PATH"] = crop_paths
    df_img["DEPTH_PATH"] = depth_paths
    df_img["TREE_BOX"] = box_jsons

    df_img = df_img[
        df_img["is_tree"] &
        df_img["RGB_CROP_PATH"].notna() &
        df_img["DEPTH_PATH"].notna()
    ].reset_index(drop=True)

    print("After full pipeline:", len(df_img))

    # ------------------------------------------
    # 4. Save manifest
    # ------------------------------------------
    keep_cols = [
        "ID",
        "MEDIA_COL",
        "MEDIA_SRC",
        "RGB_CROP_PATH",
        "DEPTH_PATH",
        "TREE_BOX",
        "HEIGHT_VALUE_M",
        "HEIGHT_CLASS_STR",
        "HEIGHT_CLASS_IDX",
        "CIRCUMFERENCE_IN_CM_CLEAN",
        "DBH_CM",
        "DIAMETER_CLASS_STR",
        "DIAMETER_CLASS_IDX",
        "tree_score",
        "top_prompt",
    ]
    keep_cols = [c for c in keep_cols if c in df_img.columns]

    manifest_path = manifest_dir / "tree_dataset_manifest.csv"
    df_img[keep_cols].to_csv(manifest_path, index=False)

    summary = {
        "created": timestamp,
        "codename": fake.name().replace(" ", ""),
        "dataset_dir": str(dataset_dir),
        "input_file": str(input_path),
        "n_rows_final": int(len(df_img)),
        "n_unique_ids": int(df_img["ID"].nunique()),
        "n_height_labeled": int(df_img["HEIGHT_CLASS_IDX"].notna().sum()),
        "n_diameter_labeled": int(df_img["DIAMETER_CLASS_IDX"].notna().sum()),
    }

    with open(manifest_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("Saved manifest:", manifest_path)
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()