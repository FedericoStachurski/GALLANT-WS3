#!/usr/bin/env python3
"""
tree_dataset_cache.py

Build a local training manifest for tree size classification:
- Explode MEDIA columns -> one row per image
- Resolve MEDIA_SRC to local cached images in rgb_dir (no downloading required)
- Compute Depth Anything V2 depth maps -> .npy
- Run GroundingDINO (HF Transformers local) -> boxes (json) + optional overlay image
- Clean labels -> HEIGHT_Y, TRUNK_Y (class indices), with -1 for missing
- Save manifest as CSV/Parquet

Usage example:
python tree_dataset_cache.py \
  --input "/mnt/c/Users/fss6k/OneDrive - University of Glasgow/Desktop/GALLANT_WS3_documents_data/CommuniMap/Jan_26_data/spots 16_01_2026.xlsx" \
  --out_manifest /home/fss6k/embedded_data_CM/tree_manifest_jan26.csv \
  --rgb_dir /home/fss6k/embedded_data_CM/tree_image_cache_Jan26 \
  --depth_dir /home/fss6k/embedded_data_CM/tree_image_cache_Jan26_depth \
  --box_dir /home/fss6k/embedded_data_CM/tree_image_cache_Jan26_boxesDINO \
  --depth_repo /home/fss6k/models/Depth-Anything-V2 \
  --depth_ckpt /home/fss6k/models/depth_anything_v2_vitb.pth \
  --dino_model_path /home/fss6k/models/grounding_dino_tiny \
  --dino_labels "tree,tree trunk" \
  --box_thresh 0.35 \
  --text_thresh 0.25 \
  --pad_frac 0.30 \
  --save_box_overlay
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# -------------------------
# Label cleaning
# -------------------------
HEIGHT_MAP = {
    "0–2 metres": 0,
    "0-2 metres": 0,
    "0–2 meters": 0,
    "0-2 meters": 0,
    "2–5 metres": 1,
    "2-5 metres": 1,
    "2–5 meters": 1,
    "2-5 meters": 1,
    "5–10 metres": 2,
    "5-10 metres": 2,
    "5–10 meters": 2,
    "5-10 meters": 2,
    "10+ metres": 3,
    "10+ meters": 3,
}

TRUNK_MAP = {
    "0–30 cm": 0,
    "0-30 cm": 0,
    "30–60 cm": 1,
    "30-60 cm": 1,
    "60–100 cm": 2,
    "60-100 cm": 2,
    "100+ cm": 3,
}


def _norm_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # normalize different dash types
    s = s.replace("—", "–")
    # unify spacing
    s = " ".join(s.split())
    return s


def height_to_y(row: pd.Series) -> int:
    """
    Use ESTIMATED_TREE_HEIGHT (categorical) if present; else can bin numeric if available.
    - Return -1 if missing.
    """
    est = _norm_str(row.get("ESTIMATED_TREE_HEIGHT", ""))
    if est in HEIGHT_MAP:
        return HEIGHT_MAP[est]

    h = row.get("TREE_HEIGHT_IN_METERS", np.nan)
    try:
        if not pd.isna(h):
            h = float(h)
            if h < 2.0:
                return 0
            if h < 5.0:
                return 1
            if h < 10.0:
                return 2
            return 3
    except Exception:
        pass

    return -1


def trunk_to_y(row: pd.Series) -> int:
    """
    Use TREE_TRUNK_SIZE (categorical) if present; else bin numeric circumference if available.
    - Return -1 if missing.
    """
    trunk = _norm_str(row.get("TREE_TRUNK_SIZE", ""))
    if trunk in TRUNK_MAP:
        return TRUNK_MAP[trunk]

    circ = row.get("CIRCUMFERENCE_IN_CM", np.nan)
    try:
        if not pd.isna(circ):
            c = float(circ)
            if c < 30:
                return 0
            if c < 60:
                return 1
            if c < 100:
                return 2
            return 3
    except Exception:
        pass

    return -1


# -------------------------
# I/O + cache resolving
# -------------------------
def is_http(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def stable_name(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def resolve_cached_image(src: str, rgb_dir: Path) -> Optional[str]:
    """
    Resolve MEDIA_SRC to a local file in rgb_dir (no downloading).

    Handles:
      1) full local path already existing
      2) URL that was cached as md5(url).jpg
      3) bare filename in the cache folder
      4) filename with different extension
      5) last resort: scan cache for a partial stem match
    """
    if src is None:
        return None
    s = str(src).strip()
    if s == "":
        return None

    # 1) If it's already an existing local path, use it
    p = Path(s)
    if p.exists():
        return str(p)

    rgb_dir = Path(rgb_dir)
    if not rgb_dir.exists():
        return None

    # 2) If src looks like a URL, assume cached as md5(url).jpg
    if is_http(s):
        candidate = rgb_dir / f"{stable_name(s)}.jpg"
        if candidate.exists():
            return str(candidate)

    # 3) If src is a filename, try direct join
    base = Path(s).name
    candidate = rgb_dir / base
    if candidate.exists():
        return str(candidate)

    # 4) Try common extensions
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = rgb_dir / (Path(base).stem + ext)
        if candidate.exists():
            return str(candidate)

    # 5) Last resort scan (comment out if your folder is massive)
    stem = Path(base).stem
    try:
        for f in rgb_dir.glob(f"*{stem}*"):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                return str(f)
    except Exception:
        pass

    return None


# -------------------------
# Depth Anything V2 (local repo + local ckpt)
# -------------------------
def load_depth_anything(depth_repo: Path, ckpt: Path, device: str):
    import sys

    sys.path.append(str(depth_repo))
    from depth_anything_v2.dpt import DepthAnythingV2
    import torch

    model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def infer_depth(depth_model, image_path: str, device: str) -> np.ndarray:
    """
    Returns float32 depth map same resolution as image.
    """
    import cv2
    import torch

    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    inp = cv2.resize(bgr, (518, 518), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(inp).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        d = depth_model(x)
    d = d.squeeze().detach().cpu().numpy()

    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    d = d.astype(np.float32)
    return d


def cache_depth(
    depth_model, image_path: str, depth_dir: Path, device: str, overwrite: bool = False
) -> Optional[str]:
    depth_dir.mkdir(parents=True, exist_ok=True)
    out = depth_dir / (Path(image_path).stem + "_depth.npy")
    if out.exists() and not overwrite:
        return str(out)
    try:
        d = infer_depth(depth_model, image_path, device=device)
        np.save(out, d)
        return str(out)
    except Exception:
        return None


# -------------------------
# GroundingDINO via Transformers (HF local model)
# -------------------------
def load_dino_hf(model_path: Path, device: str):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(model_path), local_files_only=True
    ).to(device).eval()

    return processor, model


def dino_predict_box_hf(
    processor,
    model,
    image_path: str,
    text_labels: List[str],
    box_thresh: float,
    text_thresh: float,
    device: str,
) -> Optional[Dict]:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, text=[text_labels], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[image.size[::-1]],  # (H, W)
    )

    det = results[0]
    if det is None or len(det.get("boxes", [])) == 0:
        return None

    boxes = det["boxes"]  # pixel xyxy
    scores = det["scores"]
    labels = det["labels"]

    best_i = int(torch.argmax(scores).item())
    box = boxes[best_i].detach().cpu().tolist()
    score = float(scores[best_i].detach().cpu().item())
    label = labels[best_i] if isinstance(labels, list) else str(labels)

    W, H = image.size
    return {
        "box_xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
        "score": score,
        "phrase": str(label),
        "image_size": [int(W), int(H)],
        "labels_used": list(text_labels),
        "box_thresh": box_thresh,
        "text_thresh": text_thresh,
    }


def pad_box_xyxy(box_xyxy: List[float], W: int, H: int, pad_frac: float) -> List[int]:
    x1, y1, x2, y2 = box_xyxy
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = bw * pad_frac, bh * pad_frac
    x1p = max(0, int(x1 - px))
    y1p = max(0, int(y1 - py))
    x2p = min(W, int(x2 + px))
    y2p = min(H, int(y2 + py))
    return [x1p, y1p, x2p, y2p]


def save_box_assets(
    image_path: str,
    box_info: Dict,
    box_dir: Path,
    pad_frac: float,
    save_overlay: bool,
) -> Tuple[str, Optional[str]]:
    box_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    json_path = box_dir / f"{stem}_box.json"
    overlay_path = box_dir / f"{stem}_box.jpg"

    W, H = box_info["image_size"]
    padded = pad_box_xyxy(box_info["box_xyxy"], W=W, H=H, pad_frac=pad_frac)
    box_info2 = dict(box_info)
    box_info2["box_xyxy_padded"] = padded
    with open(json_path, "w") as f:
        json.dump(box_info2, f, indent=2)

    if save_overlay:
        img = Image.open(image_path).convert("RGB")
        dr = ImageDraw.Draw(img)
        x1, y1, x2, y2 = padded
        dr.rectangle([x1, y1, x2, y2], width=4)
        img.save(overlay_path, quality=92)
        return str(json_path), str(overlay_path)

    return str(json_path), None


# -------------------------
# Main manifest building
# -------------------------
def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, sep=",", engine="python")
    raise ValueError(f"Unsupported input: {path}")


def explode_media(df: pd.DataFrame, media_prefix="MEDIA_2635_") -> pd.DataFrame:
    media_cols = [c for c in df.columns if c.startswith(media_prefix)]
    keep_cols = [
        "ID",
        "TREE",
        "ESTIMATED_TREE_HEIGHT",
        "TREE_HEIGHT_IN_METERS",
        "TREE_TRUNK_SIZE",
        "CIRCUMFERENCE_IN_CM",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    base = df[keep_cols + media_cols].copy()

    rows = []
    for _, r in base.iterrows():
        tree_id = r.get("ID")
        for c in media_cols:
            v = r.get(c)
            if pd.isna(v) or str(v).strip() == "":
                continue
            row = {k: r.get(k) for k in keep_cols}
            row["TREE_ID"] = tree_id
            row["MEDIA_COL"] = c
            row["MEDIA_SRC"] = str(v)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--out_manifest", required=True, type=str)

    ap.add_argument("--rgb_dir", required=True, type=str)
    ap.add_argument("--depth_dir", required=True, type=str)
    ap.add_argument("--box_dir", required=True, type=str)

    ap.add_argument("--depth_repo", required=True, type=str)
    ap.add_argument("--depth_ckpt", required=True, type=str)

    ap.add_argument("--dino_model_path", required=True, type=str)
    ap.add_argument("--dino_labels", default="tree,tree trunk", type=str)

    ap.add_argument("--box_thresh", default=0.35, type=float)
    ap.add_argument("--text_thresh", default=0.25, type=float)
    ap.add_argument("--pad_frac", default=0.20, type=float)
    ap.add_argument("--save_box_overlay", action="store_true")

    ap.add_argument("--overwrite_depth", action="store_true")
    ap.add_argument("--overwrite_boxes", action="store_true")

    # optional: keep only labeled rows (training)
    ap.add_argument(
        "--keep_only_labeled",
        action="store_true",
        help="If set, keep only rows with HEIGHT_Y != -1 or TRUNK_Y != -1",
    )

    args = ap.parse_args()

    inp = Path(args.input)
    rgb_dir = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir)
    box_dir = Path(args.box_dir)

    device = (
        "cuda"
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
        else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    )

    labels = [s.strip() for s in args.dino_labels.split(",") if s.strip()]
    if not labels:
        raise ValueError("No labels parsed from --dino_labels")

    df = load_table(inp)

    # keep only "TREE == yes"
    if "TREE" in df.columns:
        df = df[df["TREE"].astype(str).str.lower().str.strip().eq("yes")].copy()

    df_img = explode_media(df)
    if df_img.empty:
        raise RuntimeError("No media rows after explode. Check MEDIA_2635_* columns and TREE filtering.")

    print("After explode_media:", len(df_img))
    print("MEDIA_SRC examples:", df_img["MEDIA_SRC"].head(3).tolist())

    # labels
    df_img["HEIGHT_Y"] = df_img.apply(height_to_y, axis=1)
    df_img["TRUNK_Y"] = df_img.apply(trunk_to_y, axis=1)

    # Resolve from existing cache folder
    df_img["IMAGE_PATH"] = df_img["MEDIA_SRC"].apply(lambda s: resolve_cached_image(s, rgb_dir))

    # drop missing RGB
    df_img = df_img[df_img["IMAGE_PATH"].notna()].copy()
    exists_mask = df_img["IMAGE_PATH"].apply(lambda p: Path(p).exists())
    print("IMAGE_PATH exists:", int(exists_mask.sum()), "/", len(df_img))
    df_img = df_img[exists_mask].reset_index(drop=True)

    print("After IMAGE_PATH filter:", len(df_img))
    if df_img.empty:
        raise RuntimeError(
            "All images were filtered out. MEDIA_SRC values do not resolve to files in --rgb_dir.\n"
            "Check that the cached filenames match the MEDIA_2635_* values, or that URLs were cached as md5(url).jpg."
        )

    # Load Depth Anything
    depth_model = load_depth_anything(Path(args.depth_repo), Path(args.depth_ckpt), device=device)

    # Cache depth
    df_img["DEPTH_PATH"] = df_img["IMAGE_PATH"].apply(
        lambda p: cache_depth(depth_model, p, depth_dir, device=device, overwrite=args.overwrite_depth)
    )
    df_img = df_img[df_img["DEPTH_PATH"].notna()].reset_index(drop=True)
    print("After depth cache:", len(df_img))

    if df_img.empty:
        raise RuntimeError("All rows dropped after depth caching. Check DepthAnything repo/ckpt/cv2.")

    # Load DINO model (HF local)
    processor, dino_model = load_dino_hf(Path(args.dino_model_path), device=device)

    # Boxes
    box_json_paths = []
    for p in df_img["IMAGE_PATH"].tolist():
        stem = Path(p).stem
        json_path = box_dir / f"{stem}_box.json"
        if json_path.exists() and not args.overwrite_boxes:
            box_json_paths.append(str(json_path))
            continue

        info = dino_predict_box_hf(
            processor=processor,
            model=dino_model,
            image_path=p,
            text_labels=labels,
            box_thresh=args.box_thresh,
            text_thresh=args.text_thresh,
            device=device,
        )
        if info is None:
            box_json_paths.append(None)
            continue

        jp, _ = save_box_assets(
            image_path=p,
            box_info=info,
            box_dir=box_dir,
            pad_frac=args.pad_frac,
            save_overlay=args.save_box_overlay,
        )
        box_json_paths.append(jp)

    df_img["BOX_JSON"] = box_json_paths
    print("Boxes found:", int(pd.Series(box_json_paths).notna().sum()), "/", len(box_json_paths))

    # Keep only rows where a box was found (your original behavior)
    df_img = df_img[df_img["BOX_JSON"].notna()].reset_index(drop=True)
    print("After BOX_JSON filter:", len(df_img))

    # Keep only rows with at least one label (optional)
    df_img["HAS_LABEL"] = (df_img["HEIGHT_Y"] != -1) | (df_img["TRUNK_Y"] != -1)
    if args.keep_only_labeled:
        df_img = df_img[df_img["HAS_LABEL"]].reset_index(drop=True)
        print("After keep_only_labeled:", len(df_img))

    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df_img.to_parquet(out_path, index=False)
    else:
        df_img.to_csv(out_path, index=False)

    print("Saved manifest:", out_path)
    print("Rows:", len(df_img))
    print("Trees:", df_img["TREE_ID"].nunique())
    print("Height labeled:", int((df_img["HEIGHT_Y"] != -1).sum()))
    print("Trunk labeled:", int((df_img["TRUNK_Y"] != -1).sum()))
    print(
        "Example columns:",
        [c for c in df_img.columns if c in ["TREE_ID", "HEIGHT_Y", "TRUNK_Y", "IMAGE_PATH", "DEPTH_PATH", "BOX_JSON"]],
    )


if __name__ == "__main__":
    main()
