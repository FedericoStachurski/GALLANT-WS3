#!/usr/bin/env python3
"""
embed_local_multimodal.py

Local multimodal embeddings for CommuniMap.

What it does
------------
1. Load CommuniMap data from CSV or XLSX
2. If DESCRIPTION/text is missing or too short, generate a BLIP caption from the image
3. Embed text + images using a chosen vision-language model family:
   - clip   -> sentence-transformers CLIP
   - siglip -> Google SigLIP2
4. Save:
   - *_text.npy
   - *_image.npy
   - *_meta.json

Usage
-----
python embed_local.py <path_to_data> [output_name] \
    --vlm siglip \
    --vl_model google/siglip2-base-patch16-384 \
    --blip_model Salesforce/blip-image-captioning-base \
    -batch_size_BLIP 16 \
    -batch_size_TEXT 128 \
    -batch_size_IMG 8 \
    -MIN_TEXT_LEN 10

Examples
--------
# Use default SigLIP2
python embed_local_.py "/path/to/data.xlsx" out_name --vlm siglip

# Use default CLIP
python embed_local.py "/path/to/data.xlsx" out_name --vlm clip

# Use a local SigLIP2 folder
python embed_local.py "/path/to/data.xlsx" out_name \
    --vlm siglip \
    --vl_model /home/fss6k/models/siglip2-base-patch16-384

# Use a local BLIP folder
python embed_local.py "/path/to/data.xlsx" out_name \
    --blip_model /home/fss6k/models/blip-image-captioning-base
"""

import os
import sys
import json
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =============================================================
# Import CommuniMap loader
# =============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR, "..", "utilities_scr")
sys.path.insert(0, os.path.abspath(UTILS_DIR))

from load_data_communimap import load_communimap_data  # noqa: E402


# =============================================================
# DEFAULT CONFIG
# =============================================================

DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"

MODEL_REGISTRY = {
    "clip": {
        "type": "sentence_transformer_clip",
        "default": "sentence-transformers/clip-ViT-B-32",
        "description": "SentenceTransformers CLIP (shared text-image space)",
    },
    "siglip": {
        "type": "siglip",
        "default": "google/siglip2-base-patch16-384",
        "description": "Google SigLIP2 (shared text-image space)",
    },
}

TEXT_BATCH = 64
IMG_BATCH = 64
BLIP_BATCH = 1
MIN_TEXT_LEN = 0


# =============================================================
# ARGUMENT PARSING
# =============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed CommuniMap text and images with CLIP or SigLIP2."
    )

    parser.add_argument(
        "data_path",
        help="Path to input CommuniMap CSV or XLSX file",
    )
    parser.add_argument(
        "output_name",
        nargs="?",
        default=None,
        help="Output prefix name (default: derived from input filename)",
    )

    parser.add_argument(
        "--vlm",
        choices=list(MODEL_REGISTRY.keys()),
        default="siglip",
        help="Vision-language model family to use",
    )
    parser.add_argument(
        "--vl_model",
        default=None,
        help="Optional local path or model ID overriding the default model for the chosen VLM family",
    )
    parser.add_argument(
        "--blip_model",
        default=DEFAULT_BLIP_MODEL,
        help="Local path or model ID for BLIP captioning model",
    )

    parser.add_argument("-batch_size_BLIP", type=int, default=1)
    parser.add_argument("-batch_size_TEXT", type=int, default=64)
    parser.add_argument("-batch_size_IMG", type=int, default=64)
    parser.add_argument("-MIN_TEXT_LEN", type=int, default=0)

    parser.add_argument(
        "--max_blip_tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens for BLIP caption generation",
    )

    return parser.parse_args()


# =============================================================
# MODEL RESOLUTION
# =============================================================

def resolve_model_source(model_spec: str) -> str:
    """
    If model_spec exists locally, return its absolute path.
    Otherwise treat it as a remote model ID.
    """
    p = Path(model_spec).expanduser()
    if p.exists():
        resolved = str(p.resolve())
        print(f"[MODEL] Using local path: {resolved}")
        return resolved

    print(f"[MODEL] Using model ID: {model_spec}")
    return model_spec


# =============================================================
# IMAGE DOWNLOAD
# =============================================================

def download_image(url, timeout=7):
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


# =============================================================
# BLIP CAPTIONING
# =============================================================

def load_blip(model_spec: str):
    source = resolve_model_source(model_spec)
    print(f"[BLIP] Loading model: {source}")
    processor = BlipProcessor.from_pretrained(source)
    model = BlipForConditionalGeneration.from_pretrained(source).to(device)
    model.eval()
    return processor, model


def caption_image_blip(image, processor, model, max_new_tokens=20):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            do_sample=True,
            temperature=0.7,
        )

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


def caption_batch_blip(images, processor, model, max_new_tokens=20):
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            do_sample=True,
            temperature=0.7,
        )

    captions = [processor.decode(seq, skip_special_tokens=True).strip() for seq in out]
    return captions


def fill_short_descriptions_with_blip(
    df,
    blip_model_spec: str,
    batch_size: int = 1,
    max_new_tokens: int = 20,
):
    """
    Replace missing/short text using BLIP captioning.
    """
    processor, blip = load_blip(blip_model_spec)

    norm_text = df["text"].fillna("").astype(str).str.strip()
    lens = norm_text.str.len()

    print(f"[BLIP] MIN_TEXT_LEN = {MIN_TEXT_LEN}")
    print(f"[BLIP] Total rows: {len(df)}")
    print(f"[BLIP] Rows with len >= MIN_TEXT_LEN: {(lens >= MIN_TEXT_LEN).sum()}")
    print(f"[BLIP] Rows with len <  MIN_TEXT_LEN: {(lens < MIN_TEXT_LEN).sum()}")

    url_series = df["primary_image"].fillna("").astype(str)
    has_url = url_series.str.strip().ne("")
    needs_caption = lens < MIN_TEXT_LEN
    mask = needs_caption & has_url

    target_indices = df.index[mask].tolist()
    target_urls = url_series[mask].tolist()

    print(f"[BLIP] Will caption {len(target_indices)} rows out of {len(df)}")

    if not target_indices:
        print("[BLIP] No rows require captioning.")
        return df

    if batch_size <= 1:
        for idx, url in tqdm(
            list(zip(target_indices, target_urls)),
            total=len(target_indices),
            desc="[BLIP] Captioning (single)",
        ):
            img = download_image(url)
            if img is None:
                continue

            try:
                caption = caption_image_blip(
                    img,
                    processor,
                    blip,
                    max_new_tokens=max_new_tokens,
                )
                df.at[idx, "text"] = caption
                print(f"[BLIP] Row {idx}: {caption}")
            except Exception as e:
                print(f"[BLIP] Failed row {idx}: {e}")

        return df

    print(f"[BLIP] Using batch size = {batch_size}")

    for start in tqdm(
        range(0, len(target_indices), batch_size),
        desc="[BLIP] Captioning (batch)",
    ):
        end = start + batch_size
        batch_indices = target_indices[start:end]
        batch_urls = target_urls[start:end]

        images = []
        valid_indices = []

        for idx, url in zip(batch_indices, batch_urls):
            img = download_image(url)
            if img is None:
                print(f"[BLIP] Could not download image for row {idx}, skipping.")
                continue
            images.append(img)
            valid_indices.append(idx)

        if not images:
            continue

        try:
            captions = caption_batch_blip(
                images,
                processor,
                blip,
                max_new_tokens=max_new_tokens,
            )
            for idx, cap in zip(valid_indices, captions):
                df.at[idx, "text"] = cap
                print(f"[BLIP] Row {idx}: {cap}")
        except Exception as e:
            print(f"[BLIP] Batch {start}-{end} failed: {e}")

    return df


# =============================================================
# CLIP (SentenceTransformers) EMBEDDING
# =============================================================

def load_st_clip_model(model_spec: str):
    source = resolve_model_source(model_spec)
    print(f"[CLIP] Loading SentenceTransformer model: {source}")
    model = SentenceTransformer(source, device=device)
    return model


def embed_texts_clip(texts, model_spec: str, batch_size: int):
    model = load_st_clip_model(model_spec)

    vecs = []
    N = len(texts)

    for i in tqdm(range(0, N, batch_size), desc="[TEXT] Embedding (CLIP)"):
        batch = texts[i:i + batch_size]
        print(f"[TEXT] {i}–{i+len(batch)-1} / {N-1}")

        v = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        vecs.append(v)

    return np.vstack(vecs)


def embed_images_clip(urls, model_spec: str, batch_size: int):
    model = load_st_clip_model(model_spec)

    N = len(urls)
    all_vecs = None

    for i in tqdm(range(0, N, batch_size), desc="[IMG] Embedding (CLIP)"):
        batch_urls = urls[i:i + batch_size]
        imgs = []
        idxs = []

        for j, url in enumerate(batch_urls):
            img = download_image(url)
            if img is not None:
                imgs.append(img)
                idxs.append(i + j)

        if not imgs:
            continue

        print(f"[IMG] {i}–{i+len(batch_urls)-1} / {N-1}")

        v = model.encode(
            imgs,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        if all_vecs is None:
            d = v.shape[1]
            all_vecs = np.zeros((N, d), dtype=np.float32)

        for k, idx in enumerate(idxs):
            all_vecs[idx] = v[k]

    if all_vecs is None:
        print("[IMG] No images found, returning zero matrix.")
        return np.zeros((N, 1), dtype=np.float32)

    return all_vecs


# =============================================================
# SIGLIP2 EMBEDDING
# =============================================================

def load_siglip_model(model_spec: str):
    source = resolve_model_source(model_spec)
    print(f"[SIGLIP] Loading model: {source}")
    processor = AutoProcessor.from_pretrained(source)
    model = AutoModel.from_pretrained(source).to(device)
    model.eval()
    return processor, model


def embed_texts_siglip(texts, model_spec: str, batch_size: int):
    processor, model = load_siglip_model(model_spec)

    vecs = []
    N = len(texts)

    for i in tqdm(range(0, N, batch_size), desc="[TEXT] Embedding (SigLIP)"):
        batch = texts[i:i + batch_size]
        print(f"[TEXT] {i}–{i+len(batch)-1} / {N-1}")

        inputs = processor(
            text=batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            feats = model.get_text_features(**inputs)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy().astype(np.float32))

    return np.vstack(vecs)


def embed_images_siglip(urls, model_spec: str, batch_size: int):
    processor, model = load_siglip_model(model_spec)

    N = len(urls)
    all_vecs = None

    for i in tqdm(range(0, N, batch_size), desc="[IMG] Embedding (SigLIP)"):
        batch_urls = urls[i:i + batch_size]
        imgs = []
        idxs = []

        for j, url in enumerate(batch_urls):
            img = download_image(url)
            if img is not None:
                imgs.append(img)
                idxs.append(i + j)

        if not imgs:
            continue

        print(f"[IMG] {i}–{i+len(batch_urls)-1} / {N-1}")

        inputs = processor(
            images=imgs,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            feats = model.get_image_features(**inputs)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu().numpy().astype(np.float32)

        if all_vecs is None:
            d = feats.shape[1]
            all_vecs = np.zeros((N, d), dtype=np.float32)

        for k, idx in enumerate(idxs):
            all_vecs[idx] = feats[k]

    if all_vecs is None:
        print("[IMG] No images found, returning zero matrix.")
        return np.zeros((N, 1), dtype=np.float32)

    return all_vecs


# =============================================================
# MAIN
# =============================================================

def main():
    global TEXT_BATCH, IMG_BATCH, BLIP_BATCH, MIN_TEXT_LEN

    args = parse_args()

    TEXT_BATCH = args.batch_size_TEXT
    IMG_BATCH = args.batch_size_IMG
    BLIP_BATCH = args.batch_size_BLIP
    MIN_TEXT_LEN = args.MIN_TEXT_LEN

    data_path = os.path.abspath(args.data_path)

    model_info = MODEL_REGISTRY[args.vlm]
    vlm_type = model_info["type"]
    vlm_model = args.vl_model if args.vl_model else model_info["default"]

    if args.output_name is not None:
        out_name = args.output_name
    else:
        base = os.path.splitext(os.path.basename(data_path))[0]
        out_name = f"{base}_{args.vlm}_multimodal"

    # If output_name is an absolute path, keep it as-is
    if os.path.isabs(out_name):
        out_prefix = out_name
    else:
        out_prefix = os.path.join(os.path.dirname(data_path), out_name)

    print(f"Data path:      {data_path}")
    print(f"Output prefix:  {out_prefix}")
    print(f"VLM family:     {args.vlm}")
    print(f"VLM model:      {vlm_model}")
    print(f"BLIP model:     {args.blip_model}")
    print(f"TEXT_BATCH:     {TEXT_BATCH}")
    print(f"IMG_BATCH:      {IMG_BATCH}")
    print(f"BLIP_BATCH:     {BLIP_BATCH}")
    print(f"MIN_TEXT_LEN:   {MIN_TEXT_LEN}")

    # 1. Load data
    df = load_communimap_data(data_path)
    print(f"[DATA] Loaded {len(df)} rows")

    print("[DEBUG] Example text values:", df["text"].head(10).tolist())
    lens = df["text"].fillna("").astype(str).str.strip().str.len()
    print("[DEBUG] Length stats:")
    print(lens.describe())
    print("[DEBUG] Rows with len >= 1:", (lens >= 1).sum())

    # 2. Fill short descriptions with BLIP
    df = fill_short_descriptions_with_blip(
        df,
        blip_model_spec=args.blip_model,
        batch_size=BLIP_BATCH,
        max_new_tokens=args.max_blip_tokens,
    )

    # 3. Prepare payload
    texts = df["text"].fillna("").astype(str).tolist()
    texts = [" ".join(t.split()[:50]) for t in texts]  # truncate to ~50 words
    imgs = df["primary_image"].tolist()
    ids = df["id"].tolist()
    lats = df["LATITUDE"].tolist()
    lons = df["LONGITUDE"].tolist()

    # 4. Embed with chosen family
    if vlm_type == "sentence_transformer_clip":
        text_vecs = embed_texts_clip(texts, model_spec=vlm_model, batch_size=TEXT_BATCH)
        img_vecs = embed_images_clip(imgs, model_spec=vlm_model, batch_size=IMG_BATCH)

    elif vlm_type == "siglip":
        text_vecs = embed_texts_siglip(texts, model_spec=vlm_model, batch_size=TEXT_BATCH)
        img_vecs = embed_images_siglip(imgs, model_spec=vlm_model, batch_size=IMG_BATCH)

    else:
        raise ValueError(f"Unsupported VLM type: {vlm_type}")

    print("[TEXT] Shape:", text_vecs.shape)
    print("[IMG] Shape:", img_vecs.shape)

    # 5. Save embeddings
    np.save(out_prefix + "_text.npy", text_vecs)
    np.save(out_prefix + "_image.npy", img_vecs)

    # 6. Save metadata
    meta = []
    for i in range(len(df)):
        meta.append(
            {
                "id": ids[i],
                "text": texts[i],
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "primary_image": imgs[i],
            }
        )

    with open(out_prefix + "_meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] Saved:")
    print(" ", out_prefix + "_text.npy")
    print(" ", out_prefix + "_image.npy")
    print(" ", out_prefix + "_meta.json")


if __name__ == "__main__":
    main()