#!/usr/bin/env python3
"""
embed_local_multimodal.py

Local (free) embeddings for CommuniMap:

    - If DESCRIPTION is missing/short, generate a BLIP caption.
    - Embed TEXT using MiniLM.
    - Embed IMAGES using CLIP.
    - Save *_text.npy, *_image.npy, *_meta.json.

Usage:
    python embed_local_multimodal.py <path_to_csv> [output_name]
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm 

from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:', device)

# === Import CommuniMap loader ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR, "..", "utilities_scr")
sys.path.insert(0, os.path.abspath(UTILS_DIR))

from load_data_communimap import load_communimap_data


# === CONFIG ===
TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"  # Changed to CLIP for joint embedding space
IMAGE_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"





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
# BLIP caption utilities
# =============================================================

def load_blip():
    print(f"[BLIP] Loading model: {BLIP_MODEL_NAME}")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return processor, model, device


def caption_image_blip(image, processor, model, device, max_new_tokens=40):
    # image is a PIL.Image
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()

def caption_batch_blip(images, processor, model, device, max_new_tokens=40):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    captions = [
        processor.decode(seq, skip_special_tokens=True).strip()
        for seq in out
    ]
    return captions




def fill_short_descriptions_with_blip(df, batch_size: int = 1, max_new_tokens: int = 40):
    """
    Replace missing/short text using BLIP captioning.

    If batch_size == 1 -> processes each row individually.
    If batch_size > 1  -> downloads images and captions them in batches.

    Uses MIN_TEXT_LEN as: minimum number of characters required to KEEP existing text.
    If len(normalized_text) < MIN_TEXT_LEN -> BLIP will generate a caption.
    """
    processor, blip, device = load_blip()

    # --- Normalize text column once (handles NaN/None/other types) ---
    norm_text = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

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

    # First, collect all rows that actually need captioning
    target_indices = []
    target_urls = []

    for i, row in df.iterrows():
        raw_txt = row["text"]
        url = row["primary_image"]

        # --- normalize text ---
        if isinstance(raw_txt, str):
            txt = raw_txt.strip()
        elif raw_txt is None:
            txt = ""
        else:
            # handle NaN or weird types
            try:
                # pandas NaN is float and not equal to itself
                import math
                if isinstance(raw_txt, float) and math.isnan(raw_txt):
                    txt = ""
                else:
                    txt = str(raw_txt).strip()
            except Exception:
                txt = ""

        # Skip if text already long enough
        if len(txt) >= MIN_TEXT_LEN:
            continue

        # Skip if no valid image URL
        if not isinstance(url, str) or not url.strip():
            continue

        target_indices.append(i)
        target_urls.append(url)

    if not target_indices:
        print("[BLIP] No rows require captioning.")
        return df

    # ---- Non-batch mode: single image at a time ----
    if batch_size <= 1:
        for idx, url in tqdm(
            list(zip(target_indices, target_urls)),
            total=len(target_indices),
            desc="[BLIP] Captioning (single)"
        ):
            img = download_image(url)
            if img is None:
                continue

            try:
                caption = caption_image_blip(img, processor, blip, device, max_new_tokens=max_new_tokens)
                df.at[idx, "text"] = caption
                print(f"[BLIP] Row {idx}: {caption}")
            except Exception as e:
                print(f"[BLIP] Failed row {idx}: {e}")

        return df

    # ---- Batch mode ----
    print(f"[BLIP] Using batch size = {batch_size}")
    for start in tqdm(
        range(0, len(target_indices), batch_size),
        desc="[BLIP] Captioning (batch)"
    ):
        end = start + batch_size
        batch_indices = target_indices[start:end]
        batch_urls = target_urls[start:end]

        images = []
        valid_indices = []

        # Download images for this batch
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
                images, processor, blip, device, max_new_tokens=max_new_tokens
            )

            # Map captions back to dataframe rows
            for idx, cap in zip(valid_indices, captions):
                df.at[idx, "text"] = cap
                print(f"[BLIP] Row {idx}: {cap}")

        except Exception as e:
            print(f"[BLIP] Batch {start}-{end} failed: {e}")

    return df




# =============================================================
# TEXT embedding
# =============================================================

def embed_texts(texts, model_name=TEXT_MODEL_NAME):
    print(f"[TEXT] Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    vecs = []
    N = len(texts)

    for i in tqdm(range(0, N, TEXT_BATCH), desc="[TEXT] Embedding" ):
        batch = texts[i:i + TEXT_BATCH]
        print(f"[TEXT] {i}–{i+len(batch)-1} / {N-1}")
        v = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        vecs.append(v)

    return np.vstack(vecs)


# =============================================================
# IMAGE embedding
# =============================================================

def embed_images(urls, model_name=IMAGE_MODEL_NAME, device=device):
    print(f"[IMG] Loading model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    N = len(urls)
    all_vecs = None  # will allocate once we know d

    for i in tqdm(range(0, N, IMG_BATCH), desc="[IMG] Embedding"):
        batch_urls = urls[i:i + IMG_BATCH]
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

        # Runs on GPU because model is on `device`
        v = model.encode(
            imgs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

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
# MAIN
# =============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python embed_local_multimodal.py <csv> [output_name] -batch_size_BLIP <n> -batch_size_TEXT <n> -batch_size_IMG <n> -MIN_TEXT_LEN <n>")
        sys.exit(1)

    csv_path = os.path.abspath(sys.argv[1])

    batch_size_blip = None
    batch_size_text = None
    batch_size_img = None
    min_text_len = None

    if "-batch_size_BLIP" in sys.argv:
        try:
            batch_size_blip = int(sys.argv[sys.argv.index("-batch_size_BLIP") + 1])
        except (IndexError, ValueError):
            print("Invalid BLIP batch size specified.")
    if "-batch_size_TEXT" in sys.argv:
        try:
            batch_size_text = int(sys.argv[sys.argv.index("-batch_size_TEXT") + 1])
        except (IndexError, ValueError):
            print("Invalid TEXT batch size specified.")
    if "-batch_size_IMG" in sys.argv:
        try:
            batch_size_img = int(sys.argv[sys.argv.index("-batch_size_IMG") + 1])
        except (IndexError, ValueError):
            print("Invalid IMG batch size specified.")
            sys.exit(1)
    if "-MIN_TEXT_LEN" in sys.argv:
        try:
            min_text_len = int(sys.argv[sys.argv.index("-MIN_TEXT_LEN") + 1])
        except (IndexError, ValueError):
            print("Invalid MIN_TEXT_LEN specified.")
            sys.exit(1)

    # after parsing batch_size
    global TEXT_BATCH, IMG_BATCH, BLIP_BATCH, MIN_TEXT_LEN
    if batch_size_blip is not None:
        BLIP_BATCH = batch_size_blip
    else:
        BLIP_BATCH = 1  # default
    if batch_size_text is not None:
        TEXT_BATCH = batch_size_text
    else:   
        TEXT_BATCH = 64  # default
    if batch_size_img is not None:
        IMG_BATCH = batch_size_img
    else:
        IMG_BATCH = 64  # default
    if min_text_len is not None:
        MIN_TEXT_LEN = min_text_len
    else:
        MIN_TEXT_LEN = 0  # default

    if len(sys.argv) >= 3:
        out_name = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_name = f"{base}_local_multimodal"

    out_prefix = os.path.join(os.path.dirname(csv_path), out_name)

    print(f"CSV path:      {csv_path}")
    print(f"Output prefix: {out_prefix}")

    # 1. Load raw data
    df = load_communimap_data(csv_path)
    print(f"[DATA] Loaded {len(df)} rows")

    print("[DEBUG] Example text values:", df["text"].head(10).tolist())

    lens = df["text"].fillna("").astype(str).str.strip().str.len()
    print("[DEBUG] Length stats:")
    print(lens.describe())
    print("[DEBUG] Rows with len >= 1:", (lens >= 1).sum())


    # 2. Fix text using BLIP
    df = fill_short_descriptions_with_blip(df, batch_size=BLIP_BATCH)

    texts = df["text"].tolist()
    imgs = df["primary_image"].tolist()
    ids = df["id"].tolist()
    lats = df["LATITUDE"].tolist()
    lons = df["LONGITUDE"].tolist()

    # 3. Embed TEXT
    text_vecs = embed_texts(texts)
    print("[TEXT] Shape:", text_vecs.shape)

    # 4. Embed IMAGES
    img_vecs = embed_images(imgs)
    print("[IMG] Shape:", img_vecs.shape)

    # 5. Save embeddings
    np.save(out_prefix + "_text.npy", text_vecs)
    np.save(out_prefix + "_image.npy", img_vecs)

    # 6. Save metadata
    meta = []
    for i in range(len(df)):
        meta.append({
            "id": ids[i],
            "text": texts[i],
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "primary_image": imgs[i],
        })

    with open(out_prefix + "_meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] Saved:")
    print(" ", out_prefix + "_text.npy")
    print(" ", out_prefix + "_image.npy")
    print(" ", out_prefix + "_meta.json")


if __name__ == "__main__":
    main()

 


