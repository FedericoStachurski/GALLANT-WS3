#!/usr/bin/env python3
"""
load_data_communimap.py

Usage:
    from load_data_communimap import load_communimap_data
    df = load_communimap_data("/path/to/communimap_spots.csv", use_blip=True)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Optional: only import BLIP stuff if needed
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration


def sniff_delimiter(path):
    """Try to detect delimiter automatically."""
    try:
        sample = pd.read_csv(path, sep=None, engine="python", nrows=5)
        return sample
    except Exception:
        # fallback: assume comma
        return pd.read_csv(path, sep=",", nrows=5)


def load_raw_dataframe(path):
    """
    Load either a CSV or an Excel file.
    - .csv  -> tries auto/comma/semicolon
    - .xlsx/.xls -> uses read_excel
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        print(f"Detected Excel file: {path}")
        return pd.read_excel(path)

    if suffix == ".csv":
        try:
            df = pd.read_csv(path, sep=None, engine="python")
            return df
        except Exception as e1:
            print("Auto-detect failed, trying comma...")

            try:
                df = pd.read_csv(path, sep=",")
                return df
            except Exception as e2:
                print("Comma failed, trying semicolon...")

                try:
                    df = pd.read_csv(path, sep=";")
                    return df
                except Exception as e3:
                    raise ValueError(
                        f"Could not parse CSV.\nAuto: {e1}\nComma: {e2}\nSemicolon: {e3}"
                    )

    raise ValueError(f"Unsupported file type: {suffix}")


def pick_primary_image(row, media_cols):
    """
    Pick one single image per row:
      1. IMAGE column if present
      2. Otherwise first non-empty MEDIA_* column
    """
    img = row.get("IMAGE")
    if isinstance(img, str) and img.strip():
        return img.strip()

    for col in media_cols:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return None


# ---------- BLIP utils ----------

_BLIP_LOADED = False
_BLIP_PROCESSOR = None
_BLIP_MODEL = None
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"


def _load_blip():
    global _BLIP_LOADED, _BLIP_PROCESSOR, _BLIP_MODEL
    if _BLIP_LOADED:
        return _BLIP_PROCESSOR, _BLIP_MODEL
    print(f"[BLIP] Loading model: {BLIP_MODEL_NAME}")
    _BLIP_PROCESSOR = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
    _BLIP_LOADED = True
    return _BLIP_PROCESSOR, _BLIP_MODEL


def _download_image(url, timeout=5):
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"[BLIP] Failed to download {url}: {e}")
        return None


def _caption_image_blip(image, processor, model, max_new_tokens=40):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()


def fill_missing_text_with_blip(df, min_len=40, max_rows=None):
    """
    For rows where df['text'] is missing/short AND a primary_image exists,
    generate a caption with BLIP and overwrite df['text'].

    min_len: minimum length to consider text "good enough".
    max_rows: optionally limit number of rows to caption (for testing).
    """
    processor, model = _load_blip()

    texts = df["text"].fillna("").astype(str)
    needs_caption = (
        (texts.str.len() < min_len)
        & df["primary_image"].notna()
        & df["primary_image"].astype(str).str.strip().ne("")
    )

    indices = np.where(needs_caption.values)[0]
    if max_rows is not None:
        indices = indices[:max_rows]

    print(f"[BLIP] Will caption {len(indices)} rows (min_len={min_len}).")

    for idx in indices:
        url = df.loc[idx, "primary_image"]
        img = _download_image(url)
        if img is None:
            continue
        try:
            caption = _caption_image_blip(img, processor, model)
            if caption:
                df.at[idx, "text"] = caption
                print(f"[BLIP] Row {idx}: {caption}")
        except Exception as e:
            print(f"[BLIP] Caption failed for row {idx}: {e}")

    return df


def load_communimap_data(path, use_blip=False, blip_min_len=40, blip_max_rows=None):
    """
    Loads, cleans, and structures CommuniMap spot data into:

        id, source_id, text, LATITUDE, LONGITUDE, primary_image

    If use_blip=True, rows with missing/very short text but with an image
    will get a BLIP-generated caption used as text.

    Returns a cleaned pandas DataFrame.
    """

    print(f"Loading CommuniMap CSV from: {path}")

    df = load_raw_dataframe(path)
    print(f"Loaded dataframe: {df.shape[0]} rows, {df.shape[1]} columns")

    media_cols = [c for c in df.columns if c.startswith("MEDIA_")]

    needed = {"DESCRIPTION", "LATITUDE", "LONGITUDE", "IMAGE"}
    missing_cols = needed - set(df.columns)
    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")
        raise ValueError("Input CSV missing required columns.")

    df["primary_image"] = df.apply(
        lambda row: pick_primary_image(row, media_cols),
        axis=1
    )

    df["text"] = df["DESCRIPTION"].fillna("").astype(str).str.strip()

    if use_blip:
        df = fill_missing_text_with_blip(
            df,
            min_len=blip_min_len,
            max_rows=blip_max_rows,
        )

    mask = (
        df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
    )

    clean = df[mask].copy().reset_index(drop=True)

    # Preserve original source ID if available
    if "ID" in clean.columns:
        clean["source_id"] = clean["ID"].astype(str).str.strip()
    else:
        clean["source_id"] = clean.index.astype(str)

    # Optional internal embedding/local row ID
    clean["id"] = clean.index.astype(str)

    clean = clean[["id", "source_id", "text", "LATITUDE", "LONGITUDE", "primary_image"]]

    print(f"Cleaned dataset: {clean.shape[0]} usable rows.")
    return clean


# For command-line debugging
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_data_communimap.py <path_to_csv> [--blip]")
        sys.exit(1)

    path = sys.argv[1]
    use_blip = ("--blip" in sys.argv)

    df = load_communimap_data(path, use_blip=use_blip)
    print(df.head().to_string())
