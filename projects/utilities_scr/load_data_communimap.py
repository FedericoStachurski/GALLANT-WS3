#!/usr/bin/env python3
"""
load_data_communimap.py

Usage:
    from load_data_communimap import load_communimap_data
    df = load_communimap_data("/path/to/communimap_spots.csv")
"""

import pandas as pd
import numpy as np


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
    Attempts to load the CSV with auto-detected delimiter.
    Falls back to comma or semicolon if needed.
    """
    # First attempt: let pandas sniff
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


def pick_primary_image(row, media_cols):
    """
    Pick one single image per row:
      1. IMAGE column if present
      2. Otherwise first non-empty MEDIA_* column
    """
    # Prefer the main IMAGE column
    img = row.get("IMAGE")
    if isinstance(img, str) and img.strip():
        return img.strip()

    # Fallback to any MEDIA_* column
    for col in media_cols:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return None


def load_communimap_data(path):
    """
    Loads, cleans, and structures CommuniMap spot data into:

    id, text, lat, lon, primary_image

    Returns a cleaned pandas DataFrame.
    """

    print(f"Loading CommuniMap CSV from: {path}")

    df = load_raw_dataframe(path)
    print(f"Loaded dataframe: {df.shape[0]} rows, {df.shape[1]} columns")

    # Identify media columns
    media_cols = [c for c in df.columns if c.startswith("MEDIA_")]

    # Core columns we need (some may not exist in all rows)
    needed = {"DESCRIPTION", "LATITUDE", "LONGITUDE", "IMAGE"}
    missing_cols = needed - set(df.columns)

    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")

    # Build the primary_image column
    df["primary_image"] = df.apply(
        lambda row: pick_primary_image(row, media_cols),
        axis=1
    )

    # Build the text field
    df["text"] = df["DESCRIPTION"].fillna("").astype(str).str.strip()

    # Drop rows with no coordinates or no usable text
    mask = (
        df["LATITUDE"].notna() &
        df["LONGITUDE"].notna() &
        (df["text"].str.len() > 0)
    )

    clean = df[mask].copy().reset_index(drop=True)

    # Create id column (if no id exists)
    clean["id"] = clean.index.astype(str)

    # Keep only the columns needed downstream
    clean = clean[["id", "text", "LATITUDE", "LONGITUDE", "primary_image"]]

    print(f"Cleaned dataset: {clean.shape[0]} usable rows.")
    return clean


# For command-line debugging
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python load_data_communimap.py <path_to_csv>")
        sys.exit(1)

    path = sys.argv[1]
    df = load_communimap_data(path)
    print(df.head().to_string())
