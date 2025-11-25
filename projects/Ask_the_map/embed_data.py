#!/usr/bin/env python3
"""
embed_local.py

Local (free) embeddings for CommuniMap descriptions using sentence-transformers.

Usage:
    python embed_local.py <path_to_csv> [output_name]

Example:
    python embed_local.py \
      /home/staff3/fstachurski/CommuniMap/Nov_25_data/communimap_spots_20251107170112.csv \
      communimap_run_Nov_25_local
"""

import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Import the CommuniMap loader from ../utilities_scr ---

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR, "..", "utilities_scr")
sys.path.insert(0, os.path.abspath(UTILS_DIR))

from load_data_communimap import load_communimap_data  # noqa: E402


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts_local(texts, model_name=MODEL_NAME, batch_size=64):
    """
    Embed a list of texts using a local sentence-transformers model.
    Returns a numpy array of shape (n, d).
    """
    print(f"Loading local model: {model_name}")
    model = SentenceTransformer(model_name)

    vectors = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        print(f"Embedding batch {i}–{i + len(batch) - 1} / {total - 1}")
        batch_vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we'll decide later if we normalise
        )
        vectors.append(batch_vecs)

    return np.vstack(vectors)


def main():
    if len(sys.argv) < 2:
        print("Usage: python embed_local.py <path_to_csv> [output_name]")
        sys.exit(1)

    csv_path = os.path.abspath(sys.argv[1])
    out_dir = os.path.dirname(csv_path)

    # Output name (prefix)
    if len(sys.argv) >= 3:
        out_name = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_name = f"{base}_local_embeddings"

    out_prefix = os.path.join(out_dir, out_name)

    print(f"CSV path:      {csv_path}")
    print(f"Output prefix: {out_prefix}")

    # 1. Load and clean CommuniMap data
    df = load_communimap_data(csv_path)
    texts = df["text"].tolist()
    lats = df["LATITUDE"].tolist()
    lons = df["LONGITUDE"].tolist()
    ids = df["id"].tolist()
    imgs = df["primary_image"].tolist()

    print(f"Embedding {len(texts)} descriptions locally...")

    # 2. Embed
    vectors = embed_texts_local(texts)
    print("Embeddings shape:", vectors.shape)

    # 3. Save embeddings as .npy
    npy_path = f"{out_prefix}.npy"
    np.save(npy_path, vectors)
    print(f"Saved embeddings to: {npy_path}")

    # 4. Save metadata as .json
    meta = []
    for i in range(len(ids)):
        meta.append(
            {
                "id": ids[i],
                "text": texts[i],
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "primary_image": imgs[i],
            }
        )

    meta_path = f"{out_prefix}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to: {meta_path}")

    print("Done. Local embeddings ready.")


if __name__ == "__main__":
    main()


