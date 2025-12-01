#!/usr/bin/env python3
"""
Interactive querying of CommuniMap *multimodal* embeddings (text + image) + Folium map.

- Loads:
    <prefix>_text.npy
    <prefix>_image.npy
    <prefix>_meta.json

- Builds two FAISS indices:
    - text index (MiniLM)
    - image index (CLIP)

- Given a text query:
    - encode with MiniLM for text index
    - encode with CLIP (text side) for image index
    - fuse scores: score = w_text * score_text + w_img * score_image

- Shows results on a Folium map (HeatMap + markers)
- Overwrites the same HTML map file each query
"""

import os
import json
import argparse
import numpy as np
import faiss
import folium
from folium.plugins import HeatMap
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Utils
# -------------------------------------------------
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--prefix",
        required=True,
        help="Prefix for embeddings (expects <prefix>_text.npy, <prefix>_image.npy, <prefix>_meta.json).",
    )
    p.add_argument(
        "--csv",
        required=False,
        default=None,
        help="Original CommuniMap CSV (optional, for reference).",
    )
    p.add_argument(
        "--map_file",
        required=False,
        default="map_latest.html",
        help="Full path or filename for output HTML map (overwritten each query).",
    )
    p.add_argument(
        "--k",
        required=False,
        type=int,
        default=50,
        help="Number of nearest neighbours to retrieve *before* thresholding.",
    )
    p.add_argument(
        "--threshold",
        required=False,
        type=float,
        default=0.0,
        help="Minimum fused similarity score to keep a result.",
    )
    p.add_argument(
        "--w_text",
        required=False,
        type=float,
        default=0.7,
        help="Weight for text similarity in fusion (0–1).",
    )
    p.add_argument(
        "--w_img",
        required=False,
        type=float,
        default=0.3,
        help="Weight for image similarity in fusion (0–1).",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    PREFIX     = args.prefix
    CSV_PATH   = args.csv
    MAP_PATH   = args.map_file
    K_DEFAULT  = args.k
    THRESHOLD  = args.threshold
    W_TEXT     = args.w_text
    W_IMG      = args.w_img

    # If MAP_PATH is not absolute, put it in ./maps
    if not os.path.isabs(MAP_PATH):
        os.makedirs("./maps", exist_ok=True)
        MAP_PATH = os.path.join("./maps", MAP_PATH)

    TEXT_EMB_PATH = PREFIX + "_text.npy"
    IMG_EMB_PATH  = PREFIX + "_image.npy"
    META_PATH     = PREFIX + "_meta.json"

    print(f"\n[SETUP] Embeddings prefix: {PREFIX}")
    print(f"[SETUP] Text embeddings:   {TEXT_EMB_PATH}")
    print(f"[SETUP] Image embeddings:  {IMG_EMB_PATH}")
    print(f"[SETUP] Metadata file:     {META_PATH}")
    if CSV_PATH:
        print(f"[SETUP] CSV file:          {CSV_PATH}")
    print(f"[SETUP] Map will be saved to: {MAP_PATH}")
    print(f"[SETUP] k = {K_DEFAULT}, threshold = {THRESHOLD}")
    print(f"[SETUP] Fusion weights: w_text={W_TEXT}, w_img={W_IMG}\n")

    # -------------------------
    # Load embeddings + meta
    # -------------------------
    if not os.path.exists(TEXT_EMB_PATH):
        raise FileNotFoundError(f"Text embedding file not found: {TEXT_EMB_PATH}")
    if not os.path.exists(IMG_EMB_PATH):
        raise FileNotFoundError(f"Image embedding file not found: {IMG_EMB_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    text_embs = np.load(TEXT_EMB_PATH).astype("float32")
    img_embs  = np.load(IMG_EMB_PATH).astype("float32")

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    print("[LOAD] Text embeddings shape :", text_embs.shape)
    print("[LOAD] Image embeddings shape:", img_embs.shape)
    print("[LOAD] Metadata entries      :", len(meta))

    if len(meta) != text_embs.shape[0]:
        raise ValueError("Metadata length does not match number of *text* embeddings.")
    if len(meta) != img_embs.shape[0]:
        raise ValueError("Metadata length does not match number of *image* embeddings.")

    # -------------------------
    # Build FAISS indices
    # -------------------------
    text_embs_norm = normalize_rows(text_embs)
    img_embs_norm  = normalize_rows(img_embs)

    dim_t = text_embs_norm.shape[1]
    dim_i = img_embs_norm.shape[1]

    index_text = faiss.IndexFlatIP(dim_t)
    index_text.add(text_embs_norm)
    print("[FAISS] Text index ready with", index_text.ntotal, "vectors. dim =", index_text.d)

    index_img = faiss.IndexFlatIP(dim_i)
    index_img.add(img_embs_norm)
    print("[FAISS] Image index ready with", index_img.ntotal, "vectors. dim =", index_img.d)

    # -------------------------
    # Load query encoders
    # -------------------------
    # Text encoder (MiniLM) — must match TEXT_MODEL_NAME used for text embeddings
    QUERY_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"[MODEL] Loading TEXT query encoder:  {QUERY_TEXT_MODEL}")
    model_text = SentenceTransformer(QUERY_TEXT_MODEL)

    # Image-side encoder (CLIP from sentence-transformers, same as IMAGE_MODEL_NAME)
    QUERY_IMG_MODEL = "sentence-transformers/clip-ViT-B-32"
    print(f"[MODEL] Loading IMAGE query encoder: {QUERY_IMG_MODEL}")
    model_img = SentenceTransformer(QUERY_IMG_MODEL)

    def embed_query_text(q: str) -> np.ndarray:
        vec = model_text.encode([q], convert_to_numpy=True).astype("float32")
        return normalize_rows(vec)  # shape (1, dim_t)

    def embed_query_img(q: str) -> np.ndarray:
        # CLIP text encoder via SentenceTransformer(clip-ViT-B-32)
        vec = model_img.encode([q], convert_to_numpy=True).astype("float32")
        return normalize_rows(vec)  # shape (1, dim_i)

    # -------------------------
    # Multimodal search (text + image)
    # -------------------------
    def search_multimodal(query: str, k: int, threshold: float, w_text: float, w_img: float):
        # text-space query
        qv_t = embed_query_text(query)   # (1, dim_t)
        D_t, I_t = index_text.search(qv_t, k)
        scores_t = D_t[0]
        idxs_t   = I_t[0]

        # image-space query (CLIP text encoder)
        qv_i = embed_query_img(query)    # (1, dim_i)
        D_i, I_i = index_img.search(qv_i, k)
        scores_i = D_i[0]
        idxs_i   = I_i[0]

        # Build dicts for quick lookup
        score_text_dict = {int(idx): float(score) for idx, score in zip(idxs_t, scores_t) if idx >= 0}
        score_img_dict  = {int(idx): float(score) for idx, score in zip(idxs_i, scores_i) if idx >= 0}

        # Union of candidate indices
        candidate_idxs = set(score_text_dict.keys()).union(score_img_dict.keys())

        fused = []
        for idx in candidate_idxs:
            st = score_text_dict.get(idx, 0.0)
            si = score_img_dict.get(idx, 0.0)
            s  = w_text * st + w_img * si
            if s < threshold:
                continue
            fused.append((idx, s, st, si))

        # Sort by fused score descending
        fused.sort(key=lambda x: x[1], reverse=True)

        # Build result objects
        results = []
        for idx, s_fused, st, si in fused[:k]:
            item = meta[idx]
            results.append({
                "idx": int(idx),
                "score": float(s_fused),
                "score_text": float(st),
                "score_img": float(si),
                "id": item["id"],
                "text": item.get("text", ""),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "image": item.get("primary_image"),
            })

        return results

    # -------------------------
    # Map builder
    # -------------------------
    def make_map(results, query=None, center=None, zoom_start=12,
             heat_radius=25, heat_blur=30):
        """
        results: output of search_return(query, k=N)
        """

        if not results:
            print("No results provided.")
            return None

        # --- compute default center if none provided ---
        if center is None:
            # avg_lat = sum(r["lat"] for r in results) / len(results)
            # avg_lon = sum(r["lon"] for r in results) / len(results)
            # center = (avg_lat, avg_lon)
            center = (55.8721, -4.2892) # University of Glasgow coordinates
            

        # --- base map ---
        m = folium.Map(location=center, zoom_start=zoom_start)

        # --- prepare HeatMap points ---
        heat_points = [(r["lat"], r["lon"]) for r in results]

        # --- add HeatMap layer ---
        HeatMap(
            heat_points,
            radius=heat_radius,
            blur=heat_blur,
            max_zoom=13,
        ).add_to(m)

        # --- add point markers with popups ---
        for r in results:
            lat, lon = r["lat"], r["lon"]
            text = (r["text"] or "")[:200].replace("\n", " ") + "…"
            img = r["image"]
            score = r["score"]

            html = f"""
            <div style="width:240px;">
            <b>Score: {score:.3f}</b><br>
            <p style="font-size:11px;">{text}</p>
            {'<img src="' + img + '" width="220">' if img else ''}
            </div>
            """
            popup = folium.Popup(html, max_width=260)

            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                fill=True,
                fill_opacity=0.85,
                color='red',
                popup=popup,
            ).add_to(m)


        return m


    # -------------------------
    # Interactive loop
    # -------------------------
    print("\n[READY] Enter search queries.")
    print("        Commands:")
    print("          k=<int>          change number of neighbours")
    print("          threshold=<float> change similarity threshold")
    print("          quit / exit      stop")
    print()
    print(f"        Map file (overwrite each query): {MAP_PATH}\n")

    current_k = K_DEFAULT
    current_threshold = THRESHOLD

    while True:
        try:
            q = input(f"Query (k={current_k}, thr={current_threshold})> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break

        if not q:
            continue

        # -------------------------
        # Handle commands
        # -------------------------
        if q.lower() in {"q", "quit", "exit"}:
            print("[EXIT]")
            break

        # k=<number>
        if q.startswith("k =") or q.startswith("k="):
            try:
                current_k = int(q.split("=", 1)[1])
                print(f"[SET] k updated to {current_k}\n")
            except:
                print("[ERROR] Invalid k value.\n")
            continue

        # threshold=<float>
        if q.startswith("threshold =") or q.startswith("threshold="):
            try:
                current_threshold = float(q.split("=", 1)[1])
                print(f"[SET] threshold updated to {current_threshold}\n")
            except:
                print("[ERROR] Invalid threshold value.\n")
            continue

        # -------------------------
        # Run multimodal search
        # -------------------------
        print(f"[SEARCH] Multimodal search (k={current_k}, threshold={current_threshold}) for: {q!r}")

        results = search_multimodal(
            query=q,
            k=current_k,
            threshold=current_threshold,
            w_text=W_TEXT,
            w_img=W_IMG,
        )

        print(f"[SEARCH] Got {len(results)} results after fusion + threshold.\n")


        print(f"[SEARCH] Got {len(results)} results after fusion + threshold.\n")

        if not results:
            print("[MAP] No results above threshold — map not updated.")
            print("      Try lowering the threshold or changing the query.\n")
            continue  # go back to the next Query>



        m = make_map(results, query=q)
        os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)
        m.save(MAP_PATH)

        print(f"[MAP] Saved to: {MAP_PATH}\n(Refresh in your browser.)\n")


if __name__ == "__main__":
    main()
