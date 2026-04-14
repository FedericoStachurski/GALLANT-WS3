#!/usr/bin/env python3
"""
Interactive querying of CommuniMap multimodal embeddings (text + image) + Folium map.

Expected files
--------------
Given a prefix <prefix>, this script loads:
    <prefix>_text.npy
    <prefix>_image.npy
    <prefix>_meta.json

What it does
------------
- Builds two FAISS indices:
    - text index
    - image index
- Encodes the text query with the SAME vision-language model family used to create
  the embeddings:
    - clip   -> SentenceTransformers CLIP
    - siglip -> Google SigLIP2
- Searches both indices
- Fuses scores:
    score = w_text * score_text + w_img * score_image
- Only returns entries that have an associated image
- Displays results on a Folium map
- Optionally saves result images locally

Examples
--------
# Query embeddings made with SigLIP2
python prompt_the_map.py \
    --prefix /home/fss6k/embedded_data_CM/March2026 \
    --vlm siglip

# Query embeddings made with CLIP
python prompt_the_map.py \
    --prefix /home/fss6k/embedded_data_CM/March2026_clip \
    --vlm clip

# Query embeddings made with a local SigLIP2 folder
python prompt_the_map.py \
    --prefix /home/fss6k/embedded_data_CM/March2026 \
    --vlm siglip \
    --vl_model /home/fss6k/models/siglip2-base-patch16-384
"""

import os
import io
import json
import bisect
import argparse

import faiss
import folium
import numpy as np
import requests
import torch
from PIL import Image
from folium.plugins import HeatMap
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


MODEL_REGISTRY = {
    "clip": {
        "type": "sentence_transformer_clip",
        "default": "sentence-transformers/clip-ViT-B-32",
    },
    "siglip": {
        "type": "siglip",
        "default": "google/siglip2-base-patch16-384",
    },
}


# -------------------------------------------------
# Utils
# -------------------------------------------------
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def resolve_model_source(model_spec: str) -> str:
    p = Path(model_spec).expanduser()
    if p.exists():
        resolved = str(p.resolve())
        print(f"[MODEL] Using local path: {resolved}")
        return resolved
    print(f"[MODEL] Using model ID: {model_spec}")
    return model_spec


def has_valid_image(item) -> bool:
    img = item.get("primary_image")
    return isinstance(img, str) and img.strip() != ""


def download_image(url, save_path, timeout=7):
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.save(save_path)
        return True
    except Exception:
        return False


def load_siglip_model(model_spec: str):
    source = resolve_model_source(model_spec)
    print(f"[SIGLIP] Loading model: {source}")
    processor = AutoProcessor.from_pretrained(source)
    model = AutoModel.from_pretrained(source).to(device)
    model.eval()
    return processor, model


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
        help="Original CommuniMap CSV/XLSX path (optional, for reference only).",
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
        help="Number of nearest neighbours to retrieve before thresholding.",
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
    p.add_argument(
        "--vlm",
        required=False,
        choices=list(MODEL_REGISTRY.keys()),
        default="siglip",
        help="Vision-language model family used to create the embeddings.",
    )
    p.add_argument(
        "--vl_model",
        required=False,
        default=None,
        help="Optional local path or model ID overriding the default VLM.",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    PREFIX = args.prefix
    CSV_PATH = args.csv
    MAP_PATH = args.map_file
    K_DEFAULT = args.k
    THRESHOLD = args.threshold
    W_TEXT = args.w_text
    W_IMG = args.w_img

    model_info = MODEL_REGISTRY[args.vlm]
    vlm_type = model_info["type"]
    vlm_model = args.vl_model if args.vl_model else model_info["default"]

    if not os.path.isabs(MAP_PATH):
        os.makedirs("./maps", exist_ok=True)
        MAP_PATH = os.path.join("./maps", MAP_PATH)

    TEXT_EMB_PATH = PREFIX + "_text.npy"
    IMG_EMB_PATH = PREFIX + "_image.npy"
    META_PATH = PREFIX + "_meta.json"

    print(f"\n[SETUP] Embeddings prefix: {PREFIX}")
    print(f"[SETUP] Text embeddings:   {TEXT_EMB_PATH}")
    print(f"[SETUP] Image embeddings:  {IMG_EMB_PATH}")
    print(f"[SETUP] Metadata file:     {META_PATH}")
    if CSV_PATH:
        print(f"[SETUP] CSV file:          {CSV_PATH}")
    print(f"[SETUP] Map will be saved to: {MAP_PATH}")
    print(f"[SETUP] k = {K_DEFAULT}, threshold = {THRESHOLD}")
    print(f"[SETUP] Fusion weights: w_text={W_TEXT}, w_img={W_IMG}")
    print(f"[SETUP] VLM family: {args.vlm}")
    print(f"[SETUP] VLM model:  {vlm_model}\n")

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
    img_embs = np.load(IMG_EMB_PATH).astype("float32")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("[LOAD] Text embeddings shape :", text_embs.shape)
    print("[LOAD] Image embeddings shape:", img_embs.shape)
    print("[LOAD] Metadata entries      :", len(meta))

    if len(meta) != text_embs.shape[0]:
        raise ValueError("Metadata length does not match number of text embeddings.")
    if len(meta) != img_embs.shape[0]:
        raise ValueError("Metadata length does not match number of image embeddings.")

    n_with_images = sum(has_valid_image(item) for item in meta)
    print(f"[LOAD] Entries with valid images: {n_with_images} / {len(meta)}")

    # -------------------------
    # Build FAISS indices
    # -------------------------
    text_embs_norm = normalize_rows(text_embs)
    img_embs_norm = normalize_rows(img_embs)

    dim_t = text_embs_norm.shape[1]
    dim_i = img_embs_norm.shape[1]

    index_text = faiss.IndexFlatIP(dim_t)
    index_text.add(text_embs_norm)
    print("[FAISS] Text index ready with", index_text.ntotal, "vectors. dim =", index_text.d)

    index_img = faiss.IndexFlatIP(dim_i)
    index_img.add(img_embs_norm)
    print("[FAISS] Image index ready with", index_img.ntotal, "vectors. dim =", index_img.d)

    # -------------------------
    # Load query encoder(s)
    # -------------------------
    if vlm_type == "sentence_transformer_clip":
        model_vlm = SentenceTransformer(resolve_model_source(vlm_model), device=device)

        def embed_query_text(q: str) -> np.ndarray:
            truncated_q = " ".join(q.split()[:50])
            vec = model_vlm.encode(
                [truncated_q],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            return vec

        def embed_query_img(q: str) -> np.ndarray:
            truncated_q = " ".join(q.split()[:50])
            vec = model_vlm.encode(
                [truncated_q],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            return vec

    elif vlm_type == "siglip":
        processor_vlm, model_vlm = load_siglip_model(vlm_model)

        def embed_query_text(q: str) -> np.ndarray:
            truncated_q = " ".join(q.split()[:50])
            inputs = processor_vlm(
                text=[truncated_q],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                vec = model_vlm.get_text_features(**inputs)

            vec = vec / vec.norm(dim=-1, keepdim=True)
            return vec.cpu().numpy().astype("float32")

        def embed_query_img(q: str) -> np.ndarray:
            truncated_q = " ".join(q.split()[:50])
            inputs = processor_vlm(
                text=[truncated_q],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                vec = model_vlm.get_text_features(**inputs)

            vec = vec / vec.norm(dim=-1, keepdim=True)
            return vec.cpu().numpy().astype("float32")

    else:
        raise ValueError(f"Unsupported vlm type: {vlm_type}")

    # -------------------------
    # Multimodal search
    # -------------------------
    def search_multimodal(query: str, k: int, threshold: float, w_text: float, w_img: float):
        qv_t = embed_query_text(query)
        qv_i = embed_query_img(query)

        if qv_t.shape[1] != index_text.d:
            raise ValueError(
                f"Text query dim mismatch: query has {qv_t.shape[1]}, "
                f"but text index expects {index_text.d}. "
                f"Make sure --vlm/--vl_model matches the model used to create the embeddings."
            )

        if qv_i.shape[1] != index_img.d:
            raise ValueError(
                f"Image query dim mismatch: query has {qv_i.shape[1]}, "
                f"but image index expects {index_img.d}. "
                f"Make sure --vlm/--vl_model matches the model used to create the embeddings."
            )

        D_t, I_t = index_text.search(qv_t, k)
        scores_t = D_t[0]
        idxs_t = I_t[0]

        D_i, I_i = index_img.search(qv_i, k)
        scores_i = D_i[0]
        idxs_i = I_i[0]

        score_text_dict = {
            int(idx): float(score)
            for idx, score in zip(idxs_t, scores_t)
            if idx >= 0
        }
        score_img_dict = {
            int(idx): float(score)
            for idx, score in zip(idxs_i, scores_i)
            if idx >= 0
        }

        candidate_idxs = set(score_text_dict.keys()).union(score_img_dict.keys())

        text_scores = sorted(score_text_dict.values())
        img_scores = sorted(score_img_dict.values())

        def get_percentile(score, scores_list):
            if not scores_list:
                return 0.0
            pos = bisect.bisect_left(scores_list, score)
            return pos / len(scores_list)

        fused = []
        skipped_no_image = 0

        for idx in candidate_idxs:
            item = meta[idx]

            if not has_valid_image(item):
                skipped_no_image += 1
                continue

            st = score_text_dict.get(idx, 0.0)
            si = score_img_dict.get(idx, 0.0)

            p_text = 100.0 * get_percentile(st, text_scores)
            p_img = 100.0 * get_percentile(si, img_scores)

            s = w_text * st + w_img * si
            if s < threshold:
                continue

            fused.append((idx, s, st, si, p_text, p_img))

        print(f"[SEARCH] Skipped {skipped_no_image} candidate results with no image.")

        fused.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, s_fused, st, si, p_text, p_img in fused[:k]:
            item = meta[idx]
            results.append(
                {
                    "idx": int(idx),
                    "score": float(s_fused),
                    "score_text": float(st),
                    "score_img": float(si),
                    "p_text": float(p_text),
                    "p_img": float(p_img),
                    "id": item["id"],
                    "text": item.get("text", ""),
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "image": item.get("primary_image"),
                }
            )

        return results

    # -------------------------
    # Map builder
    # -------------------------
    def make_map(results, query=None, center=None, zoom_start=12, heat_radius=25, heat_blur=30):
        if not results:
            print("No results provided.")
            return None

        if center is None:
            center = (55.8721, -4.2892)

        m = folium.Map(location=center, zoom_start=zoom_start)

        heat_points = [(r["lat"], r["lon"]) for r in results]

        HeatMap(
            heat_points,
            radius=heat_radius,
            blur=heat_blur,
            max_zoom=13,
        ).add_to(m)

        for r in results:
            lat, lon = r["lat"], r["lon"]
            text = (r["text"] or "")[:200].replace("\n", " ")
            img = r["image"]
            score = r["score"]
            entry_id = r["id"]

            html = f"""
            <div style="width:240px;">
            <b>ID: {entry_id}</b><br>
            <b>Score: {score:.3f}</b><br>
            <b>Text Score: {r["score_text"]:.3f} (p: {r["p_text"]:.3f} %)</b><br>
            <b>Image Score: {r["score_img"]:.3f} (p: {r["p_img"]:.3f} %)</b><br>
            <p style="font-size:11px;">{text}...</p>
            <img src="{img}" width="220">
            </div>
            """
            popup = folium.Popup(html, max_width=260)

            # Show ID directly on the map as tooltip as well
            tooltip = folium.Tooltip(f"ID: {entry_id}")

            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                fill=True,
                fill_opacity=0.85,
                color="red",
                popup=popup,
                tooltip=tooltip,
            ).add_to(m)

        return m

    # -------------------------
    # Interactive loop
    # -------------------------
    print("\n[READY] Enter search queries.")
    print("        Commands:")
    print("          k=<int>            change number of neighbours")
    print("          threshold=<float>  change similarity threshold")
    print("          quit / exit        stop")
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

        if q.lower() in {"q", "quit", "exit"}:
            print("[EXIT]")
            break

        if q.startswith("k =") or q.startswith("k="):
            try:
                current_k = int(q.split("=", 1)[1])
                print(f"[SET] k updated to {current_k}\n")
            except Exception:
                print("[ERROR] Invalid k value.\n")
            continue

        if q.startswith("threshold =") or q.startswith("threshold="):
            try:
                current_threshold = float(q.split("=", 1)[1])
                print(f"[SET] threshold updated to {current_threshold}\n")
            except Exception:
                print("[ERROR] Invalid threshold value.\n")
            continue

        print(f"[SEARCH] Multimodal search (k={current_k}, threshold={current_threshold}) for: {q!r}")

        results = search_multimodal(
            query=q,
            k=current_k,
            threshold=current_threshold,
            w_text=W_TEXT,
            w_img=W_IMG,
        )

        print(f"[SEARCH] Got {len(results)} image-backed results after fusion + threshold.\n")

        if not results:
            print("[MAP] No image-backed results above threshold — map not updated.")
            print("      Try lowering the threshold or changing the query.\n")
            continue

        m = make_map(results, query=q)
        os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)
        m.save(MAP_PATH)

        print(f"[MAP] Saved to: {MAP_PATH}\n(Refresh in your browser.)\n")

        save_images = input("Save images to folder? (y/n): ").strip().lower()
        if save_images in {"y", "yes"}:
            map_name = os.path.splitext(os.path.basename(MAP_PATH))[0]
            folder_name = f"save_images_{q.replace(' ', '_')[:20]}_{map_name}"
            full_folder_path = os.path.join(
                "/home/fss6k/embedded_data_CM/query_maps",
                folder_name,
            )
            os.makedirs(full_folder_path, exist_ok=True)

            saved_count = 0
            for r in results:
                img_url = r.get("image")
                if img_url:
                    filename = f"{r['id']}.jpg"
                    save_path = os.path.join(full_folder_path, filename)
                    if download_image(img_url, save_path):
                        saved_count += 1
                    else:
                        print(f"Failed to save {img_url}")

            print(f"[SAVE] Saved {saved_count} images to {full_folder_path}/\n")


if __name__ == "__main__":
    main()