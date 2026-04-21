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
- Supports fusion modes:
    - weighted : w_text * norm_text + w_img * norm_img
    - rrf      : reciprocal rank fusion
    - text     : text-only ranking
    - image    : image-only ranking
- Only returns entries that have an associated image
- Displays results on a Folium map
- Optionally saves result images locally

Extras
------
- Saves score distribution plots for debugging
- Supports a VALIDATE command that evaluates retrieval for a fixed tree query
  using supervision from metadata flags such as:
      Tree Colabor, Tree Type, TREE, etc.
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
import matplotlib.pyplot as plt
import pandas as pd


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


def str_to_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "tree", "trees"}


def get_first_present(item, field_names):
    for f in field_names:
        if f in item:
            return item.get(f)
    return None


def safe_filename(s: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s.strip())
    cleaned = cleaned.strip("_")
    return cleaned[:100] if cleaned else "query"


def parse_k_list(s: str):
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


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
        help="Weight for text similarity in weighted fusion.",
    )
    p.add_argument(
        "--w_img",
        required=False,
        type=float,
        default=0.3,
        help="Weight for image similarity in weighted fusion.",
    )
    p.add_argument(
        "--fusion",
        required=False,
        choices=["weighted", "rrf", "text", "image"],
        default="weighted",
        help="Fusion mode: weighted, rrf, text, or image.",
    )
    p.add_argument(
        "--rrf_k",
        required=False,
        type=int,
        default=60,
        help="Constant used for Reciprocal Rank Fusion (larger = flatter contributions).",
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

    # Validation options
    p.add_argument(
        "--validate_query",
        required=False,
        default="trees",
        help="Query used when VALIDATE mode is triggered.",
    )
    p.add_argument(
        "--validate_flag_fields",
        required=False,
        default="TREE",
        help="Comma-separated metadata field names to try for supervision labels.",
    )
    p.add_argument(
        "--validate_k_list",
        required=False,
        default="5,10,20,50,100,200,500",
        help="Comma-separated K values for Precision@K / Recall@K during VALIDATE.",
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
    FUSION_DEFAULT = args.fusion
    RRF_K = args.rrf_k
    VALIDATE_QUERY = args.validate_query
    VALIDATE_FLAG_FIELDS = [x.strip() for x in args.validate_flag_fields.split(",") if x.strip()]
    VALIDATE_K_LIST = parse_k_list(args.validate_k_list)

    model_info = MODEL_REGISTRY[args.vlm]
    vlm_type = model_info["type"]
    vlm_model = args.vl_model if args.vl_model else model_info["default"]

    if not os.path.isabs(MAP_PATH):
        os.makedirs("./maps", exist_ok=True)
        MAP_PATH = os.path.join("./maps", MAP_PATH)

    df = None
    if CSV_PATH:
        try:
            if CSV_PATH.lower().endswith(".csv"):
                df = pd.read_csv(CSV_PATH)
            elif CSV_PATH.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(CSV_PATH)
            else:
                print(f"[WARNING] Unsupported CSV file format: {CSV_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load CSV file: {e}")

    TEXT_EMB_PATH = PREFIX + "_text.npy"
    IMG_EMB_PATH = PREFIX + "_image.npy"
    META_PATH = PREFIX + "_meta.json"
    PLOTS_PATH = PREFIX

    print(f"\n[SETUP] Embeddings prefix: {PREFIX}")
    print(f"[SETUP] Text embeddings:   {TEXT_EMB_PATH}")
    print(f"[SETUP] Image embeddings:  {IMG_EMB_PATH}")
    print(f"[SETUP] Metadata file:     {META_PATH}")
    if CSV_PATH:
        print(f"[SETUP] CSV file:          {CSV_PATH}")
    print(f"[SETUP] Map will be saved to: {MAP_PATH}")
    print(f"[SETUP] k = {K_DEFAULT}, threshold = {THRESHOLD}")
    print(f"[SETUP] Fusion mode: {FUSION_DEFAULT}")
    print(f"[SETUP] Fusion weights: w_text={W_TEXT}, w_img={W_IMG}")
    print(f"[SETUP] RRF k: {RRF_K}")
    print(f"[SETUP] VLM family: {args.vlm}")
    print(f"[SETUP] VLM model:  {vlm_model}")
    print(f"[SETUP] Validate query: {VALIDATE_QUERY!r}")
    print(f"[SETUP] Validate flag fields: {VALIDATE_FLAG_FIELDS}")
    print(f"[SETUP] Validate K list: {VALIDATE_K_LIST}\n")

    ###################
    # ----- Utils -----
    ###################

    def minmax(xs):
        if not xs:
            return []
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-8:
            return [0.0 for _ in xs]
        return [(x - lo) / (hi - lo) for x in xs]

    def minmax_dict(score_dict):
        if not score_dict:
            return {}

        values = list(score_dict.values())
        lo, hi = min(values), max(values)

        if hi - lo < 1e-8:
            return {k: 0.0 for k in score_dict}

        return {
            k: (v - lo) / (hi - lo)
            for k, v in score_dict.items()
        }

    def normalize_id(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        try:
            f = float(s)
            if f.is_integer():
                s = str(int(f))
        except ValueError:
            pass
        return s

    def build_tree_relevance_set_from_table(meta, df, flag_fields, id_field="ID"):
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        id_field_norm = str(id_field).strip().lower()
        flag_fields_norm = [str(f).strip().lower() for f in flag_fields]

        if id_field_norm not in df.columns:
            raise ValueError(
                f"ID field {id_field!r} not found in dataframe columns: {df.columns.tolist()}"
            )

        present_flag_fields = [f for f in flag_fields_norm if f in df.columns]

        if not present_flag_fields:
            raise ValueError(
                f"None of the requested flag fields were found in dataframe.\n"
                f"Requested: {flag_fields}\n"
                f"Available: {df.columns.tolist()}"
            )

        searchable_ids = set()
        for item in meta:
            if not has_valid_image(item):
                continue
            item_id = normalize_id(item.get("source_id"))
            if item_id is not None:
                searchable_ids.add(item_id)

        is_relevant = (
            df[present_flag_fields]
            .astype(str)
            .apply(lambda col: col.str.strip().str.lower())
            .eq("yes")
            .any(axis=1)
        )

        print(f"[VALIDATE] Found {is_relevant.sum()} relevant items using flag fields: {present_flag_fields}")

        relevant_ids = set(
            df.loc[is_relevant, id_field_norm]
            .dropna()
            .map(normalize_id)
            .dropna()
        )

        print(f"[VALIDATE] Relevant IDs before searchable filter: {len(relevant_ids)}")
        print(f"[VALIDATE] Searchable IDs from meta: {len(searchable_ids)}")

        overlap = relevant_ids.intersection(searchable_ids)
        print(f"[VALIDATE] Found {len(overlap)} unique relevant items.")

        print("[DEBUG] Sample relevant_ids:", list(relevant_ids)[:20])
        print("[DEBUG] Sample searchable_ids:", list(searchable_ids)[:20])
        print("[DEBUG] relevant_id types:", {type(x).__name__ for x in list(relevant_ids)[:20]})
        print("[DEBUG] searchable_id types:", {type(x).__name__ for x in list(searchable_ids)[:20]})

        return searchable_ids, overlap

    def reciprocal_rank(rank, rrf_k):
        return 1.0 / (rrf_k + rank)

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
    def search_multimodal(
        query: str,
        k: int,
        threshold: float,
        w_text: float,
        w_img: float,
        fusion_type: str,
        rrf_k: int,
        make_plots: bool = True,
    ):
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

        k_search = max(1, min(int(k), len(meta)))

        D_t, I_t = index_text.search(qv_t, k_search)
        scores_t = D_t[0]
        idxs_t = I_t[0]

        D_i, I_i = index_img.search(qv_i, k_search)
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

        rank_text_dict = {
            int(idx): rank
            for rank, idx in enumerate(idxs_t, start=1)
            if idx >= 0
        }
        rank_img_dict = {
            int(idx): rank
            for rank, idx in enumerate(idxs_i, start=1)
            if idx >= 0
        }

        candidate_idxs = set(score_text_dict.keys()).union(score_img_dict.keys())

        text_scores = sorted(score_text_dict.values())
        img_scores = sorted(score_img_dict.values())
        norm_text_dict = minmax_dict(score_text_dict)
        norm_img_dict = minmax_dict(score_img_dict)

        text_norm_scores = sorted(norm_text_dict.values())
        img_norm_scores = sorted(norm_img_dict.values())

        if make_plots:
            save_dir = os.path.dirname(PLOTS_PATH) or "."

            plt.figure()
            plt.hist(text_scores, bins=30, alpha=0.6, label="Text scores")
            plt.hist(img_scores, bins=30, alpha=0.6, label="Image scores")
            plt.legend()
            plt.title(f"Raw score distributions\n{query}")
            plt.xlabel("Score")
            plt.ylabel("Frequency")
            raw_path = os.path.join(save_dir, "hist_raw_scores.png")
            plt.savefig(raw_path)
            plt.close()

            text_norm = minmax(text_scores)
            img_norm = minmax(img_scores)

            plt.figure()
            plt.hist(text_norm, bins=30, alpha=0.6, label="Text (norm)")
            plt.hist(img_norm, bins=30, alpha=0.6, label="Image (norm)")
            plt.legend()
            plt.title(f"Normalized score distributions\n{query}")
            plt.xlabel("Normalized score")
            plt.ylabel("Frequency")
            norm_path = os.path.join(save_dir, "hist_norm_scores.png")
            plt.savefig(norm_path)
            plt.close()

            print(f"[PLOT] Saved raw scores to: {raw_path}")
            print(f"[PLOT] Saved normalized scores to: {norm_path}")

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

            st = norm_text_dict.get(idx, 0.0)
            si = norm_img_dict.get(idx, 0.0)

            p_text = 100.0 * get_percentile(st, text_norm_scores)
            p_img = 100.0 * get_percentile(si, img_norm_scores)

            rt = rank_text_dict.get(idx)
            ri = rank_img_dict.get(idx)

            if fusion_type == "weighted":
                s = w_text * st + w_img * si
            elif fusion_type == "text":
                s = st
            elif fusion_type == "image":
                s = si
            elif fusion_type == "rrf":
                s = 0.0
                if rt is not None:
                    s += reciprocal_rank(rt, rrf_k)
                if ri is not None:
                    s += reciprocal_rank(ri, rrf_k)
            else:
                raise ValueError(f"Unsupported fusion type: {fusion_type}")

            if s < threshold:
                continue

            fused.append((idx, s, st, si, p_text, p_img))

        print(f"[SEARCH] Skipped {skipped_no_image} candidate results with no image.")

        fused.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, s_fused, st, si, p_text, p_img in fused[:k_search]:
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
                    "source_id": item.get("source_id", item["id"]),
                    "text": item.get("text", ""),
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "image": item.get("primary_image"),
                }
            )

        return results

    # -------------------------
    # Validation helpers
    # -------------------------
    def normalized_average_rank(ranks, N, N_rel):
        if N <= 0 or N_rel <= 0:
            return None
        if len(ranks) != N_rel:
            return None
        return (sum(ranks) - (N_rel * (N_rel + 1)) / 2.0) / (N * N_rel)

    def run_validation():
        print("\n[VALIDATE] Running supervised validation...")
        print(f"[VALIDATE] Query: {VALIDATE_QUERY!r}")
        print(f"[VALIDATE] Flag fields: {VALIDATE_FLAG_FIELDS}")
        print(f"[VALIDATE] Fusion mode: {current_fusion}")

        if df is None:
            print("[VALIDATE] No CSV/XLSX dataframe loaded, so validation cannot run.\n")
            return

        searchable_ids, relevant_ids = build_tree_relevance_set_from_table(meta, df, VALIDATE_FLAG_FIELDS, id_field="ID")

        N = len(searchable_ids)
        N_rel = len(relevant_ids)

        print(f"[VALIDATE] Searchable image-backed universe size N = {N}")
        print(f"[VALIDATE] Relevant tree-labelled items N_rel = {N_rel}")

        if N == 0:
            print("[VALIDATE] No searchable image-backed items found.\n")
            return

        if N_rel == 0:
            print("[VALIDATE] No relevant tree-labelled items found using those flag fields.\n")
            return

        full_results = search_multimodal(
            query=VALIDATE_QUERY,
            k=N,
            threshold=-1.0,
            w_text=W_TEXT,
            w_img=W_IMG,
            fusion_type=current_fusion,
            rrf_k=RRF_K,
            make_plots=True,
        )

        ranked_ids = [
            normalize_id(r.get("source_id"))
            for r in full_results
            if r.get("source_id") is not None
        ]
        rank_lookup = {item_id: rank for rank, item_id in enumerate(ranked_ids, start=1)}

        found_relevant_ids = [item_id for item_id in relevant_ids if item_id in rank_lookup]
        found_ranks = sorted(rank_lookup[item_id] for item_id in found_relevant_ids)

        print(f"[VALIDATE] Relevant items retrieved in ranking: {len(found_relevant_ids)} / {N_rel}")

        if len(found_relevant_ids) != N_rel:
            print("[VALIDATE] Warning: not all relevant items were recovered in the ranked results.")
            print("           Rank* will only be exact when all searchable relevant items are included.")

        rank_star = normalized_average_rank(found_ranks, N=N, N_rel=N_rel) if len(found_relevant_ids) == N_rel else None

        print("\n[VALIDATE] Precision / Recall by K")
        print("----------------------------------")
        valid_k_list = sorted(set(min(max(1, k), len(full_results)) for k in VALIDATE_K_LIST))
        for k_val in valid_k_list:
            topk_ids = {
                normalize_id(r.get("source_id"))
                for r in full_results[:k_val]
                if r.get("source_id") is not None
            }
            hits = len(topk_ids.intersection(relevant_ids))
            precision_k = hits / k_val
            recall_k = hits / N_rel
            print(f"k={k_val:4d}  Precision@k={precision_k:.4f}  Recall@k={recall_k:.4f}  Hits={hits}")

        print("\n[VALIDATE] Rank summary")
        print("-----------------------")
        if found_ranks:
            mean_rank = sum(found_ranks) / len(found_ranks)
            median_rank = float(np.median(found_ranks))
            print(f"Mean relevant rank   : {mean_rank:.3f}")
            print(f"Median relevant rank : {median_rank:.3f}")
            print(f"Best relevant rank   : {min(found_ranks)}")
            print(f"Worst relevant rank  : {max(found_ranks)}")
        else:
            print("No relevant items were retrieved.")

        if rank_star is not None:
            print(f"~Rank~               : {rank_star:.6f}")
            print("Interpretation       : 0 is perfect, ~0.5 is random, closer to 1 is worse")
        else:
            print("~Rank~               : not computed exactly")
            print("Reason               : not all searchable relevant items were present in the ranking")

        print()

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
            entry_id = r["source_id"]

            html = f"""
            <div style="width:240px;">
            <b>ID: {entry_id}</b><br>
            <b>Score: {score:.3f}</b><br>
            <b>Text Score: {r["score_text"]:.3f} (p: {r["p_text"]:.1f}%)</b><br>
            <b>Image Score: {r["score_img"]:.3f} (p: {r["p_img"]:.1f}%)</b><br>
            <p style="font-size:11px;">{text}...</p>
            <img src="{img}" width="220">
            </div>
            """
            popup = folium.Popup(html, max_width=260)
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
    print("          k=<int>             change number of neighbours")
    print("          threshold=<float>   change similarity threshold")
    print("          fusion=<mode>       weighted | rrf | text | image")
    print("          VALIDATE            run weakly supervised tree validation")
    print("          quit / exit         stop")
    print()
    print(f"        Map file (overwrite each query): {MAP_PATH}\n")

    current_k = K_DEFAULT
    current_threshold = THRESHOLD
    current_fusion = FUSION_DEFAULT

    while True:
        try:
            q = input(f"Query (k={current_k}, thr={current_threshold}, fusion={current_fusion})> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break

        if not q:
            continue

        if q.lower() in {"q", "quit", "exit"}:
            print("[EXIT]")
            break

        if q.upper() == "VALIDATE":
            run_validation()
            continue

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

        if q.startswith("fusion =") or q.startswith("fusion="):
            try:
                new_fusion = q.split("=", 1)[1].strip().lower()
                if new_fusion not in {"weighted", "rrf", "text", "image"}:
                    raise ValueError("invalid fusion mode")
                current_fusion = new_fusion
                print(f"[SET] fusion updated to {current_fusion}\n")
            except Exception:
                print("[ERROR] Invalid fusion mode. Use: weighted, rrf, text, image.\n")
            continue

        print(
            f"[SEARCH] Search "
            f"(fusion={current_fusion}, k={current_k}, threshold={current_threshold}) for: {q!r}"
        )

        results = search_multimodal(
            query=q,
            k=current_k,
            threshold=current_threshold,
            w_text=W_TEXT,
            w_img=W_IMG,
            fusion_type=current_fusion,
            rrf_k=RRF_K,
            make_plots=True,
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
                    filename = f"{r['source_id']}.jpg"
                    save_path = os.path.join(full_folder_path, filename)
                    if download_image(img_url, save_path):
                        saved_count += 1
                    else:
                        print(f"Failed to save {img_url}")

            print(f"[SAVE] Saved {saved_count} images to {full_folder_path}/\n")


if __name__ == "__main__":
    main()