import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

import open_clip
from open_clip.tokenizer import HFTokenizer


# -----------------------------
# Utils
# -----------------------------
def ask(prompt: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(prompt).strip()
    s = input(f"{prompt} [{default}] ").strip()
    return s if s else default


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = []
    for e in exts:
        files.extend(folder.glob(f"*{e}"))
        files.extend(folder.glob(f"*{e.upper()}"))
    return sorted(set(files))


def clamp_box(box, W, H):
    x0, y0, x1, y1 = map(int, box)
    x0 = max(0, min(x0, W - 1))
    x1 = max(0, min(x1, W - 1))
    y0 = max(0, min(y0, H - 1))
    y1 = max(0, min(y1, H - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


# -----------------------------
# GroundingDINO selection logic
# -----------------------------
def select_tree_box(
    detections: Dict[str, Any],
    image_size: Tuple[int, int],
    score_min: float = 0.25
) -> Optional[List[int]]:
    W, H = image_size
    best_box, best_s = None, -1.0

    for box, score in zip(detections["boxes"], detections["scores"]):
        s = float(score)
        if s < score_min:
            continue

        x0, y0, x1, y1 = box.tolist()
        w, h = (x1 - x0), (y1 - y0)
        if w <= 1 or h <= 1:
            continue

        area = w * h

        touch = (x0 < 5) + (y0 < 5) + (x1 > W - 5) + (y1 > H - 5)
        edge_penalty = 0.65 ** touch

        aspect = max(w / h, h / w)
        aspect_penalty = 1.0 if aspect < 6 else 0.5

        score_combined = (area * s) * edge_penalty * aspect_penalty

        if score_combined > best_s:
            best_s = score_combined
            best_box = [int(x0), int(y0), int(x1), int(y1)]

    if best_box is None:
        return None
    return clamp_box(best_box, W, H)


def draw_selected_box(ax, box, label="SELECTED"):
    x0, y0, x1, y1 = box
    rect = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=3,
        edgecolor="lime",
        facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(
        x0,
        max(y0 - 12, 0),
        label,
        color="lime",
        fontsize=12,
        backgroundcolor="black"
    )


def expand_box(box: List[int], image_shape, pad: int = 50) -> List[int]:
    H, W = image_shape[:2]
    x0, y0, x1, y1 = map(int, box)
    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(W - 1, x1 + pad),
        min(H - 1, y1 + pad),
    ]


# -----------------------------
# BioCLIP helpers
# -----------------------------
def clean_species_name(name: str) -> str:
    name2 = re.sub(r"\s*'[^']+'", "", name)
    name2 = name2.replace("×", "x")
    return " ".join(name2.split()).strip()


def flatten_glasgow_tree_list(glasgow: dict) -> List[Dict[str, Any]]:
    candidates = []
    for block in glasgow["trees"]:
        size_class = block.get("size_class")
        genus = block.get("genus")
        for sp in block.get("species", []):
            candidates.append({
                "name": sp["name"],
                "genus": genus,
                "size_class": size_class,
                "evergreen": sp.get("evergreen", None),
            })
    return candidates


@torch.inference_mode()
def bioclip_rank_species(
    pil_img: Image.Image,
    candidates: List[Dict[str, Any]],
    model,
    preprocess,
    tokenizer,
    device: str,
    topk: int = 10
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, float]]]:

    image_in = preprocess(pil_img).unsqueeze(0).to(device)

    texts = []
    owner = []
    for j, c in enumerate(candidates):
        full = c["name"]
        clean = clean_species_name(full)
        for variant in {full, clean}:
            texts.append(f"a photo of {variant}")
            owner.append(j)

    text_tokens = tokenizer(texts).to(device)

    img_f = model.encode_image(image_in)
    txt_f = model.encode_text(text_tokens)

    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

    probs = ((img_f @ txt_f.T) * 100.0).softmax(dim=-1)[0].detach().cpu().numpy()

    cand_probs = np.zeros(len(candidates), dtype=np.float64)
    for p, j in zip(probs, owner):
        cand_probs[j] += float(p)

    idx = cand_probs.argsort()[::-1][:topk]
    results = [{
        "species": candidates[i]["name"],
        "genus": candidates[i].get("genus"),
        "size_class": candidates[i].get("size_class"),
        "evergreen": candidates[i].get("evergreen"),
        "p": float(cand_probs[i]),
    } for i in idx]

    genus_scores = {}
    for i, p in enumerate(cand_probs):
        g = candidates[i].get("genus") or "UNKNOWN"
        genus_scores[g] = genus_scores.get(g, 0.0) + float(p)
    genus_rank = sorted(genus_scores.items(), key=lambda x: x[1], reverse=True)

    return results, genus_rank


def probs_to_percent(items: List[Dict[str, Any]], key="p") -> List[Dict[str, Any]]:
    s = sum(float(x[key]) for x in items) + 1e-12
    out = []
    for x in items:
        y = dict(x)
        y["percent"] = float(x[key]) / s * 100.0
        out.append(y)
    return out


# -----------------------------
# NEW: Save pie/donut chart helper
# -----------------------------
def save_species_pie_chart(
    species: List[Dict[str, Any]],
    out_path: Path,
    entry_id: str,
    top_k: int = 5
) -> Optional[Path]:
    """
    Save a donut chart of species probabilities.
    species: list of dicts with keys ["species", "percent"]
    """
    if not species:
        return None

    top = species[:top_k]
    other_pct = sum(s["percent"] for s in species[top_k:])

    sizes = [s["percent"] for s in top] + [other_pct]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, _ = ax.pie(
        sizes,
        startangle=140,
        wedgeprops=dict(
            width=0.5,
            edgecolor="black",
            linewidth=1.2
        ),
    )

    legend_labels = (
        [f"{s['species']} — {s['percent']:.1f}%" for s in top]
        + [f"Other — {other_pct:.1f}%"]
    )

    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    ax.set_aspect("equal")

    plt.tight_layout()

    out_file = out_path / f"pie_chart_{entry_id}.png"
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    return out_file


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # ---- Terminal prompts ----
    images_dir = Path(ask("Input images folder:", "/home/fss6k/embedded_data_CM/query_maps/save_images_trees_in_the_streets_testing_query"))
    out_dir = Path(ask("Output folder:", "/home/fss6k/tree_pipeline_outputs"))
    ensure_dir(out_dir)

    words_raw = ask("GroundingDINO prompt words (comma-separated):", "tree, tree trunk")
    prompt_words = [w.strip() for w in words_raw.split(",") if w.strip()]
    if not prompt_words:
        print("ERROR: no prompt words provided.")
        sys.exit(1)

    exts_raw = ask("Image extensions (comma-separated):", ".jpg,.jpeg,.png")
    exts = tuple(e.strip() for e in exts_raw.split(",") if e.strip())

    dino_box_thresh = float(ask("GroundingDINO box threshold:", "0.35"))
    dino_text_thresh = float(ask("GroundingDINO text threshold:", "0.25"))
    select_score_min = float(ask("Min score for selection heuristic:", "0.25"))

    sam_pad = int(ask("SAM box padding (pixels):", "50"))
    sam_ckpt = ask("SAM checkpoint path:", "/home/fss6k/models/sam_vit_b_01ec64.pth")
    sam_type = ask("SAM model type (vit_b/vit_l/vit_h):", "vit_b")

    dino_model_path = ask("GroundingDINO local model path:", "/home/fss6k/models/grounding_dino_tiny")

    bioclip_weights_path = ask("BioCLIP weights (.bin) path:", "/home/fss6k/models/bioclip_v1/open_clip_pytorch_model.bin")
    bioclip_tokenizer_path = ask("BioCLIP tokenizer folder path:", "/home/fss6k/models/bioclip_v1")
    tree_species_json = ask("Glasgow tree_species.json path:", "/home/fss6k/GALLANT-WS3/projects/Ask_the_map/tree_species.json")

    topk = int(ask("Top-K species to save:", "10"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # ---- Load models once ----
    print("\nLoading GroundingDINO...")
    processor = AutoProcessor.from_pretrained(dino_model_path, local_files_only=True)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_path, local_files_only=True).to(device).eval()

    print("Loading SAM...")
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    sam.to(device).eval()
    predictor = SamPredictor(sam)

    print("Loading BioCLIP (open_clip)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained=bioclip_weights_path
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = HFTokenizer(bioclip_tokenizer_path)

    print("Loading candidate species list...")
    with open(tree_species_json, "r") as f:
        tree_data = json.load(f)
    candidates = flatten_glasgow_tree_list(tree_data)

    # ---- Enumerate images ----
    img_files = list_images(images_dir, exts)
    print(f"\nFound {len(img_files)} images in {images_dir}")
    if not img_files:
        sys.exit(0)

    # ---- Process ----
    for img_path in img_files:
        stem = img_path.stem
        out_plot = out_dir / f"{stem}_panel.png"
        out_json = out_dir / f"{stem}_results.json"
        out_pie = None  # will be created later

        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] {img_path.name}: failed to open ({e})")
            continue

        W, H = pil.size

        # -------- GroundingDINO detect --------
        text_labels = [prompt_words]

        inputs = processor(images=pil, text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=dino_box_thresh,
            text_threshold=dino_text_thresh,
            target_sizes=[(H, W)]
        )
        detections = results[0]

        best_box = select_tree_box(detections, pil.size, score_min=select_score_min)

        # If no box, still save a debug plot + JSON
        if best_box is None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            axes[0].imshow(pil); axes[0].set_title("RAW"); axes[0].axis("off")
            axes[1].imshow(pil); axes[1].set_title("NO BOX SELECTED"); axes[1].axis("off")
            axes[2].imshow(np.zeros((H, W, 3), dtype=np.uint8)); axes[2].set_title("MASKED (N/A)"); axes[2].axis("off")
            plt.tight_layout()
            fig.savefig(out_plot, dpi=160)
            plt.close(fig)

            payload = {
                "image": str(img_path),
                "status": "no_detection",
                "prompt_words": prompt_words,
                "dino": {
                    "box_threshold": dino_box_thresh,
                    "text_threshold": dino_text_thresh,
                    "num_boxes": int(len(detections.get("boxes", []))),
                },
                "selected_box": None,
                "sam": None,
                "bioclip": None,
                "outputs": {
                    "panel_plot": str(out_plot),
                    "pie_chart": None,
                    "json": str(out_json),
                }
            }
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[NO DETECTION] {img_path.name} -> saved {out_plot.name}, {out_json.name}")
            continue

        # -------- SAM mask --------
        np_img = np.array(pil, dtype=np.uint8)
        predictor.set_image(np_img)

        box_used = expand_box(best_box, np_img.shape, pad=sam_pad)
        box_np = np.array(box_used, dtype=np.float32)

        masks, scores, _ = predictor.predict(box=box_np, multimask_output=True)
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(bool)
        sam_score = float(scores[best_idx])

        masked = np_img.copy()
        masked[~mask] = 0

        # -------- BioCLIP classify (masked image) --------
        masked_pil = Image.fromarray(masked)

        top_species, top_genera = bioclip_rank_species(
            masked_pil,
            candidates=candidates,
            model=clip_model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=device,
            topk=topk
        )

        # Convert to percentages for JSON readability
        top_species_pct = probs_to_percent(top_species, key="p")
        top_genera_pct = []
        genus_sum = sum(g[1] for g in top_genera) + 1e-12
        for g, p in top_genera[:min(10, len(top_genera))]:
            top_genera_pct.append({"genus": g, "percent": float(p) / genus_sum * 100.0, "p": float(p)})

        # -------- NEW: Save pie/donut chart --------
        out_pie = save_species_pie_chart(
            species=top_species_pct,
            out_path=out_dir,
            entry_id=stem,
            top_k=5
        )

        # -------- Save 3-panel plot --------
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        axes[0].imshow(pil)
        axes[0].set_title("RAW")
        axes[0].axis("off")

        axes[1].imshow(pil)
        axes[1].set_title("SELECTED BOX")
        axes[1].axis("off")
        draw_selected_box(axes[1], best_box, label="SELECTED")

        axes[2].imshow(masked)
        axes[2].set_title("MASKED (SAM)")
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(out_plot, dpi=160)
        plt.close(fig)

        # -------- Save JSON --------
        payload = {
            "image": str(img_path),
            "status": "ok",
            "prompt_words": prompt_words,

            "dino": {
                "model_path": dino_model_path,
                "box_threshold": dino_box_thresh,
                "text_threshold": dino_text_thresh,
                "num_boxes": int(len(detections.get("boxes", []))),
                "boxes": [list(map(float, b.tolist())) for b in detections.get("boxes", [])],
                "scores": [float(s) for s in detections.get("scores", [])],
            },

            "selected_box": {
                "box_xyxy": best_box,
                "box_used_for_sam_xyxy": box_used,
            },

            "sam": {
                "model_type": sam_type,
                "checkpoint": sam_ckpt,
                "pad": sam_pad,
                "best_score": sam_score,
            },

            "bioclip": {
                "weights": bioclip_weights_path,
                "tokenizer_path": bioclip_tokenizer_path,
                "top_species": top_species_pct,
                "top_genera": top_genera_pct,
            },

            "outputs": {
                "panel_plot": str(out_plot),
                "pie_chart": str(out_pie) if out_pie else None,
                "json": str(out_json),
            }
        }

        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        best_name = top_species_pct[0]["species"] if top_species_pct else "UNKNOWN"
        best_pct = top_species_pct[0]["percent"] if top_species_pct else 0.0
        print(f"[OK] {img_path.name} -> {out_plot.name}, {out_json.name}, {Path(out_pie).name if out_pie else 'no_pie'} | top: {best_name} ({best_pct:.1f}%)")


if __name__ == "__main__":
    main()
