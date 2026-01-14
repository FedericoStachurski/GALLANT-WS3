#!/usr/bin/env python3
"""
tree_pipeline_prototype.py

Prototype pipeline:
- Segment a "tree-like" region using SAM (if installed) + simple heuristics
- Classify into functional categories with CLIP (robust)
- Optionally suggest species with CLIP zero-shot (assistive)
- Produce basic scorecards (indicative CO2 + runoff proxy) with uncertainty tiers

Usage:
  python tree_pipeline_prototype.py \
    --image_dir /path/to/tree_photos \
    --out_dir outputs \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --device cuda

If SAM isn't available or no checkpoint is provided, the script falls back to using the full image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor


# ----------------------------
# Config: labels & scorecards
# ----------------------------

DEFAULT_SPECIES_CANDIDATES = [
    # UK-ish common urban trees (edit freely to fit your city)
    "English oak (Quercus robur)",
    "sessile oak (Quercus petraea)",
    "London plane (Platanus × acerifolia)",
    "sycamore (Acer pseudoplatanus)",
    "field maple (Acer campestre)",
    "silver birch (Betula pendula)",
    "downy birch (Betula pubescens)",
    "common lime (Tilia × europaea)",
    "small-leaved lime (Tilia cordata)",
    "horse chestnut (Aesculus hippocastanum)",
    "beech (Fagus sylvatica)",
    "ash (Fraxinus excelsior)",
    "rowan (Sorbus aucuparia)",
    "hawthorn (Crataegus monogyna)",
    "holly (Ilex aquifolium)",
    "Scots pine (Pinus sylvestris)",
    "yew (Taxus baccata)",
    "Norway spruce (Picea abies)",
]

# Functional category prompts (more robust than species)
FUNCTIONAL_PROMPTS = {
    "leaf_type": ["a photo of a broadleaf tree", "a photo of a conifer tree"],
    "phenology": ["a photo of an evergreen tree", "a photo of a deciduous tree"],
    "scene_view": ["a close-up photo of tree bark", "a photo of a tree canopy with leaves", "a photo of a whole tree"],
}

@dataclass
class TreeResult:
    image_name: str
    used_sam: bool
    mask_area_frac: float  # fraction of image pixels in selected mask
    mask_green_score: float  # heuristic
    functional_leaf_type: str
    functional_leaf_type_p: float
    functional_phenology: str
    functional_phenology_p: float
    view_type: str
    view_type_p: float
    top_species: str
    top_species_p: float
    top_species_list_json: str
    confidence_tier: str
    scorecard_json: str


# ----------------------------
# Helper: CLIP zero-shot
# ----------------------------

@torch.inference_mode()
def clip_zero_shot(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    prompts: List[str],
    device: str,
) -> Tuple[List[float], int]:
    """Returns probabilities for each prompt and argmax index."""
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    # CLIP returns logits per (image, text)
    logits = outputs.logits_per_image[0]  # shape: (num_prompts,)
    probs = logits.softmax(dim=-1).detach().cpu().numpy().tolist()
    best_idx = int(np.argmax(probs))
    return probs, best_idx


# ----------------------------
# Helper: Mask selection heuristics
# ----------------------------

def image_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def compute_green_score(rgb: np.ndarray, mask: np.ndarray) -> float:
    """
    Crude heuristic: trees often have lots of green pixels (not always! winter/bark shots will fail).
    green_score in [0,1] roughly indicates "green dominance" within mask.
    """
    if mask.sum() < 10:
        return 0.0
    region = rgb[mask]
    r, g, b = region[:, 0].astype(np.float32), region[:, 1].astype(np.float32), region[:, 2].astype(np.float32)
    # "greenness": green greater than red & blue by margin
    greenish = (g > r + 10) & (g > b + 10)
    return float(greenish.mean())

def pick_best_mask(rgb: np.ndarray, masks: List[Dict]) -> Optional[np.ndarray]:
    """
    Select a mask from SAM's automatic mask generator output.
    We combine:
      - larger area (prefer canopy-ish regions)
      - green_score (prefer vegetation)
      - penalize extremely huge masks (like entire image)
    """
    H, W, _ = rgb.shape
    best = None
    best_score = -1e9

    for m in masks:
        seg = m.get("segmentation", None)
        if seg is None:
            continue
        seg = seg.astype(bool)
        area = float(seg.sum()) / float(H * W)
        if area < 0.01:  # ignore tiny segments
            continue
        gscore = compute_green_score(rgb, seg)

        # scoring: encourage area, encourage green, penalize near-full-frame
        penalty = 0.0
        if area > 0.80:
            penalty = 2.0 * (area - 0.80)  # strong penalty for "mask is basically everything"

        score = (2.0 * gscore) + (1.0 * math.sqrt(area)) - penalty

        if score > best_score:
            best_score = score
            best = seg

    return best

def apply_mask_crop(img: Image.Image, mask: np.ndarray, pad: int = 10) -> Image.Image:
    """Crop to mask bounding box and black out background."""
    rgb = image_to_np_rgb(img)
    H, W, _ = rgb.shape

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img

    x0, x1 = max(0, xs.min() - pad), min(W - 1, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(H - 1, ys.max() + pad)

    crop = rgb[y0:y1+1, x0:x1+1].copy()
    crop_mask = mask[y0:y1+1, x0:x1+1]

    # black background where not mask
    crop[~crop_mask] = 0
    return Image.fromarray(crop)


# ----------------------------
# Scorecard (indicative + tiered uncertainty)
# ----------------------------

def build_scorecard(mask_area_frac: float, leaf_type: str, phenology: str, tier: str) -> Dict:
    """
    Very rough proxy scorecard based on canopy fraction in the photo.
    This is NOT a physical canopy area unless you have scale / geo.
    Still useful as: relative ranking + transparency.

    Outputs:
      - CO2 sequestration range (relative)
      - runoff interception range (relative)
    """
    # Base multipliers by category (toy assumptions)
    leaf_mult = 1.00 if leaf_type == "broadleaf" else 0.85
    phen_mult = 1.05 if phenology == "evergreen" else 1.00

    # Convert mask fraction into a "size index" in [0,1] with mild nonlinearity
    size_index = float(np.clip(mask_area_frac, 0.0, 1.0)) ** 0.6

    # Toy baseline outputs in arbitrary units, then convert to indicative ranges.
    # You should replace these with your chosen ecological model once you have real features (DBH, crown width, rainfall).
    co2_base = 20.0  # "kg CO2/year" baseline for mid-size; purely illustrative
    runoff_base = 200.0  # "L per rainfall event" baseline; purely illustrative

    point_co2 = co2_base * size_index * leaf_mult * phen_mult
    point_runoff = runoff_base * size_index * leaf_mult * phen_mult

    # Uncertainty bands by tier
    if tier == "high":
        band = 0.30
    elif tier == "medium":
        band = 0.60
    else:
        band = 1.00

    co2_range = [max(0.0, point_co2 * (1 - band)), point_co2 * (1 + band)]
    runoff_range = [max(0.0, point_runoff * (1 - band)), point_runoff * (1 + band)]

    return {
        "tier": tier,
        "inputs": {
            "mask_area_fraction": mask_area_frac,
            "leaf_type": leaf_type,
            "phenology": phenology,
            "notes": [
                "These metrics are indicative and based on a photo-derived size proxy, not measured canopy area.",
                "Replace toy constants with ecological models once you have real features (e.g., DBH, crown width, rainfall).",
            ],
        },
        "outputs": {
            "co2_sequestration_kg_per_year_range": [round(co2_range[0], 2), round(co2_range[1], 2)],
            "runoff_interception_l_per_event_range": [round(runoff_range[0], 2), round(runoff_range[1], 2)],
        },
    }


def infer_confidence_tier(mask_area_frac: float, used_sam: bool, view_type: str) -> str:
    """
    Simple heuristic:
    - bark-only views are hard for species; canopy/whole tree is better
    - if SAM wasn't used, confidence lower
    """
    if not used_sam:
        return "low"
    if mask_area_frac < 0.05:
        return "low"
    if view_type in ("a photo of a tree canopy with leaves", "a photo of a whole tree"):
        return "medium"
    return "low"


# ----------------------------
# Main
# ----------------------------

def load_sam(sam_checkpoint: Optional[str], device: str):
    """
    Loads Segment Anything if available.
    Returns (sam_available, mask_generator)
    """
    if sam_checkpoint is None:
        return False, None

    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("[WARN] segment-anything not installed. Falling back to no-segmentation mode.")
        return False, None

    if not os.path.exists(sam_checkpoint):
        print(f"[WARN] SAM checkpoint not found: {sam_checkpoint}. Falling back to no-segmentation mode.")
        return False, None

    # You can change vit_h -> vit_l / vit_b depending on checkpoint
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=24,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=800,
    )
    return True, mask_generator


def iter_images(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, type=str, help="Folder containing tree images")
    ap.add_argument("--out_dir", required=True, type=str, help="Output folder")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--sam_checkpoint", default=None, type=str, help="Path to SAM checkpoint .pth")
    ap.add_argument("--species_topk", default=5, type=int)
    ap.add_argument("--species_list", default=None, type=str, help="Optional path to txt file, one species per line")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "json").mkdir(parents=True, exist_ok=True)

    # Load CLIP
    device = args.device
    clip_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_name)
    model = CLIPModel.from_pretrained(clip_name).to(device)
    model.eval()

    # Species list
    species_candidates = DEFAULT_SPECIES_CANDIDATES
    if args.species_list:
        sp_path = Path(args.species_list)
        if sp_path.exists():
            if sp_path.suffix.lower() == ".json":
                with open(sp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                species_candidates = []
                for tree in data.get("trees", []):
                    for sp in tree.get("species", []):
                        species_candidates.append(sp["name"])
            else:
                species_candidates = [ln.strip() for ln in sp_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # Load SAM (optional)
    sam_available, mask_generator = load_sam(args.sam_checkpoint, device=device)

    rows: List[TreeResult] = []

    files = iter_images(image_dir)
    print(f"[INFO] Found {len(files)} images in {image_dir}")

    for idx, fp in enumerate(files):
        try:
            img = Image.open(fp).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {fp}: {e}")
            continue

        used_sam = False
        mask_area_frac = 1.0
        mask_green_score = 0.0
        masked_img = img

        rgb = image_to_np_rgb(img)

        # Segment with SAM if available
        if sam_available and mask_generator is not None:
            try:
                masks = mask_generator.generate(rgb)
                best_mask = pick_best_mask(rgb, masks)
                if best_mask is not None:
                    used_sam = True
                    mask_area_frac = float(best_mask.mean())
                    mask_green_score = compute_green_score(rgb, best_mask)
                    masked_img = apply_mask_crop(img, best_mask, pad=12)
            except Exception as e:
                print(f"[WARN] SAM failed on {fp.name}: {e}")

        # Functional classification
        leaf_prompts = FUNCTIONAL_PROMPTS["leaf_type"]
        leaf_probs, leaf_best = clip_zero_shot(model, processor, masked_img, leaf_prompts, device)
        leaf_label = "broadleaf" if leaf_best == 0 else "conifer"

        phen_prompts = FUNCTIONAL_PROMPTS["phenology"]
        phen_probs, phen_best = clip_zero_shot(model, processor, masked_img, phen_prompts, device)
        phen_label = "evergreen" if phen_best == 0 else "deciduous"

        view_prompts = FUNCTIONAL_PROMPTS["scene_view"]
        view_probs, view_best = clip_zero_shot(model, processor, masked_img, view_prompts, device)
        view_label = view_prompts[view_best]

        # Species suggestions (assistive)
        # Use "a photo of <species>" prompts
        sp_prompts = [f"a photo of {s}" for s in species_candidates]
        sp_probs, sp_best = clip_zero_shot(model, processor, masked_img, sp_prompts, device)
        topk = int(np.clip(args.species_topk, 1, len(species_candidates)))
        top_idx = np.argsort(sp_probs)[::-1][:topk].tolist()
        top_species_list = [{"species": species_candidates[i], "p": float(sp_probs[i])} for i in top_idx]

        confidence_tier = infer_confidence_tier(mask_area_frac, used_sam, view_label)
        scorecard = build_scorecard(mask_area_frac, leaf_label, phen_label, confidence_tier)

        result = TreeResult(
            image_name=fp.name,
            used_sam=used_sam,
            mask_area_frac=mask_area_frac,
            mask_green_score=mask_green_score,
            functional_leaf_type=leaf_label,
            functional_leaf_type_p=float(leaf_probs[leaf_best]),
            functional_phenology=phen_label,
            functional_phenology_p=float(phen_probs[phen_best]),
            view_type=view_label,
            view_type_p=float(view_probs[view_best]),
            top_species=species_candidates[sp_best],
            top_species_p=float(sp_probs[sp_best]),
            top_species_list_json=json.dumps(top_species_list, ensure_ascii=False),
            confidence_tier=confidence_tier,
            scorecard_json=json.dumps(scorecard, ensure_ascii=False),
        )
        rows.append(result)

        # Per-image JSON
        out_json = out_dir / "json" / f"{fp.stem}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": fp.name,
                    "segmentation": {
                        "used_sam": used_sam,
                        "mask_area_fraction": mask_area_frac,
                        "mask_green_score": mask_green_score,
                    },
                    "functional": {
                        "leaf_type": {"label": leaf_label, "p": leaf_probs[leaf_best], "all": dict(zip(leaf_prompts, leaf_probs))},
                        "phenology": {"label": phen_label, "p": phen_probs[phen_best], "all": dict(zip(phen_prompts, phen_probs))},
                        "view": {"label": view_label, "p": view_probs[view_best], "all": dict(zip(view_prompts, view_probs))},
                    },
                    "species_suggestions": top_species_list,
                    "confidence_tier": confidence_tier,
                    "scorecard": scorecard,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if (idx + 1) % 10 == 0:
            print(f"[INFO] Processed {idx+1}/{len(files)}")

    # Write CSV summary
    out_csv = out_dir / "results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))

    print(f"[DONE] Wrote {len(rows)} results to {out_csv} and JSONs to {out_dir / 'json'}")


if __name__ == "__main__":
    main()
