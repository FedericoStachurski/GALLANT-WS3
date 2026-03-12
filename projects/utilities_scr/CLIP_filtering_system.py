
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Union, Optional

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


ImageLike = Union[str, Path, Image.Image]


# --------------------------------------------------
# Default prompts
# --------------------------------------------------

TREE_PROMPTS = [
    "a photograph of a tree trunk in a forest",
    "a photo of a standing tree outdoors",
    "a tall tree with a visible trunk",
    "trees in a natural outdoor setting",
    "a forest with tree trunks",
    "a tree with a clear trunk and branches",
    "a photo of a tree with visible bark and trunk",
]

NON_TREE_PROMPTS = [
    "a painting or artwork of leaves",
    "a close-up photo of leaves without a tree trunk",
    "an abstract or artistic image",
    "a photo of grass or bushes without trees",
    "an indoor photo with no trees",
    "a blank or solid colour image",
    "a photo of a person or animal without trees",
    "a photo of a cityscape or building without trees",
]


# --------------------------------------------------
# Model loader
# --------------------------------------------------

def load_clip_model(
    model_path: Union[str, Path] = "/home/fss6k/models/clip-vit-base-patch32",
    device: Optional[str] = None,
):
    """
    Load CLIP model and processor.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_path).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_path)
    print(f"Loaded CLIP model from {model_path} on {device}")
    return model, processor, device


# --------------------------------------------------
# Prompt loading (optional)
# --------------------------------------------------

def load_prompt_file(path: Union[str, Path]) -> List[str]:
    """
    Load prompts from a text file (one prompt per line).
    """
    path = Path(path)

    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    return prompts


# --------------------------------------------------
# Image loading
# --------------------------------------------------

def load_image(image_or_path: ImageLike) -> Image.Image:

    if isinstance(image_or_path, (str, Path)):
        return Image.open(image_or_path).convert("RGB")

    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image_or_path)}")


# --------------------------------------------------
# Batch scoring
# --------------------------------------------------

def score_images_batch(
    image_paths: List[Union[str, Path]],
    model,
    processor,
    device,
    tree_prompts: Optional[List[str]] = None,
    non_tree_prompts: Optional[List[str]] = None,
    batch_size: int = 16,
    threshold: float = 0.5,
) -> List[Dict]:

    if tree_prompts is None:
        tree_prompts = TREE_PROMPTS

    if non_tree_prompts is None:
        non_tree_prompts = NON_TREE_PROMPTS

    all_prompts = tree_prompts + non_tree_prompts
    n_tree = len(tree_prompts)

    results = []

    for i in range(0, len(image_paths), batch_size):

        batch_paths = image_paths[i:i + batch_size]

        images = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = load_image(p)
                images.append(img)
                valid_paths.append(p)
            except Exception:
                results.append({
                    "image_path": str(p),
                    "tree_score": 0.0,
                    "top_prompt": "LOAD_ERROR",
                    "is_tree": False,
                    "raw_probs": []
                })

        if not images:
            continue

        inputs = processor(
            text=all_prompts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

        probs = probs.cpu().numpy()

        for j, p in enumerate(valid_paths):

            pr = probs[j].tolist()

            tree_score = sum(pr[:n_tree])
            top_idx = int(np.argmax(pr))

            results.append({
                "image_path": str(p),
                "tree_score": round(tree_score, 4),
                "top_prompt": all_prompts[top_idx],
                "is_tree": tree_score > threshold,
                "raw_probs": pr
            })

    return results