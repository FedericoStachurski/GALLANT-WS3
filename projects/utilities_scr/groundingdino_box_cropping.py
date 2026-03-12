from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


ImageLike = Union[str, Path, Image.Image]
Box = List[int]


def load_grounding_dino(
    model_path: Union[str, Path] = "/home/fss6k/models/grounding_dino_tiny",
    device: Optional[str] = None,
):
    """
    Load GroundingDINO processor and model from a local folder.
    """
    model_path = str(model_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path,
        local_files_only=True
    ).to(device).eval()
    print(f"Loaded GroundingDINO model from {model_path} on {device}")
    return processor, model, device


def load_image(image_or_path: ImageLike) -> Image.Image:
    """
    Load a PIL RGB image from path or return a converted PIL image.
    """
    if isinstance(image_or_path, (str, Path)):
        return Image.open(image_or_path).convert("RGB")
    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image_or_path)}")


def select_tree_box(
    detections: Dict[str, Any],
    image_size: Tuple[int, int],
    score_min: float = 0.2,
    border_margin: int = 5,
    max_aspect_ratio: float = 6.0,
    edge_penalty_base: float = 0.65,
) -> Optional[Box]:
    """
    Select the best tree box using confidence, area, edge penalty, and aspect penalty.
    image_size is (W, H).
    """
    W, H = image_size
    best_box: Optional[Box] = None
    best_score = -1.0

    for box, score in zip(detections["boxes"], detections["scores"]):
        s = float(score)
        if s < score_min:
            continue

        x0, y0, x1, y1 = box.tolist()
        w, h = x1 - x0, y1 - y0
        if w <= 1 or h <= 1:
            continue

        area = w * h

        touch = (
            (x0 < border_margin)
            + (y0 < border_margin)
            + (x1 > W - border_margin)
            + (y1 > H - border_margin)
        )
        edge_penalty = edge_penalty_base ** touch

        aspect = max(w / h, h / w)
        aspect_penalty = 1.0 if aspect < max_aspect_ratio else 0.5

        combined = (area * s) * edge_penalty * aspect_penalty

        if combined > best_score:
            best_score = combined
            best_box = [int(x0), int(y0), int(x1), int(y1)]

    return best_box


def detect_tree_box(
    image_or_path: ImageLike,
    processor,
    model,
    device: str,
    text_labels: Optional[Sequence[Sequence[str]]] = None,
    threshold: float = 0.2,
    text_threshold: float = 0.25,
    score_min: float = 0.2,
) -> Dict[str, Any]:
    """
    Run GroundingDINO on one image and return detections + selected best box.
    """
    if text_labels is None:
        text_labels = [["tree", "tree trunk", "tree canopy"]]

    image = load_image(image_or_path)

    inputs = processor(
        images=image,
        text=text_labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (H, W)
    )

    detections = results[0]
    best_box = select_tree_box(detections, image.size, score_min=score_min)

    return {
        "image": image,
        "detections": detections,
        "best_box": best_box,
    }


def expand_box(
    box: Box,
    image_size: Tuple[int, int],
    scale: float = 1.35,
    scale_x: Optional[float] = None,
    scale_y: Optional[float] = None,
) -> Box:
    """
    Expand a bounding box around its center.
    image_size is (W, H).

    If scale_x / scale_y are provided, they override scale for anisotropic expansion.
    """
    x0, y0, x1, y1 = box
    W, H = image_size

    if scale_x is None:
        scale_x = scale
    if scale_y is None:
        scale_y = scale

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    w = (x1 - x0) * scale_x
    h = (y1 - y0) * scale_y

    x0 = max(0, int(cx - w / 2))
    y0 = max(0, int(cy - h / 2))
    x1 = min(W, int(cx + w / 2))
    y1 = min(H, int(cy + h / 2))

    return [x0, y0, x1, y1]


def crop_box(image_or_path: ImageLike, box: Optional[Box]) -> Optional[Image.Image]:
    """
    Crop a PIL image using box coordinates.
    """
    if box is None:
        return None
    image = load_image(image_or_path)
    return image.crop(box)


def show_tree_detections(
    result: Dict[str, Any],
    score_min: float = 0.2,
    figsize: Tuple[int, int] = (8, 6),
    show_selected_label: bool = True,
):
    """
    Plot all detections and highlight the selected best box.
    """
    image = result["image"]
    detections = result["detections"]
    best_box = result["best_box"]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    ax.axis("off")

    for box, score in zip(detections["boxes"], detections["scores"]):
        s = float(score)
        if s < score_min:
            continue

        x0, y0, x1, y1 = box.tolist()
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            max(y0 - 6, 0),
            f"{s:.2f}",
            color="red",
            fontsize=9,
            backgroundcolor="white",
        )

    if best_box is not None:
        x0, y0, x1, y1 = best_box
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=3,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        if show_selected_label:
            ax.text(
                x0,
                max(y0 - 12, 0),
                "SELECTED",
                color="lime",
                fontsize=11,
                backgroundcolor="black",
            )

    plt.tight_layout()
    plt.show()