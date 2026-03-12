from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import sys
import cv2
import torch
import numpy as np
from PIL import Image

# add Depth Anything repo to path
sys.path.append("/home/fss6k/models/Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2


ImageLike = Union[str, Path, Image.Image, np.ndarray]


def load_depth_anything_v2(
    repo_path: Union[str, Path] = "/home/fss6k/models/Depth-Anything-V2",
    ckpt_path: Union[str, Path] = "/home/fss6k/models/depth_anything_v2_vitb.pth",
    encoder: str = "vitb",
    features: int = 128,
    out_channels: list[int] = [96, 192, 384, 768],
    device: Optional[str] = None,
):
    """
    Load Depth Anything V2 model from local checkpoint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    repo_path = str(repo_path)
    ckpt_path = str(ckpt_path)

    if repo_path not in sys.path:
        sys.path.append(repo_path)

    model = DepthAnythingV2(
        encoder=encoder,
        features=features,
        out_channels=out_channels,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"Loaded Depth Anything V2 model from {ckpt_path} on {device}")
    return model, device


def load_image(image_or_path: ImageLike) -> Image.Image:
    """
    Load an RGB PIL image from path / PIL / numpy input.
    """
    if isinstance(image_or_path, (str, Path)):
        return Image.open(image_or_path).convert("RGB")

    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")

    if isinstance(image_or_path, np.ndarray):
        if image_or_path.ndim == 2:
            return Image.fromarray(image_or_path).convert("RGB")
        return Image.fromarray(image_or_path.astype(np.uint8)).convert("RGB")

    raise TypeError(f"Unsupported image input type: {type(image_or_path)}")


def infer_depth(
    image_or_path: ImageLike,
    model,
    device: str,
    input_size: int = 518,
) -> np.ndarray:
    """
    Infer relative depth map for one image.
    Returns a float32 numpy array of shape (H, W).
    """
    image = load_image(image_or_path)
    img = np.array(image)

    # PIL RGB -> OpenCV BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (input_size, input_size))

    x = torch.from_numpy(img_resized).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(x)

    depth = depth.squeeze().detach().cpu().numpy().astype(np.float32)
    depth = cv2.resize(depth, (w, h))

    return depth


def normalize_depth(depth: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize depth map to [0, 1].
    """
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    return (depth - depth_min) / (depth_max - depth_min + eps)


def depth_to_pil(depth: np.ndarray) -> Image.Image:
    """
    Convert depth map to grayscale PIL image.
    """
    depth_norm = normalize_depth(depth)
    return Image.fromarray((depth_norm * 255).astype(np.uint8)).convert("L")