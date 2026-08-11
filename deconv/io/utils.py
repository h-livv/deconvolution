"""Image I/O and common preprocessing utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image and return it as a float64 grayscale array in [0, 1]."""
    image = Image.open(path)
    if image.mode != "L":
        image = image.convert("L")
    array = np.asarray(image, dtype=np.float64)
    return normalize(array)


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize an array to the unit interval [0, 1]."""
    image = np.asarray(image, dtype=np.float64)
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.float64)
    return (image - min_val) / (max_val - min_val)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a grayscale image to ``size`` × ``size`` using bilinear resampling."""
    pil_image = Image.fromarray((normalize(image) * 255.0).astype(np.uint8), mode="L")
    resized = pil_image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64) / 255.0


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save a grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image in [0, 1] as PNG."""
    clipped = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    u8 = (clipped * 255.0).round().astype(np.uint8)
    if u8.ndim == 2:
        pil_image = Image.fromarray(u8, mode="L")
    elif u8.ndim == 3 and u8.shape[-1] == 3:
        pil_image = Image.fromarray(u8, mode="RGB")
    else:
        raise ValueError(f"Unsupported image shape for save_image: {u8.shape}")
    pil_image.save(path)


def create_results_folder(base_dir: str | Path | None = None) -> Path:
    """Create a uniquely timestamped results directory and return its path.

    Relative paths are resolved against the repository root so entry points
    work regardless of the caller's working directory.
    """
    from deconv.paths import REPO_ROOT, RESULTS_DIR

    if base_dir is None:
        base = RESULTS_DIR
    else:
        base = Path(base_dir)
        if not base.is_absolute():
            base = REPO_ROOT / base
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base / stamp
    suffix = 1
    while output_dir.exists():
        output_dir = base / f"{stamp}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def clip_to_unit_interval(image: np.ndarray) -> np.ndarray:
    """Clip values to [0, 1] for display and metric comparison."""
    return np.clip(image, 0.0, 1.0)


def show_image(axis, image: np.ndarray, title: str = "", *, fontsize: int | None = None) -> None:
    """imshow helper: RGB as colour, grayscale with a fixed [0, 1] scale."""
    display = clip_to_unit_interval(np.asarray(image, dtype=np.float64))
    if display.ndim == 3 and display.shape[-1] == 3:
        axis.imshow(display)
    else:
        if display.ndim == 3:
            display = display[..., 0]
        axis.imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
    if title:
        if fontsize is not None:
            axis.set_title(title, fontsize=fontsize)
        else:
            axis.set_title(title)
    axis.axis("off")
