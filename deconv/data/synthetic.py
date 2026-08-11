"""Synthetic edge-emphasizing test images for boundary-condition demos."""

from __future__ import annotations

from typing import Literal

import numpy as np

SyntheticPattern = Literal[
    "corner_pixel",
    "edge_square",
    "border_frame",
    "diagonal",
]

# Grayscale intensities in [0, 1].
BACKGROUND = 0.0       # black
FOREGROUND = 0.35      # dark grey borders / patterns


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Ensure a 2-D float64 image for blur / deconvolution.

    Already-grayscale inputs are returned unchanged (as float64). RGB inputs
    are reduced with Rec. 709 luminance.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] >= 3:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    raise ValueError(f"Unsupported image shape {image.shape}")


def generate_synthetic_image(
    size: int,
    pattern: SyntheticPattern = "border_frame",
) -> np.ndarray:
    """
    Build a ``size`` × ``size`` float64 grayscale image in [0, 1].

    Black background with dark-grey borders/patterns for boundary demos.

    Patterns
    --------
    corner_pixel:
        Single foreground pixel at the top-left corner.
    edge_square:
        Foreground square flush with the top and left borders.
    border_frame:
        One-pixel-wide foreground frame around the image.
    diagonal:
        Foreground diagonal band that intersects opposite borders.
    """
    if size < 4:
        raise ValueError("size must be at least 4")

    image = np.full((size, size), BACKGROUND, dtype=np.float64)

    if pattern == "corner_pixel":
        image[0, 0] = FOREGROUND
    elif pattern == "edge_square":
        extent = max(size // 3, 2)
        image[:extent, :extent] = FOREGROUND
    elif pattern == "border_frame":
        image[0, :] = FOREGROUND
        image[-1, :] = FOREGROUND
        image[:, 0] = FOREGROUND
        image[:, -1] = FOREGROUND
    elif pattern == "diagonal":
        thickness = max(size // 16, 1)
        for offset in range(-thickness, thickness + 1):
            rows = np.arange(size)
            cols = rows + offset
            valid = (cols >= 0) & (cols < size)
            image[rows[valid], cols[valid]] = FOREGROUND
    else:
        raise ValueError(
            f"Unknown SYNTHETIC_PATTERN '{pattern}'. "
            "Use corner_pixel, edge_square, border_frame, or diagonal."
        )

    return image
