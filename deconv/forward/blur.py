"""Apply blur to an image via convolution with a PSF."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import convolve2d

Boundary = Literal["wrap", "fill"]


def apply_blur(
    image: np.ndarray,
    psf: np.ndarray,
    boundary: Boundary = "wrap",
    fillvalue: float = 0.0,
) -> np.ndarray:
    """
    Convolve ``image`` with ``psf`` (same-size output).

    Parameters
    ----------
    boundary:
        ``"wrap"`` — circular / periodic convolution.
        ``"fill"`` — linear convolution with constant padding (default 0).
    """
    if boundary == "wrap":
        blurred = convolve2d(image, psf, mode="same", boundary="wrap")
    elif boundary == "fill":
        blurred = convolve2d(
            image, psf, mode="same", boundary="fill", fillvalue=fillvalue
        )
    else:
        raise ValueError(f"Unknown boundary '{boundary}'. Use 'wrap' or 'fill'.")
    return blurred.astype(np.float64)


def add_gaussian_noise(
    image: np.ndarray,
    std: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add i.i.d. Gaussian noise: ``image + N(0, std²)``.

    If ``std`` is zero or negative, the input is returned unchanged.
    """
    if std <= 0.0:
        return np.asarray(image, dtype=np.float64)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=image.shape)
    return np.asarray(image, dtype=np.float64) + noise
