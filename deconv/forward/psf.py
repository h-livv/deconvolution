"""Point-spread function (PSF) generation."""

from __future__ import annotations

import numpy as np


def gaussian_psf(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Build a normalized 2-D Gaussian PSF by direct evaluation of

        G(x, y) = (1 / (2 π σ²)) · exp(-(x² + y²) / (2 σ²))

    The kernel is centered on its middle pixel and normalized so that its
    entries sum to one, preserving image intensity under convolution.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    radius = kernel_size // 2
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float64)
