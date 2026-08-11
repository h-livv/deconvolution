"""Direct deconvolution via the matrix representation of convolution."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import convolve2d

Boundary = Literal["wrap", "fill"]


def build_convolution_matrix(
    psf: np.ndarray,
    height: int,
    width: int,
    boundary: Boundary = "wrap",
    fillvalue: float = 0.0,
) -> np.ndarray:
    """
    Construct the dense matrix ``A`` satisfying

        A @ x.ravel() == convolve2d(x, psf, mode='same', boundary=...).ravel()

    Each column is the response to a unit impulse at one pixel, so ``A`` is
    exactly the selected linear (``fill``) or circular (``wrap``) blur operator.
    """
    if boundary not in {"wrap", "fill"}:
        raise ValueError(f"Unknown boundary '{boundary}'. Use 'wrap' or 'fill'.")

    n_pixels = height * width
    matrix = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    conv_kwargs: dict = {"mode": "same", "boundary": boundary}
    if boundary == "fill":
        conv_kwargs["fillvalue"] = fillvalue

    for row in range(height):
        for col in range(width):
            impulse = np.zeros((height, width), dtype=np.float64)
            impulse[row, col] = 1.0
            response = convolve2d(impulse, psf, **conv_kwargs)
            matrix[:, row * width + col] = response.ravel()

    return matrix


def direct_deconvolution(
    blurred: np.ndarray,
    psf: np.ndarray,
    boundary: Boundary = "wrap",
    fillvalue: float = 0.0,
) -> np.ndarray:
    """
    Recover an image by solving the linear system

        A x = b

    where ``A`` is the convolution matrix for the chosen boundary model.
    """
    height, width = blurred.shape
    matrix = build_convolution_matrix(
        psf, height, width, boundary=boundary, fillvalue=fillvalue
    )
    observation = blurred.ravel()

    # Gaussian blur yields a poorly conditioned operator; least squares is the
    # stable NumPy solution of Ax = b without adding regularization.
    recovered_flat, _, _, _ = np.linalg.lstsq(matrix, observation, rcond=None)
    return recovered_flat.reshape(height, width)
