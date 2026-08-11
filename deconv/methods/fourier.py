"""Fourier-domain deconvolution via the convolution theorem.

Under the circular (periodic) convolution model,

    b = h * x  ⟹  B̂ = Ĥ X̂  ⟹  X̂ = B̂ / Ĥ,

the DFT diagonalizes convolution, so deconvolution is pointwise division of
Fourier coefficients. This module implements that identity on the native
image grid — not a spatial-domain matrix solve.
"""

from __future__ import annotations

import numpy as np


def _embed_psf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Embed ``psf`` into an FFT grid of size ``shape``.

    The kernel is placed in the top-left corner and then rolled so that its
    spatial centre (``psf.shape // 2``) lands at index ``(0, 0)``, matching
    the NumPy ``fft2`` convention for circular convolution. With that
    alignment,

        ifft2(fft2(x) * fft2(_embed_psf(h, x.shape)))

    coincides with ``scipy.signal.convolve2d(x, h, mode='same', boundary='wrap')``.
    """
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    return np.roll(padded, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))


def fourier_deconvolution(
    blurred: np.ndarray,
    psf: np.ndarray,
    *,
    reg: float = 0.0,
) -> np.ndarray:
    """
    Recover ``x`` from circular convolution ``b = h * x`` by Fourier division.

    Parameters
    ----------
    blurred:
        Observed image ``b`` (same shape as the unknown ``x``).
    psf:
        Convolution kernel ``h``. Centred on its middle pixel; odd sizes are
        the usual case so the centre is unambiguous.
    reg:
        Non-negative Tikhonov / Wiener parameter ``λ``. With ``λ = 0`` the
        unregularized inverse is used:

            X̂ = B̂ / Ĥ

        With ``λ > 0`` the stabilized filter

            X̂ = Ĥ* / (|Ĥ|² + λ) · B̂

        is used instead. The latter preserves the phase of ``Ĥ`` (it never
        replaces small complex coefficients by a real scalar).

    Returns
    -------
    Real-valued reconstruction ``x`` on the same grid as ``blurred``.

    Notes
    -----
    Boundary model: circular / periodic only. Zero-padded (``fill``) linear
    convolution is *not* diagonalized by an image-sized DFT, so it is outside
    the scope of this function.
    """
    if reg < 0.0:
        raise ValueError("reg must be non-negative")

    blurred = np.asarray(blurred, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)

    B_ft = np.fft.fft2(blurred)
    H_ft = np.fft.fft2(_embed_psf(psf, blurred.shape))

    if reg == 0.0:
        # Convolution theorem: X̂ = B̂ / Ĥ
        X_ft = B_ft / H_ft
    else:
        # Tikhonov inverse: X̂ = Ĥ* / (|Ĥ|² + λ) · B̂
        X_ft = np.conj(H_ft) / (np.abs(H_ft) ** 2 + reg) * B_ft

    return np.real(np.fft.ifft2(X_ft))
