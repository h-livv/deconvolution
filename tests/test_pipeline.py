"""End-to-end physical-accuracy tests for the matched circular pipeline."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from deconv.data.synthetic import generate_synthetic_image
from deconv.forward.blur import apply_blur
from deconv.forward.psf import gaussian_psf
from deconv.methods.direct import build_convolution_matrix, direct_deconvolution
from deconv.methods.fourier import _embed_psf, fourier_deconvolution
from deconv.metrics import mean_squared_error


def test_psf_is_normalized_and_centered() -> None:
    h = gaussian_psf(11, 1.5)
    assert h.shape == (11, 11)
    np.testing.assert_allclose(h.sum(), 1.0, atol=1e-15)
    # Peak at geometric centre.
    peak = np.unravel_index(np.argmax(h), h.shape)
    assert peak == (5, 5)
    # Radial symmetry of a centred isotropic Gaussian.
    np.testing.assert_allclose(h, h.T, atol=1e-15)
    np.testing.assert_allclose(h, np.flip(h), atol=1e-15)


def test_wrap_blur_matches_fft_multiplication() -> None:
    x = generate_synthetic_image(24, "border_frame")
    h = gaussian_psf(7, 1.2)
    spatial = apply_blur(x, h, boundary="wrap")
    via_fft = np.real(
        np.fft.ifft2(np.fft.fft2(x) * np.fft.fft2(_embed_psf(h, x.shape)))
    )
    np.testing.assert_allclose(spatial, via_fft, atol=1e-12)


def test_direct_matrix_matches_convolve2d_wrap() -> None:
    h = gaussian_psf(5, 1.0)
    height = width = 10
    A = build_convolution_matrix(h, height, width, boundary="wrap")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(height, width))
    Ax = (A @ x.ravel()).reshape(height, width)
    ref = convolve2d(x, h, mode="same", boundary="wrap")
    np.testing.assert_allclose(Ax, ref, atol=1e-12)


def test_matched_circular_pipeline_recovers_image() -> None:
    """Blur and both inverses share the wrap operator; recovery must succeed."""
    x = generate_synthetic_image(20, "edge_square")
    h = gaussian_psf(7, 1.0)
    b = apply_blur(x, h, boundary="wrap")

    x_direct = direct_deconvolution(b, h, boundary="wrap")
    x_fourier = fourier_deconvolution(b, h, reg=0.0)

    assert mean_squared_error(x, x_direct) < 1e-10
    assert mean_squared_error(x, x_fourier) < 1e-10
    assert mean_squared_error(x_direct, x_fourier) < 1e-10


def test_mismatch_boundary_breaks_fourier_but_not_direct() -> None:
    """Linear blur + circular Fourier inverse is the wrong physical model."""
    x = generate_synthetic_image(20, "border_frame")
    h = gaussian_psf(7, 1.2)
    b_fill = apply_blur(x, h, boundary="fill", fillvalue=0.0)

    x_direct = direct_deconvolution(b_fill, h, boundary="fill", fillvalue=0.0)
    x_fourier = fourier_deconvolution(b_fill, h, reg=0.0)

    assert mean_squared_error(x, x_direct) < 1e-10
    # Fourier assumes wrap; fill observation is inconsistent → large error.
    assert mean_squared_error(x, x_fourier) > 1e-2
