"""Tests for circular Fourier-domain deconvolution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import convolve2d

from deconv.fourier import _embed_psf, fourier_deconvolution
from deconv.psf import gaussian_psf


def _circular_blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Reference circular blur matching the FFT convention."""
    return convolve2d(image, psf, mode="same", boundary="wrap").astype(np.float64)


def _fft_circular_blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Same operator as ``_circular_blur``, written with FFTs."""
    return np.real(
        np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(_embed_psf(psf, image.shape)))
    )


def test_embed_psf_matches_convolve2d_wrap() -> None:
    rng = np.random.default_rng(0)
    image = rng.normal(size=(16, 20))
    # Asymmetric kernel: wrong centre shift cannot pass.
    psf = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.5],
            [0.0, 0.2, 0.2],
        ],
        dtype=np.float64,
    )
    psf /= psf.sum()

    expected = _circular_blur(image, psf)
    via_fft = _fft_circular_blur(image, psf)
    np.testing.assert_allclose(via_fft, expected, atol=1e-12)


def test_identity_psf_recovers_image() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 10))
    h = np.array([[1.0]])
    b = _circular_blur(x, h)
    recovered = fourier_deconvolution(b, h)
    np.testing.assert_allclose(recovered, x, atol=1e-12)


def test_delta_psf_at_centre_recovers_image() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(14, 14))
    h = np.zeros((5, 5), dtype=np.float64)
    h[2, 2] = 1.0
    b = _circular_blur(x, h)
    recovered = fourier_deconvolution(b, h)
    np.testing.assert_allclose(recovered, x, atol=1e-12)


def test_gaussian_circular_deconvolution_identity() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(32, 32))
    h = gaussian_psf(7, sigma=1.2)
    b = _circular_blur(x, h)
    recovered = fourier_deconvolution(b, h)
    np.testing.assert_allclose(recovered, x, atol=1e-10, rtol=1e-8)


def test_asymmetric_psf_alignment() -> None:
    """Wrong roll of the PSF centre must fail; the correct one recovers x."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 18))
    # Asymmetric and Fourier-invertible (no near-null frequencies).
    h = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.04, 0.50, 0.10],
            [0.05, 0.15, 0.10],
        ],
        dtype=np.float64,
    )
    h /= h.sum()
    assert not np.allclose(h, h[::-1, ::-1])

    b = _circular_blur(x, h)
    recovered = fourier_deconvolution(b, h)
    np.testing.assert_allclose(recovered, x, atol=1e-12)

    # Deliberately mis-centred embed: recovery must fail.
    wrong = np.zeros_like(b)
    kh, kw = h.shape
    wrong[:kh, :kw] = h  # no roll to (0, 0)
    wrong_ft = np.fft.fft2(b) / np.fft.fft2(wrong)
    wrong_rec = np.real(np.fft.ifft2(wrong_ft))
    assert np.max(np.abs(wrong_rec - x)) > 0.1


def test_unregularized_division_blows_up_on_null_frequency() -> None:
    """A kernel with an exact Fourier zero makes B̂/Ĥ undefined there."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(8, 8))
    # Averaging along rows: Ĥ = 0 for odd horizontal frequencies.
    h = np.array([[0.5, 0.5]], dtype=np.float64)
    b = _circular_blur(x, h)
    recovered = fourier_deconvolution(b, h, reg=0.0)
    assert not np.all(np.isfinite(recovered))


def test_tikhonov_stabilizes_near_zero_coefficients() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(8, 8))
    h = np.array([[0.5, 0.5]], dtype=np.float64)
    b = _circular_blur(x, h)
    recovered = fourier_deconvolution(b, h, reg=1e-6)
    assert np.all(np.isfinite(recovered))


def test_tikhonov_preserves_psf_phase() -> None:
    """Stabilized filter is Ĥ*/(|Ĥ|²+λ); phase of Ĥ is not replaced by a real ε."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 16))
    h = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.0, 0.2, 0.3],
            [0.1, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    h /= h.sum()
    b = _circular_blur(x, h)
    lam = 1e-4

    B_ft = np.fft.fft2(b)
    H_ft = np.fft.fft2(_embed_psf(h, b.shape))
    expected_ft = np.conj(H_ft) / (np.abs(H_ft) ** 2 + lam) * B_ft
    expected = np.real(np.fft.ifft2(expected_ft))

    recovered = fourier_deconvolution(b, h, reg=lam)
    np.testing.assert_allclose(recovered, expected, atol=1e-14)


def test_reg_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fourier_deconvolution(np.ones((4, 4)), np.ones((1, 1)), reg=-1.0)


def test_fill_boundary_api_removed() -> None:
    import inspect

    signature = inspect.signature(fourier_deconvolution)
    assert "boundary" not in signature.parameters
    assert "fillvalue" not in signature.parameters
    assert "epsilon" not in signature.parameters
