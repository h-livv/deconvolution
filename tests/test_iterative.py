"""Validation tests for matrix-free fill convolution and CGLS."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from deconv.methods.direct import build_convolution_matrix, direct_deconvolution
from deconv.methods.iterative import FillConvolutionOperator, iterative_deconvolution_with_info
from deconv.forward.psf import gaussian_psf
from deconv.data.synthetic import generate_synthetic_image


def test_fill_forward_matches_convolve2d_and_direct_matrix() -> None:
    psf = gaussian_psf(7, 1.2)
    height, width = 18, 16
    op = FillConvolutionOperator(psf, (height, width))
    A = build_convolution_matrix(psf, height, width, boundary="fill")
    rng = np.random.default_rng(0)

    for _ in range(5):
        x = rng.normal(size=(height, width))
        ax_fft = op.forward(x)
        ax_sp = convolve2d(x, psf, mode="same", boundary="fill")
        ax_mat = (A @ x.ravel()).reshape(height, width)
        rel_sp = np.linalg.norm(ax_fft - ax_sp) / np.linalg.norm(ax_sp)
        rel_mat = np.linalg.norm(ax_fft - ax_mat) / np.linalg.norm(ax_mat)
        assert rel_sp < 1e-12
        assert rel_mat < 1e-12


def test_fill_adjoint_identity() -> None:
    psf = gaussian_psf(7, 1.2)
    height, width = 18, 16
    op = FillConvolutionOperator(psf, (height, width))
    A = build_convolution_matrix(psf, height, width, boundary="fill")
    rng = np.random.default_rng(1)

    for _ in range(8):
        x = rng.normal(size=(height, width))
        y = rng.normal(size=(height, width))
        lhs = float(np.vdot(op.forward(x), y).real)
        rhs = float(np.vdot(x, op.adjoint(y)).real)
        denom = max(abs(lhs), abs(rhs), 1.0)
        assert abs(lhs - rhs) / denom < 1e-12

        at_y = (A.T @ y.ravel()).reshape(height, width)
        rel = np.linalg.norm(op.adjoint(y) - at_y) / np.linalg.norm(at_y)
        assert rel < 1e-12


def test_cgls_matches_direct_on_well_conditioned_fill() -> None:
    """Mild blur: CGLS must reproduce dense fill lstsq."""
    psf = gaussian_psf(5, 0.8)
    x = generate_synthetic_image(24, "border_frame")
    b = convolve2d(x, psf, mode="same", boundary="fill")

    x_direct = direct_deconvolution(b, psf, boundary="fill")
    result = iterative_deconvolution_with_info(b, psf, tol=1e-10, maxiter=2000)

    assert result.converged
    rel_vs_direct = np.linalg.norm(result.image - x_direct) / np.linalg.norm(x_direct)
    rel_vs_truth = np.linalg.norm(result.image - x) / np.linalg.norm(x)
    assert rel_vs_direct < 1e-8
    assert rel_vs_truth < 1e-8
    assert result.residual_history[-1] < 1e-8


def test_iterative_is_not_circular_quotient() -> None:
    """Fill CGLS must disagree with wrap Fourier on a fill observation."""
    psf = gaussian_psf(7, 1.0)
    x = generate_synthetic_image(20, "edge_square")
    b = convolve2d(x, psf, mode="same", boundary="fill")

    from deconv.methods.fourier import fourier_deconvolution

    x_iter = iterative_deconvolution_with_info(b, psf, tol=1e-10, maxiter=2000).image
    x_four = fourier_deconvolution(b, psf, reg=0.0)
    # Both finite, but they solve different operators.
    assert np.all(np.isfinite(x_iter))
    assert np.linalg.norm(x_iter - x_four) / np.linalg.norm(x_iter) > 1e-2
