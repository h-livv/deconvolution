"""Validation tests for matrix-free fill gradient descent."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d

from deconv.data.synthetic import generate_synthetic_image
from deconv.forward.psf import gaussian_psf
from deconv.methods.direct import direct_deconvolution
from deconv.methods.gradient_descent import gradient_deconvolution_with_info
from deconv.methods.iterative import iterative_deconvolution_with_info


def test_gd_matches_direct_on_well_conditioned_fill() -> None:
    """Mild blur: steepest descent must approach dense fill lstsq."""
    psf = gaussian_psf(5, 0.8)
    x = generate_synthetic_image(20, "border_frame")
    b = convolve2d(x, psf, mode="same", boundary="fill")

    x_direct = direct_deconvolution(b, psf, boundary="fill")
    # Steepest descent reaches Direct closely; tol is looser than CGLS defaults
    # because GD residual often plateaus slightly above 1e-10.
    result = gradient_deconvolution_with_info(b, psf, tol=1e-7, maxiter=20000)

    assert result.converged
    rel_vs_direct = np.linalg.norm(result.image - x_direct) / np.linalg.norm(x_direct)
    assert rel_vs_direct < 1e-3
    assert min(result.residual_history) < 1e-5


def test_gd_is_not_circular_quotient() -> None:
    """Fill GD must disagree with wrap Fourier on a fill observation."""
    from deconv.methods.fourier import fourier_deconvolution

    psf = gaussian_psf(7, 1.0)
    x = generate_synthetic_image(20, "edge_square")
    b = convolve2d(x, psf, mode="same", boundary="fill")

    x_gd = gradient_deconvolution_with_info(b, psf, tol=1e-7, maxiter=10000).image
    x_four = fourier_deconvolution(b, psf, reg=0.0)
    assert np.all(np.isfinite(x_gd))
    assert np.linalg.norm(x_gd - x_four) / np.linalg.norm(x_gd) > 1e-2


def test_gd_slower_convergence_than_cgls_on_same_problem() -> None:
    """On a mild fill problem, CGLS should need fewer iterations than GD."""
    psf = gaussian_psf(5, 0.8)
    x = generate_synthetic_image(16, "border_frame")
    b = convolve2d(x, psf, mode="same", boundary="fill")

    cgls = iterative_deconvolution_with_info(b, psf, tol=1e-7, maxiter=5000)
    gd = gradient_deconvolution_with_info(b, psf, tol=1e-7, maxiter=20000)

    assert cgls.converged and gd.converged
    assert cgls.iterations < gd.iterations
