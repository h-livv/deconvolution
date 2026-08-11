"""
Matrix-free FFT fill deconvolution via plain gradient descent.

Same forward/adjoint operator as ``deconv.methods.iterative`` (zero-padded
FFT realisation of fill/`same` convolution). Instead of CGLS, this module
minimises

    f(x) = (1/2) ||A x - b||_2^2

by steepest descent: compute the gradient ``g = Aᵀ(A x - b)``, then take a
gradient step. The step length uses the exact line search for this quadratic
(still ordinary GD — no conjugate directions).

    g = Aᵀ (A x - b)
    α = ||g||_2^2 / ||A g||_2^2
    x ← x - α g
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deconv.methods.iterative import FillConvolutionOperator

# Match CGLS defaults so the two iterative fill solvers are comparable.
DEFAULT_GD_TOL = 1e-10
DEFAULT_GD_MAXITER = 10000


@dataclass
class GradientDescentResult:
    """Outcome of matrix-free gradient descent for ``min ||A x - b||_2^2``."""

    image: np.ndarray
    iterations: int
    residual_history: list[float]
    normal_residual_history: list[float]
    n_forward: int
    n_adjoint: int
    converged: bool
    tol: float
    maxiter: int


def gradient_descent(
    operator: FillConvolutionOperator,
    observation: np.ndarray,
    *,
    tol: float = DEFAULT_GD_TOL,
    maxiter: int = DEFAULT_GD_MAXITER,
    x0: np.ndarray | None = None,
) -> GradientDescentResult:
    """
    Steepest descent on ``min_x ||A x - b||_2^2`` using only forward/adjoint.

    Stopping accepts either the relative normal residual

        ||Aᵀ (A x - b)||_2 / ||Aᵀ b||_2  <  tol

    or the relative data residual ``||A x - b|| / ||b|| < tol`` (steepest
    descent often stagnates on the normal residual near the solution).
    The data residual is recorded each iteration.
    """
    b = np.asarray(observation, dtype=np.float64)
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()

    operator.reset_counters()
    ax = operator.forward(x)
    # Data residual r = A x - b (matches ∇f = Aᵀ r).
    r = ax - b
    g = operator.adjoint(r)

    b_norm = max(float(np.linalg.norm(b)), 1e-30)
    # Normal residual scale ||Aᵀ b|| (x=0 reference), matching CGLS convention.
    atb = operator.adjoint(b)
    z0_norm = max(float(np.linalg.norm(atb)), 1e-30)

    residual_history = [float(np.linalg.norm(r) / b_norm)]
    normal_residual_history = [float(np.linalg.norm(g) / z0_norm)]
    converged = False
    iterations = 0

    for k in range(1, maxiter + 1):
        if (
            normal_residual_history[-1] < tol
            or residual_history[-1] < tol
        ):
            converged = True
            iterations = k - 1
            break

        ag = operator.forward(g)
        ag_norm_sq = float(np.vdot(ag, ag).real)
        g_norm_sq = float(np.vdot(g, g).real)
        if ag_norm_sq <= 0.0 or g_norm_sq <= 0.0:
            iterations = k - 1
            break

        alpha = g_norm_sq / ag_norm_sq
        x = x - alpha * g
        # Rank-1 update of residual: r ← r - α A g.
        r = r - alpha * ag
        g = operator.adjoint(r)

        residual_history.append(float(np.linalg.norm(r) / b_norm))
        normal_residual_history.append(float(np.linalg.norm(g) / z0_norm))
        iterations = k

        # Accept either normal-residual (CGLS-style) or data-residual stopping.
        # Steepest descent often stagnates on the normal residual near the
        # solution while the data residual is already tiny.
        if (
            normal_residual_history[-1] < tol
            or residual_history[-1] < tol
        ):
            converged = True
            break

    return GradientDescentResult(
        image=x,
        iterations=iterations,
        residual_history=residual_history,
        normal_residual_history=normal_residual_history,
        n_forward=operator.n_forward,
        n_adjoint=operator.n_adjoint,
        converged=converged,
        tol=tol,
        maxiter=maxiter,
    )


def gradient_deconvolution(
    blurred: np.ndarray,
    psf: np.ndarray,
    *,
    tol: float = DEFAULT_GD_TOL,
    maxiter: int = DEFAULT_GD_MAXITER,
) -> np.ndarray:
    """
    Fill-boundary deconvolution via matrix-free gradient descent.

    Same call signature family as ``direct_deconvolution`` /
    ``fourier_deconvolution`` / ``iterative_deconvolution``.
    """
    blurred = np.asarray(blurred, dtype=np.float64)
    operator = FillConvolutionOperator(psf, blurred.shape)
    return gradient_descent(operator, blurred, tol=tol, maxiter=maxiter).image


def gradient_deconvolution_with_info(
    blurred: np.ndarray,
    psf: np.ndarray,
    *,
    tol: float = DEFAULT_GD_TOL,
    maxiter: int = DEFAULT_GD_MAXITER,
) -> GradientDescentResult:
    """Like ``gradient_deconvolution`` but returns solver diagnostics."""
    blurred = np.asarray(blurred, dtype=np.float64)
    operator = FillConvolutionOperator(psf, blurred.shape)
    return gradient_descent(operator, blurred, tol=tol, maxiter=maxiter)
