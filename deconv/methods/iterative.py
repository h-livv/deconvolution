"""
Matrix-free FFT-accelerated iterative deconvolution with fill boundaries.

Scientific context
------------------
Circular Fourier division diagonalizes periodic convolution, but the physical
finite-domain (zero-fill) operator used by ``direct_deconvolution(...,
boundary='fill')`` is *not* diagonalized by an image-sized DFT. Following the
spirit of Simões et al. (2016) and Almeida et al. (2013) — FFT efficiency
without pretending the observation is periodic — this module applies the
**exact** fill/"same" operator via zero-padded FFTs and inverts it with
matrix-free CGLS:

    minimize_x  ||A x - b||_2^2

where ``A`` never exists as an explicit matrix. Only

    forward(x)  -> A x
    adjoint(y)  -> A^T y

are evaluated, each via one padded FFT multiply.

This does **not** replace ``fourier_deconvolution`` (circular quotient) or
``direct_deconvolution`` (dense ``lstsq``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _pad_psf_linear(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Embed ``psf`` at the origin of an FFT grid (no circular roll).

    Together with a top-left image embed on a grid of size at least
    ``(M+K-1, N+L-1)``, FFT multiplication realises linear convolution and
    matches ``scipy.signal.convolve2d(..., mode='full', boundary='fill')``.
    """
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    padded[:kh, :kw] = np.asarray(psf, dtype=np.float64)
    return padded


@dataclass
class FillConvolutionOperator:
    """
    Matrix-free realisation of

        A x = convolve2d(x, psf, mode='same', boundary='fill', fillvalue=0)

    via zero-padded FFTs. The FFT is only a fast linear-convolution engine;
    it does **not** impose wraparound on the physical ``M×N`` domain.
    """

    psf: np.ndarray
    shape: tuple[int, int]
    n_forward: int = field(default=0, init=False)
    n_adjoint: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.psf = np.asarray(self.psf, dtype=np.float64)
        self.height, self.width = self.shape
        kh, kw = self.psf.shape
        # Full linear-convolution support; circular wrap on this larger grid
        # coincides with linear convolution of the embedded (zero-padded) image.
        self.pad_h = self.height + kh - 1
        self.pad_w = self.width + kw - 1
        # SciPy 'same' crop of 'full' for the usual odd kernels.
        self.origin_h = (kh - 1) // 2
        self.origin_w = (kw - 1) // 2
        self._H_ft = np.fft.fft2(_pad_psf_linear(self.psf, (self.pad_h, self.pad_w)))

    def reset_counters(self) -> None:
        self.n_forward = 0
        self.n_adjoint = 0

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Apply ``A``: embed → FFT×H → iFFT → crop 'same' window."""
        self.n_forward += 1
        image = np.asarray(image, dtype=np.float64)
        embedded = np.zeros((self.pad_h, self.pad_w), dtype=np.float64)
        embedded[: self.height, : self.width] = image
        full = np.real(np.fft.ifft2(np.fft.fft2(embedded) * self._H_ft))
        return full[
            self.origin_h : self.origin_h + self.height,
            self.origin_w : self.origin_w + self.width,
        ]

    def adjoint(self, residual: np.ndarray) -> np.ndarray:
        """
        Apply ``Aᵀ``.

        Adjoint of crop → embed residual into the 'same' window;
        adjoint of circular convolution with ``H`` → multiply by ``conj(H)``;
        adjoint of top-left embed → crop the object domain.
        For a real PSF this is equivalent to fill-convolution with the
        spatially flipped kernel, cropped consistently.
        """
        self.n_adjoint += 1
        residual = np.asarray(residual, dtype=np.float64)
        embedded = np.zeros((self.pad_h, self.pad_w), dtype=np.float64)
        embedded[
            self.origin_h : self.origin_h + self.height,
            self.origin_w : self.origin_w + self.width,
        ] = residual
        full = np.real(np.fft.ifft2(np.fft.fft2(embedded) * np.conj(self._H_ft)))
        return full[: self.height, : self.width]


@dataclass
class CGLSResult:
    """Outcome of matrix-free CGLS for ``min ||A x - b||_2^2``."""

    image: np.ndarray
    iterations: int
    residual_history: list[float]
    normal_residual_history: list[float]
    n_forward: int
    n_adjoint: int
    converged: bool
    tol: float
    maxiter: int


def cgls(
    operator: FillConvolutionOperator,
    observation: np.ndarray,
    *,
    tol: float = 1e-10,
    maxiter: int = 2000,
    x0: np.ndarray | None = None,
) -> CGLSResult:
    """
    Conjugate Gradient Least Squares (CGLS / CGNR).

    Solves ``min_x ||A x - b||_2^2`` using only ``operator.forward`` and
    ``operator.adjoint``. Stopping uses the relative normal residual

        ||Aᵀ (b - A x)||_2 / ||Aᵀ b||_2  <  tol

    which vanishes at critical points of the least-squares objective.
    The data residual ``||A x - b|| / ||b||`` is recorded each iteration.
    """
    b = np.asarray(observation, dtype=np.float64)
    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = np.asarray(x0, dtype=np.float64).copy()
        r = b - operator.forward(x)

    z = operator.adjoint(r)
    p = z.copy()
    gamma = float(np.vdot(z, z).real)
    b_norm = max(float(np.linalg.norm(b)), 1e-30)
    z0_norm = max(float(np.linalg.norm(z)), 1e-30)

    residual_history = [float(np.linalg.norm(r) / b_norm)]
    normal_residual_history = [float(np.linalg.norm(z) / z0_norm)]
    converged = False
    iterations = 0

    for k in range(1, maxiter + 1):
        q = operator.forward(p)
        q_norm_sq = float(np.vdot(q, q).real)
        if q_norm_sq <= 0.0:
            iterations = k - 1
            break

        alpha = gamma / q_norm_sq
        x = x + alpha * p
        r = r - alpha * q
        z = operator.adjoint(r)
        gamma_new = float(np.vdot(z, z).real)

        residual_history.append(float(np.linalg.norm(r) / b_norm))
        normal_residual_history.append(float(np.linalg.norm(z) / z0_norm))
        iterations = k

        if normal_residual_history[-1] < tol:
            converged = True
            break
        if gamma <= 0.0:
            break

        beta = gamma_new / gamma
        p = z + beta * p
        gamma = gamma_new

    return CGLSResult(
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


# Default solver knobs shared by the demo and the scaling benchmark.
DEFAULT_CGLS_TOL = 1e-10
DEFAULT_CGLS_MAXITER = 5000


def iterative_deconvolution(
    blurred: np.ndarray,
    psf: np.ndarray,
    *,
    tol: float = DEFAULT_CGLS_TOL,
    maxiter: int = DEFAULT_CGLS_MAXITER,
) -> np.ndarray:
    """
    Fill-boundary deconvolution via matrix-free CGLS.

    Same call signature family as ``direct_deconvolution`` /
    ``fourier_deconvolution`` for use with ``time_deconvolution``.
    """
    blurred = np.asarray(blurred, dtype=np.float64)
    operator = FillConvolutionOperator(psf, blurred.shape)
    result = cgls(operator, blurred, tol=tol, maxiter=maxiter)
    return result.image


def iterative_deconvolution_with_info(
    blurred: np.ndarray,
    psf: np.ndarray,
    *,
    tol: float = DEFAULT_CGLS_TOL,
    maxiter: int = DEFAULT_CGLS_MAXITER,
) -> CGLSResult:
    """Like ``iterative_deconvolution`` but returns solver diagnostics."""
    blurred = np.asarray(blurred, dtype=np.float64)
    operator = FillConvolutionOperator(psf, blurred.shape)
    return cgls(operator, blurred, tol=tol, maxiter=maxiter)
