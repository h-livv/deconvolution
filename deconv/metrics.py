"""Runtime and reconstruction-quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np


@dataclass
class DeconvolutionResult:
    """Recovered image together with the wall-clock runtime of the solver."""

    image: np.ndarray
    runtime_seconds: float


def time_deconvolution(
    method: Callable[[np.ndarray, np.ndarray], np.ndarray],
    blurred: np.ndarray,
    psf: np.ndarray,
) -> DeconvolutionResult:
    """
    Time a single deconvolution call only.

    The clock covers ``method(blurred, psf)`` and nothing else — not image
    loading, PSF construction, forward blur, post-processing, or metrics.
    """
    start = perf_counter()
    recovered = method(blurred, psf)
    elapsed = perf_counter() - start
    return DeconvolutionResult(image=recovered, runtime_seconds=elapsed)


def mean_squared_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Mean squared error between two images of equal shape."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    return float(np.mean((reference - estimate) ** 2))


def relative_l2_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """||estimate - reference||_2 / ||reference||_2."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return float(np.linalg.norm(estimate))
    return float(np.linalg.norm(estimate - reference) / denom)


def relative_residual(
    operator_apply,
    estimate: np.ndarray,
    observation: np.ndarray,
) -> float:
    """||A estimate - observation||_2 / ||observation||_2."""
    observation = np.asarray(observation, dtype=np.float64)
    residual = operator_apply(estimate) - observation
    denom = float(np.linalg.norm(observation))
    if denom == 0.0:
        return float(np.linalg.norm(residual))
    return float(np.linalg.norm(residual) / denom)


def compute_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    runtime_seconds: float,
) -> dict[str, float]:
    """Bundle MSE, relative L2 error, and runtime into a single dictionary."""
    return {
        "runtime_seconds": float(runtime_seconds),
        "mse": mean_squared_error(reference, estimate),
        "rel_error": relative_l2_error(reference, estimate),
    }
