"""
Image deconvolution demonstration.

Edit the configuration block, then run:

    python scripts/main.py

Compares Direct (dense matrix), Fourier quotient (circular), matrix-free
iterative FFT CGLS, and fill gradient descent under configurable boundaries.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deconv.forward.blur import add_gaussian_noise, apply_blur
from deconv.methods.direct import direct_deconvolution
from deconv.methods.fourier import fourier_deconvolution
from deconv.methods.gradient_descent import (
    DEFAULT_GD_MAXITER,
    DEFAULT_GD_TOL,
    gradient_deconvolution_with_info,
)
from deconv.methods.iterative import (
    DEFAULT_CGLS_MAXITER,
    DEFAULT_CGLS_TOL,
    iterative_deconvolution_with_info,
)
from deconv.metrics import compute_metrics, time_deconvolution
from deconv.paths import IMAGES_DIR
from deconv.forward.psf import gaussian_psf
from deconv.data.synthetic import generate_synthetic_image
from deconv.io.utils import (
    clip_to_unit_interval,
    create_results_folder,
    load_image,
    resize_image,
    save_image,
    show_image,
)

# ---------------------------------------------------------------------------
# Configuration — edit these values only
# ---------------------------------------------------------------------------

# Image
IMAGE_SOURCE = "file"
# "file" | "synthetic"
IMAGE_PATH = str(IMAGES_DIR / "ring.png")
SYNTHETIC_PATTERN = "edge_square"
# "corner_pixel" | "edge_square" | "border_frame" | "diagonal"

# Methods to run — any non-empty subset, in any order.
METHODS: tuple[str, ...] = ("direct", "fourier", "gradient")
# Allowed names: "direct", "fourier", "iterative", "gradient"
# Examples:
#   METHODS = ("direct", "fourier")
#   METHODS = ("iterative", "gradient")
#   METHODS = ("gradient",)

# Image settings — matched to analyze_scaling defaults
IMAGE_SIZE = 64

# Gaussian PSF — matched to analyze_scaling (SIGMA=1.0, KERNEL_SIZE=7)
SIGMA = 1.0
KERNEL_SIZE = 7

# Observation
NOISE_STD = 0.0

# Reconstruction (set equal to SIGMA for a matched model)
RECONSTRUCTION_SIGMA = SIGMA

# Boundary model for blur / direct (Fourier is always circular).
# Matched to analyze_scaling: fill observation + Direct fill; Fourier circular.
BOUNDARY_MODE = "mismatch"
# "equivalent" — wrap / wrap   (Direct ≈ Fourier under circular convolution)
# "mismatch"   — fill / fill   (Direct, Iterative, Gradient solve fill; Fourier assumes wrap)

# CGLS (iterative) — same defaults as analyze_scaling
CGLS_TOL = DEFAULT_CGLS_TOL
CGLS_MAXITER = DEFAULT_CGLS_MAXITER

# Gradient descent (same fill operator; shared tol/maxiter defaults)
GD_TOL = DEFAULT_GD_TOL
GD_MAXITER = DEFAULT_GD_MAXITER

# ---------------------------------------------------------------------------

BOUNDARY_MODE_MAP: dict[str, tuple[str, str]] = {
    "equivalent": ("wrap", "wrap"),
    "mismatch": ("fill", "fill"),
}

BOUNDARY_MODE_LABELS: dict[str, str] = {
    "equivalent": "Equivalent",
    "mismatch": "Mismatch",
}


ALLOWED_METHODS: tuple[str, ...] = ("direct", "fourier", "iterative", "gradient")


def resolve_methods(methods: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """
    Normalise a method selection to a stable ordered tuple.

    Accepts a sequence of method names, or a single comma-separated string.
    Legacy aliases ``"all"`` and ``"both"`` are still recognised.
    """
    if isinstance(methods, str):
        raw = tuple(p.strip().lower() for p in methods.split(",") if p.strip())
    else:
        raw = tuple(str(m).strip().lower() for m in methods if str(m).strip())

    if not raw:
        raise ValueError("METHODS must list at least one method.")

    # Legacy single-token aliases.
    if raw == ("all",):
        return ALLOWED_METHODS
    if raw == ("both",):
        return ("direct", "fourier")

    unknown = set(raw) - set(ALLOWED_METHODS)
    if unknown:
        raise ValueError(
            f"Unknown method(s) {sorted(unknown)}; "
            f"choose from {list(ALLOWED_METHODS)} "
            "(or legacy aliases 'all' / 'both')."
        )
    # Preserve canonical order, drop duplicates.
    return tuple(m for m in ALLOWED_METHODS if m in raw)


def resolve_boundaries(mode: str) -> tuple[str, str]:
    """Map BOUNDARY_MODE to (blur, direct) boundary settings."""
    key = mode.lower().strip()
    if key not in BOUNDARY_MODE_MAP:
        raise ValueError(
            f"Unknown BOUNDARY_MODE '{mode}'. "
            "Use 'equivalent' or 'mismatch'."
        )
    return BOUNDARY_MODE_MAP[key]


def load_working_image() -> tuple[np.ndarray, str]:
    """Load or synthesize the working image."""
    if IMAGE_SOURCE == "synthetic":
        return (
            generate_synthetic_image(IMAGE_SIZE, SYNTHETIC_PATTERN),
            f"synthetic:{SYNTHETIC_PATTERN}",
        )
    if IMAGE_SOURCE == "file":
        path = Path(IMAGE_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return resize_image(load_image(path), IMAGE_SIZE), path.name
    raise ValueError("IMAGE_SOURCE must be 'file' or 'synthetic'.")


def save_comparison(
    original: np.ndarray,
    observation: np.ndarray,
    recovered_direct: np.ndarray | None,
    recovered_fourier: np.ndarray | None,
    path: Path,
    recovered_iterative: np.ndarray | None = None,
    recovered_gradient: np.ndarray | None = None,
) -> None:
    """Save Original | Observation | Direct | Fourier | Iterative | Gradient."""
    panels: list[tuple[np.ndarray, str]] = [
        (original, "Original"),
        (observation, "Observation"),
    ]
    if recovered_direct is not None:
        panels.append((recovered_direct, "Direct"))
    if recovered_fourier is not None:
        panels.append((recovered_fourier, "Fourier"))
    if recovered_iterative is not None:
        panels.append((recovered_iterative, "Iterative CGLS"))
    if recovered_gradient is not None:
        panels.append((recovered_gradient, "Gradient descent"))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]
    for axis, (image, title) in zip(axes, panels):
        show_image(axis, image, title)
    fig.suptitle("Forward model and reconstructions", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_error_maps(
    original: np.ndarray,
    recovered_direct: np.ndarray,
    recovered_fourier: np.ndarray,
    path: Path,
    recovered_iterative: np.ndarray | None = None,
    recovered_gradient: np.ndarray | None = None,
) -> None:
    """Absolute-error panels with one shared color scale."""
    panels_data: list[tuple[np.ndarray, str]] = [
        (np.abs(original - recovered_direct), "Absolute Error (Direct)"),
        (np.abs(original - recovered_fourier), "Absolute Error (Fourier)"),
        (np.abs(recovered_direct - recovered_fourier), "|Direct − Fourier|"),
    ]
    if recovered_iterative is not None:
        panels_data.append(
            (np.abs(original - recovered_iterative), "Absolute Error (Iterative)")
        )
        panels_data.append(
            (
                np.abs(recovered_direct - recovered_iterative),
                "|Direct − Iterative|",
            )
        )
    if recovered_gradient is not None:
        panels_data.append(
            (np.abs(original - recovered_gradient), "Absolute Error (Gradient)")
        )
        panels_data.append(
            (
                np.abs(recovered_direct - recovered_gradient),
                "|Direct − Gradient|",
            )
        )

    vmax = float(max(max(img.max() for img, _ in panels_data), 1e-16))
    n = len(panels_data)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.4))
    if n == 1:
        axes = [axes]
    images = []
    for axis, (image, title) in zip(axes, panels_data):
        im = axis.imshow(image, cmap="magma", vmin=0.0, vmax=vmax)
        images.append(im)
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(images[-1], cax=cax)

    fig.suptitle("Error maps (shared color scale)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_figure(
    metrics_direct: dict[str, float] | None,
    metrics_fourier: dict[str, float] | None,
    path: Path,
    metrics_iterative: dict[str, float] | None = None,
    metrics_gradient: dict[str, float] | None = None,
) -> None:
    """Runtime (log) and MSE bars for available methods."""
    labels: list[str] = []
    runtime: list[float] = []
    mse: list[float] = []
    colors_cycle = ("#3b6d9c", "#c46b3a", "#2f6f4e", "#8b5a2b")
    if metrics_direct is not None:
        labels.append("Direct")
        runtime.append(metrics_direct["runtime_seconds"])
        mse.append(metrics_direct["mse"])
    if metrics_fourier is not None:
        labels.append("Fourier")
        runtime.append(metrics_fourier["runtime_seconds"])
        mse.append(metrics_fourier["mse"])
    if metrics_iterative is not None:
        labels.append("Iterative")
        runtime.append(metrics_iterative["runtime_seconds"])
        mse.append(metrics_iterative["mse"])
    if metrics_gradient is not None:
        labels.append("Gradient")
        runtime.append(metrics_gradient["runtime_seconds"])
        mse.append(metrics_gradient["mse"])
    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = colors_cycle[: len(labels)]

    runtime_plot = [max(t, 1e-12) for t in runtime]
    bars = axes[0].bar(labels, runtime_plot, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Seconds (log scale)")
    axes[0].set_title("Runtime")
    for bar, value in zip(bars, runtime):
        axes[0].annotate(
            f"{value:.4g} s",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    mse_plot = [max(v, 1e-30) for v in mse]
    bars_mse = axes[1].bar(labels, mse_plot, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("MSE (log scale)")
    axes[1].set_title("Mean Squared Error")
    for bar, value in zip(bars_mse, mse):
        axes[1].annotate(
            f"{value:.3e}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for axis in axes:
        axis.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Runtime vs reconstruction quality", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_iterative_convergence(residual_history: list[float], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.0, 3.6))
    axis.semilogy(np.arange(len(residual_history)), residual_history, color="#2f6f4e")
    axis.set_xlabel("CGLS iteration")
    axis.set_ylabel(r"$\|Ax_k - b\|_2 / \|b\|_2$")
    axis.set_title("Iterative fill CGLS residual")
    axis.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gradient_convergence(residual_history: list[float], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.0, 3.6))
    axis.semilogy(np.arange(len(residual_history)), residual_history, color="#8b5a2b")
    axis.set_xlabel("Gradient-descent iteration")
    axis.set_ylabel(r"$\|Ax_k - b\|_2 / \|b\|_2$")
    axis.set_title("Fill gradient-descent residual")
    axis.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_metrics_file(
    path: Path,
    *,
    method: str,
    boundary_mode: str,
    sigma: float,
    kernel_size: int,
    noise_std: float,
    metrics_direct: dict[str, float] | None,
    metrics_fourier: dict[str, float] | None,
    metrics_iterative: dict[str, float] | None = None,
    iterative_iterations: int | None = None,
    metrics_gradient: dict[str, float] | None = None,
    gradient_iterations: int | None = None,
) -> None:
    """Write a concise metrics.txt for the presentation."""
    lines = [
        "Method(s):",
        method,
        "",
        "Boundary mode:",
        BOUNDARY_MODE_LABELS[boundary_mode],
        "",
        "Boundaries:",
        "  Direct: same as blur (wrap or fill)",
        "  Fourier: circular",
        "  Iterative: fill (matrix-free FFT CGLS)",
        "  Gradient: fill (matrix-free FFT gradient descent)",
        "",
        "Sigma:",
        str(sigma),
        "",
        "Kernel size:",
        str(kernel_size),
        "",
        "Noise:",
        str(noise_std),
        "",
    ]
    if metrics_direct is not None:
        lines.extend(
            [
                "Direct runtime:",
                f"{metrics_direct['runtime_seconds']:.6f} s",
                "",
                "Direct MSE:",
                f"{metrics_direct['mse']:.6e}",
                "",
                "Direct rel_error:",
                f"{metrics_direct.get('rel_error', float('nan')):.6e}",
                "",
            ]
        )
    if metrics_fourier is not None:
        lines.extend(
            [
                "Fourier runtime:",
                f"{metrics_fourier['runtime_seconds']:.6f} s",
                "",
                "Fourier MSE:",
                f"{metrics_fourier['mse']:.6e}",
                "",
                "Fourier rel_error:",
                f"{metrics_fourier.get('rel_error', float('nan')):.6e}",
                "",
            ]
        )
    if metrics_iterative is not None:
        lines.extend(
            [
                "Iterative runtime:",
                f"{metrics_iterative['runtime_seconds']:.6f} s",
                "",
                "Iterative MSE:",
                f"{metrics_iterative['mse']:.6e}",
                "",
                "Iterative rel_error:",
                f"{metrics_iterative.get('rel_error', float('nan')):.6e}",
                "",
            ]
        )
        if iterative_iterations is not None:
            lines.extend(["Iterative iterations:", str(iterative_iterations), ""])
    if metrics_gradient is not None:
        lines.extend(
            [
                "Gradient runtime:",
                f"{metrics_gradient['runtime_seconds']:.6f} s",
                "",
                "Gradient MSE:",
                f"{metrics_gradient['mse']:.6e}",
                "",
                "Gradient rel_error:",
                f"{metrics_gradient.get('rel_error', float('nan')):.6e}",
                "",
            ]
        )
        if gradient_iterations is not None:
            lines.extend(["Gradient iterations:", str(gradient_iterations), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(
    *,
    boundary_mode: str,
    sigma: float,
    noise_std: float,
    metrics_direct: dict[str, float] | None,
    metrics_fourier: dict[str, float] | None,
    output_dir: Path,
    metrics_iterative: dict[str, float] | None = None,
    iterative_iterations: int | None = None,
    metrics_gradient: dict[str, float] | None = None,
    gradient_iterations: int | None = None,
) -> None:
    """Print a concise console summary."""
    print("================================")
    print()
    print("Boundary model:")
    print(BOUNDARY_MODE_LABELS[boundary_mode])
    print()
    print("Sigma:")
    print(sigma)
    print()
    print("Noise:")
    print(noise_std)
    print()
    if metrics_direct is not None:
        print("Direct")
        print()
        print("Runtime")
        print(f"{metrics_direct['runtime_seconds']:.6f} s")
        print()
        print("MSE")
        print(f"{metrics_direct['mse']:.6e}")
        print()
    if metrics_fourier is not None:
        print("Fourier")
        print()
        print("Runtime")
        print(f"{metrics_fourier['runtime_seconds']:.6f} s")
        print()
        print("MSE")
        print(f"{metrics_fourier['mse']:.6e}")
        print()
    if metrics_iterative is not None:
        print("Iterative FFT (fill CGLS)")
        print()
        print("Runtime")
        print(f"{metrics_iterative['runtime_seconds']:.6f} s")
        print()
        print("MSE")
        print(f"{metrics_iterative['mse']:.6e}")
        print()
        if iterative_iterations is not None:
            print("Iterations")
            print(iterative_iterations)
            print()
    if metrics_gradient is not None:
        print("Gradient descent (fill)")
        print()
        print("Runtime")
        print(f"{metrics_gradient['runtime_seconds']:.6f} s")
        print()
        print("MSE")
        print(f"{metrics_gradient['mse']:.6e}")
        print()
        if gradient_iterations is not None:
            print("Iterations")
            print(gradient_iterations)
            print()
    print("Results folder")
    print()
    print(f"{output_dir}/")
    print()
    print("================================")


def run() -> Path:
    """Run one presentation experiment and write figures + metrics."""
    if RECONSTRUCTION_SIGMA <= 0:
        raise ValueError("RECONSTRUCTION_SIGMA must be positive")

    selected = resolve_methods(METHODS)
    want_direct = "direct" in selected
    want_fourier = "fourier" in selected
    want_iterative = "iterative" in selected
    want_gradient = "gradient" in selected
    method_label = ",".join(selected)

    boundary_mode = BOUNDARY_MODE.lower().strip()
    blur_boundary, direct_boundary = resolve_boundaries(boundary_mode)

    original, _image_name = load_working_image()
    blur_psf = gaussian_psf(KERNEL_SIZE, SIGMA)
    t_blur0 = perf_counter()
    blurred = apply_blur(original, blur_psf, boundary=blur_boundary, fillvalue=0.0)
    blur_s = perf_counter() - t_blur0
    observation = add_gaussian_noise(blurred, NOISE_STD, seed=0)
    reconstruction_psf = gaussian_psf(KERNEL_SIZE, RECONSTRUCTION_SIGMA)

    recovered_direct = None
    recovered_fourier = None
    recovered_iterative = None
    recovered_gradient = None
    metrics_direct = None
    metrics_fourier = None
    metrics_iterative = None
    metrics_gradient = None
    iterative_iterations = None
    gradient_iterations = None
    iterative_residual_history: list[float] | None = None
    gradient_residual_history: list[float] | None = None

    if want_direct:
        direct_fn = partial(
            direct_deconvolution,
            boundary=direct_boundary,
            fillvalue=0.0,
        )
        result = time_deconvolution(direct_fn, observation, reconstruction_psf)
        recovered_direct = result.image
        metrics_direct = compute_metrics(
            original, recovered_direct, result.runtime_seconds
        )

    if want_fourier:
        # Circular Fourier division only; matches blur when BOUNDARY_MODE is
        # "equivalent" (wrap). Optional Tikhonov λ via reg=...
        fourier_fn = partial(fourier_deconvolution, reg=0.0)
        result = time_deconvolution(fourier_fn, observation, reconstruction_psf)
        recovered_fourier = result.image
        metrics_fourier = compute_metrics(
            original, recovered_fourier, result.runtime_seconds
        )

    if want_iterative:
        # Fill operator + CGLS. Runtime includes the forward convolution that
        # formed ``b`` plus the solve (Direct / Fourier stay deconvolution-only).
        t0 = perf_counter()
        info = iterative_deconvolution_with_info(
            observation,
            reconstruction_psf,
            tol=CGLS_TOL,
            maxiter=CGLS_MAXITER,
        )
        elapsed = perf_counter() - t0 + blur_s
        recovered_iterative = info.image
        metrics_iterative = compute_metrics(original, recovered_iterative, elapsed)
        iterative_iterations = info.iterations
        iterative_residual_history = info.residual_history

    if want_gradient:
        # Same fill operator + steepest descent (no CGLS).
        t0 = perf_counter()
        info = gradient_deconvolution_with_info(
            observation,
            reconstruction_psf,
            tol=GD_TOL,
            maxiter=GD_MAXITER,
        )
        elapsed = perf_counter() - t0 + blur_s
        recovered_gradient = info.image
        metrics_gradient = compute_metrics(original, recovered_gradient, elapsed)
        gradient_iterations = info.iterations
        gradient_residual_history = info.residual_history

    observation_display = clip_to_unit_interval(observation)
    direct_display = (
        clip_to_unit_interval(recovered_direct)
        if recovered_direct is not None
        else None
    )
    fourier_display = (
        clip_to_unit_interval(recovered_fourier)
        if recovered_fourier is not None
        else None
    )
    iterative_display = (
        clip_to_unit_interval(recovered_iterative)
        if recovered_iterative is not None
        else None
    )
    gradient_display = (
        clip_to_unit_interval(recovered_gradient)
        if recovered_gradient is not None
        else None
    )

    output_dir = create_results_folder("results")
    save_image(original, output_dir / "original.png")
    save_image(observation_display, output_dir / "observation.png")
    if direct_display is not None:
        save_image(direct_display, output_dir / "recovered_direct.png")
    if fourier_display is not None:
        save_image(fourier_display, output_dir / "recovered_fourier.png")
    if iterative_display is not None:
        save_image(iterative_display, output_dir / "recovered_iterative.png")
    if gradient_display is not None:
        save_image(gradient_display, output_dir / "recovered_gradient.png")

    save_comparison(
        original,
        observation_display,
        direct_display,
        fourier_display,
        output_dir / "comparison.png",
        recovered_iterative=iterative_display,
        recovered_gradient=gradient_display,
    )
    if recovered_direct is not None and recovered_fourier is not None:
        save_error_maps(
            original,
            recovered_direct,
            recovered_fourier,
            output_dir / "error_maps.png",
            recovered_iterative=recovered_iterative,
            recovered_gradient=recovered_gradient,
        )
    save_metrics_figure(
        metrics_direct,
        metrics_fourier,
        output_dir / "metrics.png",
        metrics_iterative=metrics_iterative,
        metrics_gradient=metrics_gradient,
    )
    if iterative_residual_history is not None:
        save_iterative_convergence(
            iterative_residual_history,
            output_dir / "iterative_convergence.png",
        )
    if gradient_residual_history is not None:
        save_gradient_convergence(
            gradient_residual_history,
            output_dir / "gradient_convergence.png",
        )

    write_metrics_file(
        output_dir / "metrics.txt",
        method=method_label,
        boundary_mode=boundary_mode,
        sigma=SIGMA,
        kernel_size=KERNEL_SIZE,
        noise_std=NOISE_STD,
        metrics_direct=metrics_direct,
        metrics_fourier=metrics_fourier,
        metrics_iterative=metrics_iterative,
        iterative_iterations=iterative_iterations,
        metrics_gradient=metrics_gradient,
        gradient_iterations=gradient_iterations,
    )
    print_summary(
        boundary_mode=boundary_mode,
        sigma=SIGMA,
        noise_std=NOISE_STD,
        metrics_direct=metrics_direct,
        metrics_fourier=metrics_fourier,
        output_dir=output_dir,
        metrics_iterative=metrics_iterative,
        iterative_iterations=iterative_iterations,
        metrics_gradient=metrics_gradient,
        gradient_iterations=gradient_iterations,
    )
    return output_dir



if __name__ == "__main__":
    run()
