"""
Multi-method deconvolution benchmark (fill observation).

Compares, on the *same* physical fill-boundary test problem:

1. Direct — dense ``A``, ``lstsq`` (boundary=fill)
2. Fourier quotient — circular division (intentionally different model)
3. Iterative FFT — matrix-free CGLS with the exact fill operator
4. Gradient descent — same fill operator, steepest descent (no CGLS)

Runtimes are deconvolution-only (via ``time_deconvolution`` / equivalent
clocks around the solver), except the matrix-free iterative solvers which
also include the forward fill convolution that formed ``b``. Setup
(image, PSF) is outside the clock.
"""

from __future__ import annotations

import json
import tracemalloc
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from deconv.data.synthetic import generate_synthetic_image
from deconv.forward.blur import apply_blur
from deconv.forward.psf import gaussian_psf
from deconv.io.utils import clip_to_unit_interval, create_results_folder, save_image, show_image
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
    FillConvolutionOperator,
    iterative_deconvolution_with_info,
)
from deconv.metrics import mean_squared_error, relative_l2_error, time_deconvolution

# Default size grid for the fill-method benchmark entry point.
DEFAULT_SIZES = (16, 24, 32, 40)

ALL_METHODS: tuple[str, ...] = ("direct", "fourier", "iterative", "gradient")


@dataclass
class MethodResult:
    name: str
    boundary: str
    runtime_s: float
    setup_s: float
    rel_error: float
    mse: float
    rel_residual: float
    iterations: int | None
    n_forward: int | None
    n_adjoint: int | None
    peak_memory_mib: float | None
    residual_history: list[float] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class SizeCase:
    image_size: int
    n_pixels: int
    n_matrix_entries: int
    sigma: float
    kernel_size: int
    methods: list[MethodResult]
    validation: dict[str, float]


def _peak_memory_mib(fn) -> tuple[object, float]:
    """Run ``fn`` under tracemalloc; return (result, peak MiB)."""
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak / (1024.0 * 1024.0)


def _relative_residual_fill(
    psf: np.ndarray, estimate: np.ndarray, observation: np.ndarray
) -> float:
    op = FillConvolutionOperator(psf, observation.shape)
    return float(
        np.linalg.norm(op.forward(estimate) - observation)
        / max(np.linalg.norm(observation), 1e-30)
    )


def validate_operators(psf: np.ndarray, size: int, trials: int = 4) -> dict[str, float]:
    """Forward agreement with Direct matrix + adjoint identity."""
    from deconv.methods.direct import build_convolution_matrix

    op = FillConvolutionOperator(psf, (size, size))
    A = build_convolution_matrix(psf, size, size, boundary="fill")
    rng = np.random.default_rng(0)
    fwd_errs: list[float] = []
    adj_errs: list[float] = []
    for _ in range(trials):
        x = rng.normal(size=(size, size))
        y = rng.normal(size=(size, size))
        ax = op.forward(x)
        ax_mat = (A @ x.ravel()).reshape(size, size)
        fwd_errs.append(float(np.linalg.norm(ax - ax_mat) / np.linalg.norm(ax_mat)))
        lhs = float(np.vdot(op.forward(x), y).real)
        rhs = float(np.vdot(x, op.adjoint(y)).real)
        adj_errs.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0))
    return {
        "forward_rel_error_max": max(fwd_errs),
        "forward_rel_error_mean": float(np.mean(fwd_errs)),
        "adjoint_rel_error_max": max(adj_errs),
        "adjoint_rel_error_mean": float(np.mean(adj_errs)),
    }


def run_size_case(
    size: int,
    *,
    sigma: float = 1.0,
    kernel_size: int = 7,
    cgls_tol: float = DEFAULT_CGLS_TOL,
    cgls_maxiter: int = DEFAULT_CGLS_MAXITER,
    methods: tuple[str, ...] | list[str] | None = None,
    direct_boundary: str = "fill",
) -> SizeCase:
    """
    Run selected reconstruction methods on a shared **fill** observation.

    Parameters
    ----------
    methods:
        Subset of ``{"direct", "fourier", "iterative", "gradient"}``.
        Default: all four.
    direct_boundary:
        Boundary used by Direct reconstruction (``"fill"`` or ``"wrap"``).
        The forward observation is always ``fill``. Fourier is always circular;
        iterative and gradient always use the matrix-free fill operator.
    """
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(3, min(kernel_size, size if size % 2 == 1 else size - 1))

    if direct_boundary not in {"fill", "wrap"}:
        raise ValueError("direct_boundary must be 'fill' or 'wrap'")

    selected = tuple(methods) if methods is not None else ALL_METHODS
    unknown = set(selected) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if not selected:
        raise ValueError("At least one method must be selected")

    original = generate_synthetic_image(size, "border_frame")
    psf = gaussian_psf(kernel_size, sigma)
    # Physical fill observation — always, regardless of reconstruction boundary.
    t_blur0 = perf_counter()
    observation = apply_blur(original, psf, boundary="fill", fillvalue=0.0)
    blur_s = perf_counter() - t_blur0

    validation = validate_operators(psf, size)
    method_results: list[MethodResult] = []
    n_pixels = size * size

    want_direct = "direct" in selected
    want_fourier = "fourier" in selected
    want_iterative = "iterative" in selected
    want_gradient = "gradient" in selected

    recovered_direct = None
    recovered_fourier = None
    recovered_iterative = None
    recovered_gradient = None
    cgls_info = None
    gd_info = None

    # --- Direct ---
    if want_direct:
        direct_fn = partial(
            direct_deconvolution,
            boundary=direct_boundary,
            fillvalue=0.0,
        )

        def _direct_call():
            return time_deconvolution(direct_fn, observation, psf)

        timed, peak = _peak_memory_mib(_direct_call)
        x_d = timed.image
        method_results.append(
            MethodResult(
                name="direct",
                boundary=direct_boundary,
                runtime_s=timed.runtime_seconds,
                setup_s=0.0,
                rel_error=relative_l2_error(original, x_d),
                mse=mean_squared_error(original, x_d),
                rel_residual=_relative_residual_fill(psf, x_d, observation),
                iterations=None,
                n_forward=None,
                n_adjoint=None,
                peak_memory_mib=peak,
            )
        )
        recovered_direct = x_d

    # --- Fourier quotient (circular; model mismatch for fill data) ---
    if want_fourier:
        fourier_fn = partial(fourier_deconvolution, reg=0.0)

        def _fourier_call():
            return time_deconvolution(fourier_fn, observation, psf)

        timed, peak = _peak_memory_mib(_fourier_call)
        x_f = timed.image
        method_results.append(
            MethodResult(
                name="fourier",
                boundary="circular",
                runtime_s=timed.runtime_seconds,
                setup_s=0.0,
                rel_error=relative_l2_error(original, x_f),
                mse=mean_squared_error(original, x_f),
                rel_residual=_relative_residual_fill(psf, x_f, observation),
                iterations=None,
                n_forward=None,
                n_adjoint=None,
                peak_memory_mib=peak,
            )
        )
        recovered_fourier = x_f

    # --- Iterative FFT fill CGLS ---
    # Runtime includes the forward fill convolution that formed ``b`` plus the
    # CGLS solve (Direct / Fourier remain deconvolution-only).
    if want_iterative:
        t_setup0 = perf_counter()
        _ = FillConvolutionOperator(psf, observation.shape)
        setup_s = perf_counter() - t_setup0

        def _iter_call():
            t0 = perf_counter()
            info = iterative_deconvolution_with_info(
                observation, psf, tol=cgls_tol, maxiter=cgls_maxiter
            )
            elapsed = perf_counter() - t0
            return info, elapsed

        (info, elapsed), peak = _peak_memory_mib(_iter_call)
        method_results.append(
            MethodResult(
                name="iterative",
                boundary="fill",
                runtime_s=elapsed + blur_s,
                setup_s=setup_s,
                rel_error=relative_l2_error(original, info.image),
                mse=mean_squared_error(original, info.image),
                rel_residual=info.residual_history[-1],
                iterations=info.iterations,
                n_forward=info.n_forward,
                n_adjoint=info.n_adjoint,
                peak_memory_mib=peak,
                residual_history=list(info.residual_history),
            )
        )
        recovered_iterative = info.image
        cgls_info = info

    # --- Gradient descent on the same fill FFT operator ---
    # Same timing convention as iterative (forward blur + solve).
    if want_gradient:
        t_setup0 = perf_counter()
        _ = FillConvolutionOperator(psf, observation.shape)
        setup_s = perf_counter() - t_setup0

        def _gd_call():
            t0 = perf_counter()
            info = gradient_deconvolution_with_info(
                observation, psf, tol=cgls_tol, maxiter=cgls_maxiter
            )
            elapsed = perf_counter() - t0
            return info, elapsed

        (info, elapsed), peak = _peak_memory_mib(_gd_call)
        method_results.append(
            MethodResult(
                name="gradient",
                boundary="fill",
                runtime_s=elapsed + blur_s,
                setup_s=setup_s,
                rel_error=relative_l2_error(original, info.image),
                mse=mean_squared_error(original, info.image),
                rel_residual=info.residual_history[-1],
                iterations=info.iterations,
                n_forward=info.n_forward,
                n_adjoint=info.n_adjoint,
                peak_memory_mib=peak,
                residual_history=list(info.residual_history),
            )
        )
        recovered_gradient = info.image
        gd_info = info

    case = SizeCase(
        image_size=size,
        n_pixels=n_pixels,
        n_matrix_entries=n_pixels * n_pixels,
        sigma=sigma,
        kernel_size=kernel_size,
        methods=method_results,
        validation=validation,
    )
    case._artifacts = {  # type: ignore[attr-defined]
        "original": original,
        "observation": observation,
        "psf": psf,
        "recovered_direct": recovered_direct,
        "recovered_fourier": recovered_fourier,
        "recovered_iterative": recovered_iterative,
        "recovered_gradient": recovered_gradient,
        "cgls": cgls_info,
        "gradient": gd_info,
        "direct_boundary": direct_boundary,
        "methods": selected,
    }
    return case


def _save_case_images(case: SizeCase, output_dir: Path) -> None:
    arts = case._artifacts  # type: ignore[attr-defined]
    sub = output_dir / f"N{case.image_size}"
    sub.mkdir(parents=True, exist_ok=True)
    save_image(arts["original"], sub / "ground_truth.png")
    save_image(clip_to_unit_interval(arts["observation"]), sub / "observation.png")
    if arts.get("recovered_direct") is not None:
        save_image(
            clip_to_unit_interval(arts["recovered_direct"]),
            sub / "recovered_direct.png",
        )
    if arts.get("recovered_fourier") is not None:
        save_image(
            clip_to_unit_interval(arts["recovered_fourier"]),
            sub / "recovered_fourier.png",
        )
    if arts.get("recovered_iterative") is not None:
        save_image(
            clip_to_unit_interval(arts["recovered_iterative"]),
            sub / "recovered_iterative.png",
        )
    if arts.get("recovered_gradient") is not None:
        save_image(
            clip_to_unit_interval(arts["recovered_gradient"]),
            sub / "recovered_gradient.png",
        )

    direct_boundary = arts.get("direct_boundary", "fill")
    panels = [
        (arts["original"], "Ground truth"),
        (clip_to_unit_interval(arts["observation"]), "Observation (fill)"),
    ]
    if arts.get("recovered_direct") is not None:
        panels.append(
            (
                clip_to_unit_interval(arts["recovered_direct"]),
                f"Direct ({direct_boundary})",
            )
        )
    if arts.get("recovered_fourier") is not None:
        panels.append(
            (clip_to_unit_interval(arts["recovered_fourier"]), "Fourier (circular)")
        )
    if arts.get("recovered_iterative") is not None:
        panels.append(
            (
                clip_to_unit_interval(arts["recovered_iterative"]),
                "Iterative CGLS (fill)",
            )
        )
    if arts.get("recovered_gradient") is not None:
        panels.append(
            (
                clip_to_unit_interval(arts["recovered_gradient"]),
                "Gradient descent (fill)",
            )
        )

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2))
    if len(panels) == 1:
        axes = [axes]
    for axis, (image, title) in zip(axes, panels):
        show_image(axis, image, title, fontsize=9)
    fig.suptitle(f"N={case.image_size} reconstructions (fill observation)")
    fig.tight_layout()
    fig.savefig(sub / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cgls = arts.get("cgls")
    if cgls is not None:
        hist = cgls.residual_history
        fig, axis = plt.subplots(figsize=(6.0, 3.6))
        axis.semilogy(np.arange(len(hist)), hist, color="#2f6f4e")
        axis.set_xlabel("CGLS iteration")
        axis.set_ylabel(r"$\|Ax_k - b\|_2 / \|b\|_2$")
        axis.set_title(f"Iterative fill CGLS residual (N={case.image_size})")
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(sub / "iterative_convergence.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    gd = arts.get("gradient")
    if gd is not None:
        hist = gd.residual_history
        fig, axis = plt.subplots(figsize=(6.0, 3.6))
        axis.semilogy(np.arange(len(hist)), hist, color="#8b5a2b")
        axis.set_xlabel("Gradient-descent iteration")
        axis.set_ylabel(r"$\|Ax_k - b\|_2 / \|b\|_2$")
        axis.set_title(f"Fill gradient-descent residual (N={case.image_size})")
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(sub / "gradient_convergence.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_benchmark(cases: list[SizeCase], output_dir: Path) -> None:
    rows = []
    for case in cases:
        _save_case_images(case, output_dir)
        for method in case.methods:
            rows.append(
                {
                    "image_size": case.image_size,
                    "n_pixels": case.n_pixels,
                    "n_matrix_entries": case.n_matrix_entries,
                    "sigma": case.sigma,
                    "kernel_size": case.kernel_size,
                    "validation": case.validation,
                    **{
                        k: v
                        for k, v in asdict(method).items()
                        if k != "residual_history"
                    },
                    "residual_history_len": len(method.residual_history),
                }
            )

    payload = {
        "protocol": {
            "observation": "fill / zero-pad linear convolution (mode=same)",
            "methods": {
                "direct": "dense A + lstsq, boundary configurable",
                "fourier": "circular Fourier quotient (model mismatch by design)",
                "iterative": "matrix-free CGLS on FFT fill operator",
                "gradient": "matrix-free steepest descent on FFT fill operator",
            },
            "timing": (
                "deconvolution only for direct/fourier; "
                "iterative/gradient include forward fill blur + solve"
            ),
            "cgls": {
                "tol": DEFAULT_CGLS_TOL,
                "maxiter": DEFAULT_CGLS_MAXITER,
                "stopping": "||A^T r|| / ||A^T b|| < tol",
                "x0": "zeros",
                "dtype": "float64",
            },
            "gradient": {
                "tol": DEFAULT_GD_TOL,
                "maxiter": DEFAULT_GD_MAXITER,
                "stopping": (
                    "||A^T (Ax-b)|| / ||A^T b|| < tol  OR  ||Ax-b|| / ||b|| < tol"
                ),
                "step": "exact line search for quadratic LS",
                "x0": "zeros",
                "dtype": "float64",
            },
            "memory": "tracemalloc peak during deconvolution call (MiB)",
        },
        "rows": rows,
    }
    (output_dir / "benchmark_methods.json").write_text(json.dumps(payload, indent=2))

    header = (
        "image_size,n_pixels,n_matrix_entries,method,boundary,"
        "runtime_s,setup_s,rel_error,mse,rel_residual,iterations,"
        "n_forward,n_adjoint,peak_memory_mib,skipped\n"
    )
    lines = []
    for case in cases:
        for m in case.methods:
            lines.append(
                f"{case.image_size},{case.n_pixels},{case.n_matrix_entries},"
                f"{m.name},{m.boundary},"
                f"{m.runtime_s:.8f},{m.setup_s:.8f},"
                f"{m.rel_error:.12e},{m.mse:.12e},{m.rel_residual:.12e},"
                f"{'' if m.iterations is None else m.iterations},"
                f"{'' if m.n_forward is None else m.n_forward},"
                f"{'' if m.n_adjoint is None else m.n_adjoint},"
                f"{'' if m.peak_memory_mib is None else f'{m.peak_memory_mib:.6f}'},"
                f"{int(m.skipped)}"
            )
    (output_dir / "benchmark_methods.csv").write_text(header + "\n".join(lines) + "\n")

    # Human-readable table
    table_lines = [
        "Fill-observation multi-method benchmark",
        "=======================================",
        "",
        f"{'Size':>4} {'Method':>10} {'Bound':>9} {'Runtime_s':>12} "
        f"{'RelErr':>12} {'RelRes':>12} {'Iters':>6} {'MemMiB':>8}",
        "-" * 84,
    ]
    for case in cases:
        for m in case.methods:
            if m.skipped:
                table_lines.append(
                    f"{case.image_size:>4} {m.name:>10} {m.boundary:>9} "
                    f"{'SKIPPED':>12} {m.skip_reason}"
                )
                continue
            iters = "" if m.iterations is None else str(m.iterations)
            mem = "" if m.peak_memory_mib is None else f"{m.peak_memory_mib:.2f}"
            table_lines.append(
                f"{case.image_size:>4} {m.name:>10} {m.boundary:>9} "
                f"{m.runtime_s:12.6f} {m.rel_error:12.4e} {m.rel_residual:12.4e} "
                f"{iters:>6} {mem:>8}"
            )
        table_lines.append(
            f"     validation forward_max={case.validation['forward_rel_error_max']:.3e} "
            f"adjoint_max={case.validation['adjoint_rel_error_max']:.3e}"
        )
        table_lines.append("")
    (output_dir / "benchmark_methods.txt").write_text("\n".join(table_lines) + "\n")
    # Analysis suite (runtime/memory/error vs pixels, etc.).
    from deconv.experiments.analysis import write_analysis_plots

    write_analysis_plots(cases, output_dir)


def run_three_method_benchmark(
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    *,
    sigma: float = 1.0,
    kernel_size: int = 7,
) -> Path:
    """Run the shared fill-observation multi-method benchmark (legacy name)."""
    output_dir = create_results_folder("results")
    print("Fill-observation multi-method benchmark")
    print(f"Writing to {output_dir}")
    print(f"PSF: sigma={sigma}, kernel={kernel_size}")
    print(
        f"CGLS/GD: tol={DEFAULT_CGLS_TOL}, maxiter={DEFAULT_CGLS_MAXITER}, x0=0, float64"
    )
    cases: list[SizeCase] = []
    for size in sizes:
        print(f"\n--- N={size} ---")
        case = run_size_case(size, sigma=sigma, kernel_size=kernel_size)
        cases.append(case)
        print(
            f"  validation: forward_max={case.validation['forward_rel_error_max']:.3e}, "
            f"adjoint_max={case.validation['adjoint_rel_error_max']:.3e}"
        )
        for m in case.methods:
            if m.skipped:
                print(f"  {m.name}: SKIPPED ({m.skip_reason})")
            else:
                print(
                    f"  {m.name:10s}  t={m.runtime_s:.4g}s  "
                    f"rel_err={m.rel_error:.3e}  rel_res={m.rel_residual:.3e}  "
                    f"iters={m.iterations}  mem={m.peak_memory_mib:.2f}MiB"
                )
    save_benchmark(cases, output_dir)
    print(f"\nSaved tables and figures under {output_dir}/")
    return output_dir


def main() -> Path:
    return run_three_method_benchmark()


if __name__ == "__main__":
    main()
