"""
Multi-size scaling analysis for Direct, Fourier, and Iterative FFT methods.

The forward observation is **always** fill / zero-pad blur. Reconstruction
methods and Direct's boundary are configurable.

Figures written
---------------
* runtime vs pixels
* peak memory vs pixels
* relative reconstruction error vs pixels
* relative data residual vs pixels
* MSE vs pixels
* CGLS iterations vs pixels (if iterative is selected)
* iterative vs direct agreement (if both selected)

Edit the configuration block below, or pass CLI flags.

Usage
-----
    python scripts/analyze_scaling.py
    python scripts/analyze_scaling.py --methods direct,iterative --direct-boundary fill
    python scripts/analyze_scaling.py --methods fourier --maxiter 2000
    python scripts/analyze_scaling.py --methods direct,fourier,iterative --direct-boundary wrap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deconv.experiments.experiment import SizeCase, run_size_case, save_benchmark
from deconv.methods.iterative import DEFAULT_CGLS_MAXITER, DEFAULT_CGLS_TOL
from deconv.io.utils import create_results_folder

# ---------------------------------------------------------------------------
# Configuration — edit these, or override via CLI
# ---------------------------------------------------------------------------

# Which reconstruction methods to run (any non-empty subset).
METHODS: tuple[str, ...] = ("direct", "fourier")
# "direct" | "fourier" | "iterative"

# Direct reconstruction boundary. Observation is always fill.
# Fourier is always circular; iterative is always matrix-free fill.
DIRECT_BOUNDARY = "fill"
# "fill" | "wrap"

# CGLS controls (iterative method only)
CGLS_MAXITER = DEFAULT_CGLS_MAXITER
CGLS_TOL = DEFAULT_CGLS_TOL

# Image-size sweep and PSF
ANALYSIS_SIZES: tuple[int, ...] = (16, 24, 32, 40, 48, 64, 72, 80)
SIGMA = 1.0
KERNEL_SIZE = 7

# ---------------------------------------------------------------------------

METHOD_STYLE = {
    "direct": ("o-", "#3b6d9c", "Direct"),
    "fourier": ("s-", "#c46b3a", "Fourier (circular)"),
    "iterative": ("^-", "#2f6f4e", "Iterative FFT (fill)"),
}


def _method_label(name: str, cases: list[SizeCase]) -> str:
    base = METHOD_STYLE[name][2]
    if name != "direct":
        return base
    for case in cases:
        method = next((m for m in case.methods if m.name == "direct"), None)
        if method is not None:
            return f"Direct ({method.boundary})"
    return base


def _series(
    cases: list[SizeCase],
    method_name: str,
    value_attr: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (n_pixels, values) for one method, dropping skipped / non-finite."""
    xs: list[float] = []
    ys: list[float] = []
    for case in cases:
        method = next((m for m in case.methods if m.name == method_name), None)
        if method is None or method.skipped:
            continue
        value = getattr(method, value_attr)
        if value is None or not np.isfinite(value):
            continue
        xs.append(float(case.n_pixels))
        ys.append(float(value))
    return np.asarray(xs), np.asarray(ys)


def _active_methods(cases: list[SizeCase]) -> tuple[str, ...]:
    names: list[str] = []
    for name in ("direct", "fourier", "iterative"):
        if any(m.name == name for case in cases for m in case.methods):
            names.append(name)
    return tuple(names)


def _plot_metric_vs_pixels(
    cases: list[SizeCase],
    *,
    value_attr: str,
    ylabel: str,
    title: str,
    path: Path,
    log_y: bool = True,
    log_x: bool = True,
    methods: tuple[str, ...] | None = None,
) -> None:
    if methods is None:
        methods = _active_methods(cases)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    plotted = False
    for name in methods:
        fmt, color, _ = METHOD_STYLE[name]
        xs, ys = _series(cases, name, value_attr)
        if xs.size == 0:
            continue
        plotted = True
        if log_y:
            ys = np.maximum(ys, 1e-16)
        if log_x and log_y:
            plot = axis.loglog
        elif log_x:
            plot = axis.semilogx
        elif log_y:
            plot = axis.semilogy
        else:
            plot = axis.plot
        plot(xs, ys, fmt, color=color, label=_method_label(name, cases), markersize=6)

    if not plotted:
        plt.close(fig)
        return

    axis.set_xlabel(r"Pixels $N^2$")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", linestyle=":", alpha=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_agreement_direct_iterative(cases: list[SizeCase], path: Path) -> None:
    """||x_iter - x_direct|| / ||x_direct|| when both recoveries exist."""
    xs: list[float] = []
    ys: list[float] = []
    for case in cases:
        arts = getattr(case, "_artifacts", None)
        if arts is None:
            continue
        x_d = arts.get("recovered_direct")
        x_i = arts.get("recovered_iterative")
        if x_d is None or x_i is None:
            continue
        denom = float(np.linalg.norm(x_d))
        if denom == 0.0:
            continue
        xs.append(float(case.n_pixels))
        ys.append(float(np.linalg.norm(x_i - x_d) / denom))

    if not xs:
        return

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.loglog(xs, np.maximum(ys, 1e-16), "D-", color="#6b4c9a", markersize=6)
    axis.set_xlabel(r"Pixels $N^2$")
    axis.set_ylabel(
        r"$\|x_{\mathrm{iter}}-x_{\mathrm{direct}}\|_2"
        r" / \|x_{\mathrm{direct}}\|_2$"
    )
    axis.set_title("Iterative vs Direct agreement")
    axis.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_analysis_plots(cases: list[SizeCase], output_dir: Path) -> list[Path]:
    """Write the scaling-analysis figure set; return paths written."""
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    active = _active_methods(cases)

    specs = [
        (
            "runtime_vs_pixels.png",
            "runtime_s",
            "Deconvolution runtime (s)",
            "Runtime vs pixels (fill observation)",
            True,
        ),
        (
            "memory_vs_pixels.png",
            "peak_memory_mib",
            "Peak memory (MiB)",
            "Peak memory vs pixels",
            True,
        ),
        (
            "error_vs_pixels.png",
            "rel_error",
            r"Relative error $\|x_{\mathrm{recon}}-x\|_2 / \|x\|_2$",
            "Reconstruction error vs pixels",
            True,
        ),
        (
            "rel_residual_vs_pixels.png",
            "rel_residual",
            r"Relative residual $\|Ax-b\|_2 / \|b\|_2$",
            "Relative data residual vs pixels (fill $A$)",
            True,
        ),
        (
            "mse_vs_pixels.png",
            "mse",
            "MSE vs ground truth",
            "Mean squared error vs pixels",
            True,
        ),
    ]
    for filename, attr, ylabel, title, log_y in specs:
        path = analysis_dir / filename
        _plot_metric_vs_pixels(
            cases,
            value_attr=attr,
            ylabel=ylabel,
            title=title,
            path=path,
            log_y=log_y,
            methods=active,
        )
        if path.is_file():
            written.append(path)

    if "iterative" in active:
        iters_path = analysis_dir / "iterations_vs_pixels.png"
        _plot_metric_vs_pixels(
            cases,
            value_attr="iterations",
            ylabel="CGLS iterations",
            title="Iterative solver iterations vs pixels",
            path=iters_path,
            log_y=False,
            methods=("iterative",),
        )
        if iters_path.is_file():
            written.append(iters_path)

    if "direct" in active and "iterative" in active:
        agree_path = analysis_dir / "iterative_vs_direct_agreement.png"
        _plot_agreement_direct_iterative(cases, agree_path)
        if agree_path.is_file():
            written.append(agree_path)

    # Convenience copies at the results root.
    for src_name, dst_name in (
        ("runtime_vs_pixels.png", "runtime_vs_pixels.png"),
        ("memory_vs_pixels.png", "memory_vs_pixels.png"),
        ("error_vs_pixels.png", "error_vs_pixels.png"),
    ):
        src = analysis_dir / src_name
        if src.is_file():
            dst = output_dir / dst_name
            dst.write_bytes(src.read_bytes())
            written.append(dst)

    return written


def _parse_methods(text: str) -> tuple[str, ...]:
    parts = tuple(p.strip().lower() for p in text.split(",") if p.strip())
    allowed = {"direct", "fourier", "iterative"}
    unknown = set(parts) - allowed
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown methods {sorted(unknown)}; choose from {sorted(allowed)}"
        )
    if not parts:
        raise argparse.ArgumentTypeError("Provide at least one method")
    # Preserve a stable order.
    return tuple(m for m in ("direct", "fourier", "iterative") if m in parts)


def _parse_sizes(text: str) -> tuple[int, ...]:
    sizes = tuple(int(p.strip()) for p in text.split(",") if p.strip())
    if not sizes or any(s < 4 for s in sizes):
        raise argparse.ArgumentTypeError("Sizes must be integers >= 4")
    return sizes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-size scaling analysis. Observation is always fill; "
            "choose reconstruction methods and Direct boundary."
        )
    )
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=None,
        help="Comma-separated subset: direct,fourier,iterative "
        f"(default: {','.join(METHODS)})",
    )
    parser.add_argument(
        "--direct-boundary",
        choices=("fill", "wrap"),
        default=None,
        help="Boundary for Direct reconstruction only "
        f"(default: {DIRECT_BOUNDARY}). Observation is always fill.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help=f"CGLS max iterations for iterative method (default: {CGLS_MAXITER})",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help=f"CGLS relative normal-residual tolerance (default: {CGLS_TOL})",
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=None,
        help="Comma-separated image sizes, e.g. 12,16,24,32 "
        f"(default: {','.join(str(s) for s in ANALYSIS_SIZES)})",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help=f"Gaussian PSF sigma (default: {SIGMA})",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        help=f"Odd Gaussian kernel size (default: {KERNEL_SIZE})",
    )
    return parser


def run_scaling_analysis(
    sizes: tuple[int, ...] | None = None,
    *,
    methods: tuple[str, ...] | None = None,
    direct_boundary: str | None = None,
    cgls_maxiter: int | None = None,
    cgls_tol: float | None = None,
    sigma: float | None = None,
    kernel_size: int | None = None,
) -> Path:
    """Sweep image sizes, save tables/images, and write analysis plots."""
    sizes = sizes if sizes is not None else ANALYSIS_SIZES
    methods = methods if methods is not None else METHODS
    direct_boundary = (
        direct_boundary if direct_boundary is not None else DIRECT_BOUNDARY
    )
    cgls_maxiter = CGLS_MAXITER if cgls_maxiter is None else cgls_maxiter
    cgls_tol = CGLS_TOL if cgls_tol is None else cgls_tol
    sigma = SIGMA if sigma is None else sigma
    kernel_size = KERNEL_SIZE if kernel_size is None else kernel_size

    if cgls_maxiter < 1:
        raise ValueError("cgls_maxiter must be >= 1")
    if cgls_tol <= 0:
        raise ValueError("cgls_tol must be positive")

    output_dir = create_results_folder("results")
    print("Multi-size scaling analysis")
    print(f"Writing to {output_dir}")
    print("Observation boundary: fill (fixed)")
    print(f"Methods: {', '.join(methods)}")
    print(f"Direct reconstruction boundary: {direct_boundary}")
    print("Fourier reconstruction boundary: circular")
    print("Iterative reconstruction boundary: fill")
    print(f"Sizes: {sizes}")
    print(f"PSF: sigma={sigma}, kernel={kernel_size}")
    if "iterative" in methods:
        print(f"CGLS: tol={cgls_tol}, maxiter={cgls_maxiter}, x0=0, float64")

    cases: list[SizeCase] = []
    for size in sizes:
        print(f"\n--- N={size} ({size * size} pixels) ---")
        case = run_size_case(
            size,
            sigma=sigma,
            kernel_size=kernel_size,
            cgls_tol=cgls_tol,
            cgls_maxiter=cgls_maxiter,
            methods=methods,
            direct_boundary=direct_boundary,
        )
        cases.append(case)
        for m in case.methods:
            if m.skipped:
                print(f"  {m.name}: SKIPPED ({m.skip_reason})")
            else:
                mem = (
                    f"{m.peak_memory_mib:.2f} MiB"
                    if m.peak_memory_mib is not None
                    else "n/a"
                )
                extra = f"  iters={m.iterations}" if m.iterations is not None else ""
                print(
                    f"  {m.name:10s}  [{m.boundary}]  t={m.runtime_s:.4g}s  "
                    f"rel_err={m.rel_error:.3e}  rel_res={m.rel_residual:.3e}  "
                    f"mem={mem}{extra}"
                )

    save_benchmark(cases, output_dir)
    written = write_analysis_plots(cases, output_dir)

    print("\nAnalysis figures:")
    for path in written:
        print(f"  {path}")
    print(f"\nDone → {output_dir}/")
    return output_dir


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    return run_scaling_analysis(
        sizes=args.sizes,
        methods=args.methods,
        direct_boundary=args.direct_boundary,
        cgls_maxiter=args.maxiter,
        cgls_tol=args.tol,
        sigma=args.sigma,
        kernel_size=args.kernel_size,
    )


if __name__ == "__main__":
    main()
