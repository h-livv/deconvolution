# Deconvolution Lab

Exploratory study of **convolution-based inverse problems**, using controlled
2D synthetic experiments. Solar Gravitational Lens (SGL) imaging is a motivating
application; the present code is not an SGL reconstruction pipeline.

## Problem

Given a known convolution kernel (PSF) \(h\) and an unknown field \(x\), the
observation is

\[
b = h * x
\]

or, in matrix form, \(b = A x\). The lab asks how to recover \(x\) when the
operator \(A\) (and especially its **boundary model**) is chosen deliberately,
and how different inverses trade accuracy against cost.

Current work focuses on:

- direct inversion of an explicit convolution operator
- Fourier-domain quotient deconvolution
- wrap vs fill (finite-domain / zero-pad) boundary conditions
- matrix-free iterative reconstruction with FFT-applied forward/adjoint maps
  (CGLS and plain gradient descent)
- computational scaling with image size

## Current methods

| Method | Approach | Boundary |
|--------|----------|----------|
| Direct | Explicit dense convolution matrix \(A\), least squares | Fill or wrap |
| Fourier quotient | \(\hat X = \hat B / \hat H\) on the native DFT grid | Circular / periodic |
| Iterative FFT | Matrix-free CGLS; FFTs apply the fill forward and adjoint | Fill |
| Gradient descent | Same fill FFT operator; steepest descent (no conjugacy) | Fill |

The Fourier quotient is extremely efficient because it exploits the
diagonalization of the **circular** convolution operator, but therefore does
not generally invert a finite-domain/fill operator.

The two matrix-free fill methods use FFTs only to apply the fill forward map
and its adjoint; the inverse problem is solved iteratively. CGLS accelerates
least-squares progress with conjugate search directions; gradient descent uses
the same gradient \(A^{\mathsf T}(Ax-b)\) with an exact line-search step and no
conjugacy — a simpler baseline on the same operator.

Default scaling experiments form \(b\) with **fill** blur. Direct and the
matrix-free fill solvers then invert that fill model; Fourier still assumes
circular convolution (intentional model mismatch). See
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) and [docs/MATH.md](docs/MATH.md).

## Current observations

On fill-boundary synthetic tests (Gaussian PSF, no noise; outputs under
`results/`):

- The Fourier quotient is dramatically faster than Direct or the iterative
  fill solvers, but can fail badly when \(b\) was formed with fill
  boundaries—the circular assumption is wrong for that forward model.
- Direct reconstructs the fill problem accurately (when the system is well
  conditioned) but stores a dense \(N^2 \times N^2\) operator and scales poorly
  in time and memory.
- Matrix-free CGLS targets the same fill least-squares problem without storing
  \(A\), trading memory for repeated FFT-based iterations; agreement with
  Direct is strong on mild blur.
- Plain gradient descent uses the same fill operator and objective, but
  typically needs more iterations than CGLS to reach the same residual
  tolerance.

When blur and Direct both use wrap, Direct and the Fourier quotient agree
closely—the circular model is then matched. That comparison is available in
the demo’s `equivalent` boundary mode.

## Running

From the repository root:

```bash
pip install -r requirements.txt
pip install pytest          # for tests

python -m pytest -v
python scripts/main.py                    # single-size demo
python scripts/analyze_scaling.py         # size sweep, fill observation
python scripts/benchmark.py               # shared fill multi-method harness
```

Outputs go to `results/<timestamp>/`. Configuration details and CLI flags:
[docs/RUNNING.md](docs/RUNNING.md).

## Repository structure

```
deconv/
├── forward/        # PSF + blur (forms b = h*x)
├── methods/        # Direct, Fourier, iterative CGLS, gradient descent
├── data/           # Synthetic test fields
├── io/             # Load / save / display / results folders
├── experiments/    # Multi-method harness + scaling analysis
├── metrics.py      # Error metrics and timing helpers
└── paths.py        # Repository / images / results roots
scripts/            # main.py, analyze_scaling.py, benchmark.py
tests/              # Fourier, fill-operator, GD, and pipeline checks
docs/               # HOW_IT_WORKS, RUNNING, MATH
images/             # Optional file inputs
results/            # Timestamped experiment outputs
```

## Scope

**Now:** controlled synthetic convolution/deconvolution under known PSFs and
explicit boundary models—a numerical laboratory for inverse-method behaviour.

**Motivation:** reconstruction questions relevant to SGL imaging (blur,
boundaries, scalable inverses), not yet an end-to-end SGL forward model.

**Later (not implemented):** more realistic SGL operators, noise, regularization,
spatially varying PSFs, and additional inverse methods.
