# Deconvolution Lab

Exploratory study of **convolution-based inverse problems**, using controlled 2D synthetic experiments. Solar Gravitational Lens (SGL) imaging is a motivating application; the present code is not an SGL reconstruction pipeline.

## Problem

Given a known convolution kernel (PSF) \(h\) and an unknown field \(x\), the observation is

\[
b = h * x
\]

or, in matrix form, \(b = A x\). The lab asks how to recover \(x\) when the operator \(A\) (and especially its **boundary model**) is chosen deliberately, and how different inverses trade accuracy against cost.

Current work focuses on:

- direct inversion of an explicit convolution operator
- Fourier-domain quotient deconvolution
- wrap vs fill (finite-domain / zero-pad) boundary conditions
- matrix-free iterative reconstruction with FFT-applied forward/adjoint maps
- computational scaling with image size

## Current methods

| Method | Approach | Boundary |
|--------|----------|----------|
| Direct | Explicit dense convolution matrix \(A\), least squares | Fill or wrap |
| Fourier quotient | \(\hat X = \hat B / \hat H\) on the native DFT grid | Circular / periodic |
| Iterative FFT | Matrix-free CGLS; FFTs apply the fill forward and adjoint | Fill |

The Fourier quotient is extremely efficient because it exploits the diagonalization of the **circular** convolution operator, but therefore does not generally invert a finite-domain/fill operator.

The iterative method uses FFTs only to apply the fill forward map and its adjoint cheaply; the inverse problem itself is solved by CGLS. That is not the same cost as a single Fourier quotient.

Default scaling experiments form \(b\) with **fill** blur. Direct and iterative then invert that fill model; Fourier still assumes circular convolution (intentional model mismatch). See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) and [docs/MATH.md](docs/MATH.md).

## Current observations

On fill-boundary synthetic tests (Gaussian PSF, no noise; outputs under
`results/`):

- The Fourier quotient is dramatically faster than Direct or Iterative, but can fail badly when \(b\) was formed with fill boundaries—the circular assumption is wrong for that forward model.
- Direct reconstructs the fill problem accurately (when the system is well conditioned) but stores a dense \(N^2 \times N^2\) operator and scales poorly in time and memory.
- Matrix-free iterative reconstruction targets the same fill least-squares problem without storing \(A\), trading memory for repeated FFT-based iterations; agreement with Direct is strong on mild blur, while iteration counts grow as the problem gets harder.

When blur and Direct both use wrap, Direct and the Fourier quotient agree closely—the circular model is then matched. That comparison is available in  the demo’s `equivalent` boundary mode.

## Running

From the repository root:

```bash
pip install -r requirements.txt
pip install pytest          # for tests

python -m pytest -v
python scripts/main.py                    # single-size demo
python scripts/analyze_scaling.py         # size sweep, fill observation
python scripts/benchmark.py               # shared three-method fill harness
```

Outputs go to `results/<timestamp>/`. Configuration details and CLI flags:
[docs/RUNNING.md](docs/RUNNING.md).

## Repository structure

```
deconv/      # operators, solvers, experiment harness, scaling analysis
scripts/     # main.py, analyze_scaling.py, benchmark.py
tests/       # Fourier, fill-operator, and pipeline checks
docs/        # HOW_IT_WORKS, RUNNING, MATH
images/      # optional file inputs
results/     # timestamped experiment outputs
```

## Scope

**Now:** controlled synthetic convolution/deconvolution under known PSFs and
explicit boundary models—a numerical laboratory for inverse-method behaviour.

**Motivation:** reconstruction questions relevant to SGL imaging (blur,
boundaries, scalable inverses), not yet an end-to-end SGL forward model.

**Later (not implemented):** more realistic SGL operators, noise, regularization,
spatially varying PSFs, and additional inverse methods.
