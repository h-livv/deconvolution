# Deconvolution Lab

Exploratory study of **convolution-based inverse problems**, using controlled
2D experiments. Solar Gravitational Lens (SGL) imaging is a motivating
application; the present code is not an SGL reconstruction pipeline.

> **Status: Archived / Exploratory.**
> <br>
> This repository is preserved as a record of experiments in numerical inverse
> problems, convolution operators, and deconvolution methods.

## Problem

Given a known convolution kernel (PSF) \(h\) and an unknown field \(x\), the
observation is

\[
b = h * x
\]

or, in matrix form, \(b = A x\). The lab asks how to recover \(x\) when the
operator \(A\) (and especially its **boundary model**) is chosen deliberately,
and how different inverses trade accuracy against cost.

This project focuses on:

- direct inversion of an explicit convolution operator
- Fourier-domain quotient deconvolution
- wrap vs fill (finite-domain / zero-pad) boundary conditions
- matrix-free iterative reconstruction with FFT-applied forward/adjoint maps
- computational scaling with image size

## Implemented methods

| Method | Approach | Boundary |
|--------|----------|----------|
| Direct | Explicit dense convolution matrix \(A\), least squares | Fill or wrap |
| Fourier quotient | $(\hat X = \hat B / \hat H\)$ on the native DFT grid | Circular / periodic |
| Iterative FFT | Matrix-free CGLS; FFTs apply the fill forward and adjoint | Fill |
| Gradient descent | Same fill FFT operator; steepest descent (no conjugacy) | Fill |

The Fourier quotient is extremely efficient because it exploits the
diagonalization of the **circular** convolution operator, but therefore does
not invert a finite-domain/fill operator.

The matrix-free methods do not explicitly store \(A\). Padded FFTs apply the fill forward
map and its adjoint; least squares is solved iteratively. **Gradient descent**
is the iterative method in the default comparison: steepest descent with an
exact line-search step, no conjugacy.

Default experiments form \(b\) with **fill** blur. Direct (fill) and gradient
descent invert that model; Fourier still assumes circular convolution
(intentional mismatch). See
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) and [docs/MATH.md](docs/MATH.md).

## Current observations

<img src="assets/comparison.png" alt="Fill-blurred ring with Direct, Fourier, and gradient-descent reconstructions" width="750">


- On the fill-boundary observation shown above, Fourier is far faster than Direct or gradient
  descent, but the circular inverse is the wrong model: wrap-around ringing
  appears even though the ring is still recognizable.
- Direct reconstructs the fill problem accurately (when the system is well
  conditioned) but stores a dense $\(N^2 \times N^2\)$ operator and scales poorly
  in time and memory.
- Gradient descent targets the same fill least-squares problem without storing
  \(A\). It recovers the ring much more faithfully than Fourier, with more
  leftover structure than Direct.

When blur and Direct both use wrap, Direct and the Fourier quotient agree
closely — the circular model is then matched. That comparison is available in
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
tests/              # Fourier, fill-operator, CGLS, GD, and pipeline checks
docs/               # HOW_IT_WORKS, RUNNING, MATH
images/             # Optional file inputs
results/            # Timestamped experiment outputs
```

## Scope

**Implemented:** Controlled convolution/deconvolution under known PSFs and
explicit boundary models — a numerical laboratory for inverse-method
behaviour.

**Motivation:** Reconstruction questions relevant to SGL imaging (blur,
boundaries, scalable inverses), not an end-to-end SGL forward model.

**Not implemented:** More realistic SGL operators, noise, regularization,
spatially varying PSFs, and additional inverse methods.

## Reference papers

[Viktor T. Toth, Slava G. Turyshev, Image recovery with the solar gravitational lens (2021)](https://arxiv.org/abs/2012.05477v2)
