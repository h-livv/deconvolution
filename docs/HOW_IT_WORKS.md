# How this project works

## Role of the codebase

This repository is an **experimental imaging / numerical lab**, not a production
restoration library. It exists to:

1. Encode a clear forward model \(b = A x\) (Gaussian blur, wrap or fill).
2. Invert that model with three numerically different algorithms.
3. Measure **runtime**, **memory**, and **reconstruction error** under
   controlled conditions.
4. Show when the cheap Fourier quotient is *valid* (circular blur) versus when
   it is *fast but wrong* (fill blur + circular inverse).

If you are new here: start with `python scripts/main.py`, open the figures in
`results/…/`, then read the method sections below.

---

## End-to-end pipeline

```text
  ground-truth image x
           │
           ▼
   gaussian_psf(h)     ← normalized, odd-sized kernel
           │
           ▼
   apply_blur(x, h)    ← wrap or fill  →  observation b
           │
           ├──────────────┬──────────────────┐
           ▼              ▼                  ▼
        Direct         Fourier            Iterative
      build A,lstsq   X̂ = B̂ / Ĥ        CGLS on fill A
           │              │                  │
           └──────────────┴──────────────────┘
                           │
                           ▼
              metrics + figures → results/<timestamp>/
```

| Stage | Module | Notes |
|-------|--------|-------|
| Image | `deconv/synthetic.py` / `deconv/utils.py` | Synthetic: black bg, dark-grey features; or load a file |
| PSF | `deconv/psf.py` | Discrete Gaussian, sum-normalized to 1 |
| Blur | `deconv/blur.py` | `scipy.signal.convolve2d(..., mode='same')` |
| Direct | `deconv/direct.py` | Dense \(A\), `numpy.linalg.lstsq` |
| Fourier | `deconv/fourier.py` | Circular division only |
| Iterative | `deconv/iterative.py` | Zero-padded FFT apply of **fill** \(A\), CGLS |
| Timing | `deconv/metrics.py` | Clocks **deconvolution only** (not blur / I/O) |
| Demo | `scripts/main.py` | One size, configurable |
| Scaling | `scripts/analyze_scaling.py` + `deconv/experiment.py` | Many sizes, fill observation |

---

## Features

### 1. Direct spatial reconstruction

- Builds the full convolution matrix whose action matches `convolve2d`.
- Supports **`boundary='wrap'`** and **`boundary='fill'`**.
- Solves \(\min_x \|A x - b\|_2\) with dense least squares.
- Exact (up to conditioning / floating point) for the chosen boundary, but
  memory grows like \(O(N^4)\) for an \(N\times N\) image.

### 2. Fourier quotient deconvolution

- Implements the convolution theorem under **periodic / circular** blur:

  \[
  \hat B = \hat H\,\hat X
  \quad\Rightarrow\quad
  \hat X = \hat B / \hat H
  \]

  (optional Tikhonov form with `reg=λ`).
- Extremely fast: a few FFTs + pointwise division.
- **Does not** implement fill/linear convolution on the native grid — that
  operator is not diagonalized by an image-sized DFT.

### 3. Iterative FFT (fill) reconstruction

- Never stores \(A\).
- Applies the **same** fill/`same` operator as Direct via zero-padded FFTs.
- Uses CGLS (`min \|Ax-b\|_2^2`) with exact adjoint (verified by tests).
- Memory stays \(O(N^2)\); cost is roughly \(O(K N^2 \log N)\) for \(K\)
  iterations — not as fast as a single Fourier quotient.

### 4. Boundary-aware experiments

| Mode | Observation | Direct | Fourier | Iterative |
|------|-------------|--------|---------|-----------|
| `equivalent` (demo) | wrap | wrap | circular | fill\* |
| `mismatch` / scaling default | **fill** | fill | circular | fill |

\*Iterative always uses the fill operator. Prefer `mismatch` (or scaling
defaults) when comparing all three on the same physical fill problem.

### 5. Scaling analysis

`scripts/analyze_scaling.py` sweeps image sizes and writes:

- runtime vs pixels  
- peak memory vs pixels (`tracemalloc` during the solve)  
- relative error, residual, MSE vs pixels  
- CGLS iteration counts (if iterative is selected)

Observation is **always fill**. You choose which methods run and Direct’s
reconstruction boundary.

### 6. Validation suite

```bash
python -m pytest -v
```

Covers Fourier circular identities, fill forward ≡ Direct matrix, adjoint
identity, CGLS ≈ Direct on mild blur, and intentional Fourier mismatch on fill
data.

---

## Synthetic images

`synthetic.py` builds grayscale test fields in \([0,1]\):

- **Background:** black (`0.0`)
- **Borders / patterns:** dark grey (`0.35`)

Patterns: `corner_pixel`, `edge_square`, `border_frame`, `diagonal`.

These stress edges and boundaries so wrap vs fill differences are visible.

---

## What “success” looks like

On a **matched** problem (fill observation + Direct/Iterative fill, mild
Gaussian blur):

- Direct recovers \(x\) to near machine precision (subject to conditioning).
- Iterative approaches the same least-squares solution (may need many CGLS
  iterations if the blur is strong).
- Fourier remains fast but typically shows large error (wrong boundary model).

On a **wrap** observation with Direct wrap + Fourier circular, Direct and
Fourier should agree closely — that is the `equivalent` demo mode.

---

## Related reading (conceptual)

- Simões, Almeida, Bioucas-Dias, Chanussot — *A Framework for Fast Image
  Deconvolution with Incomplete Observations* (2016).
- Almeida et al. — *Deconvolving Images with Unknown Boundaries Using ADMM*
  (2013).

This lab uses the simplest matrix-free LS solver (CGLS) on the exact fill
operator rather than reproducing those papers’ full ADMM frameworks.

For equations and operator details, see **[MATH.md](MATH.md)**.  
For install / run instructions, see **[RUNNING.md](RUNNING.md)**.
