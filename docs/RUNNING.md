# Running the project

## Install

```bash
cd fourier-deconv
pip install -r requirements.txt
pip install pytest    # needed for tests
```

Dependencies: `numpy`, `scipy`, `Pillow`, `matplotlib`.

---

## 1. Tests (do this first)

```bash
python -m pytest -v
```

| File | What it checks |
|------|----------------|
| `tests/test_fourier.py` | Circular Fourier identities, PSF alignment, Tikhonov |
| `tests/test_pipeline.py` | PSF normalization, wrap FFT≡`convolve2d`, matched wrap recovery |
| `tests/test_iterative.py` | Fill forward≡Direct \(A\), adjoint test, CGLS≈Direct, ≠Fourier |

All tests should pass before trusting new experiments.

---

## 2. Single-run demo — `scripts/main.py`

Edit the configuration block at the top of `scripts/main.py`, then:

```bash
python scripts/main.py
```

### Important settings

| Variable | Meaning | Typical value |
|----------|---------|---------------|
| `IMAGE_SOURCE` | `"synthetic"` or `"file"` | `"synthetic"` |
| `SYNTHETIC_PATTERN` | Pattern name | `"border_frame"` / `"diagonal"` / … |
| `IMAGE_SIZE` | \(N\) for \(N\times N\) | `32` or `64` |
| `SIGMA`, `KERNEL_SIZE` | Gaussian PSF | `1.0`, `7` (aligned with scaling) |
| `METHOD` | Which solvers | `"all"`, `"both"`, `"direct"`, `"fourier"`, `"iterative"` |
| `BOUNDARY_MODE` | Blur + Direct pair | `"mismatch"` (fill) or `"equivalent"` (wrap) |
| `CGLS_TOL`, `CGLS_MAXITER` | Iterative solver | `1e-10`, `5000` |
| `NOISE_STD` | Optional Gaussian noise | `0.0` |

### Boundary modes in the demo

- **`mismatch`** — blur & Direct use **fill**; Fourier stays circular. Best for
  comparing all three on a fill observation (same protocol as scaling).
- **`equivalent`** — blur & Direct use **wrap**; Fourier circular. Direct ≈ Fourier.

### Outputs (`results/<timestamp>/`)

| File | Content |
|------|---------|
| `original.png` | Ground truth |
| `observation.png` | Blurred (and optionally noisy) image |
| `recovered_*.png` | Per-method reconstructions |
| `comparison.png` | Side-by-side panel |
| `error_maps.png` | Absolute errors (when Direct + Fourier ran) |
| `metrics.png` / `metrics.txt` | Runtime and MSE |
| `iterative_convergence.png` | CGLS residual history (if iterative ran) |

Timing reported here is **deconvolution only** (blur and I/O are outside the clock).

---

## 3. Multi-size scaling — `scripts/analyze_scaling.py`

Preferred entry point for runtime / memory / error vs pixels.
Config lives in `deconv/experiments/analysis.py`; the script is a thin CLI wrapper.

```bash
python scripts/analyze_scaling.py
```

### Config (header) or CLI

```bash
# Header defaults in deconv/experiments/analysis.py: METHODS, DIRECT_BOUNDARY, SIGMA=1.0,
# KERNEL_SIZE=7, ANALYSIS_SIZES=(16,24,32,40,48,64), CGLS_MAXITER, CGLS_TOL

python scripts/analyze_scaling.py \
  --methods direct,fourier,iterative \
  --direct-boundary fill \
  --maxiter 5000 \
  --sizes 16,24,32,40

python scripts/analyze_scaling.py --methods fourier --sizes 32,64
python scripts/analyze_scaling.py --methods direct,iterative --direct-boundary wrap
```

| Flag | Effect |
|------|--------|
| `--methods` | Subset of `direct,fourier,iterative` |
| `--direct-boundary` | `fill` or `wrap` for Direct only |
| `--maxiter` / `--tol` | CGLS controls |
| `--sizes` | Comma-separated \(N\) values |
| `--sigma` / `--kernel-size` | PSF |

**Fixed:** observation blur is always **fill**.  
**Fixed:** Fourier reconstruction is always **circular**.  
**Fixed:** Iterative reconstruction is always **fill**.

### Outputs

```
results/<timestamp>/
├── benchmark_three_method.{csv,json,txt}
├── runtime_vs_pixels.png
├── memory_vs_pixels.png
├── error_vs_pixels.png
├── analysis/
│   ├── runtime_vs_pixels.png
│   ├── memory_vs_pixels.png
│   ├── error_vs_pixels.png
│   ├── rel_residual_vs_pixels.png
│   ├── mse_vs_pixels.png
│   ├── iterations_vs_pixels.png          # if iterative selected
│   └── iterative_vs_direct_agreement.png # if both selected
└── N16/, N24/, …                         # per-size images + convergence
```

**Warning:** Direct builds a dense \(N^2 \times N^2\) matrix. Large \(N\) (e.g. 64+)
can use a lot of RAM and time. There is no automatic skip — choose sizes
deliberately.

---

## 4. Benchmark harness — `scripts/benchmark.py`

```bash
python scripts/benchmark.py
```

Runs the same fill-observation three-method protocol (default sizes
`16,24,32,40`) and writes tables + analysis plots. Prefer
`scripts/analyze_scaling.py` when you want CLI control over methods and sizes;
the harness itself lives in `deconv/experiments/experiment.py`.

---

## 5. Interpreting Fourier vs Direct/Iterative

On the default fill observation:

- **Direct / Iterative** invert the fill operator → small error when the blur
  is well conditioned and CGLS is allowed enough iterations.
- **Fourier** assumes wrap → often large relative error even when it looks
  “smoother” than a failed inverse. Smooth ≠ correct.

To see Fourier succeed, switch the demo to `BOUNDARY_MODE = "equivalent"`
(wrap blur) so observation and Fourier share the circular model.

---

## 6. Typical first-session checklist

1. `pip install -r requirements.txt && pip install pytest`
2. `python -m pytest -v`
3. `python scripts/main.py` → open `results/…/comparison.png`
4. `python scripts/analyze_scaling.py --sizes 16,24,32` → open `runtime_vs_pixels.png`
   and `error_vs_pixels.png`
5. Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md) and [MATH.md](MATH.md) as needed

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Fourier looks wild / huge MSE | Fill observation + circular inverse, and/or strong blur (\(\sigma\) large) |
| Iterative error ≫ Direct | Hit `CGLS_MAXITER`; raise `--maxiter` or use milder PSF |
| Direct extremely slow / OOM | \(N\) too large for dense \(A\); reduce `--sizes` |
| Tests fail after edits | Re-check fill forward/adjoint and PSF centre roll conventions |
