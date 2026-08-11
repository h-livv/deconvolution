# Mathematics and operators

This note records how the code maps to the imaging equations. For a conceptual
overview see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

---

## 1. Forward model

Unknown object \(x\), known PSF \(h\), observation

\[
b = A x + n.
\]

In this lab \(n\) is usually zero. \(A\) is discrete convolution with
`mode='same'` and either:

- **wrap** — periodic / circular convolution, or  
- **fill** — linear convolution with zeros outside the frame.

The Gaussian PSF (`psf.gaussian_psf`) is evaluated on an odd window, centred on
the middle pixel, and **renormalized** so \(\sum_{ij} h_{ij} = 1\).

Synthetic images use background \(0\) (black) and feature level \(0.35\)
(dark grey).

---

## 2. Direct method

Construct \(A\in\mathbb{R}^{MN\times MN}\) so that

\[
(A x)_{\mathrm{vec}}
=
\mathrm{vec}\bigl(\mathrm{convolve2d}(x,h,\mathrm{mode{=}same},\,\mathrm{boundary})\bigr).
\]

Each column is the response to a unit impulse at one pixel
(`direct.build_convolution_matrix`). Recover

\[
x^\star = \arg\min_x \|A x - b\|_2
\]

via `numpy.linalg.lstsq`.

---

## 3. Fourier quotient (circular only)

For **circular** convolution the DFT diagonalizes \(A\):

\[
\hat B_{kl} = \hat H_{kl}\,\hat X_{kl}
\quad\Rightarrow\quad
\hat X_{kl} = \frac{\hat B_{kl}}{\hat H_{kl}}
\quad (\hat H_{kl}\neq 0).
\]

### PSF embedding (NumPy FFT convention)

Pad \(h\) to the image size and **roll** so the spatial centre sits at index
`(0,0)`:

```text
ifft2( fft2(x) * fft2(embed_and_roll(h)) )
  ≡  convolve2d(x, h, mode='same', boundary='wrap')
```

Implemented in `fourier._embed_psf` / `fourier_deconvolution`.

### Optional Tikhonov / Wiener form (`reg=λ>0`)

\[
\hat X
=
\frac{\hat H^{*}}{|\hat H|^2 + \lambda}\,\hat B.
\]

This preserves the phase of \(\hat H\) (it does not replace small complex
values by a real \(\varepsilon\)).

**Important:** image-sized DFT division is **not** the inverse of fill/`same`
linear convolution. Using Fourier on fill data is a deliberate mismatch in the
default experiments.

---

## 4. Iterative fill operator (FFT-accelerated)

Let the image be \(M\times N\) and the PSF \(K\times L\). Full linear
convolution lives on a grid of size \((M+K-1)\times(N+L-1)\).

### Forward \(A_{\mathrm{fill}}\)

1. Embed \(x\) at the top-left of the padded grid (zeros elsewhere).  
2. Embed \(h\) at the origin (**no** circular roll).  
3. Multiply FFTs; inverse FFT.  
4. Crop the SciPy `'same'` window starting at
   \(\bigl((K-1)//2,\,(L-1)//2\bigr)\).

This matches

```text
convolve2d(x, h, mode='same', boundary='fill', fillvalue=0)
```

to ~\(10^{-16}\) relative error (`test_iterative.py`).

### Adjoint \(A^{\mathsf T}\)

1. Embed the residual into the `'same'` window of the padded grid.  
2. Multiply by \(\overline{\hat H}\) in Fourier space.  
3. Inverse FFT; crop the top-left \(M\times N\) object domain.

Mandatory adjoint test:

\[
\frac{|\langle Ax,y\rangle - \langle x, A^{\mathsf T} y\rangle|}
{\max(|\langle Ax,y\rangle|,\,|\langle x,A^{\mathsf T} y\rangle|,\,1)}
\sim 10^{-15}.
\]

### CGLS

Solve \(\min_x \|A x - b\|_2^2\) with only `forward` / `adjoint` calls.
Default stopping:

\[
\frac{\|A^{\mathsf T} r\|_2}{\|A^{\mathsf T} b\|_2} < 10^{-10},
\qquad
x_0 = 0,
\qquad
\mathrm{maxiter} = 5000,
\qquad
\mathrm{dtype}=\mathrm{float64}.
\]

Strong blur makes \(A\) ill-conditioned: dense `lstsq` (SVD truncation) may
still beat iteration-capped CGLS on near-nullspace components. Milder PSFs
(\(\sigma=1\), \(7\times7\)) show close agreement.

Cost scale: \(O(K N^2 \log N)\) vs Fourier quotient \(O(N^2 \log N)\).

---



### Gradient descent

Same fill operator and objective \(f(x)=\tfrac12\|A x-b\|_2^2\). Instead of
CGLS, take steepest-descent steps:

\[
g = A^{\mathsf T}(A x - b),
\qquad
\alpha = \frac{\|g\|_2^2}{\|A g\|_2^2},
\qquad
x \leftarrow x - \alpha g.
\]

Exact line search for this quadratic chooses \(\alpha\); there are **no**
conjugate directions. Default stopping accepts either

\[
\frac{\|A^{\mathsf T}(Ax-b)\|_2}{\|A^{\mathsf T} b\|_2} < 10^{-10}
\quad\text{or}\quad
\frac{\|Ax-b\|_2}{\|b\|_2} < 10^{-10},
\]

with \(x_0=0\), \(\mathrm{maxiter}=5000\), float64.

On mild fill problems GD reaches the Direct solution but typically needs more
iterations than CGLS.

---

## 5. Metrics

| Metric | Definition |
|--------|------------|
| Runtime | Wall time of the deconvolution call only (`time_deconvolution` / equivalent) |
| Rel. reconstruction error | \(\|x_{\mathrm{recon}}-x\|_2 / \|x\|_2\) |
| Rel. residual | \(\|A_{\mathrm{fill}} x_{\mathrm{recon}} - b\|_2 / \|b\|_2\) |
| Peak memory | `tracemalloc` peak during the timed solve (MiB) |

Residuals in the scaling study are always measured against the **fill**
operator, even for Fourier reconstructions — that exposes model mismatch.

---

## 6. Complexity summary

| Method | Dominant memory | Dominant time |
|--------|-----------------|---------------|
| Direct | Dense \(A\): \(O(N^4)\) entries | Build \(A\) + dense LS |
| Fourier | \(O(N^2)\) FFT buffers | Few FFTs |
| Iterative (CGLS) | \(O(N^2)\) work vectors + padded FFTs | \(K\) forward/adjoint FFT pairs |
| Gradient descent | Same as iterative | Typically larger \(K\) than CGLS |

---

## 7. References

- M. Simões, L. B. Almeida, J. Bioucas-Dias, J. Chanussot,  
  *A Framework for Fast Image Deconvolution with Incomplete Observations*,
  IEEE TIP, 2016.

- M. S. C. Almeida et al.,  
  *Deconvolving Images with Unknown Boundaries Using the Alternating Direction
  Method of Multipliers*, IEEE TIP, 2013.
