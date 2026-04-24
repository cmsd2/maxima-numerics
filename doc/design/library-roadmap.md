# Standard Library Roadmap

## Motivation

The [architecture](architecture.md) roadmap focuses on wrapping Maxima's bundled
solvers (ODEPACK, MINPACK, COBYLA) to expose them through the ndarray handle.
This document addresses a different gap: the **utility layer** that sits between
raw ndarray operations and domain-specific toolboxes.

NumPy, SciPy, and MATLAB succeed not just because they have fast arrays, but
because they ship hundreds of small, composable functions that save users from
reimplementing routine numerical tasks. Our demo notebooks reveal the same
pattern — each notebook reinvents helpers like `makelist(f(i), i, 1, N)` →
ndarray, log-spaced grids, convergence rate estimation, windowed FFTs, and
finite-difference stencils. Promoting these into the package eliminates
boilerplate and makes Maxima competitive for everyday numeric work.

The guiding principle: **ship utilities, not toolboxes.** We are not rebuilding
MATLAB's Signal Processing Toolbox or SciPy's `scipy.stats`. We provide the
building blocks that make those workflows natural in Maxima, letting the
symbolic engine handle what proprietary tools cannot.

---

## What Already Exists

Before proposing additions, here is what the package already provides (130+
functions). Proposals below are careful not to duplicate these.

| Category | Count | Key functions |
|----------|-------|---------------|
| Constructors | 17 | `np_zeros`, `np_ones`, `np_eye`, `np_rand`, `np_randn`, `np_linspace`, `np_arange`, `np_diag`, `np_copy` |
| Element-wise | 51 | Full arithmetic, trig, hyperbolic, activation functions, comparisons, logical ops, `np_map`, `np_where`, `np_extract` |
| Linear algebra | 28 | `np_matmul`, `np_inv`, `np_svd`, `np_eig`, `np_qr`, `np_lu`, `np_cholesky`, `np_solve`, `np_lstsq`, `np_pinv`, `np_expm`, `np_norm` |
| Slicing | 11 | `np_ref`, `np_set`, `np_row`, `np_col`, `np_slice`, `np_reshape`, `np_flatten`, `np_hstack`, `np_vstack` |
| Aggregation | 14 | `np_sum`, `np_mean`, `np_std`, `np_var`, `np_min`, `np_max`, `np_cumsum`, `np_sort`, `np_argsort`, `np_dot`, `np_discount` |
| Signal | 7 | `np_fft`, `np_ifft`, `np_fft2d`, `np_ifft2d`, `np_convolve`, `np_convolve2d`, `np_trapz` |
| Optimisation | 8 | `np_minimize`, `np_minimize_cobyla`, `np_fsolve`, `np_lsq_nonlinear`, `np_compile_gradient` |
| ODE | 1 | `np_odeint` |
| Image | 5 | `np_read_image`, `np_mandrill`, etc. |
| Arrow / I/O | 6 | `np_read_csv`, `np_write_csv`, table operations |

---

## The Differentiator: Symbolic Bridge Functions

Maxima's unique advantage over NumPy/MATLAB is its symbolic engine. The
`np_compile_gradient` pattern — take a symbolic expression, differentiate it,
compile both the expression and its derivative into fast numeric closures —
is something no purely numeric tool can offer. Every new utility should ask:
**can this leverage symbolic computation?**

Examples already in the package:
- `np_compile_gradient(expr, vars)` — symbolic gradient → compiled closures
- `np_minimize(expr, vars, x0)` — expression mode: symbolic diff + L-BFGS
- `np_odeint(rhs_exprs, vars, y0, tspan)` — expression mode: symbolic Jacobian
- `np_fsolve(fcns, vars, x0)` — expression mode: symbolic Jacobian → MINPACK

The pattern generalises. Any workflow where a user writes a formula and the
system derives auxiliary quantities (gradients, Jacobians, Hessians, adjoint
equations, sensitivity matrices) is a natural fit. The proposals below identify
where this pattern can be extended.

---

## Tier 1: Numeric Utilities

Small, self-contained functions that eliminate common boilerplate. Each is
implementable in a few dozen lines of Lisp or Maxima code. No new native
dependencies.

### 1.1 Grid and mesh construction

**Gap:** Notebooks repeatedly build 2D grids via nested `makelist`. A
`np_meshgrid` would make contour plots and surface evaluations one-liners.

```maxima
/* Current pattern (from PDE notebooks): */
xx : makelist(makelist(x_grid[j], j, 1, nx), i, 1, ny)$
yy : makelist(makelist(y_grid[i], j, 1, nx), i, 1, ny)$

/* Proposed: */
[XX, YY] : np_meshgrid(np_linspace(0, 1, nx), np_linspace(0, 1, ny))$
```

| Function | Description | Implementation |
|----------|-------------|----------------|
| `np_meshgrid(x, y)` | Coordinate matrices from 1D vectors | Lisp: broadcast + reshape |
| `np_logspace(a, b, n)` | Log-spaced points (10^a to 10^b) | Lisp: `np_linspace` + `np_pow` |
| `np_geomspace(a, b, n)` | Geometrically spaced points (a to b) | Lisp: `np_logspace` variant |

### 1.2 Differencing and cumulative operations

**Gap:** The beam-deflection notebook builds finite-difference stencils by hand.
The spectral analysis notebook computes cumulative sums with `np_cumsum` (which
exists) but has no `np_diff`.

Wait — `np_diff` is not currently in the package. The `np_cumsum` exists but
`np_diff` (the inverse operation: first differences) does not.

| Function | Description | Implementation |
|----------|-------------|----------------|
| `np_diff(A)` | First differences along axis | Lisp: `np_sub(A[2:], A[:-1])` via slicing |
| `np_gradient(A, dx)` | Numeric gradient (central differences, one-sided at edges) | Lisp: stencil loop |
| `np_cumprod(A)` | Cumulative product | Lisp: parallel to `np_cumsum` |
| `np_clip(A, lo, hi)` | Clamp values to range | Lisp: `np_where` chain |
| `np_interp(xp, x, y)` | 1D linear interpolation (NumPy-style) | Lisp: binary search + lerp |
| `np_unwrap(A)` | Phase unwrapping (add ±2pi to remove discontinuities) | Lisp: diff + cumsum |

### 1.3 Signal / spectral helpers

**Gap:** The spectral analysis notebook implements magnitude spectrum, PSD,
windowing, and STFT from scratch using raw FFT. These are standard operations
that belong in the package.

| Function | Description | Implementation |
|----------|-------------|----------------|
| `np_magnitude_spectrum(x)` | `np_abs(np_fft(x))` normalised | Lisp: trivial wrapper |
| `np_psd(x)` | Power spectral density (one-sided) | Lisp: `abs(fft)^2 / N`, fold negative freqs |
| `np_window(name, N)` | Window functions: hann, hamming, blackman, rectangular | Lisp: formula evaluation |
| `np_stft(x, seg_len, hop)` | Short-time Fourier transform | Lisp: segment + window + FFT loop |
| `np_fftfreq(N, fs)` | Frequency bin centres for FFT output | Lisp: arithmetic |
| `np_rfft(x)` | Real-input FFT (returns positive frequencies only) | Lisp: wraps fftpack5 `rfft` |

The existing `np_fft` returns a full complex spectrum including negative
frequencies. `np_rfft` would return only the positive half, which is what most
spectral analysis workflows actually want. fftpack5 already has `rfft`
internally.

### 1.4 Statistical functions

**Gap:** `np_mean`, `np_std`, `np_var` exist but there are no correlation,
covariance, or histogram functions. The PCA notebook computes covariance
manually.

| Function | Description | Implementation |
|----------|-------------|----------------|
| `np_corrcoef(A)` | Pearson correlation matrix | Lisp: center + `np_matmul` |
| `np_cov(A)` | Covariance matrix | Lisp: center + `np_matmul` |
| `np_histogram(A, bins)` | Bin counts and edges | Lisp: sort + bin search |
| `np_percentile(A, q)` | Percentile (linear interpolation) | Lisp: sort + index |
| `np_median(A)` | Median value | Lisp: `np_percentile(A, 50)` |

### 1.5 Miscellaneous

| Function | Description | Implementation |
|----------|-------------|----------------|
| `np_outer(a, b)` | Outer product of two 1D arrays | Lisp: reshape + broadcast multiply |
| `np_kron(A, B)` | Kronecker product | Lisp: block construction |
| `np_tile(A, reps)` | Repeat array | Lisp: allocate + copy |
| `np_flip(A)` / `np_flipud` / `np_fliplr` | Reverse along axis | Lisp: index reversal |
| `np_unique(A)` | Unique sorted values | Lisp: sort + dedup |
| `np_searchsorted(A, v)` | Binary search in sorted array | Lisp: `position` with bounds |
| `np_allclose(A, B, tol)` | Element-wise approximate equality | Lisp: `np_abs(np_sub(A,B))` + threshold |

---

## Tier 2: Numeric Recipes

Slightly higher-level functions that combine multiple ndarray operations into
a reusable workflow. These are the patterns that keep appearing across demo
notebooks. Still general-purpose — not domain-specific.

### 2.1 Convergence analysis

**Gap:** The beam-deflection notebook implements convergence rate estimation
by hand (log-log slope computation). This is a universal pattern in numerical
methods.

```maxima
/* Proposed: */
[rate, log_h, log_err] : np_convergence_rate(h_list, err_list)$
/* Returns estimated convergence order and log-scale vectors for plotting */
```

| Function | Description |
|----------|-------------|
| `np_convergence_rate(h, err)` | Estimate convergence order from refinement study |
| `np_richardson_extrapolate(h, val, order)` | Richardson extrapolation to improve accuracy |

### 2.2 Finite difference operators

**Gap:** Multiple notebooks build FD stencils manually. Standard stencils
for first and second derivatives on uniform grids are well-known.

```maxima
/* Proposed: */
D1 : np_fd_matrix(n, 1, dx)$    /* n×n first-derivative matrix, spacing dx */
D2 : np_fd_matrix(n, 2, dx)$    /* n×n second-derivative matrix */
```

| Function | Description |
|----------|-------------|
| `np_fd_matrix(n, order, dx)` | Sparse-ish FD differentiation matrix (tridiagonal/pentadiagonal) |
| `np_fd_gradient(f, x, h)` | Finite-difference gradient of a compiled function at a point |

Note: `np_fd_matrix` returns a dense ndarray (we have no sparse type). For
moderate `n` (< ~2000) this is fine. For larger problems, this motivates the
banded solver discussed in the architecture roadmap.

### 2.3 Symbolic-to-numeric compilation

**Gap:** `np_compile_gradient` exists but the pattern generalises. Users
frequently need to compile a symbolic expression into a fast numeric function
for evaluation over a grid or inside a loop. Currently they use
`compile(f) := buildq([f], lambda([x], f))` patterns or fall back to the
Maxima evaluator.

```maxima
/* Proposed: */
f_num : np_compile(sin(x)*exp(-x^2), [x])$
/* f_num is a compiled closure: f_num(0.5) → 0.3678... */

ys : np_eval(sin(x)*exp(-x^2), [x], xs)$
/* Evaluate symbolic expression over an ndarray, returning an ndarray */
```

| Function | Description |
|----------|-------------|
| `np_compile(expr, vars)` | Compile symbolic expression to numeric closure via `coerce-float-fun` |
| `np_eval(expr, vars, values)` | Evaluate symbolic expression element-wise over ndarray(s) |
| `np_compile_hessian(expr, vars)` | Compile expression + gradient + Hessian |
| `np_compile_jacobian(exprs, vars)` | Compile vector expression + Jacobian matrix |

`np_eval` is the key function here. It bridges the symbolic and numeric worlds
in a single call: write a formula in Maxima's symbolic language, evaluate it
over an ndarray grid, get an ndarray back. No manual compilation, no
element-wise loops, no `makelist`.

This is the function that most directly addresses the "symbolic CAS with
numeric muscle" value proposition. NumPy users write `np.sin(x) * np.exp(-x**2)`
and get vectorised evaluation. Maxima users should be able to write the same
formula in natural mathematical notation and get the same performance.

### 2.4 Interpolation

**Gap:** No interpolation functions exist. Cubic spline interpolation is a
tridiagonal system — solvable with existing `np_solve`.

| Function | Description |
|----------|-------------|
| `np_interp(xp, x, y)` | Piecewise linear interpolation (listed above in Tier 1) |
| `np_cubic_spline(x, y)` | Fit natural cubic spline, return coefficient object |
| `np_spline_eval(spline, xp)` | Evaluate fitted spline at new points |

The spline object would be a Maxima list of coefficient arrays — no new type
needed. The tridiagonal solve is O(n) with the Thomas algorithm, which could
be a specialised Lisp function or simply use `np_solve` for moderate n.

### 2.5 Quadrature

**Gap:** `np_trapz` exists for trapezoidal integration. Simpson's rule and
adaptive quadrature are natural extensions.

| Function | Description |
|----------|-------------|
| `np_simps(y, x)` | Simpson's rule over sampled data |
| `np_quad(expr, var, a, b)` | Adaptive quadrature of symbolic expression (wraps Maxima's `quad_qags`) |

`np_quad` would bridge Maxima's existing `quad_qags` (which wraps QUADPACK)
with the ndarray ecosystem — compiling the integrand via `coerce-float-fun`
for speed.

---

## What We Are Not Building

This section complements the exclusions in architecture.md.

- **Domain-specific toolboxes**: No "control systems toolbox", "signal
  processing toolbox", or "statistics toolbox". We provide `np_fft`, not
  `bandpass_filter`. We provide `np_eig`, not `bode_plot`. The demo notebooks
  show that domain workflows can be built naturally from utilities + Maxima's
  symbolic engine.

- **DataFrame / data wrangling**: The Arrow table type provides basic columnar
  storage, but we are not building pandas. Data manipulation (joins, groupby,
  pivot) is better handled by feeding data through DuckDB or polars via Arrow
  zero-copy.

- **Plotting**: The ax-plots package handles visualisation. Numerics provides
  data; plotting consumes it. No overlap.

- **Symbolic computation**: Maxima's symbolic engine is the foundation we
  build on, not something we extend. We compile symbolic results to numeric
  form; we do not add new symbolic algorithms.

---

## Prioritisation

Ranked by impact-to-effort ratio, informed by which gaps the demo notebooks
hit most frequently:

| Priority | Item | Effort | Impact | Rationale |
|----------|------|--------|--------|-----------|
| **1** | `np_eval` (2.3) | Medium | High | Eliminates the most common pain point: getting from symbolic expression to numeric array |
| **2** | `np_meshgrid` (1.1) | Low | High | Unlocks 2D/3D surface evaluation, contour plots |
| **3** | Signal helpers (1.3) | Low | Medium | Spectral analysis is a common demo category; all are thin wrappers |
| **4** | `np_diff`, `np_gradient` (1.2) | Low | Medium | Fundamental numeric operations, used across many domains |
| **5** | `np_cov`, `np_corrcoef` (1.4) | Low | Medium | PCA and statistics notebooks need these |
| **6** | FD matrices (2.2) | Medium | Medium | Beam-deflection and PDE notebooks |
| **7** | Interpolation (2.4) | Medium | Medium | No current solution; spline is a natural fit |
| **8** | `np_compile_hessian/jacobian` (2.3) | Medium | Medium | Extends the symbolic bridge pattern |
| **9** | Convergence helpers (2.1) | Low | Low | Niche but clean |
| **10** | `np_outer`, `np_kron` (1.5) | Low | Low | Useful but less frequently needed |

---

## Implementation Strategy

### Language choice

- **Lisp** for anything performance-sensitive or that touches ndarray internals
  (meshgrid, diff, gradient, signal helpers, eval, compile variants).
- **Maxima** (.mac) for pure-arithmetic recipes that compose existing functions
  (convergence rate, Richardson extrapolation, FD matrix construction).

### Testing

Each function gets a test file in `tests/` following the existing `batch(file, test)`
pattern. Cross-validation tests verify compositions (e.g., `np_diff(np_cumsum(A))`
should recover `A` up to the first element).

### Documentation

Each function gets a docstring-style entry in `doc/`. The demo notebooks in
maxima-demos serve as integration-level documentation — after adding a utility,
update the relevant notebook to use it.

### Versioning

These additions are backwards-compatible (new functions only, no API changes).
They can be released incrementally — each Tier 1 function is independently
useful.

---

## Relationship to Architecture Roadmap

The [architecture roadmap](architecture.md) covers:
- **Solver wrapping** (ODEPACK DLSODA/DLSODAR/DLSODI/DLSODES)
- **Sparse / banded matrices** (longer-term)
- **SUNDIALS IDA** (longer-term)

This document covers the **utility layer** that sits above those solvers:
- Grid construction and evaluation helpers
- Signal processing building blocks
- Statistical aggregations
- Symbolic→numeric bridge extensions
- Numeric recipes (convergence, FD, interpolation)

The two roadmaps are complementary. Solver wrapping provides the heavy
computational infrastructure; the standard library provides the ergonomic layer
that makes everyday use pleasant. Both are needed for Maxima to be competitive
with NumPy/SciPy for routine numerical work.
