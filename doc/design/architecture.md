# Numerical Computing in Maxima: Architecture

## Overview

The `numerics` package brings NumPy-like numerical computing to Maxima. The
central idea is an **opaque handle type** that keeps data in contiguous foreign
memory across operations, avoiding the marshalling overhead that has historically
made Maxima's numerical performance impractical.

| Layer | Library | Role |
|-------|---------|------|
| 1 | magicl | Dense numerics, linear algebra (BLAS/LAPACK) |
| 2 | Arrow | Zero-copy columnar interop via C Data Interface |

Target: **SBCL only**. magicl and CFFI require SBCL (or CCL); GCL is explicitly
unsupported. This matches the direction the Maxima community is heading — a 2022
mailing list thread was titled "Trying to accommodate GCL is keeping us from
making progress."

---

## Prior Art in Maxima

### What exists

- **LAPACK via f2cl**: Maxima ships LAPACK/BLAS translated from Fortran to Common
  Lisp via `f2cl` (`share/lapack/`). Covers `dgeev`, `dgesvd`, `dgesv`, `dgeqrf`,
  `dgemm`, plus complex variants.
- **Marshalling functions**: `lapack-lispify-matrix` / `lapack-maxify-matrix`
  convert between Maxima nested-list matrices and column-major `double-float`
  arrays — but pay the conversion cost on every call.
- **AMatrix**: `share/amatrix/` provides a struct-based matrix with submatrix views.
- **FFT**: `share/fftpack5/` uses a converter-factory pattern.
- **Custom display**: `displa-def` + `dim-$typename` for custom rendering
  (see `inference_result` in `share/stats/`, `amatrix` in `share/amatrix/`).

### The marshalling problem

Raymond Toy (2014, 2020): "We still have to marshall all of maxima's arrays and
I think the conversion overhead would eat any gains from using native BLAS."

This is the central problem our design solves. The **opaque handle type** keeps
data in foreign memory across operations; only explicit `np_to_matrix()` or
`np_to_list()` materializes back to Maxima form. A chain like
`np_inv(np_matmul(A, B))` stays in foreign memory the whole time.

---

## Package Structure

```
maxima-numerics/
├── manifest.toml                    # mxpm package metadata
├── numerics.mac                     # Maxima entry point
├── rtest_numerics.mac               # Test runner (batches individual test files)
├── tests/                           # Individual test files (batch-test format)
├── doc/                             # Function reference docs
│   └── design/
│       └── architecture.md          # This file
│
├── lisp/                            # Common Lisp implementation
│   ├── packages.lisp                # #:numerics package
│   ├── numerics.asd                 # ASDF system definitions
│   │
│   ├── core/                        # numerics/core: magicl wrapper
│   │   ├── util.lisp                # Dtype helpers, number conversion
│   │   ├── handle.lisp              # ndarray struct + GC finalizers
│   │   ├── display.lisp             # Custom Maxima display (console + TeX)
│   │   ├── convert.lisp             # Maxima matrix/list <-> ndarray
│   │   ├── constructors.lisp        # zeros, ones, eye, rand, etc.
│   │   ├── linalg.lisp              # inv, det, svd, eig, solve, etc.
│   │   ├── elementwise.lisp         # +, -, *, /, exp, log, sin, etc.
│   │   ├── slicing.lisp             # Indexing, slicing, reshaping
│   │   ├── signal.lisp              # FFT, IFFT, convolution (wraps fftpack5)
│   │   └── aggregation.lisp         # sum, mean, min, max, std, etc.
│   │
│   ├── optimize/                    # numerics/optimize: L-BFGS wrapper
│   │   └── optimize.lisp            # np_minimize (wraps Maxima's lbfgs)
│   │
│   ├── image/                       # numerics/image: image I/O
│   │   └── image.lisp               # np_read_image, np_mandrill (wraps opticl)
│   │
│   └── arrow/                       # Layer 2: Arrow integration
│       ├── schema.lisp              # ArrowSchema CFFI struct
│       ├── array.lisp               # ArrowArray CFFI struct
│       ├── table.lisp               # Table type (named columns)
│       ├── bridge.lisp              # ndarray <-> Arrow zero-copy
│       └── io.lisp                  # CSV I/O
│
└── .github/workflows/               # CI
```

Four ASDF systems are defined:
- **`numerics/core`** — core ndarray + linear algebra + signal processing
  (depends on magicl, trivial-garbage, alexandria)
- **`numerics/optimize`** — L-BFGS optimizer (wraps Maxima's `lbfgs` package)
- **`numerics/image`** — image I/O (depends on opticl via Quicklisp)
- **`numerics`** — full system including Arrow (adds cffi, static-vectors)

`numerics.mac` loads `numerics/core` by default. `numerics/optimize` is loaded
separately via `numerics-optimize-loader.lisp`. The full system can be loaded
explicitly via `:lisp (asdf:load-system "numerics")`.

---

## Core Design: The Handle Type

### Why an opaque handle

Maxima's internal representation is nested S-expressions — ideal for symbolic
computation but terrible for numerical performance. Every LAPACK call via
Maxima's existing `share/lapack/` has to:

1. Walk the nested list, coerce each element to `double-float`
2. Pack into a contiguous column-major array
3. Call LAPACK
4. Unpack results back into nested lists

The ndarray handle eliminates steps 1-2 and 4 for all intermediate operations.
Data enters foreign memory once (via `ndarray()`) and leaves once (via
`np_to_matrix()` or `np_to_list()`).

### How it works

An `ndarray` is a Lisp struct wrapping a `magicl:tensor`. At the Maxima level it
appears as `(($ndarray simp) <struct>)`, where the `simp` flag tells the
simplifier to leave it alone and the struct is opaque to Maxima.

Every `$np_*` function calls `numerics-unwrap` on inputs and `numerics-wrap` on
outputs — this is the single chokepoint for type safety.

### Memory management

When an ndarray handle becomes unreachable, `trivial-garbage:finalize` fires a
cleanup closure. The closure captures the **tensor** (not the handle) to avoid
preventing GC of the handle itself. magicl's own finalizer on the underlying
static-vector then frees the foreign memory.

For Arrow-imported data, the finalizer calls the Arrow `release` callback
instead, handing lifetime control back to the Arrow producer.

### Display

Small 2D arrays (≤6×6) display their contents as a matrix with a header line.
Larger arrays show a compact summary with shape and dtype. TeX output is also
supported for notebook rendering. The implementation follows the `displa-def` +
`dim-$typename` pattern used by `inference_result` and `amatrix` in Maxima's
standard library.

### Dtype support

Two element types are supported: `double-float` and `complex-double-float`.
Constructors accept an optional `complex` argument (e.g., `np_zeros([3,3], complex)`).
Complex numbers cross the Maxima/Lisp boundary as `a + b*%i` ↔ `#C(a b)`.

Helper functions in `util.lisp` handle all dtype logic:
- `numerics-element-type` / `numerics-result-dtype` — dtype dispatch
- `maxima-to-lisp-number` — safe Maxima→CL conversion (never drops imaginary parts, unlike `$float`)
- `numerics-require-real` — guard for operations that reject complex input (comparisons, sorting)
- `numerics-parse-dtype` — parse the optional `complex` keyword

---

## API Surface

### Conversion

| Function | Description |
|---|---|
| `ndarray(matrix)` | Maxima matrix → ndarray |
| `ndarray(matrix, complex)` | Maxima matrix → complex ndarray |
| `ndarray(list, shape)` | Maxima list → ndarray with given shape |
| `np_to_matrix(A)` | ndarray → Maxima matrix |
| `np_to_list(A)` | ndarray → flat Maxima list |

### Constructors

| Function | Description |
|---|---|
| `np_zeros(shape)` | Zero-filled (optional dtype) |
| `np_ones(shape)` | Ones-filled (optional dtype) |
| `np_eye(n)` | Identity matrix (optional dtype) |
| `np_full(shape, val)` | Constant-filled (optional dtype) |
| `np_empty(shape)` | Uninitialized (optional dtype) |
| `np_diag(list)` | Diagonal matrix (optional dtype) |
| `np_rand(shape)` | Uniform random [0,1) — always real |
| `np_randn(shape)` | Standard normal — always real |
| `np_arange(n)` | 0..n-1 as 1D — always real |
| `np_linspace(a,b,n)` | Evenly spaced points — always real |
| `np_copy(A)` | Deep copy (preserves dtype) |

### Linear algebra

| Function | Description |
|---|---|
| `np_matmul(A, B)` | Matrix multiplication (BLAS dgemm/zgemm) |
| `np_inv(A)` | Matrix inverse |
| `np_det(A)` | Determinant (returns scalar) |
| `np_solve(A, b)` | Solve Ax = b |
| `np_svd(A)` | SVD → `[U, S, Vt]` |
| `np_eig(A)` | Eigendecomposition → `[eigenvalues, eigenvectors]` |
| `np_qr(A)` | QR decomposition → `[Q, R]` |
| `np_lu(A)` | LU decomposition → `[L, U, P]` |
| `np_cholesky(A)` | Cholesky decomposition |
| `np_norm(A)` | Matrix/vector norm |
| `np_rank(A)` | Numerical rank |
| `np_trace(A)` | Matrix trace (returns scalar) |
| `np_transpose(A)` | Transpose |
| `np_ctranspose(A)` | Conjugate transpose (Hermitian adjoint) |
| `np_conj(A)` | Element-wise conjugate |
| `np_real(A)` / `np_imag(A)` | Extract real/imaginary parts |
| `np_angle(A)` | Element-wise phase angle |
| `np_expm(A)` | Matrix exponential |
| `np_lstsq(A, b)` | Least-squares solution → `[x, residuals, rank, S]` |
| `np_pinv(A)` | Moore-Penrose pseudo-inverse |

#### magicl delegation

Most linalg functions delegate directly to magicl (`magicl:@`, `magicl:inv`,
`magicl:det`, `magicl:svd`, `magicl:eig`, `magicl:qr`, `magicl:lu`,
`magicl:linear-solve`, `magicl:transpose`, `magicl:conjugate-transpose`,
`magicl:trace`). A few are implemented in pure Lisp because magicl either lacks
the function or requires an extension with additional native dependencies:

- **`np_expm`**: Pade approximation with scaling-and-squaring (Higham 2005).
  magicl provides `magicl:expm` but only via `magicl/ext-expokit`, which requires
  gfortran to compile the Expokit Fortran source. Our pure-Lisp implementation
  avoids this build dependency.
- **`np_lstsq`**: SVD-based least-squares solver. No magicl equivalent.
- **`np_pinv`**: SVD-based Moore-Penrose pseudo-inverse. No magicl equivalent.
- **`np_norm`**: 1/2/inf/Frobenius norms for both vectors and matrices. magicl's
  `magicl:norm` handles vector p-norms only, so we implement matrix norms ourselves.
- **`np_rank`**: Singular value thresholding. No magicl equivalent.

### Element-wise operations

| Function | Description |
|---|---|
| `np_add`, `np_sub`, `np_mul`, `np_div` | Arithmetic (ndarray or scalar) |
| `np_pow`, `np_sqrt`, `np_exp`, `np_log` | Power/exponential |
| `np_sin`, `np_cos`, `np_tan` | Trigonometric |
| `np_abs`, `np_neg` | Absolute value, negation |
| `np_scale(alpha, A)` | Scalar multiplication |
| `np_greater`, `np_less`, etc. | Comparisons (real only) |
| `np_equal`, `np_not_equal` | Equality (supports complex) |
| `np_logical_and`, `np_logical_or`, `np_logical_not` | Logical |
| `np_where(cond, A, B)` | Conditional selection |
| `np_map(f, A)` | Apply named function element-wise |
| `np_extract(cond, A)` | Select elements where condition is nonzero |

Element-wise ops use two internal helpers: `numerics-binary-op` dispatches
ndarray×ndarray, ndarray×scalar, and scalar×ndarray cases via magicl's `.+`,
`.*`, etc. `numerics-unary-op` applies a function via `magicl:map!` on a deep
copy. `np_abs` is special-cased: for complex input it returns magnitudes as
`double-float` (matching NumPy behaviour).

### Slicing and indexing

| Function | Description |
|---|---|
| `np_ref(A, i, j)` | Single element (0-indexed) |
| `np_set(A, i, j, val)` | Set element (mutating) |
| `np_row(A, i)` / `np_col(A, j)` | Extract row/column as 1D ndarray |
| `np_slice(A, rows, cols)` | Sub-matrix (half-open ranges, negative indices) |
| `np_reshape(A, shape)` | Reshape (total size must match) |
| `np_flatten(A)` | Flatten to 1D |
| `np_hstack(A, B)` / `np_vstack(A, B)` | Concatenation (promotes to complex if mixed) |
| `np_shape(A)` | Shape as Maxima list |
| `np_size(A)` | Total element count |
| `np_dtype(A)` | Element type string |

### Aggregation

| Function | Description |
|---|---|
| `np_sum(A)` / `np_sum(A, axis)` | Sum (total or along axis) |
| `np_mean(A)` / `np_mean(A, axis)` | Mean |
| `np_min(A)` / `np_max(A)` | Min/max (real only) |
| `np_argmin(A)` / `np_argmax(A)` | Index of min/max (real only) |
| `np_std(A)` / `np_var(A)` | Standard deviation / variance (result always real) |
| `np_cumsum(A)` | Cumulative sum |
| `np_dot(a, b)` | Dot product (1D vectors) |
| `np_sort(A)` / `np_argsort(A)` | Sort / sort indices (real only) |

### Signal processing

| Function | Description |
|---|---|
| `np_fft(A)` | Complex FFT (wraps fftpack5) |
| `np_ifft(A)` | Inverse complex FFT |
| `np_fft2d(A)` | 2D FFT (separable: columns then rows) |
| `np_ifft2d(A)` | Inverse 2D FFT |
| `np_convolve(A, B)` | 1D convolution (direct, O(n·k)) |
| `np_convolve2d(A, K)` | 2D convolution (direct) |
| `np_trapz(A)` / `np_trapz(A, X)` | Trapezoidal integration |

### Optimization

| Function | Description |
|---|---|
| `np_minimize(f, grad, x0)` | Unconstrained minimization (wraps Maxima's L-BFGS) |

`np_minimize` takes Maxima lambda functions for the objective and gradient,
with ndarray arguments and return values. It manages the L-BFGS work arrays
internally and returns `[x_opt, f_opt, converged]`.

---

## Layer 2: Arrow Integration

### Why Arrow

Arrow is not an alternative to magicl — it's a **memory format** for zero-copy
interop with other tools (polars, DuckDB, Python, R). The key use cases:

1. Exchange columnar data with external processes
2. Read/write CSV via Arrow-native I/O
3. Receive query results from analytical engines without copying

### Approach

Rather than linking to the large `libarrow` C++ library, we implement just the
two C structs from the Arrow C Data Interface (`ArrowSchema` and `ArrowArray`).
This minimizes native dependencies while remaining compatible with any Arrow
producer/consumer.

Release callbacks manage lifetime: when an Arrow-imported ndarray is GC'd, the
release callback fires, handing memory ownership back to the original producer.

### Table type

A `table` is a list of named 1D ndarray columns — the columnar equivalent of a
dataframe. The Maxima API provides column access, shape inspection, and
conversion to/from 2D ndarrays.

### Zero-copy data flow

```
                    ZERO COPY                    ZERO COPY
  Arrow column  ─────────────> ndarray (magicl)  ─────────────> BLAS/LAPACK
  (C heap ptr)                 (static-vector)                  operates on
                                                                contiguous buffer
```

Both magicl (with `static-vectors`) and Arrow Float64 columns store data as
contiguous `double-float` buffers. The bridge exploits this: `static-vector-pointer`
gives the raw C pointer without copying. Only `float64` gets zero-copy; other
Arrow types are copied/converted.

**Where copies are unavoidable:**

1. **Maxima matrix → ndarray**: Maxima matrices are nested lists of
   arbitrary-precision numbers; coercion to `double-float` and contiguous layout
   is unavoidable. One-time boundary cost.
2. **ndarray → Maxima matrix**: The user explicitly opts into materialization.
3. **magicl operations**: Functions like `magicl:inv` allocate new storage for
   results — intrinsic to the computation. The key point is that chaining
   operations never round-trips through Maxima.

---

## Error Handling

All errors surface to Maxima via `merror`, catchable with `errcatch()`:

```maxima
result : errcatch(np_inv(singular_matrix));
if result = [] then print("Matrix was singular");
```

Error categories:
- **Type errors**: `numerics-unwrap` rejects non-ndarray inputs
- **Dimension mismatch**: checked before BLAS calls
- **Singular matrix**: caught from magicl via `handler-case`
- **Non-numeric data**: caught during Maxima→ndarray conversion
- **Complex where real required**: `numerics-require-real` guard

---

## Testing

Tests use Maxima's `batch(file, test)` format — alternating input/expected-output
pairs. Individual test files live in `tests/` and share helpers from
`tests/test_helpers.mac`. The test runner `rtest_numerics.mac` batches all files.

Test files are listed in `manifest.toml` for `mxpm test` discovery:

| File | Covers |
|------|--------|
| `test_constructors.mac` | All constructor functions |
| `test_linalg.mac` | Linear algebra operations |
| `test_elementwise.mac` | Element-wise ops, comparisons, logical, map/where/extract |
| `test_slicing.mac` | Indexing, slicing, reshape, concat, shape/size/dtype |
| `test_aggregation.mac` | Sum, mean, min, max, std, var, sort, cumsum, dot |
| `test_complex.mac` | Complex dtype across all function categories |
| `test_crossval.mac` | Cross-validation: SVD reconstruction, QR orthogonality, etc. |

Run all tests: `maxima --very-quiet -b rtest_numerics.mac`

---

## Build and Dependencies

### System requirements

1. **SBCL** (>= 2.0)
2. **Quicklisp** — for magicl and transitive CL dependencies
3. **BLAS/LAPACK** — macOS: Accelerate.framework (bundled); Linux: `liblapack-dev libblas-dev`

### Loading mechanism

When `load("numerics")` runs in Maxima:

1. `(require "asdf")` — ensures ASDF is available
2. Pushes the package's `lisp/` directory onto `asdf:*central-registry*`
3. `(asdf:load-system "numerics/core")` — Quicklisp's ASDF integration
   auto-resolves dependencies from `~/quicklisp/`

---

## Interoperability with Maxima Packages

Maxima ships several numerical packages in `share/`. They all follow the same
data flow: Maxima expression → `coerce-float-fun` / `$float` → Lisp
`double-float` arrays → solver → Maxima lists/matrices. The conversion cost is
paid on every call. The ndarray handle sits naturally at the "Lisp double-float
arrays" stage, so interop can bypass marshalling in both directions.

### Integration approach

The bundled packages live in Maxima's `share/` tree and are maintained upstream.
Modifying them would require either contributing changes upstream or maintaining
a fork — neither is practical for an external package. All interop must therefore
be non-invasive: numerics wraps the bundled packages from the outside, without
modifying their source.

This is feasible because the packages that matter most expose their internals at
the right level:

- **FFTPACK5**: Exports `cfft`, `inverse-cfft`, `rfft`, `inverse-rfft` which
  accept raw CL arrays. We call these directly with `magicl:storage`.
- **ODEPACK**: Exposes `dlsode_init` / `dlsode_step` for stepping and `dlsode`
  for batch integration. We collect step outputs into ndarrays.
- **COBYLA**: Exposes `fmin_cobyla` which accepts Maxima expressions and
  constraint lists. We convert ndarray initial guesses at the boundary.
- **MINPACK**: Exposes `minpack_solve` (root finding) and `minpack_lsquares`
  (nonlinear least squares). Both accept Maxima expressions and initial guesses.
- **LBFGS**: Already wrapped by `np_minimize`.
- **DISTRIB**: The RNG and distribution functions are callable from Lisp. We
  write new vectorized wrappers that loop internally.
- **NUMERICALIO**: We implement our own file I/O rather than wrapping theirs.
  Binary `double-float` I/O is trivial; CSV parsing is independent.

### Wrapped packages

#### FFTPACK5 (wrapped)

FFTPACK5 (`share/fftpack5/`) uses `(simple-array (complex double-float) (*))`
internally — exactly our complex ndarray storage format. `np_fft` / `np_ifft`
extract `magicl:storage` from the ndarray, call the fftpack5 functions, and
wrap the result — no modification to `share/fftpack5/` required. 2D transforms
are implemented as separable 1D transforms (columns then rows).

#### LBFGS (wrapped)

`np_minimize` wraps Maxima's `lbfgs` package. It manages L-BFGS work arrays
internally, shuttles data between ndarrays and the solver's raw CL arrays, and
returns `[x_opt, f_opt, converged]`. The user provides Maxima lambda functions
for the objective and gradient that operate on ndarrays.

#### LAPACK (superseded)

Our magicl wrapper calls the same BLAS/LAPACK routines that `share/lapack/`
wraps via f2cl, but without per-call marshalling. There is no need to interop;
our API replaces theirs. The one feature `share/lapack/` exposes that we don't
is `$dgemm`'s full `alpha*A*B + beta*C` interface — a niche use case.

### Unwrapped packages (roadmap candidates)

#### ODEPACK — numeric ODE solver

`share/odepack/` provides `dlsode`, an adaptive ODE solver with both stiff
(BDF) and non-stiff (Adams) methods. The Maxima interface exposes:

- `dlsode_init(f, vars, mf)` — initialize solver state. `mf` selects the
  method: `10` (Adams, no Jacobian), `21` (BDF, user Jacobian), `22` (BDF,
  internal Jacobian).
- `dlsode_step(y, t, tout, rtol, atol, istate, state)` — step to `tout`.
- `dlsode(f, vars, init_y, trange, rtol, atol, mf)` — batch integration.

The solver compiles Maxima expressions via `coerce-float-fun` for the inner
loop. The bottleneck is expression evaluation, not data format. An ndarray
wrapper would collect trajectory output into a 2D ndarray (rows = timesteps,
columns = state variables) for natural downstream use in plotting, spectral
analysis, and parameter fitting.

Note: ODEPACK also includes LSODA (automatic stiff/non-stiff switching) but
only DLSODE is exposed in the Maxima interface.

#### COBYLA — constrained optimization

`share/cobyla/` provides `fmin_cobyla`, a derivative-free constrained optimizer
using linear approximation. The Maxima interface:

- `fmin_cobyla(f, vars, init, constraints, ...)` — minimize `f` subject to
  constraints of the form `g1 >= g2`, `g1 <= g2`, or `g1 = g2`.
- Returns `[solution, min_value, n_evals, return_code]`.

This fills a major gap: `np_minimize` only handles unconstrained problems.
COBYLA supports general nonlinear inequality and equality constraints without
requiring gradients. The tradeoff is slower convergence than gradient-based
methods for smooth problems.

#### MINPACK — nonlinear equations and least squares

`share/minpack/` provides two solvers:

- `minpack_solve(fcns, vars, init)` — solve a system of n nonlinear equations
  in n unknowns (wraps HYBRD1/HYBRJ1).
- `minpack_lsquares(fcns, vars, init)` — nonlinear least squares (wraps
  LMDIF1/LMDER1, Levenberg-Marquardt).

Both can compute Jacobians automatically via symbolic differentiation or use
finite differences. This is the Maxima equivalent of SciPy's `fsolve` and
`least_squares`.

#### MNEWTON — Newton's method for nonlinear systems

`share/mnewton/` provides `mnewton(fcns, vars, guess)` — a pure-Maxima
implementation of Newton's method for nonlinear systems. Automatically computes
the Jacobian via symbolic differentiation. Simpler than MINPACK but less robust
(no trust region, no fallback to finite differences on Jacobian failure).

#### DISTRIB — probability distributions

`share/distrib/` provides scalar PDF, CDF, quantile, random variate, and MLE
functions for ~20 distributions (normal, t, chi-squared, F, exponential, gamma,
beta, Poisson, binomial, etc.). Vectorized ndarray versions would be valuable
for Monte Carlo workflows — `np_random_normal(mu, sigma, shape)` returning an
ndarray, or element-wise PDF/CDF evaluation over ndarrays.

#### NUMERICALIO — direct file loading

`share/numericalio/` reads CSV and binary files into Maxima matrices. Binary
format uses IEEE 754 `double-float` — the same storage as ndarray — so direct
loading via `read-sequence` into ndarray storage is possible with zero parsing
overhead. CSV loading could similarly bypass Maxima matrix construction.

### Performance expectations

The ndarray type eliminates marshalling overhead — but that only matters when
marshalling is the bottleneck. The packages above fall into two categories:

**Data-dominated** (FFTPACK5, NUMERICALIO, LAPACK): The solver operates on raw
arrays and the cost is proportional to data size. ndarray interop eliminates
O(n) marshalling per call, which is significant for large arrays and repeated
operations. FFTPACK5 in particular benefits because FFT workflows typically
chain multiple transforms with windowing and filtering between them.

**Evaluation-dominated** (ODEPACK, MINPACK, COBYLA, LBFGS): The inner loop
evaluates Maxima expressions compiled via `coerce-float-fun`. Each solver
iteration invokes a Lisp closure that evaluates the user's formula — the cost
is in the expression evaluation, not in moving data. For these packages, ndarray
wrappers improve ergonomics (collect results into a 2D ndarray, pass ndarray
initial conditions) but don't change the computational complexity. The
marshalling overhead at the boundary is O(n) for n state variables, which is
typically small (tens, not millions).

In short: ndarray interop is a performance win where the data is large and the
solver touches it directly (FFT, file I/O, BLAS). Where the solver's inner loop
evaluates symbolic expressions, the benefit is API convenience, not speed.

---

## Roadmap

The following features are prioritized based on the gap analysis from the
maxima-demos notebooks. The guiding constraint is to **wrap existing Maxima
built-in packages and Quicklisp libraries** rather than implementing algorithms
from scratch in Lisp.

### Tier 1: Wrap existing Maxima packages

These follow the same wrapping pattern already established by `np_minimize`
(wrapping `lbfgs`) and `np_fft` (wrapping `fftpack5`). The algorithms exist;
the work is shuttling data between ndarrays and the solver interfaces.

#### 1. Numeric ODE solver — wrap ODEPACK

Wrap `share/odepack/` as `np_odeint(f, y0, tspan)` or similar. DLSODE supports
both stiff (BDF, `mf=22`) and non-stiff (Adams, `mf=10`) methods with adaptive
step control.

**Motivation:** The demos currently rely on either `ode2()` (which only works
when a closed-form solution exists) or manual matrix-exponential time-stepping
(which only works for linear systems). A numeric integrator unlocks nonlinear
dynamics, stiff systems, and removes the requirement that `ode2()` succeed
symbolically. The inverse-problems and control-tuning notebooks would become
dramatically simpler.

**Design considerations:**
- Collect trajectory output as a 2D ndarray (rows = timesteps, columns = state
  variables) for natural use in plotting and downstream analysis.
- The user-supplied RHS function will be evaluated via `coerce-float-fun` in
  the inner loop — expression evaluation is the bottleneck, not data format.
- Consider whether to expose the step-level API (`dlsode_step`) for event
  detection, or just the batch API (`dlsode`) for simplicity.

#### 2. Constrained optimization — wrap COBYLA

Wrap `share/cobyla/` as `np_minimize_cobyla(f, x0, constraints)`.

**Motivation:** `np_minimize` only handles unconstrained problems. Many
practical engineering problems have bounds or constraints (e.g., "find optimal
PID gains where Kp > 0"). COBYLA is derivative-free and supports general
nonlinear inequality/equality constraints.

**Design considerations:**
- Constraint format: accept Maxima expressions like `[x1 >= 0, x2 + x3 <= 10]`
  and translate to COBYLA's internal representation.
- Return format: match `np_minimize` conventions where possible.
- COBYLA converges slower than gradient-based methods; document this tradeoff.

#### 3. Root finding — wrap MINPACK

Wrap `share/minpack/` as `np_fsolve(fcns, vars, x0)` for nonlinear systems of
equations, and `np_lsq_nonlinear(fcns, vars, x0)` for nonlinear least squares
(Levenberg-Marquardt).

**Motivation:** Maxima has `solve()` for symbolic solutions and `find_root()`
for scalar equations, but no numeric solver for systems of nonlinear equations
where symbolic solving fails. MINPACK's HYBRD1 (root finding) and LMDER1
(Levenberg-Marquardt) fill this gap — they are the same algorithms behind
SciPy's `fsolve` and `least_squares`.

**Design considerations:**
- MINPACK can compute Jacobians via symbolic differentiation or fall back to
  finite differences. Expose this as an option.
- Levenberg-Marquardt is better than L-BFGS for nonlinear least-squares
  problems (curve fitting, inverse problems) because it exploits the
  sum-of-squares structure.

### Tier 2: Maxima-level features (no new Lisp algorithms)

These can be implemented primarily in Maxima code using existing ndarray
operations and Maxima built-ins, without writing new numerical algorithms in
Lisp.

#### 4. Symbolic-to-numeric gradient bridge

Provide a convenience function that takes a symbolic loss expression,
differentiates it with `diff()`, and produces numeric callback functions
suitable for `np_minimize`. Something like:

```maxima
[f_num, grad_num] : np_compile_gradient(loss_expr, [w1, w2, w3]);
result : np_minimize(f_num, grad_num, x0);
```

**Motivation:** The ML and optimization notebooks manually implement this
pattern (symbolic `diff()` → hand-coded numeric gradient loop). Automating it
leverages Maxima's unique symbolic strength — no other numeric tool can derive
exact gradients this naturally. The control-tuning notebook currently resorts to
finite-difference gradients because this bridge doesn't exist.

**Design considerations:**
- The compiled function needs to accept and return ndarrays.
- Maxima's `compile()` or `coerce-float-fun` can convert symbolic expressions
  to efficient Lisp closures.
- This is mostly glue code, not algorithmic work.

#### 5. Statistical hypothesis tests

Build t-test, chi-squared test, and ANOVA on top of Maxima's existing `distrib`
package (which provides all the required CDF/quantile functions).

**Motivation:** The statistics notebooks cover PCA, Monte Carlo, and
correlation, but have no hypothesis testing. These tests are arithmetic on top
of existing distribution functions — computing test statistics and looking up
p-values via `cdf_student_t`, `cdf_chi2`, etc.

**Design considerations:**
- Can be implemented entirely in Maxima (.mac) using `distrib` functions.
- Return an inference-result-style object with test statistic, p-value,
  confidence interval, and degrees of freedom.

#### 6. Interpolation and splines

Maxima does not ship an interpolation package. However, cubic spline
interpolation is a tridiagonal linear system — solvable with existing `np_solve`
and pure Maxima coefficient setup. Linear interpolation is trivial.

**Motivation:** SciPy's `interp1d` and `CubicSpline` are used constantly in
practice. The least-squares notebook uses Vandermonde matrices for polynomial
fitting but has no general-purpose interpolation.

**Design considerations:**
- Cubic spline: set up the tridiagonal system in Maxima, solve with `np_solve`.
- For large N, this motivates sparse/banded solvers — but for typical
  interpolation sizes (hundreds to low thousands of points), dense solvers are
  adequate.
- Could also wrap a Quicklisp spline library if one exists.

### Tier 3: Longer-term / higher-effort

#### 7. Sparse matrices

No suitable Quicklisp library exists for sparse linear algebra. magicl focuses
on dense BLAS/LAPACK. Options:

- **CFFI bindings to SuiteSparse**: high performance but significant build
  dependency and wrapping effort.
- **Pure Lisp CSR/COO**: feasible for moderate sizes but defeats the purpose of
  avoiding algorithm implementation.
- **Banded solvers only**: a practical middle ground — tridiagonal and banded
  systems cover FEM and spline use cases without full sparse support. LAPACK's
  `dgbsv` (banded solver) is available via f2cl.

**Motivation:** The beam-deflection FEM notebook constructs a tridiagonal system
as a full dense matrix. For practical FEM beyond ~1000 DOF, sparse or banded
solvers are essential.

**Current recommendation:** Defer full sparse support. Consider banded solvers
(wrapping LAPACK's `dgbsv` via magicl or f2cl) as a targeted solution for FEM
and spline problems.

#### 8. Vectorized distribution sampling

Wrap `share/distrib/` random variate functions with ndarray output:
`np_random_normal(mu, sigma, shape)`, `np_random_uniform(a, b, shape)`, etc.
Currently `np_randn` provides standard normal only; parameterized distributions
require element-wise post-processing.

### What we are not building

The following are explicitly out of scope — they would require implementing
substantial algorithms from scratch or pulling in heavy external dependencies:

- **Deep learning / neural networks**: fundamentally different compute model
  (GPU, autograd, large-scale SGD). Use PyTorch/JAX.
- **Advanced image processing**: morphological operations, feature detection,
  ML-based denoising. Use OpenCV.
- **Wavelet transforms**: no existing Maxima or Quicklisp library.
- **Full sparse linear algebra**: no suitable library to wrap without
  SuiteSparse CFFI (see tier 3 discussion above).
- **Automatic differentiation**: Maxima's symbolic `diff()` plus the
  gradient bridge (tier 2, item 4) covers the primary use case. Tape-based
  AD through numeric code is a different paradigm.

---

## Reference: Key Maxima Files

These files in the Maxima source tree informed the design patterns used here:

| File | Pattern |
|------|---------|
| `share/lapack/eigensys.lisp` | Matrix marshalling (the problem we're solving) |
| `share/stats/inference_result.lisp` | Custom type display via `displa-def` |
| `share/amatrix/amatrix.lisp` | Struct-based matrix with custom display |
| `share/fftpack5/fftpack5-interface.lisp` | Converter-factory pattern |
| `share/cobyla/cobyla-interface.lisp` | Constrained optimizer wrapping pattern |
| `share/minpack/minpack-interface.lisp` | Nonlinear solver wrapping pattern |
| `share/odepack/dlsode-interface.lisp` | ODE solver wrapping pattern |
| `src/numerical/f2cl-lib.lisp` | Column-major indexing conventions |
