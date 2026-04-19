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
│   ├── core/                        # Layer 1: magicl wrapper
│   │   ├── util.lisp                # Dtype helpers, number conversion
│   │   ├── handle.lisp              # ndarray struct + GC finalizers
│   │   ├── display.lisp             # Custom Maxima display (console + TeX)
│   │   ├── convert.lisp             # Maxima matrix/list <-> ndarray
│   │   ├── constructors.lisp        # zeros, ones, eye, rand, etc.
│   │   ├── linalg.lisp              # inv, det, svd, eig, solve, etc.
│   │   ├── elementwise.lisp         # +, -, *, /, exp, log, sin, etc.
│   │   ├── slicing.lisp             # Indexing, slicing, reshaping
│   │   └── aggregation.lisp         # sum, mean, min, max, std, etc.
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

Two ASDF systems are defined:
- **`numerics/core`** — core ndarray operations only (depends on magicl,
  trivial-garbage, alexandria)
- **`numerics`** — full system including Arrow (adds cffi, static-vectors)

`numerics.mac` loads `numerics/core` by default. The full system can be loaded
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

### FFTPACK5 — best candidate

FFTPACK5 (`share/fftpack5/`) uses `(simple-array (complex double-float) (*))`
internally — exactly our complex ndarray storage format. The lower-level
functions (`fftpack5:cfft`, `fftpack5:inverse-cfft`, `fftpack5:rfft`,
`fftpack5:inverse-rfft`) are exported and accept raw CL arrays directly. An
`np_fft` / `np_ifft` pair can extract `magicl:storage` from the ndarray, call
these functions, and wrap the result — no modification to `share/fftpack5/`
required.

This enables natural workflows like windowing, spectral analysis, and filtering
entirely in ndarray space, eliminating two marshalling steps per call.

### LAPACK — already superseded

Our magicl wrapper calls the same BLAS/LAPACK routines that `share/lapack/`
wraps via f2cl, but without per-call marshalling. There is no need to interop;
our API replaces theirs. The one feature `share/lapack/` exposes that we don't
is `$dgemm`'s full `alpha*A*B + beta*C` interface — a niche use case.

### DISTRIB — vectorization opportunity

`share/distrib/` operates on scalars: `pdf_normal(x, mu, sigma)` returns a
single float, and `random_normal(0, 1, 100)` returns a Maxima list. Vectorized
versions (`np_random_normal(mu, sigma, shape)` returning an ndarray, or
element-wise PDF/CDF evaluation over ndarrays) would be valuable but don't
require deep interop — they'd be new functions reusing the same underlying RNG
and distribution math.

### ODEPACK — trajectory output

`share/odepack/` returns ODE solutions step-by-step as Maxima lists. Wrapping
the output as a 2D ndarray (rows = timesteps, columns = state variables) would
make downstream computation (plotting, spectral analysis, parameter fitting)
more natural. The solver's inner loop compiles Maxima expressions via
`coerce-float-fun`, so the bottleneck is expression evaluation, not data format.

### NUMERICALIO — direct file loading

`share/numericalio/` reads CSV and binary files into Maxima matrices. Binary
format uses IEEE 754 `double-float` — the same storage as ndarray — so direct
loading via `read-sequence` into ndarray storage is possible with zero parsing
overhead. CSV loading could similarly bypass Maxima matrix construction.

### MINPACK / COBYLA / LBFGS — limited benefit

The optimization packages (`share/minpack/`, `share/cobyla/`, `share/lbfgs/`)
pass scalar objectives and small parameter vectors. Their bottleneck is
evaluating Maxima expressions in the inner loop, not data format conversion.
Accepting ndarray initial guesses or returning ndarray solutions would be a
minor convenience but not a performance win.

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
- **DISTRIB**: The RNG and distribution functions are callable from Lisp. We
  write new vectorized wrappers that loop internally.
- **NUMERICALIO**: We implement our own file I/O rather than wrapping theirs.
  Binary `double-float` I/O is trivial; CSV parsing is independent.
- **ODEPACK**: We call `dlsode_step` at the Maxima level and collect results
  into an ndarray. The per-step output is a small Maxima list (one per state
  variable), so the marshalling cost is negligible.
- **MINPACK / COBYLA / LBFGS**: We convert ndarray initial guesses to Maxima
  lists at the boundary, call the existing Maxima functions, and convert back.
  Parameter vectors are small, so this is not a bottleneck.

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

## Reference: Key Maxima Files

These files in the Maxima source tree informed the design patterns used here:

| File | Pattern |
|------|---------|
| `share/lapack/eigensys.lisp` | Matrix marshalling (the problem we're solving) |
| `share/stats/inference_result.lisp` | Custom type display via `displa-def` |
| `share/amatrix/amatrix.lisp` | Struct-based matrix with custom display |
| `share/fftpack5/fftpack5-interface.lisp` | Converter-factory pattern |
| `src/numerical/f2cl-lib.lisp` | Column-major indexing conventions |
