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

See also:
- [Core feature roadmap](core-roadmap.md) — solver wrapping, DAE support, sparse matrices
- [Library roadmap](library-roadmap.md) — utility layer, signal helpers, symbolic bridge

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

The implementation lives in `lisp/`, organized into ASDF subsystems that can be
loaded independently. Each subsystem has its own `*-loader.lisp` and `*.mac`
entry point.

| System | Role | Extra deps |
|--------|------|------------|
| `numerics/core` | ndarray handle, linalg, elementwise, signal, aggregation | magicl, trivial-garbage, alexandria |
| `numerics/optimize` | Optimization, root finding, least squares (wraps lbfgs, COBYLA, MINPACK) | — |
| `numerics/integrate` | ODE integration (wraps ODEPACK DLSODE) | — |
| `numerics/learn` | RL algorithm shells: CEM, rollout, Q-learning | — |
| `numerics/image` | Image I/O (wraps opticl) | opticl |
| `numerics` | Full system: core + Arrow bridge | cffi, static-vectors |

`numerics.mac` loads `numerics/core` by default. Extension systems are loaded
separately (e.g., `load("numerics-optimize")`). See `lisp/numerics.asd` for the
full system definitions.

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

The API follows NumPy naming conventions. All functions take/return ndarray
handles; explicit `np_to_matrix()` or `np_to_list()` materializes to Maxima form.
For the full function list, see `doc/numerics.md` and the per-module docs.

### Core (`numerics/core`)

**Conversion & constructors** — `ndarray(matrix)`, `np_to_matrix`, `np_to_list`,
`np_zeros`, `np_ones`, `np_eye`, `np_rand`, `np_randn`, `np_arange`,
`np_linspace`, `np_copy`, etc.

**Linear algebra** — `np_matmul`, `np_inv`, `np_det`, `np_solve`, `np_svd`,
`np_eig`, `np_qr`, `np_lu`, `np_cholesky`, `np_norm`, `np_expm`, `np_lstsq`,
`np_pinv`, etc. Most delegate to magicl; a few (`np_expm`, `np_lstsq`,
`np_pinv`, `np_norm`, `np_rank`) are pure-Lisp because magicl lacks them or
requires extra native deps.

**Element-wise ops** — arithmetic (`np_add`, `np_mul`, ...), math functions
(`np_exp`, `np_sin`, `np_tanh`, ...), comparisons (`np_greater`, `np_equal`, ...),
logical ops, `np_where`, `np_map`, `np_extract`. Internally dispatched via
`numerics-binary-op` (handles broadcasting) and `numerics-unary-op`.

**Slicing & indexing** — `np_ref`, `np_set`, `np_row`, `np_col`, `np_slice`,
`np_reshape`, `np_flatten`, `np_hstack`, `np_vstack`, `np_shape`, `np_size`.

**Aggregation** — `np_sum`, `np_mean`, `np_min`, `np_max`, `np_std`, `np_var`,
`np_cumsum`, `np_dot`, `np_sort`, `np_argsort`, `np_discount`.

**Signal processing** — `np_fft`, `np_ifft`, `np_fft2d`, `np_ifft2d`,
`np_convolve`, `np_convolve2d`, `np_trapz` (wraps fftpack5).

### Optimization (`numerics/optimize`)

`np_minimize` (L-BFGS), `np_minimize_cobyla` (COBYLA), `np_fsolve` (MINPACK
root finding), `np_lsq_nonlinear` (MINPACK Levenberg-Marquardt),
`np_compile_gradient`. All support expression mode (symbolic → compiled) and
function mode (user-provided Maxima callbacks).

### ODE integration (`numerics/integrate`)

`np_odeint` — wraps ODEPACK DLSODE. Supports Adams (non-stiff) and BDF (stiff)
methods. Returns 2D ndarray (rows = timesteps).

### RL / Learning (`numerics/learn`)

`np_cem` (Cross-Entropy Method), `np_rollout` (episode collection),
`np_qlearn` (tabular Q-learning). `np_cem` and `np_qlearn` accept keyword
options via `key=value` syntax (e.g., `n_samples=50`, `epsilon=0.5`).
`np_rollout` pairs with `np_discount` for policy-gradient methods.

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

Individual test files in `tests/` cover each subsystem (constructors, linalg,
elementwise, slicing, aggregation, complex, optimize, integrate, learn, etc.)
and are listed in `manifest.toml` for `mxpm test` discovery.

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
the right level. For example, FFTPACK5 accepts raw CL arrays (we pass
`magicl:storage` directly), ODEPACK exposes stepping functions, and
COBYLA/MINPACK/LBFGS accept Maxima expressions and initial guesses.

We wrap FFTPACK5, LBFGS, COBYLA, MINPACK, and ODEPACK from the outside without
modifying their source. Our LAPACK wrapper (via magicl) supersedes
`share/lapack/` entirely.

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

Design patterns drawn from the Maxima source tree:
- `share/lapack/eigensys.lisp` — matrix marshalling (the problem we solve)
- `share/stats/inference_result.lisp`, `share/amatrix/amatrix.lisp` — custom type display via `displa-def`
- `share/{cobyla,minpack,odepack}/*-interface.lisp` — solver wrapping patterns
- `share/fftpack5/fftpack5-interface.lisp` — converter-factory pattern
- `src/numerical/f2cl-lib.lisp` — column-major indexing conventions
