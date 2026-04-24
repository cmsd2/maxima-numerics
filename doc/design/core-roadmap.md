# Core Feature Roadmap

The guiding constraint is to **wrap existing Maxima built-in packages and
Quicklisp libraries** rather than implementing algorithms from scratch in Lisp.

See also:
- [Architecture](architecture.md) — package structure, handle type, API surface
- [Library roadmap](library-roadmap.md) — utility layer, signal helpers, symbolic bridge

---

## Completed

The following have been implemented and tested:

- **ODEPACK DLSODE** → `np_odeint` (expression + function modes, Adams + BDF)
- **COBYLA** → `np_minimize_cobyla` (derivative-free constrained optimization)
- **MINPACK** → `np_fsolve` (nonlinear root finding) + `np_lsq_nonlinear`
  (Levenberg-Marquardt least squares), both with expression + function modes
- **Symbolic gradient bridge** → `np_compile_gradient` + expression-mode
  `np_minimize` (symbolic `diff()` → compiled numeric closures)
- **Dual-mode APIs** — all solver functions accept either symbolic expressions
  (compiled via `coerce-float-fun`, Jacobian/gradient derived automatically) or
  user-provided Maxima functions (called through the evaluator)

---

## Unwrapped Maxima Packages

These bundled packages are compiled to CL and available inside Maxima but have
no ndarray interface yet. Each is a candidate for wrapping.

### ODEPACK — additional solvers

DLSODE is wrapped (see Completed above). The remaining ODEPACK solvers —
DLSODA, DLSODAR, DLSODI, DLSODES, and others — are compiled to CL via f2cl
but have no Maxima interface.

### MNEWTON — Newton's method for nonlinear systems

`share/mnewton/` provides `mnewton(fcns, vars, guess)` — a pure-Maxima
implementation of Newton's method for nonlinear systems. Automatically computes
the Jacobian via symbolic differentiation. Simpler than MINPACK but less robust
(no trust region, no fallback to finite differences on Jacobian failure).

### DISTRIB — probability distributions

`share/distrib/` provides scalar PDF, CDF, quantile, random variate, and MLE
functions for ~20 distributions (normal, t, chi-squared, F, exponential, gamma,
beta, Poisson, binomial, etc.). Vectorized ndarray versions would be valuable
for Monte Carlo workflows — `np_random_normal(mu, sigma, shape)` returning an
ndarray, or element-wise PDF/CDF evaluation over ndarrays.

### NUMERICALIO — direct file loading

`share/numericalio/` reads CSV and binary files into Maxima matrices. Binary
format uses IEEE 754 `double-float` — the same storage as ndarray — so direct
loading via `read-sequence` into ndarray storage is possible with zero parsing
overhead. CSV loading could similarly bypass Maxima matrix construction.

---

## Tier 1: Expand ODEPACK Coverage

ODEPACK ships **9 solvers** in `share/odepack/fortran/`, all compiled to Common
Lisp via f2cl. Only DLSODE is currently exposed. The others are available as
compiled CL code inside Maxima, requiring only a Maxima-level interface — the
same wrapping pattern used by `np_odeint`.

### 1. Event detection — wrap DLSODAR

DLSODAR extends DLSODA with **root-finding**: the user supplies constraint
functions `g(t, y)` and the solver reports when any `g_i` crosses zero,
returning which root was found via a `JROOT` array. This enables discontinuous
dynamics (switches, collisions, threshold triggers) without polling.

**API sketch:**

```maxima
np_odeint([-y], [t, y], [1.0], tspan,
          events = [y - 0.5],          /* stop when y = 0.5 */
          event_action = stop)         /* or: restart, record */
```

**Motivation:** Simulink-style simulation requires event detection for hybrid
systems (relays, switches, saturation). DLSODAR is the standard solution and
is already compiled to CL in Maxima.

**Design considerations:**
- Could extend `np_odeint` with optional `events` argument, or provide a
  separate `np_odeint_events` function.
- Need to decide on event actions: stop integration, record event time and
  continue, or call a user callback that modifies state.
- DLSODAR also includes automatic stiff/nonstiff switching (DLSODA behaviour),
  which would replace the manual `method = adams/bdf` choice.

### 2. Automatic stiff/nonstiff switching — wrap DLSODA

DLSODA automatically switches between Adams (non-stiff) and BDF (stiff) methods
based on the problem's behaviour during integration. This removes the burden of
choosing a method from the user — important when stiffness varies over the
integration interval.

**Motivation:** Many practical systems (chemical kinetics, circuit transients)
transition between stiff and non-stiff regimes. Choosing the wrong method
either wastes compute (BDF on non-stiff) or fails (Adams on stiff).

**Design considerations:**
- Could be exposed as `method = auto` in the existing `np_odeint` interface.
- DLSODAR (item 1) includes DLSODA behaviour, so wrapping DLSODAR may
  subsume this item.

### 3. Implicit / DAE systems — wrap DLSODI

DLSODI solves linearly implicit systems of the form `A(t,y) · dy/dt = g(t,y)`
where the matrix `A` can be singular. This covers **index-1 DAEs** — the most
common class arising from circuit simulation (KCL/KVL constraints), mechanism
kinematics (holonomic constraints), and chemical equilibrium.

**API sketch:**

```maxima
np_odeint_implicit(A_expr, g_expr, vars, y0, tspan)
```

**Motivation:** Many engineering systems have algebraic constraints alongside
differential equations. Currently users must manually reduce to explicit ODE
form (eliminating algebraic variables), which is error-prone and obscures the
physical structure. DLSODI handles the implicit form directly.

**Design considerations:**
- DLSODI requires the user to supply both `A(t,y)` and `g(t,y)` as functions
  or expressions, plus a routine to solve `A · x = b` (or the option to let
  DLSODI factor `A` internally).
- Only handles index-1 DAEs. Higher-index DAEs require index reduction
  (a symbolic preprocessing step) or a specialised solver like SUNDIALS IDA.
- Expression mode can compile both `A` and `g` via `coerce-float-fun`.

### 4. Sparse Jacobian support — wrap DLSODES

DLSODES handles ODE systems with **sparse Jacobian structure**. For large
systems (hundreds+ of state variables), most variables interact with only a few
others. DLSODES exploits this sparsity for memory and performance.

**Motivation:** Large coupled systems (discretised PDEs, multi-body dynamics,
chemical reaction networks) produce sparse Jacobians. Dense Jacobian storage
in DLSODE scales as O(n²), limiting practical system size.

**Design considerations:**
- Requires sparse structure specification (which variables appear in which
  equations). Could infer this from symbolic expressions via `diff()`.
- Lower priority than items 1-3 since most Maxima use cases involve
  moderate-sized systems.

---

## Tier 2: Maxima-level Features (no new Lisp algorithms)

These can be implemented primarily in Maxima code using existing ndarray
operations and Maxima built-ins.

### 5. Statistical hypothesis tests

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

### 6. Interpolation and splines

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

### 7. Vectorized distribution sampling

Wrap `share/distrib/` random variate functions with ndarray output:
`np_random_normal(mu, sigma, shape)`, `np_random_uniform(a, b, shape)`, etc.
Currently `np_randn` provides standard normal only; parameterized distributions
require element-wise post-processing.

---

## Tier 3: Longer-term / Higher-effort

### 8. Sparse matrices

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

### 9. Higher-index DAE support — SUNDIALS IDA

DLSODI (item 3) handles index-1 DAEs only. For higher-index systems (common
in constrained multibody dynamics), a solver with built-in index reduction or
constraint stabilisation is needed. SUNDIALS IDA is the standard choice.

**Landscape:** No Common Lisp bindings to SUNDIALS exist. GSLL (GNU Scientific
Library for Lisp, in Quicklisp) provides ODE steppers but no DAE support. The
Quicklisp ecosystem has no DAE solver of any kind. Rust crates `sundials` /
`sundials-sys` exist but don't help — calling Rust from CL adds an unnecessary
FFI layer vs binding the C library directly.

**Approach if needed:** Write CFFI bindings to SUNDIALS IDA (pure C library,
well-documented API). Tools like `cl-autowrap` / `c2ffi` (both in Quicklisp)
can auto-generate low-level CFFI definitions from C headers, though an
idiomatic Lisp wrapper would still be needed. This is significant effort and
should only be pursued if index-1 coverage via DLSODI proves insufficient.

---

## What We Are Not Building

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
  gradient bridge covers the primary use case. Tape-based AD through numeric
  code is a different paradigm.
- **Discrete-time / hybrid systems**: no solver infrastructure exists for
  mixed continuous-discrete dynamics. Event detection (item 1) covers the
  continuous side; discrete logic would be application-level Maxima code.
- **Block diagram compiler**: translating a graphical block diagram into a
  system of ODEs is a substantial software project (model compilation,
  algebraic loop detection, signal routing). Users write their equations
  directly — which Maxima's symbolic capabilities make relatively natural.
