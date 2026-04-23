## Optimization and Root Finding

Numerical optimization, constrained optimization, root finding, and nonlinear least squares. Wraps L-BFGS, COBYLA, and MINPACK to work directly with ndarrays.

The optimization functions are in a separate sub-module (loaded separately from the core `numerics` package):

```maxima
(%i1) load("numerics")$
(%i2) load("numerics-optimize")$
```

### Function: np_minimize (expr, vars, x0)
### Function: np_minimize (expr, vars, x0, tolerance, max_iter)
### Function: np_minimize (f, grad, x0)
### Function: np_minimize (f, grad, x0, tolerance, max_iter)

Minimize a scalar-valued function using gradient-based L-BFGS optimization. Supports two calling conventions:

**Expression mode** (gradient computed automatically):
- `expr` — a symbolic scalar expression
- `vars` — `[x1, x2, ...]` list of variables
- `x0` — initial point (real ndarray)

**Function mode** (user provides gradient):
- `f` — objective function taking an ndarray, returning a scalar
- `grad` — gradient function taking an ndarray, returning an ndarray of the same shape
- `x0` — initial point (real ndarray, any shape)

**Common optional arguments:**
- `tolerance` — convergence tolerance (default: `1e-8`)
- `max_iter` — maximum number of iterations (default: `200`)

**Returns:** a list `[x_opt, f_opt, converged]` where:
- `x_opt` — the optimized ndarray (same shape as `x0`)
- `f_opt` — the final objective value
- `converged` — `true` if L-BFGS converged, `false` otherwise

In function mode, `f` and `grad` can be named functions or lambda expressions. The shape of `x0` is preserved in the output. In expression mode, the gradient is computed via `diff()` and compiled automatically.

#### Examples

Expression mode -- just write the math:

```maxima
(%i1) [x_opt, f_opt, ok] : np_minimize(x1^2 + x2^2, [x1, x2],
                              ndarray([3.0, 4.0]));
(%o1)       [ndarray([2], DOUBLE-FLOAT), 1.97e-30, true]
```

Rosenbrock via expression mode:

```maxima
(%i1) [x_opt, f_opt, ok] : np_minimize(
        (1-x)^2 + 100*(y-x^2)^2, [x, y],
        ndarray([-1.0, 1.0]), 1e-10, 500);
(%o1)       [ndarray([2], DOUBLE-FLOAT), 0.0, true]
(%i2) np_to_list(x_opt);
(%o2)       [1.0, 1.0]
```

Function mode -- for procedural or high-dimensional objectives:

```maxima
(%i1) f(x) := np_sum(np_pow(x, 2))$
(%i2) grad(x) := np_scale(2.0, x)$
(%i3) [x_opt, f_opt, ok] : np_minimize(f, grad, ndarray([3.0, 4.0]));
(%o3)       [ndarray([2], DOUBLE-FLOAT), 1.97e-30, true]
(%i4) np_to_list(x_opt);
(%o4)       [2.80e-16, 3.55e-16]
```

Rosenbrock function (classic optimization benchmark):

```maxima
(%i1) f_rosen(x) := block([x1, x2],
        x1 : np_ref(x, 0), x2 : np_ref(x, 1),
        (1 - x1)^2 + 100 * (x2 - x1^2)^2)$
(%i2) g_rosen(x) := block([x1, x2],
        x1 : np_ref(x, 0), x2 : np_ref(x, 1),
        ndarray([-2*(1 - x1) - 400*x1*(x2 - x1^2),
                 200*(x2 - x1^2)]))$
(%i3) [x_opt, f_opt, ok] : np_minimize(f_rosen, g_rosen,
                              ndarray([-1.0, 1.0]), 1e-10, 500);
(%o3)       [ndarray([2], DOUBLE-FLOAT), 0.0, true]
(%i4) np_to_list(x_opt);
(%o4)       [1.0, 1.0]
```

Linear regression (symbolic-numeric bridge):

```maxima
(%i1) /* Derive gradient symbolically */
      L_i(a, b, x_i, y_i) := (1/2) * (y_i - a - b*x_i)^2$
(%i2) diff(L_i(a, b, x[i], y[i]), a);
(%o2)       -(y_i - b*x_i - a)
(%i3) /* Implement as ndarray functions */
      cost(w) := block([pred, res],
        pred : np_matmul(X, w),
        res : np_sub(pred, y),
        np_sum(np_pow(res, 2)) / (2*n))$
(%i4) grad(w) := np_scale(1.0/n,
        np_matmul(np_transpose(X), np_sub(np_matmul(X, w), y)))$
(%i5) [w_opt, loss, ok] : np_minimize(cost, grad, np_zeros([2, 1]))$
```

With lambda expressions:

```maxima
(%i1) [x_opt, _, ok] : np_minimize(
        lambda([x], np_sum(np_pow(x, 2))),
        lambda([x], np_scale(2.0, x)),
        ndarray([5.0, 5.0]));
(%o1)       [ndarray([2], DOUBLE-FLOAT), 1.42e-29, true]
```

**Performance notes:**
Expression mode compiles the objective and gradient into native Lisp closures (via `coerce-float-fun`) that are called directly by L-BFGS with no Maxima evaluator involvement in the hot loop. Function mode calls `f` and `grad` through the Maxima evaluator on every L-BFGS iteration. The function-mode path does allow use of vectorized ndarray operations (`np_matmul`, `np_dot`, etc.) backed by BLAS/LAPACK, which expression mode cannot leverage.

See also: `np_compile_gradient`, `np_lstsq` (for linear least-squares problems)

### Function: np_compile_gradient (expr, vars)

Compile a symbolic expression and its gradient into numeric functions compatible with `np_minimize`. This automates the "symbolic differentiation then numeric evaluation" pattern — you write the loss expression symbolically, and `np_compile_gradient` handles `diff()`, compilation, and ndarray bridging.

**Arguments:**
- `expr` — a scalar symbolic expression (the objective/loss function)
- `vars` — `[x1, x2, ...]` list of variables

**Returns:** a list `[f_func, grad_func]` where:
- `f_func` — a function taking an ndarray, returning a scalar
- `grad_func` — a function taking an ndarray, returning a 1D ndarray

Both functions are compatible with `np_minimize` — pass them directly as `f` and `grad`.

Compile and minimize a quadratic:

```maxima
(%i1) [f, g] : np_compile_gradient(w1^2 + w2^2, [w1, w2]);
(%o1)       [np_compiled_f1, np_compiled_g1]
(%i2) [x_opt, f_opt, ok] : np_minimize(f, g, ndarray([3.0, 4.0]));
(%o2)       [ndarray([2], DOUBLE-FLOAT), 1.97e-30, true]
```

Rosenbrock function — one line instead of hand-coded derivatives:

```maxima
(%i1) [f, g] : np_compile_gradient((1-x)^2 + 100*(y-x^2)^2, [x, y]);
(%o1)       [np_compiled_f2, np_compiled_g2]
(%i2) [x_opt, f_opt, ok] : np_minimize(f, g, ndarray([-1.0, 1.0]), 1e-10, 500);
(%o2)       [ndarray([2], DOUBLE-FLOAT), 0.0, true]
(%i3) np_to_list(x_opt);
(%o3)       [1.0, 1.0]
```

Call `f_func` and `grad_func` independently:

```maxima
(%i1) [f, g] : np_compile_gradient(3*a^2 + 2*b^2, [a, b]);
(%o1)       [np_compiled_f3, np_compiled_g3]
(%i2) f(ndarray([1.0, 2.0]));
(%o2)                         11.0
(%i3) np_to_list(g(ndarray([1.0, 2.0])));
(%o3)                      [6.0, 8.0]
```

See also: `np_minimize`

### Function: np_minimize_cobyla (f, vars, x0, constraints)
### Function: np_minimize_cobyla (f, vars, x0, constraints, rhobeg, rhoend, maxfun)

Minimize a function subject to constraints using COBYLA (Constrained Optimization BY Linear Approximation). This is a derivative-free method — no gradient is required.

**Arguments:**
- `f` — objective expression in `vars`
- `vars` — `[x1, x2, ...]` list of variables
- `x0` — initial guess (ndarray or Maxima list)
- `constraints` — list of constraints using `>=`, `<=`, or `=`
- `rhobeg` — initial trust region radius (default: `1.0`)
- `rhoend` — final accuracy (default: `1e-6`)
- `maxfun` — maximum function evaluations (default: `1000`)

**Returns:** a list `[x_opt, f_opt, n_evals, info]` where:
- `x_opt` — optimized point as a 1D ndarray
- `f_opt` — final objective value
- `n_evals` — number of function evaluations used
- `info` — `0` = success, `1` = maxfun reached, `2` = rounding errors, `-1` = constraints violated

Constraints are written naturally using `>=`, `<=`, or `=`. Equality constraints are converted internally to a pair of inequality constraints.

#### Examples

Minimize on the unit circle:

```maxima
(%i1) [x_opt, f_opt, nev, info] : np_minimize_cobyla(
        x1*x2, [x1, x2], [1, 1], [1 - x1^2 - x2^2 >= 0]);
(%o1)       [ndarray([2], DOUBLE-FLOAT), -0.5, 52, 0]
```

Bound constraint:

```maxima
(%i1) [x_opt, f_opt, nev, info] : np_minimize_cobyla(
        (x1 - 3)^2, [x1], [0.5], [x1 >= 0]);
(%o1)       [ndarray([1], DOUBLE-FLOAT), 0.0, 32, 0]
(%i2) np_ref(x_opt, 0);
(%o2)                          3.0
```

Equality constraint (minimize `x1^2 + x2^2` subject to `x1 + x2 = 1`):

```maxima
(%i1) [x_opt, f_opt, nev, info] : np_minimize_cobyla(
        x1^2 + x2^2, [x1, x2], [0.8, 0.2], [x1 + x2 = 1]);
(%o1)       [ndarray([2], DOUBLE-FLOAT), 0.5, 42, 0]
(%i2) np_to_list(x_opt);
(%o2)                    [0.5, 0.5]
```

See also: `np_minimize` (unconstrained, gradient-based)

### Function: np_fsolve (fcns, vars, x0)
### Function: np_fsolve (fcns, vars, x0, jacobian, tolerance)
### Function: np_fsolve (f, x0)
### Function: np_fsolve (f, x0, tolerance)

Solve a system of `n` nonlinear equations in `n` unknowns using MINPACK. Finds a point where all expressions/function values are simultaneously zero.

**Expression mode:**
- `fcns` — `[f1, f2, ...]` list of expressions (implicitly set equal to zero)
- `vars` — `[x1, x2, ...]` list of unknowns (must be same length as `fcns`)
- `x0` — initial guess (ndarray or Maxima list)
- `jacobian` — `true` (compute symbolic Jacobian, default) or `false` (finite differences)
- `tolerance` — solution tolerance (default: `sqrt(machine_epsilon)`)

**Function mode:**
- `f` — function taking a 1D ndarray, returning a Maxima list of `n` values
- `x0` — initial guess (ndarray or Maxima list)
- `tolerance` — solution tolerance (default: `sqrt(machine_epsilon)`)

Function mode always uses finite differences (HYBRD1). Expression mode uses symbolic Jacobian by default (HYBRJ1) for better convergence.

**Returns:** a list `[x_opt, residual_norm, info]` where:
- `x_opt` — solution as a 1D ndarray
- `residual_norm` — Euclidean norm of the residual at the solution
- `info` — `1` = converged, `0` = improper input, other values indicate issues

#### Examples

Expression mode:

```maxima
(%i1) [x_opt, rnorm, info] : np_fsolve([x^2 - 2], [x], [1.0]);
(%o1)       [ndarray([1], DOUBLE-FLOAT), 0.0, 1]
(%i2) np_ref(x_opt, 0);
(%o2)                   1.414213562373095
```

2D system:

```maxima
(%i1) [x_opt, rnorm, info] : np_fsolve(
        [x^2 + y^2 - 4, x - y], [x, y], [1.0, 1.0]);
(%o1)       [ndarray([2], DOUBLE-FLOAT), 0.0, 1]
```

Function mode:

```maxima
(%i1) f(x) := [np_ref(x, 0)^2 - 2]$
(%i2) [x_opt, rnorm, info] : np_fsolve(f, [1.0]);
(%o2)       [ndarray([1], DOUBLE-FLOAT), 0.0, 1]
```

**Performance notes:**
Expression mode compiles both the system and its symbolic Jacobian into native Lisp closures (via `coerce-float-fun`) and uses HYBRJ1 (analytic Jacobian). The solver hot loop calls these closures directly with no Maxima evaluator involvement. Function mode calls `f` through the Maxima evaluator on every solver iteration and uses HYBRD1 (finite-difference Jacobian), which requires additional function evaluations per step to approximate the Jacobian numerically.

See also: `np_lsq_nonlinear` (overdetermined systems)

### Function: np_lsq_nonlinear (fcns, vars, x0)
### Function: np_lsq_nonlinear (fcns, vars, x0, jacobian, tolerance)
### Function: np_lsq_nonlinear (f, x0)
### Function: np_lsq_nonlinear (f, x0, tolerance)

Nonlinear least-squares fitting: minimize `sum(fi^2)` for `m` residual functions in `n` unknowns (`m >= n`). Uses MINPACK.

**Expression mode:**
- `fcns` — `[f1, f2, ...]` list of residual expressions (`m >= n`)
- `vars` — `[x1, x2, ...]` list of unknowns
- `x0` — initial guess (ndarray or Maxima list)
- `jacobian` — `true` (compute symbolic Jacobian, default) or `false` (finite differences)
- `tolerance` — solution tolerance (default: `sqrt(machine_epsilon)`)

**Function mode:**
- `f` — function taking a 1D ndarray, returning a Maxima list of `m` residuals
- `x0` — initial guess (ndarray or Maxima list)
- `tolerance` — solution tolerance (default: `sqrt(machine_epsilon)`)

Function mode always uses finite differences (LMDIF1). Expression mode uses symbolic Jacobian by default (LMDER1).

**Returns:** a list `[x_opt, residual_norm, info]` where:
- `x_opt` — solution as a 1D ndarray
- `residual_norm` — Euclidean norm of the residual vector at the solution
- `info` — `1`-`3` = converged, `0` = improper input, other values indicate issues

#### Examples

Expression mode:

```maxima
(%i1) [x_opt, rnorm, info] : np_lsq_nonlinear(
        [a*1 - 2, a*2 - 3.9, a*3 - 6.1], [a], [1.0]);
(%o1)       [ndarray([1], DOUBLE-FLOAT), 0.1, 1]
(%i2) np_ref(x_opt, 0);
(%o2)                          2.0
```

Function mode:

```maxima
(%i1) f(x) := block([a], a : np_ref(x, 0),
        [a*1 - 2, a*2 - 3.9, a*3 - 6.1])$
(%i2) [x_opt, rnorm, info] : np_lsq_nonlinear(f, [1.0]);
(%o2)       [ndarray([1], DOUBLE-FLOAT), 0.1, 1]
```

**Performance notes:**
Same tradeoff as `np_fsolve`: expression mode compiles residuals and their symbolic Jacobian into native Lisp closures (LMDER1, analytic Jacobian), keeping the Maxima evaluator out of the hot loop. Function mode calls `f` through the Maxima evaluator on every iteration and uses LMDIF1 (finite-difference Jacobian), adding extra function evaluations per step.

See also: `np_fsolve` (square systems), `np_lstsq` (linear least-squares)
