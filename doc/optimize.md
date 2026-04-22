## Optimization

Numerical optimization using the L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) algorithm. Wraps Maxima's built-in L-BFGS implementation to work directly with ndarrays.

The optimization functions are in a separate sub-module (loaded separately from the core `numerics` package):

```maxima
(%i1) load("numerics")$
(%i2) load("numerics-optimize")$
```

### Function: np_minimize (f, grad, x0)
### Function: np_minimize (f, grad, x0, tolerance)
### Function: np_minimize (f, grad, x0, tolerance, max_iter)

Minimize a scalar-valued function using gradient-based L-BFGS optimization.

**Arguments:**
- `f` — objective function taking an ndarray, returning a scalar
- `grad` — gradient function taking an ndarray, returning an ndarray of the same shape
- `x0` — initial point (real ndarray, any shape)
- `tolerance` — convergence tolerance (default: `1e-8`)
- `max_iter` — maximum number of iterations (default: `200`)

**Returns:** a list `[x_opt, f_opt, converged]` where:
- `x_opt` — the optimized ndarray (same shape as `x0`)
- `f_opt` — the final objective value
- `converged` — `true` if L-BFGS converged, `false` otherwise

Both `f` and `grad` can be named functions or lambda expressions. The shape of `x0` is preserved in the output — if you pass a `[p, 1]` column vector, you get a `[p, 1]` result.

#### Examples

Simple quadratic:

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

See also: `np_lstsq` (for linear least-squares problems)
