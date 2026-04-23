## ODE Integration

Numerical integration of ordinary differential equations using ODEPACK/DLSODE. Supports both non-stiff (Adams) and stiff (BDF) systems.

The integration functions are in a separate sub-module (loaded separately from the core `numerics` package):

```maxima
(%i1) load("numerics")$
(%i2) load("numerics-integrate")$
```

### Function: np_odeint (f, vars, y0, tspan)
### Function: np_odeint (f, vars, y0, tspan, rtol, atol, method)
### Function: np_odeint (f_func, y0, tspan)
### Function: np_odeint (f_func, y0, tspan, rtol, atol, method)

Integrate a system of ODEs `dy/dt = f(t, y)` and collect the trajectory at specified output times. Supports two calling conventions:

**Expression mode:**
- `f` — `[f1, f2, ...]` list of right-hand-side expressions for each equation
- `vars` — `[t, y1, y2, ...]` the independent variable (time) first, then state variables
- `y0` — initial conditions (ndarray or Maxima list)
- `tspan` — output times (ndarray or Maxima list)

**Function mode:**
- `f_func` — function `f(t, y)` taking scalar `t` and 1D ndarray `y`, returning a Maxima list of derivatives
- `y0` — initial conditions (ndarray or Maxima list; length determines `neq`)
- `tspan` — output times (ndarray or Maxima list)

**Common optional arguments:**
- `rtol` — relative tolerance (default: `1e-8`)
- `atol` — absolute tolerance (default: `1e-8`)
- `method` — `adams` (default, non-stiff) or `bdf` (stiff systems)

**Returns:** a 2D ndarray of shape `[n_times, 1 + neq]` where:
- Column 0 contains the time values
- Columns 1 through `neq` contain the corresponding state variables

The first row corresponds to the initial conditions at `tspan[0]`.

#### Examples

Exponential decay `dy/dt = -y`, `y(0) = 1` (exact solution: `y(t) = exp(-t)`):

```maxima
(%i1) tspan : [0.0, 0.5, 1.0, 1.5, 2.0]$
(%i2) result : np_odeint([-y], [t, y], [1.0], tspan);
(%o2)            ndarray([5, 2], DOUBLE-FLOAT)
(%i3) np_ref(result, 2, 0);  /* t = 1.0 */
(%o3)                          1.0
(%i4) np_ref(result, 2, 1);  /* y(1.0) ≈ exp(-1) */
(%o4)                   0.36787944117144233
```

Harmonic oscillator `dx/dt = v`, `dv/dt = -x`:

```maxima
(%i1) tspan : makelist(i * 0.1, i, 0, 40)$
(%i2) result : np_odeint([v, -x], [t, x, v], [1.0, 0.0], tspan);
(%o2)            ndarray([41, 3], DOUBLE-FLOAT)
(%i3) /* x(pi) ≈ cos(pi) = -1 */
      np_ref(result, 31, 1);
(%o3)                  -0.9999923780697327
```

Stiff system with BDF method (`dy/dt = -50*y`, fast decay):

```maxima
(%i1) result : np_odeint([-50*y], [t, y], [1.0],
                          [0.0, 0.1, 0.5, 1.0], 1e-8, 1e-8, bdf);
(%o1)            ndarray([4, 2], DOUBLE-FLOAT)
(%i2) np_ref(result, 3, 1);  /* y(1.0) ≈ exp(-50) */
(%o2)                   1.9287498479639178e-22
```

Using ndarrays for `y0` and `tspan`:

```maxima
(%i1) y0 : ndarray([1.0])$
(%i2) tspan : np_linspace(0.0, 1.0, 5)$
(%i3) result : np_odeint([-y], [t, y], y0, tspan);
(%o3)            ndarray([5, 2], DOUBLE-FLOAT)
```

Function mode -- for procedural or conditional RHS:

```maxima
(%i1) f(t, y) := [-np_ref(y, 0)]$
(%i2) result : np_odeint(f, [1.0], [0.0, 0.5, 1.0, 1.5, 2.0]);
(%o2)            ndarray([5, 2], DOUBLE-FLOAT)
(%i3) np_ref(result, 2, 1);  /* y(1.0) ≈ exp(-1) */
(%o3)                   0.36787944117144233
```

Function mode with lambda:

```maxima
(%i1) result : np_odeint(
        lambda([t, y], [np_ref(y, 1), -np_ref(y, 0)]),
        [1.0, 0.0], makelist(i * 0.1, i, 0, 40));
(%o1)            ndarray([41, 3], DOUBLE-FLOAT)
```

**Performance notes:**
Expression mode compiles the RHS (and Jacobian, for BDF) into native Lisp closures via `coerce-float-fun`. DLSODE calls these closures directly with no Maxima evaluator involvement — the RHS may be called thousands of times during integration. Function mode routes each RHS call through the Maxima evaluator and copies data in/out of a callback ndarray on every step. The function-mode path does allow use of vectorized ndarray operations backed by BLAS/LAPACK, which expression mode cannot leverage.

See also: `np_minimize` (optimization), `np_fsolve` (root finding)
