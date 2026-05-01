## Control Systems

### Function: np_lqr (a, b, q, r)

Continuous-time linear-quadratic regulator.

Returns the optimal feedback gain matrix `K` such that the control law `u = -K x` minimises the cost functional

$$J = \int_0^\infty \left( x^\top Q\, x + u^\top R\, u \right) dt$$

for the linear time-invariant system $\dot x = A x + B u$.

**Arguments:**
- `a` — `n x n` state matrix (ndarray)
- `b` — `n x m` input matrix (ndarray)
- `q` — `n x n` symmetric positive-semidefinite state-penalty matrix (ndarray)
- `r` — `m x m` symmetric positive-definite input-penalty matrix (ndarray)

**Returns:** `m x n` ndarray containing the gain matrix $K = R^{-1} B^\top P$, where $P$ is the unique stabilising solution of the continuous-time algebraic Riccati equation $A^\top P + P A - P B R^{-1} B^\top P + Q = 0$.

**Method.** Forms the Hamiltonian
$$H = \begin{bmatrix} A & -B R^{-1} B^\top \\ -Q & -A^\top \end{bmatrix}$$
(size $2n \times 2n$), eigendecomposes it, selects the $n$ eigenvectors whose eigenvalues have negative real parts, partitions $V = [V_{\text{top}};\, V_{\text{bot}}]$, and computes $P = V_{\text{bot}} V_{\text{top}}^{-1}$.

**Numerical note.** This eigenvalue method is straightforward but can lose digits on near-singular Hamiltonians. The standard robust alternative is the Hamiltonian Schur form, which requires a real Schur decomposition primitive that `numerics` does not yet provide. For well-conditioned hobbyist-scale problems the eigenvalue method is fine; if your closed-loop eigenvalues come out clearly far from the imaginary axis, you can trust the result.

`np_lqr` errors out if the Hamiltonian doesn't have exactly $n$ eigenvalues with strictly negative real parts — typically a sign that $(A, B)$ is not stabilisable or $(A, \sqrt{Q})$ is not detectable, or that $Q$ or $R$ is ill-formed.

#### Examples

Double integrator with unit weights — closed-form $K = [1,\, \sqrt 3]$, closed-loop poles at $-\sqrt 3/2 \pm j/2$:

```maxima
(%i1) A : ndarray(matrix([0.0, 1.0], [0.0, 0.0]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) B : ndarray(matrix([0.0], [1.0]));
(%o2)            ndarray([2, 1], DOUBLE-FLOAT)
(%i3) K : np_lqr(A, B, ndarray(matrix([1.0,0.0],[0.0,1.0])),
                       ndarray(matrix([1.0])));
(%o3)            ndarray([1, 2], DOUBLE-FLOAT)
(%i4) np_to_list(np_reshape(K, [2]));
(%o4)         [1.0, 1.7320508075688772]
```

Stable plant with `Q = 0` — optimal control is to do nothing, so $K = 0$:

```maxima
(%i1) A : ndarray(matrix([-1.0, 0.0], [0.0, -2.0]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) B : ndarray(matrix([1.0, 0.0], [0.0, 1.0]));
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) K : np_lqr(A, B, np_zeros([2,2]), B);
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_norm(K);
(%o4)                  0.0
```

#### See also
`np_eig` — used internally to factor the Hamiltonian.
