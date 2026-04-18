## Linear Algebra

### Function: np_matmul (a, b)

Matrix multiplication.

Computes the matrix product of two 2D ndarrays using BLAS. The number of columns of `a` must equal the number of rows of `b`.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) I : np_eye(2);
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) np_to_matrix(np_matmul(I, A));
(%o3)         matrix([1.0, 2.0], [3.0, 4.0])
(%i4) np_to_matrix(np_matmul(A, A));
(%o4)       matrix([7.0, 10.0], [15.0, 22.0])
```

See also: `np_dot`, `np_inv`, `np_solve`

### Function: np_inv (a)

Matrix inverse.

Computes the inverse of a square matrix using LAPACK. Signals an error if the matrix is singular or nearly singular.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) Ainv : np_inv(A);
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) np_to_matrix(np_matmul(A, Ainv));
(%o3)         matrix([1.0, 0.0], [0.0, 1.0])
(%i4) errcatch(np_inv(ndarray(matrix([1, 0], [0, 0]))));
(%o4)                          []
```

See also: `np_solve`, `np_det`, `np_pinv`

### Function: np_det (a)

Determinant of a square matrix.

Returns a scalar. Uses LAPACK for computation.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_det(A);
(%o2)                         -2.0
(%i3) np_det(np_eye(5));
(%o3)                          1.0
```

See also: `np_inv`, `np_rank`

### Function: np_solve (a, b)

Solve the linear system Ax = b.

Computes the solution to a system of linear equations where `a` is a square coefficient matrix and `b` is a right-hand side vector or matrix. Uses LAPACK.

#### Examples

```maxima
(%i1) A : ndarray(matrix([2, 1], [5, 3]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) b : ndarray(matrix([4], [7]));
(%o2)            ndarray([2, 1], DOUBLE-FLOAT)
(%i3) x : np_solve(A, b);
(%o3)            ndarray([2, 1], DOUBLE-FLOAT)
(%i4) np_to_matrix(np_matmul(A, x));
(%o4)            matrix([4.0], [7.0])
```

See also: `np_inv`, `np_lstsq`

### Function: np_svd (a)

Singular Value Decomposition.

Decomposes `a` into U, S, and Vt such that A = U * diag(S) * Vt, where U and Vt are orthogonal matrices. S is returned as a **1D ndarray** of singular values (not a diagonal matrix). Returns a Maxima list `[U, S, Vt]`.

For an m-by-n matrix, U is m-by-m, S has min(m,n) elements, and Vt is n-by-n.

To reconstruct A, build a diagonal matrix from S: `np_matmul(np_matmul(U, np_diag(np_to_list(S))), Vt)`.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 0], [0, 2], [0, 0]));
(%o1)            ndarray([3, 2], DOUBLE-FLOAT)
(%i2) [U, S, Vt] : np_svd(A);
(%o2)        [ndarray, ndarray, ndarray]
(%i3) np_shape(S);
(%o3)                          [2]
(%i4) np_to_list(S);
(%o4)                      [2.0, 1.0]
(%i5) /* Reconstruct */
      S_mat : np_diag(np_to_list(S));
(%o5)            ndarray([2, 2], DOUBLE-FLOAT)
```

See also: `np_eig`, `np_rank`, `np_lstsq`, `np_pinv`

### Function: np_eig (a)

Eigendecomposition.

Computes eigenvalues and eigenvectors of a square matrix. Returns a Maxima list `[eigenvalues, eigenvectors]` where `eigenvalues` is a 1D ndarray and `eigenvectors` is a 2D ndarray whose columns are the eigenvectors.

When eigenvalues have non-negligible imaginary parts (e.g. rotation matrices, non-symmetric matrices), the returned ndarrays use `complex-double-float` dtype. When all eigenvalues are real, returns `double-float` ndarrays.

#### Examples

```maxima
(%i1) A : ndarray(matrix([2, 1, 0], [1, 3, 1], [0, 1, 2]));
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) [vals, vecs] : np_eig(A);
(%o2)             [ndarray, ndarray]
(%i3) np_to_list(vals);
(%o3)                [1.0, 2.0, 4.0]
(%i4) /* Verify A*v = lambda*v */
      np_to_matrix(np_sub(np_matmul(A, vecs),
                          np_matmul(vecs, np_diag(np_to_list(vals)))));
(%o4)  matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
```

See also: `np_svd`

### Function: np_qr (a)

QR decomposition.

Decomposes `a` into an orthogonal matrix Q and an upper triangular matrix R such that A = Q * R. Returns a Maxima list `[Q, R]`.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4], [5, 6]));
(%o1)            ndarray([3, 2], DOUBLE-FLOAT)
(%i2) [Q, R] : np_qr(A);
(%o2)              [ndarray, ndarray]
(%i3) np_shape(Q);
(%o3)                        [3, 3]
(%i4) np_shape(R);
(%o4)                        [3, 2]
```

See also: `np_lu`, `np_svd`

### Function: np_lu (a)

LU decomposition with partial pivoting.

Decomposes `a` into a lower triangular matrix L (unit diagonal), an upper triangular matrix U, and a permutation matrix P such that P * A = L * U. Returns a Maxima list `[L, U, P]`.

For an m-by-n matrix, L is m-by-min(m,n), U is min(m,n)-by-n, and P is m-by-m.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2, 3], [4, 5, 6], [7, 8, 10]));
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) [L, U, P] : np_lu(A);
(%o2)        [ndarray, ndarray, ndarray]
(%i3) /* Verify P*A = L*U */
      np_to_matrix(np_sub(np_matmul(P, A), np_matmul(L, U)));
(%o3)  matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
```

See also: `np_qr`, `np_svd`, `np_solve`

### Function: np_norm (a) / np_norm (a, ord)

Matrix or vector norm.

By default, computes the 2-norm for vectors and the Frobenius norm for matrices. The optional `ord` parameter selects the norm type.

Calling forms:

- `np_norm(a)` -- default norm (2-norm for vectors, Frobenius for matrices)
- `np_norm(a, 1)` -- 1-norm (sum of abs for vectors, max column sum for matrices)
- `np_norm(a, 2)` -- 2-norm (Euclidean for vectors, spectral/largest singular value for matrices)
- `np_norm(a, inf)` -- infinity norm (max abs for vectors, max row sum for matrices)
- `np_norm(a, fro)` -- Frobenius norm (matrices only, same as default)

#### Examples

```maxima
(%i1) v : ndarray([1, -2, 3], [3]);
(%o1)            ndarray([3], DOUBLE-FLOAT)
(%i2) np_norm(v);
(%o2)                    3.7416573867739413
(%i3) np_norm(v, 1);
(%o3)                          6.0
(%i4) np_norm(v, inf);
(%o4)                          3.0
(%i5) A : ndarray(matrix([1, -7], [2, -3]));
(%o5)            ndarray([2, 2], DOUBLE-FLOAT)
(%i6) np_norm(A, 1);
(%o6)                         10.0
(%i7) np_norm(A, inf);
(%o7)                          8.0
(%i8) np_norm(A, 2);
(%o8)                    7.649700064568801
```

See also: `np_det`, `np_rank`

### Function: np_rank (a)

Numerical rank of a matrix.

Computed via SVD. Counts the number of singular values above a tolerance threshold (proportional to machine epsilon and the largest singular value). Returns an integer.

#### Examples

```maxima
(%i1) np_rank(np_eye(3));
(%o1)                           3
(%i2) np_rank(ndarray(matrix([1, 2], [2, 4])));
(%o2)                           1
(%i3) np_rank(np_zeros([3, 3]));
(%o3)                           0
```

See also: `np_svd`, `np_det`

### Function: np_trace (a)

Matrix trace (sum of diagonal elements).

Returns a scalar equal to the sum of the elements on the main diagonal.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_trace(A);
(%o2)                          5.0
(%i3) np_trace(np_eye(5));
(%o3)                          5.0
```

See also: `np_det`, `np_diag`

### Function: np_transpose (a)

Matrix transpose.

Returns a new ndarray with rows and columns swapped.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_transpose(A));
(%o2)         matrix([1.0, 3.0], [2.0, 4.0])
(%i3) np_shape(np_transpose(ndarray(matrix([1, 2, 3]))));
(%o3)                        [3, 1]
```

See also: `np_conj`, `np_ctranspose`, `np_reshape`

### Function: np_conj (a)

Element-wise complex conjugate.

Returns a new ndarray where each element is the complex conjugate of the corresponding element in `a`. For real (double-float) arrays, this returns a copy of `a` unchanged. For complex arrays, conjugates each element (negates the imaginary part).

#### Examples

```maxima
(%i1) A : ndarray(matrix([1+%i, 2-3*%i]), complex);
(%o1)     ndarray([1, 2], COMPLEX-DOUBLE-FLOAT)
(%i2) np_to_list(np_conj(A));
(%o2)              [1.0 - 1.0*%i, 2.0 + 3.0*%i]
(%i3) /* Real arrays are unchanged */
      np_to_list(np_conj(ndarray([1, 2, 3], [3])));
(%o3)                     [1.0, 2.0, 3.0]
```

See also: `np_ctranspose`, `np_transpose`, `np_real`, `np_imag`

### Function: np_ctranspose (a)

Conjugate transpose (Hermitian transpose).

For real matrices, this is the same as `np_transpose`. For complex matrices, it transposes and takes the complex conjugate of each element.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_ctranspose(A));
(%o2)         matrix([1.0, 3.0], [2.0, 4.0])
```

See also: `np_conj`, `np_transpose`

### Function: np_real (a)

Extract real parts element-wise.

Returns a new double-float ndarray containing the real part of each element. For real arrays, this is equivalent to `np_copy`.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1+2*%i, 3-4*%i]), complex);
(%o1)     ndarray([1, 2], COMPLEX-DOUBLE-FLOAT)
(%i2) np_to_list(np_real(A));
(%o2)                     [1.0, 3.0]
```

See also: `np_imag`, `np_angle`, `np_conj`

### Function: np_imag (a)

Extract imaginary parts element-wise.

Returns a new double-float ndarray containing the imaginary part of each element. For real arrays, returns all zeros.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1+2*%i, 3-4*%i]), complex);
(%o1)     ndarray([1, 2], COMPLEX-DOUBLE-FLOAT)
(%i2) np_to_list(np_imag(A));
(%o2)                    [2.0, -4.0]
```

See also: `np_real`, `np_angle`, `np_conj`

### Function: np_angle (a)

Element-wise phase angle.

Returns a new double-float ndarray containing the phase angle (argument) of each element in radians. For real positive numbers, returns 0; for real negative numbers, returns pi.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, -1, %i]), complex);
(%o1)     ndarray([1, 3], COMPLEX-DOUBLE-FLOAT)
(%i2) np_to_list(np_angle(A));
(%o2)          [0.0, 3.141592653589793, 1.5707963267948966]
```

See also: `np_real`, `np_imag`, `np_abs`

### Function: np_expm (a)

Matrix exponential.

Computes the matrix exponential e^A, which is defined as the infinite series I + A + A^2/2! + A^3/3! + .... This is different from element-wise `np_exp`, which applies the scalar exponential to each element independently.

Uses the scaling-and-squaring method with adaptive diagonal Pade approximation (orders 3, 5, 7, 9, or 13 selected automatically based on the matrix 1-norm). The algorithm follows Higham (2005), "The Scaling and Squaring Method for the Matrix Exponential Revisited".

#### Examples

```maxima
(%i1) /* expm(0) = I */
      np_to_matrix(np_expm(np_zeros([2, 2])));
(%o1)         matrix([1.0, 0.0], [0.0, 1.0])
(%i2) /* expm(I) = e*I */
      np_ref(np_expm(np_eye(2)), 0, 0);
(%o2)                    2.718281828459045
(%i3) /* Rotation matrix from skew-symmetric */
      R : ndarray(matrix([0, -1], [1, 0]));
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) E : np_to_matrix(np_expm(R));
(%o4)  matrix([0.5403023058681398, -0.8414709848078965],
              [0.8414709848078965,  0.5403023058681398])
(%i5) /* expm(A) * expm(-A) = I */
      A : ndarray(matrix([1, 2], [3, 4]));
(%o5)            ndarray([2, 2], DOUBLE-FLOAT)
(%i6) np_to_matrix(np_matmul(np_expm(A), np_expm(np_neg(A))));
(%o6)         matrix([1.0, 0.0], [0.0, 1.0])
```

See also: `np_exp`, `np_matmul`

### Function: np_lstsq (a, b)

Least-squares solution to Ax = b.

Finds x that minimizes ||Ax - b||_2 using SVD. Works for over-determined systems (more equations than unknowns) and under-determined systems. For square full-rank matrices, gives the same result as `np_solve`.

Returns a Maxima list `[x, residuals, rank, S]` where:

- **x** — n-by-p solution ndarray
- **residuals** — 1D ndarray of squared residual norms `||Ax_j - b_j||^2` for each column j of b. Only non-empty when `m > n` (overdetermined) and A is full rank; otherwise an empty list `[]`.
- **rank** — effective rank of A (integer)
- **S** — 1D ndarray of singular values of A

**Breaking change:** Previous versions returned only `x`. Update callers from `x : np_lstsq(A, b)` to `[x, residuals, rank, S] : np_lstsq(A, b)`.

#### Examples

```maxima
(%i1) /* Linear fit: y = a + b*x through (1,1), (2,2), (3,3) */
      A : ndarray(matrix([1, 1], [1, 2], [1, 3]));
(%o1)            ndarray([3, 2], DOUBLE-FLOAT)
(%i2) b : ndarray(matrix([1], [2], [3]));
(%o2)            ndarray([3, 1], DOUBLE-FLOAT)
(%i3) [x, residuals, rank, S] : np_lstsq(A, b);
(%o3)        [ndarray, ndarray, 2, ndarray]
(%i4) np_to_list(x);
(%o4)                  [0.0, 1.0]
(%i5) rank;
(%o5)                           2
(%i6) np_to_list(S);
(%o6)             [3.86..., 0.64...]
```

See also: `np_solve`, `np_pinv`, `np_svd`

### Function: np_pinv (a)

Moore-Penrose pseudo-inverse.

Computed via SVD as A+ = V * S+ * Ut, where S+ inverts the non-zero singular values. For invertible square matrices, this is equivalent to `np_inv`. For non-square or rank-deficient matrices, it gives the best least-squares inverse.

The pseudo-inverse satisfies the four Moore-Penrose conditions: A*A+*A = A, A+*A*A+ = A+, (A*A+)^T = A*A+, and (A+*A)^T = A+*A.

For an m-by-n matrix, the pseudo-inverse is n-by-m.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4], [5, 6]));
(%o1)            ndarray([3, 2], DOUBLE-FLOAT)
(%i2) Ap : np_pinv(A);
(%o2)            ndarray([2, 3], DOUBLE-FLOAT)
(%i3) /* Verify A * pinv(A) * A = A */
      np_to_matrix(np_sub(np_matmul(np_matmul(A, Ap), A), A));
(%o3)  matrix([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
(%i4) /* pinv(A) * b gives least-squares solution */
      b : ndarray(matrix([1], [2], [3]));
(%o4)            ndarray([3, 1], DOUBLE-FLOAT)
(%i5) np_to_list(np_matmul(Ap, b));
(%o5)                  [0.0, 1.0]
```

See also: `np_inv`, `np_lstsq`, `np_svd`
