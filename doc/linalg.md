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

Only the real parts of eigenvalues and eigenvectors are returned (imaginary parts are discarded). For matrices with complex eigenvalues (e.g. rotation matrices), the results will be incomplete.

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

### Function: np_norm (a)

Matrix or vector norm.

For 1D arrays (vectors), computes the Euclidean (L2) norm: sqrt(sum of squares). For 2D arrays (matrices), computes the Frobenius norm: sqrt(sum of all squared elements). Returns a scalar.

#### Examples

```maxima
(%i1) v : np_arange(5);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_norm(v);
(%o2)                    5.477225575051661
(%i3) A : ndarray(matrix([1, 2], [3, 4]));
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_norm(A);
(%o4)                    5.477225575051661
(%i5) np_norm(np_eye(3));
(%o5)                    1.7320508075688772
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

See also: `np_conj`, `np_reshape`

### Function: np_conj (a)

Conjugate transpose (Hermitian transpose).

For real matrices, this is the same as `np_transpose`. For complex matrices, it transposes and takes the complex conjugate of each element.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_conj(A));
(%o2)         matrix([1.0, 3.0], [2.0, 4.0])
```

See also: `np_transpose`

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

For an m-by-n matrix A and an m-by-p right-hand side b, returns an n-by-p solution x.

#### Examples

```maxima
(%i1) /* Linear fit: y = a + b*x through (1,1), (2,2), (3,3) */
      A : ndarray(matrix([1, 1], [1, 2], [1, 3]));
(%o1)            ndarray([3, 2], DOUBLE-FLOAT)
(%i2) b : ndarray(matrix([1], [2], [3]));
(%o2)            ndarray([3, 1], DOUBLE-FLOAT)
(%i3) x : np_lstsq(A, b);
(%o3)            ndarray([2, 1], DOUBLE-FLOAT)
(%i4) np_to_list(x);
(%o4)                  [0.0, 1.0]
(%i5) /* Residual is zero (consistent system) */
      np_norm(np_sub(np_matmul(A, x), b));
(%o5)                          0.0
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
