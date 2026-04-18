# Differences from NumPy

This document catalogues where the numerics package deviates from NumPy's API.
These are either deliberate design choices, Maxima-imposed constraints, or gaps
that should be addressed in future work.

## Semantic mismatches

These are cases where a function's name or behaviour conflicts with what a NumPy
user would expect. They are the most likely source of bugs.

### `np_conj` performs conjugate transpose, not conjugate

NumPy's `np.conj()` conjugates each element without transposing. Our `np_conj`
performs a full Hermitian (conjugate) transpose, equivalent to NumPy's
`A.conj().T`. For real matrices the distinction is invisible, but for complex
matrices the results differ in shape and meaning.

**Options:**
- Rename to `np_ctranspose` or `np_hermitian`
- Keep `np_conj` as element-wise conjugate and add `np_ctranspose` for the
  Hermitian transpose

### `np_eig` silently discards imaginary parts

NumPy's `np.linalg.eig()` returns complex eigenvalues and eigenvectors when the
matrix is non-symmetric. Our implementation coerces everything to
`double-float`, dropping imaginary parts without warning. This means rotation
matrices, defective matrices, and other non-symmetric inputs give quietly wrong
answers.

**Options:**
- Support complex ndarrays (significant effort — requires `complex-double-float`
  tensors throughout)
- Signal an error when eigenvalues have non-negligible imaginary parts
- Return a second value or flag indicating whether complex values were truncated

## Missing parameters

These functions exist and work correctly, but accept fewer arguments than their
NumPy counterparts.

### `np_arange` — only `np_arange(n)`

NumPy signature: `arange(start, stop, step)`. Ours only accepts a single
argument n, producing `[0, 1, ..., n-1]`. There is no way to specify start or
step.

### `np_norm` — no `ord` parameter

Always returns the 2-norm for vectors and the Frobenius norm for matrices. NumPy
supports `ord=1`, `ord=np.inf`, `ord='nuc'`, `ord=-1`, etc.

### `np_lstsq` — returns only the solution

NumPy returns a 4-tuple `(x, residuals, rank, singular_values)`. Ours returns
only `x`. The residuals, rank, and singular values are computed internally but
discarded.

### `np_diag` — creation only

NumPy's `np.diag()` serves two purposes: create a diagonal matrix from a 1D
array, or extract the diagonal of a 2D array. Ours only creates.

### Aggregations missing `axis`

`np_min`, `np_max`, `np_std`, `np_var`, `np_argmin`, `np_argmax` always reduce
to a scalar. They have no `axis` parameter. `np_sum` and `np_mean` do support
an optional axis (0 or 1, 2D only).

### `np_dot` — 1D vectors only

NumPy's `np.dot()` is overloaded: dot product for 1D, matrix multiply for 2D.
Ours only accepts 1D vectors. Use `np_matmul` for matrix multiplication.

## Structural differences

These reflect Maxima language constraints rather than design oversights.

### No operator overloading

NumPy allows `A + B`, `A @ B`, `A * B` via Python operators. Maxima does not
support operator overloading on custom types, so all operations require explicit
function calls: `np_add(A, B)`, `np_matmul(A, B)`, `np_mul(A, B)`.

### `np_hstack` / `np_vstack` take two arguments, not a list

NumPy's `np.hstack([A, B, C])` accepts a sequence of any length. Ours takes
exactly two: `np_hstack(A, B)`. Concatenating more requires chaining.

### `np_lu` returns a permutation matrix

SciPy's `scipy.linalg.lu()` returns pivot indices by default. Ours returns the
full permutation matrix P. This is arguably more convenient but differs from the
SciPy convention.

### Return types for decompositions

NumPy decompositions return tuples. Maxima has no tuple type, so ours return
Maxima lists, destructured with `[U, S, Vt] : np_svd(A)`.

### 0-based indexing

All indexing (`np_ref`, `np_set`, `np_row`, `np_col`, `np_slice`) is 0-based,
matching NumPy but differing from Maxima's native 1-based indexing for matrices
and lists. This is documented but will surprise Maxima users who are not familiar
with NumPy.

## Not implemented

Features present in NumPy/SciPy that are absent here.

| Feature | NumPy/SciPy | Notes |
|---------|-------------|-------|
| Complex dtype | `np.complex128` | Only `double-float` supported |
| Integer dtype | `np.int64` | Only `double-float` supported |
| Broadcasting | Implicit shape matching | Only scalar-array broadcasting via `np_add(A, 5)` etc. |
| Fancy indexing | `A[[0,2], :]` | Not supported |
| Boolean indexing | `A[A > 0]` | Not supported |
| `np.concatenate` | General axis concatenation | Only `np_hstack`/`np_vstack` for 2D |
| `np.sort` / `np.argsort` | Sorting | Not implemented |
| `np.where` | Conditional selection | Not implemented |
| `np.linalg.cholesky` | Cholesky decomposition | Not implemented |
| `np.linalg.cond` | Condition number | Not implemented |
| `np.linalg.slogdet` | Log-determinant | Not implemented |
| `scipy.linalg.expm` balancing | `gebal` pre/post-processing | Our `np_expm` lacks LAPACK balancing; loses accuracy on badly-scaled matrices |
