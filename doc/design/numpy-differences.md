# Differences from NumPy

This document catalogues where the numerics package deviates from NumPy's API.
These are either deliberate design choices, Maxima-imposed constraints, or gaps
that should be addressed in future work.

## Usability summary

The package has two very different halves:

**Linear algebra — excellent.** SVD, QR, LU, lstsq, expm, pinv are all
polished, correct, and close to NumPy semantics. A user doing matrix
computations would feel at home.

**Array manipulation — painful.** Filtering, masking, and data transformation
workflows that NumPy users rely on are largely missing or require verbose
workarounds.

### Key friction points

**1. No broadcasting.** Only scalar-array works (`np_add(A, 3.0)`). You cannot:

- Add a row vector to every row of a matrix
- Normalize columns by their means (subtract `np_mean(A, 0)` from `A`)

General broadcasting (shape alignment) is architecturally complex and likely not
worth the effort for a Maxima package.

**2. No fancy indexing.** Without `A[[0, 2, 4]]`, selecting elements by position
requires `np_row`/`np_col` one at a time. A `np_take(A, indices)` function would
address this gap.

### Resolved friction points

These were previously identified as major gaps and have been implemented:

- **Comparison functions**: `np_greater`, `np_less`, `np_equal`,
  `np_greater_equal`, `np_less_equal`, `np_not_equal` returning 1.0/0.0 ndarrays
- **Logical combinators**: `np_logical_and`, `np_logical_or`, `np_logical_not`
- **Scalar broadcasting in `np_where`**: `np_where(cond, x, 0)` now works
- **Boolean indexing**: `np_extract(mask, A)` selects elements where mask is
  nonzero
- **Predicate testing**: `np_test(f, A)` applies named or lambda predicates
- **Sorting**: `np_sort`, `np_argsort` with optional axis

The filter-and-transform workflow now works:

```maxima
/* NumPy: A[A > 5] */
np_extract(np_greater(A, 5), A);

/* NumPy: np.where(A > 5, A, 0) */
np_where(np_greater(A, 5), A, 0);

/* NumPy: A[(A > 2) & (A < 8)] */
np_extract(np_logical_and(np_greater(A, 2), np_less(A, 8)), A);
```

## Semantic mismatches

These are cases where a function's name or behaviour conflicts with what a NumPy
user would expect. They are the most likely source of bugs.

### `np_eig` — no remaining semantic issues

`np_eig` delegates to `magicl:eig` which calls `dgeev` for real matrices and
`zgeev` for complex matrices. For real input with complex eigenvalues (e.g.
rotation matrices), it detects non-negligible imaginary parts and returns
complex-typed ndarrays. Complex input always produces complex output. This
matches NumPy's `np.linalg.eig()` behaviour.

## Missing parameters

These functions exist and work correctly, but accept fewer arguments than their
NumPy counterparts.

### `np_diag` — creation only

NumPy's `np.diag()` serves two purposes: create a diagonal matrix from a 1D
array, or extract the diagonal of a 2D array. Ours only creates.

### Aggregations — 2D axis only

`np_sum`, `np_mean`, `np_min`, `np_max`, `np_std`, `np_var`, `np_argmin`, and
`np_argmax` support an optional `axis` parameter (0 or 1), but only for 2D
arrays. NumPy supports arbitrary dimensions and axis values.

### `np_dot` — 1D vectors only

NumPy's `np.dot()` is overloaded: dot product for 1D, matrix multiply for 2D.
Ours only accepts 1D vectors. Use `np_matmul` for matrix multiplication.

## Structural differences

These reflect Maxima language constraints rather than design oversights.

### No operator overloading

NumPy allows `A + B`, `A @ B`, `A * B` via Python operators. Maxima does not
support operator overloading on custom types, so all operations require explicit
function calls: `np_add(A, B)`, `np_matmul(A, B)`, `np_mul(A, B)`.

### No comparison operators on ndarrays

NumPy's `A > 5` returns a boolean array. Maxima's `>` returns a single boolean
and cannot operate element-wise on ndarrays. Use the explicit comparison
functions instead: `np_greater(A, 5)`, `np_less(A, 3)`, etc. These return
1.0/0.0 ndarrays suitable for use with `np_where`, `np_extract`, and logical
combinators.

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

## Complex number support

Complex ndarrays are supported with dtype `:complex-double-float`. Create them
via `ndarray(matrix(...), complex)` or constructors with the `complex` flag:

```maxima
A : ndarray(matrix([1+%i, 2-3*%i]), complex);
B : np_zeros([3,3], complex);
C : np_full([2,2], 1+2*%i, complex);
```

Most operations propagate dtype correctly: element-wise arithmetic, matrix
multiply, transpose, SVD, QR, LU, etc. Complex-specific functions:

- `np_conj(A)` — element-wise conjugate
- `np_ctranspose(A)` — conjugate transpose
- `np_real(A)`, `np_imag(A)` — extract real/imaginary parts (always real result)
- `np_angle(A)` — element-wise phase angle (always real result)
- `np_eig(A)` — returns complex eigenvalues when they have non-negligible
  imaginary parts

**Not supported for complex arrays** (signals error):
`np_greater`, `np_less`, `np_greater_equal`, `np_less_equal`, `np_min`, `np_max`,
`np_argmin`, `np_argmax`, `np_sort`, `np_argsort`. Use `np_equal`/`np_not_equal`
for equality testing on complex arrays.

## Not implemented

Features present in NumPy/SciPy that are absent here.

| Feature | NumPy/SciPy | Notes |
|---------|-------------|-------|
| Integer dtype | `np.int64` | Only `double-float` and `complex-double-float` supported |
| Broadcasting | Implicit shape matching | Only scalar-array broadcasting via `np_add(A, 5)` etc. |
| Fancy indexing | `A[[0,2], :]` | Not supported; use `np_row`/`np_col` per element |
| `np.concatenate` | General axis concatenation | Only `np_hstack`/`np_vstack` for 2D |
| `np.linalg.cholesky` | Cholesky decomposition | Not implemented |
| `np.linalg.cond` | Condition number | Not implemented |
| `np.linalg.slogdet` | Log-determinant | Not implemented |
| `scipy.linalg.expm` balancing | `gebal` pre/post-processing | Our `np_expm` lacks LAPACK balancing; loses accuracy on badly-scaled matrices |
