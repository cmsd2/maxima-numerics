## Aggregation

### Function: np_sum (a)

Sum of array elements. Supports complex arrays.

Without an axis argument, returns the scalar sum of all elements. With an axis argument, sums along that axis and returns an ndarray.

Calling forms:

- `np_sum(a)` -- total sum, returns a scalar
- `np_sum(a, 0)` -- sum across rows (column sums), returns a 1D ndarray of length `ncol`
- `np_sum(a, 1)` -- sum across columns (row sums), returns a 1D ndarray of length `nrow`

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_sum(A);
(%o2)                         10.0
(%i3) np_to_list(np_sum(A, 0));
(%o3)                     [4.0, 6.0]
(%i4) np_to_list(np_sum(A, 1));
(%o4)                     [3.0, 7.0]
```

See also: `np_mean`, `np_cumsum`

### Function: np_mean (a)

Mean (average) of array elements. Supports complex arrays.

Without an axis argument, returns the scalar mean of all elements. With an axis argument, computes the mean along that axis.

Calling forms:

- `np_mean(a)` -- total mean, returns a scalar
- `np_mean(a, 0)` -- column means, returns a 1D ndarray
- `np_mean(a, 1)` -- row means, returns a 1D ndarray

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_mean(A);
(%o2)                          2.5
(%i3) np_to_list(np_mean(A, 0));
(%o3)                     [2.0, 3.0]
(%i4) np_to_list(np_mean(A, 1));
(%o4)                    [1.5, 3.5]
```

See also: `np_sum`, `np_std`, `np_var`

### Function: np_min (a) / np_min (a, axis)

Minimum element of an ndarray. Signals an error for complex arrays.

Without an axis argument, returns the smallest element as a scalar. With an axis argument, computes the minimum along that axis and returns an ndarray.

Calling forms:

- `np_min(a)` -- global minimum, returns a scalar
- `np_min(a, 0)` -- minimum across rows (column mins), returns a 1D ndarray of length `ncol`
- `np_min(a, 1)` -- minimum across columns (row mins), returns a 1D ndarray of length `nrow`

#### Examples

```maxima
(%i1) A : ndarray(matrix([3, 1], [4, 2]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_min(A);
(%o2)                          1.0
(%i3) np_to_list(np_min(A, 0));
(%o3)                     [3.0, 1.0]
(%i4) np_to_list(np_min(A, 1));
(%o4)                     [1.0, 2.0]
```

See also: `np_max`, `np_argmin`

### Function: np_max (a) / np_max (a, axis)

Maximum element of an ndarray. Signals an error for complex arrays.

Without an axis argument, returns the largest element as a scalar. With an axis argument, computes the maximum along that axis and returns an ndarray.

Calling forms:

- `np_max(a)` -- global maximum, returns a scalar
- `np_max(a, 0)` -- maximum across rows (column maxes), returns a 1D ndarray of length `ncol`
- `np_max(a, 1)` -- maximum across columns (row maxes), returns a 1D ndarray of length `nrow`

#### Examples

```maxima
(%i1) A : ndarray(matrix([3, 1], [4, 2]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_max(A);
(%o2)                          4.0
(%i3) np_to_list(np_max(A, 0));
(%o3)                     [4.0, 2.0]
(%i4) np_to_list(np_max(A, 1));
(%o4)                     [3.0, 4.0]
```

See also: `np_min`, `np_argmax`

### Function: np_argmin (a) / np_argmin (a, axis)

Index of the minimum element. Signals an error for complex arrays.

Without an axis argument, returns a 0-based integer index into the flattened (column-major) storage. With an axis argument, returns a 1D ndarray of indices along the specified axis.

Calling forms:

- `np_argmin(a)` -- flat index of global minimum, returns an integer
- `np_argmin(a, 0)` -- row index of minimum per column, returns a 1D ndarray
- `np_argmin(a, 1)` -- column index of minimum per row, returns a 1D ndarray

#### Examples

```maxima
(%i1) A : ndarray(matrix([3, 1], [4, 2]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_argmin(A);
(%o2)                           2
(%i3) np_to_list(np_argmin(A, 0));
(%o3)                     [0.0, 0.0]
(%i4) np_to_list(np_argmin(A, 1));
(%o4)                     [1.0, 1.0]
```

See also: `np_argmax`, `np_min`

### Function: np_argmax (a) / np_argmax (a, axis)

Index of the maximum element. Signals an error for complex arrays.

Without an axis argument, returns a 0-based integer index into the flattened (column-major) storage. With an axis argument, returns a 1D ndarray of indices along the specified axis.

Calling forms:

- `np_argmax(a)` -- flat index of global maximum, returns an integer
- `np_argmax(a, 0)` -- row index of maximum per column, returns a 1D ndarray
- `np_argmax(a, 1)` -- column index of maximum per row, returns a 1D ndarray

#### Examples

```maxima
(%i1) A : ndarray(matrix([3, 1], [4, 2]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_argmax(A);
(%o2)                           1
(%i3) np_to_list(np_argmax(A, 0));
(%o3)                     [1.0, 1.0]
(%i4) np_to_list(np_argmax(A, 1));
(%o4)                     [0.0, 0.0]
```

See also: `np_argmin`, `np_max`

### Function: np_var (a) / np_var (a, axis)

Variance of array elements.

Computes the population variance (divides by N, not N-1). Without an axis argument, returns a scalar. With an axis argument, computes variance along that axis. For complex arrays, computes |x - mean|^2; the result is always `double-float`.

Calling forms:

- `np_var(a)` -- total variance, returns a scalar
- `np_var(a, 0)` -- column variances, returns a 1D ndarray
- `np_var(a, 1)` -- row variances, returns a 1D ndarray

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 5], [3, 7]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_var(A);
(%o2)                          5.0
(%i3) np_to_list(np_var(A, 0));
(%o3)                     [1.0, 1.0]
(%i4) np_to_list(np_var(A, 1));
(%o4)                     [4.0, 4.0]
```

See also: `np_std`, `np_mean`

### Function: np_std (a) / np_std (a, axis)

Standard deviation of array elements.

Computes the population standard deviation (divides by N, not N-1). Without an axis argument, returns a scalar. With an axis argument, computes standard deviation along that axis. The result is always `double-float`, even for complex input.

Calling forms:

- `np_std(a)` -- total standard deviation, returns a scalar
- `np_std(a, 0)` -- column standard deviations, returns a 1D ndarray
- `np_std(a, 1)` -- row standard deviations, returns a 1D ndarray

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 5], [3, 7]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_std(A);
(%o2)                    2.23606797749979
(%i3) np_to_list(np_std(A, 0));
(%o3)                     [1.0, 1.0]
(%i4) np_to_list(np_std(A, 1));
(%o4)                     [2.0, 2.0]
```

See also: `np_var`, `np_mean`

### Function: np_cumsum (a)

Cumulative sum of a 1D ndarray. Supports complex arrays.

Returns a new 1D ndarray where element `i` is the sum of elements 0 through `i` of the input.

#### Examples

```maxima
(%i1) A : np_arange(5);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_to_list(np_cumsum(A));
(%o2)           [0.0, 1.0, 3.0, 6.0, 10.0]
(%i3) np_to_list(np_cumsum(np_ones(4)));
(%o3)              [1.0, 2.0, 3.0, 4.0]
```

See also: `np_sum`

### Function: np_dot (a, b)

Dot product of two 1D vectors. Supports complex arrays.

Both arguments must be 1D ndarrays of the same length. Returns a scalar.

#### Examples

```maxima
(%i1) a : np_arange(3);
(%o1)            ndarray([3], DOUBLE-FLOAT)
(%i2) b : np_ones(3);
(%o2)            ndarray([3], DOUBLE-FLOAT)
(%i3) np_dot(a, b);
(%o3)                          3.0
(%i4) np_dot(a, a);
(%o4)                          5.0
```

See also: `np_matmul`, `np_sum`

### Function: np_sort (a) / np_sort (a, axis)

Sort array elements in ascending order. Signals an error for complex arrays.

Without an axis argument, flattens to 1D and sorts. With an axis argument, sorts along the specified axis (preserving shape). Uses a stable sort — equal elements maintain their relative order. Returns a new ndarray; the input is not modified.

Calling forms:

- `np_sort(a)` -- flatten and sort, returns a 1D ndarray
- `np_sort(a, 0)` -- sort each column independently, returns same shape
- `np_sort(a, 1)` -- sort each row independently, returns same shape

#### Examples

```maxima
(%i1) A : ndarray([3, 1, 4, 1, 5], [5]);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_to_list(np_sort(A));
(%o2)           [1.0, 1.0, 3.0, 4.0, 5.0]
(%i3) B : ndarray(matrix([3, 1], [1, 4]));
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_to_matrix(np_sort(B, 0));
(%o4)         matrix([1.0, 1.0], [3.0, 4.0])
(%i5) np_to_matrix(np_sort(B, 1));
(%o5)         matrix([1.0, 3.0], [1.0, 4.0])
```

See also: `np_argsort`, `np_min`, `np_max`

### Function: np_argsort (a) / np_argsort (a, axis)

Indices that would sort array elements in ascending order. Signals an error for complex arrays.

Without an axis argument, returns flat indices for the flattened array. With an axis argument, returns indices along the specified axis. Indices are stored as double-float values. Uses a stable sort.

Calling forms:

- `np_argsort(a)` -- flat indices, returns a 1D ndarray
- `np_argsort(a, 0)` -- row indices that sort each column, returns same shape
- `np_argsort(a, 1)` -- column indices that sort each row, returns same shape

#### Examples

```maxima
(%i1) A : ndarray([3, 1, 2], [3]);
(%o1)            ndarray([3], DOUBLE-FLOAT)
(%i2) np_to_list(np_argsort(A));
(%o2)                  [1.0, 2.0, 0.0]
(%i3) B : ndarray(matrix([3, 1], [1, 4]));
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_to_matrix(np_argsort(B, 0));
(%o4)         matrix([1.0, 0.0], [0.0, 1.0])
(%i5) np_to_matrix(np_argsort(B, 1));
(%o5)         matrix([1.0, 0.0], [0.0, 1.0])
```

See also: `np_sort`, `np_argmin`, `np_argmax`

### Function: np_trapz (y) / np_trapz (y, x)

Trapezoidal numerical integration of a 1D ndarray.

Without an `x` argument, uses unit spacing (dx = 1). With an `x` argument, uses the spacing defined by the `x` array. Both `y` and `x` must be 1D ndarrays of the same length. Returns a scalar.

Calling forms:

- `np_trapz(y)` -- integrate with unit spacing
- `np_trapz(y, x)` -- integrate with variable spacing defined by x

#### Examples

```maxima
(%i1) /* Unit spacing: integral of [1, 2, 3] = 0.5*(1+2) + 0.5*(2+3) = 4 */
      np_trapz(ndarray([1, 2, 3], [3]));
(%o1)                          4.0
(%i2) /* Variable spacing: y=x^2 on [0,1,2] */
      x : ndarray([0, 1, 2], [3]);
(%o2)            ndarray([3], DOUBLE-FLOAT)
(%i3) y : ndarray([0, 1, 4], [3]);
(%o3)            ndarray([3], DOUBLE-FLOAT)
(%i4) np_trapz(y, x);
(%o4)                          3.0
```

See also: `np_sum`, `np_cumsum`, `np_diff`

### Function: np_diff (a)

First-order finite differences of a 1D ndarray.

Returns a new 1D ndarray of length n-1 where element i is `a[i+1] - a[i]`. The input must be a 1D ndarray with at least 2 elements.

#### Examples

```maxima
(%i1) np_to_list(np_diff(ndarray([1, 3, 6, 10], [4])));
(%o1)                  [2.0, 3.0, 4.0]
(%i2) np_size(np_diff(np_arange(10)));
(%o2)                           9
(%i3) /* Constant array: all diffs are zero */
      np_to_list(np_diff(np_ones([5])));
(%o3)              [0.0, 0.0, 0.0, 0.0]
```

See also: `np_cumsum`, `np_trapz`

### Function: np_cov (a)

Sample covariance matrix of a 2D ndarray.

Treats each column as a variable and each row as an observation. Returns a p-by-p covariance matrix (where p is the number of columns) using the sample covariance formula (divides by n-1). The result is symmetric.

#### Examples

```maxima
(%i1) /* 3 observations of 2 variables, perfectly correlated */
      A : ndarray(matrix([1, 2], [3, 6], [5, 10]));
(%o1)            ndarray([3, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_cov(A));
(%o2)         matrix([4.0, 8.0], [8.0, 16.0])
(%i3) /* Symmetry: C[i,j] = C[j,i] */
      C : np_to_matrix(np_cov(ndarray(matrix([1,2,3],[4,5,6],[7,8,9]))));
(%o3)  matrix([9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0])
```

See also: `np_corrcoef`, `np_var`, `np_std`

### Function: np_corrcoef (a)

Pearson correlation coefficient matrix of a 2D ndarray.

Treats each column as a variable and each row as an observation. Returns a p-by-p correlation matrix where element (i,j) is the Pearson correlation between columns i and j. Diagonal elements are always 1.0. Built on `np_cov`.

#### Examples

```maxima
(%i1) /* Perfect positive correlation gives 1 */
      A : ndarray(matrix([1, 2], [3, 6], [5, 10]));
(%o1)            ndarray([3, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_corrcoef(A));
(%o2)         matrix([1.0, 1.0], [1.0, 1.0])
(%i3) /* Anti-correlation gives -1 */
      B : ndarray(matrix([1, 10], [3, 6], [5, 2]));
(%o3)            ndarray([3, 2], DOUBLE-FLOAT)
(%i4) np_to_matrix(np_corrcoef(B));
(%o4)        matrix([1.0, -1.0], [-1.0, 1.0])
```

See also: `np_cov`, `np_var`, `np_std`
