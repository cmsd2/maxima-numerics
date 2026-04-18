## Aggregation

### Function: np_sum (a)

Sum of array elements.

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

Mean (average) of array elements.

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

### Function: np_min (a)

Minimum element of an ndarray.

Returns the smallest element as a scalar.

#### Examples

```maxima
(%i1) A : ndarray(matrix([3, 1], [4, 2]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_min(A);
(%o2)                          1.0
(%i3) np_min(np_arange(10));
(%o3)                          0.0
```

See also: `np_max`, `np_argmin`

### Function: np_max (a)

Maximum element of an ndarray.

Returns the largest element as a scalar.

#### Examples

```maxima
(%i1) A : ndarray(matrix([3, 1], [4, 2]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_max(A);
(%o2)                          4.0
(%i3) np_max(np_arange(10));
(%o3)                          9.0
```

See also: `np_min`, `np_argmax`

### Function: np_argmin (a)

Index of the minimum element in storage order.

Returns a 0-based integer index into the flattened (column-major) storage of the array.

#### Examples

```maxima
(%i1) A : ndarray(matrix([3, 1], [4, 2]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_argmin(A);
(%o2)                           2
(%i3) np_argmin(np_arange(5));
(%o3)                           0
```

See also: `np_argmax`, `np_min`

### Function: np_argmax (a)

Index of the maximum element in storage order.

Returns a 0-based integer index into the flattened (column-major) storage of the array.

#### Examples

```maxima
(%i1) A : ndarray(matrix([3, 1], [4, 2]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_argmax(A);
(%o2)                           1
(%i3) np_argmax(np_arange(5));
(%o3)                           4
```

See also: `np_argmin`, `np_max`

### Function: np_std (a)

Standard deviation of array elements.

Computes the population standard deviation (divides by N, not N-1). Returns a scalar.

#### Examples

```maxima
(%i1) A : ndarray(matrix([2, 4], [4, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_std(A);
(%o2)                    0.8660254037844386
```

See also: `np_var`, `np_mean`

### Function: np_var (a)

Variance of array elements.

Computes the population variance (divides by N, not N-1). Returns a scalar.

#### Examples

```maxima
(%i1) A : ndarray(matrix([2, 4], [4, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_var(A);
(%o2)                          0.75
(%i3) is(np_std(A)^2 - np_var(A) < 1e-10);
(%o3)                         true
```

See also: `np_std`, `np_mean`

### Function: np_cumsum (a)

Cumulative sum of a 1D ndarray.

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

Dot product of two 1D vectors.

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
