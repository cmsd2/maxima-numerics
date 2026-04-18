## Constructors

### Function: np_zeros (shape)

Create a zero-filled ndarray.

The `shape` argument is either an integer (for 1D) or a Maxima list of dimensions (for 2D or higher).

#### Examples

```maxima
(%i1) np_zeros(5);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_to_list(np_zeros(3));
(%o2)                  [0.0, 0.0, 0.0]
(%i3) np_zeros([2, 3]);
(%o3)            ndarray([2, 3], DOUBLE-FLOAT)
(%i4) np_ref(np_zeros([2, 2]), 0, 0);
(%o4)                          0.0
```

See also: `np_ones`, `np_full`, `np_empty`

### Function: np_ones (shape)

Create an ndarray filled with ones.

The `shape` argument is either an integer (for 1D) or a Maxima list of dimensions.

#### Examples

```maxima
(%i1) np_ones(4);
(%o1)            ndarray([4], DOUBLE-FLOAT)
(%i2) np_to_list(np_ones(3));
(%o2)                  [1.0, 1.0, 1.0]
(%i3) np_ones([2, 2]);
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_ref(np_ones([3, 3]), 1, 1);
(%o4)                          1.0
```

See also: `np_zeros`, `np_full`, `np_empty`

### Function: np_eye (n)

Create an identity matrix.

Returns an `n` x `n` identity matrix (ones on the diagonal, zeros elsewhere). An optional second argument `m` creates an `n` x `m` rectangular identity matrix.

Calling forms:

- `np_eye(n)` -- square identity matrix
- `np_eye(n, m)` -- rectangular identity matrix

#### Examples

```maxima
(%i1) np_eye(3);
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_eye(3));
(%o2)  matrix([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
(%i3) np_ref(np_eye(3), 0, 0);
(%o3)                          1.0
(%i4) np_ref(np_eye(3), 0, 1);
(%o4)                          0.0
```

See also: `np_diag`, `np_zeros`, `np_ones`

### Function: np_rand (shape)

Create an ndarray filled with uniform random values in [0, 1).

Each element is independently drawn from a uniform distribution on the interval [0, 1).

#### Examples

```maxima
(%i1) np_rand([2, 3]);
(%o1)            ndarray([2, 3], DOUBLE-FLOAT)
(%i2) A : np_rand(5);
(%o2)            ndarray([5], DOUBLE-FLOAT)
(%i3) is(np_min(A) >= 0 and np_max(A) < 1);
(%o3)                         true
```

See also: `np_randn`, `np_zeros`, `np_ones`

### Function: np_randn (shape)

Create an ndarray filled with standard normal random values.

Each element is independently drawn from a normal distribution with mean 0 and standard deviation 1, using the Box-Muller transform.

#### Examples

```maxima
(%i1) np_randn([3, 3]);
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) A : np_randn(10000);
(%o2)            ndarray([10000], DOUBLE-FLOAT)
```

See also: `np_rand`

### Function: np_arange (n)

Create a 1D ndarray with values 0, 1, 2, ..., n-1.

The argument `n` is truncated to an integer. Returned values are double-float.

#### Examples

```maxima
(%i1) np_arange(5);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_to_list(np_arange(5));
(%o2)           [0.0, 1.0, 2.0, 3.0, 4.0]
(%i3) np_ref(np_arange(10), 7);
(%o3)                          7.0
```

See also: `np_linspace`, `np_zeros`

### Function: np_linspace (start, stop, n)

Create `n` evenly spaced points from `start` to `stop` (inclusive).

Both endpoints are included. If `n` is 1, returns a single-element array containing `start`.

#### Examples

```maxima
(%i1) np_linspace(0, 1, 5);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_to_list(np_linspace(0, 1, 5));
(%o2)           [0.0, 0.25, 0.5, 0.75, 1.0]
(%i3) np_to_list(np_linspace(0, 10, 3));
(%o3)                 [0.0, 5.0, 10.0]
```

See also: `np_arange`

### Function: np_full (shape, val)

Create an ndarray filled with a constant value.

Every element is set to `val`, which is coerced to double-float.

#### Examples

```maxima
(%i1) np_full([2, 3], 7);
(%o1)            ndarray([2, 3], DOUBLE-FLOAT)
(%i2) np_ref(np_full([2, 2], 42), 0, 0);
(%o2)                         42.0
(%i3) np_to_list(np_full(3, %pi));
(%o3)  [3.141592653589793, 3.141592653589793, 3.141592653589793]
```

See also: `np_zeros`, `np_ones`

### Function: np_empty (shape)

Create an uninitialized ndarray.

The contents are unspecified and may contain arbitrary values. Use this when you plan to fill every element before reading, as it avoids the cost of zero-initialization.

#### Examples

```maxima
(%i1) A : np_empty([3, 3]);
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) np_shape(A);
(%o2)                        [3, 3]
```

See also: `np_zeros`, `np_full`

### Function: np_diag (list)

Create a diagonal matrix from a Maxima list.

The argument must be a Maxima list of numbers. Returns a square ndarray with the list elements on the main diagonal and zeros elsewhere.

#### Examples

```maxima
(%i1) np_diag([1, 2, 3]);
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_diag([1, 2, 3]));
(%o2)  matrix([1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0])
(%i3) np_ref(np_diag([5, 10]), 0, 1);
(%o3)                          0.0
```

See also: `np_eye`, `np_zeros`

### Function: np_copy (a)

Create a deep copy of an ndarray.

Returns a new ndarray with the same shape and values. Modifications to the copy do not affect the original.

#### Examples

```maxima
(%i1) A : np_ones([2, 2]);
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) B : np_copy(A);
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) np_set(B, 0, 0, 99.0);
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_ref(A, 0, 0);
(%o4)                          1.0
(%i5) np_ref(B, 0, 0);
(%o5)                         99.0
```

See also: `ndarray`, `np_reshape`
