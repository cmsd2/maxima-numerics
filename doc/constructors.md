## Constructors

### Function: np_zeros (shape) / np_zeros (shape, dtype)

Create a zero-filled ndarray.

The `shape` argument is either an integer (for 1D) or a Maxima list of dimensions (for 2D or higher). Pass `complex` as the optional second argument to create a complex array.

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
(%i5) np_zeros([2, 2], complex);
(%o5)            ndarray([2, 2], COMPLEX-DOUBLE-FLOAT)
```

See also: `np_ones`, `np_full`, `np_empty`

### Function: np_ones (shape) / np_ones (shape, dtype)

Create an ndarray filled with ones.

The `shape` argument is either an integer (for 1D) or a Maxima list of dimensions. Pass `complex` as the optional second argument to create a complex array.

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
(%i5) np_ones([2, 2], complex);
(%o5)            ndarray([2, 2], COMPLEX-DOUBLE-FLOAT)
```

See also: `np_zeros`, `np_full`, `np_empty`

### Function: np_eye (n) / np_eye (n, m) / np_eye (n, m, dtype)

Create an identity matrix.

Returns an `n` x `n` identity matrix (ones on the diagonal, zeros elsewhere). An optional second argument `m` creates an `n` x `m` rectangular identity matrix. Pass `complex` as the last argument to create a complex array.

Calling forms:

- `np_eye(n)` -- square identity matrix
- `np_eye(n, m)` -- rectangular identity matrix
- `np_eye(n, m, complex)` -- complex identity matrix

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

Each element is independently drawn from a uniform distribution on the interval [0, 1). Always returns `double-float`.

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

Each element is independently drawn from a normal distribution with mean 0 and standard deviation 1, using the Box-Muller transform. Always returns `double-float`.

#### Examples

```maxima
(%i1) np_randn([3, 3]);
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) A : np_randn(10000);
(%o2)            ndarray([10000], DOUBLE-FLOAT)
```

See also: `np_rand`

### Function: np_arange (stop) / np_arange (start, stop) / np_arange (start, stop, step)

Create a 1D ndarray of evenly spaced values.

Generates values from `start` (inclusive) to `stop` (exclusive) with the given `step`. Supports non-integer step values. Always returns `double-float`.

Calling forms:

- `np_arange(stop)` -- values 0, 1, ..., stop-1
- `np_arange(start, stop)` -- values start, start+1, ..., stop-1
- `np_arange(start, stop, step)` -- values start, start+step, start+2*step, ...

#### Examples

```maxima
(%i1) np_to_list(np_arange(5));
(%o1)           [0.0, 1.0, 2.0, 3.0, 4.0]
(%i2) np_to_list(np_arange(2, 6));
(%o2)           [2.0, 3.0, 4.0, 5.0]
(%i3) np_to_list(np_arange(0, 1, 0.25));
(%o3)            [0.0, 0.25, 0.5, 0.75]
(%i4) np_to_list(np_arange(10, 0, -2));
(%o4)          [10.0, 8.0, 6.0, 4.0, 2.0]
```

See also: `np_linspace`, `np_zeros`

### Function: np_linspace (start, stop, n)

Create `n` evenly spaced points from `start` to `stop` (inclusive).

Both endpoints are included. If `n` is 1, returns a single-element array containing `start`. Always returns `double-float`.

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

### Function: np_full (shape, val) / np_full (shape, val, dtype)

Create an ndarray filled with a constant value.

Every element is set to `val`. Pass `complex` as the optional third argument to create a complex array; otherwise values are coerced to `double-float`.

#### Examples

```maxima
(%i1) np_full([2, 3], 7);
(%o1)            ndarray([2, 3], DOUBLE-FLOAT)
(%i2) np_ref(np_full([2, 2], 42), 0, 0);
(%o2)                         42.0
(%i3) np_to_list(np_full(3, %pi));
(%o3)  [3.141592653589793, 3.141592653589793, 3.141592653589793]
(%i4) np_full([2, 2], 1+2*%i, complex);
(%o4)            ndarray([2, 2], COMPLEX-DOUBLE-FLOAT)
```

See also: `np_zeros`, `np_ones`

### Function: np_empty (shape) / np_empty (shape, dtype)

Create an uninitialized ndarray.

The contents are unspecified and may contain arbitrary values. Use this when you plan to fill every element before reading, as it avoids the cost of zero-initialization. Pass `complex` as the optional second argument to create a complex array.

#### Examples

```maxima
(%i1) A : np_empty([3, 3]);
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) np_shape(A);
(%o2)                        [3, 3]
(%i3) np_empty([2, 2], complex);
(%o3)            ndarray([2, 2], COMPLEX-DOUBLE-FLOAT)
```

See also: `np_zeros`, `np_full`

### Function: np_diag (list) / np_diag (list, dtype)

Create a diagonal matrix from a Maxima list.

The argument must be a Maxima list of numbers. Returns a square ndarray with the list elements on the main diagonal and zeros elsewhere. Pass `complex` as the optional second argument to create a complex array.

#### Examples

```maxima
(%i1) np_diag([1, 2, 3]);
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_diag([1, 2, 3]));
(%o2)  matrix([1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0])
(%i3) np_ref(np_diag([5, 10]), 0, 1);
(%o3)                          0.0
(%i4) np_diag([1+%i, 2-%i], complex);
(%o4)            ndarray([2, 2], COMPLEX-DOUBLE-FLOAT)
```

See also: `np_eye`, `np_zeros`

### Function: np_copy (a)

Create a deep copy of an ndarray.

Returns a new ndarray with the same shape, dtype, and values. Modifications to the copy do not affect the original.

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

### Function: np_logspace (start, stop, n)

Create `n` logarithmically spaced points from 10^start to 10^stop (inclusive).

Both endpoints are included. If `n` is 1, returns a single-element array containing 10^start. Always returns `double-float`.

#### Examples

```maxima
(%i1) np_to_list(np_logspace(0, 3, 4));
(%o1)           [1.0, 10.0, 100.0, 1000.0]
(%i2) np_to_list(np_logspace(2, 5, 1));
(%o2)                      [100.0]
(%i3) np_size(np_logspace(0, 1, 50));
(%o3)                         50
```

See also: `np_linspace`, `np_arange`

---

### Function: np_seed (n)

Set the random seed for reproducibility. Affects `np_rand`, `np_randn`, `np_randint`, `np_choice`, and `np_shuffle`.

**Parameters:** `n` — integer seed value

#### Examples

```maxima
(%i1) np_seed(42)$
(%i2) A : np_rand([3])$
(%i3) np_seed(42)$
(%i4) B : np_rand([3])$
(%i5) is(np_max(np_abs(np_sub(A, B))) = 0.0);
(%o5)                          true
```

See also: `np_rand`, `np_randn`

### Function: np_randint (lo, hi, shape)

Create an ndarray of random integers in [lo, hi). Values are stored as `double-float`.

**Parameters:**
- `lo` — lower bound (inclusive)
- `hi` — upper bound (exclusive)
- `shape` — integer or list of dimensions

#### Examples

```maxima
(%i1) np_seed(42)$
(%i2) A : np_randint(0, 10, [2, 3]);
(%o2)            ndarray([2, 3], DOUBLE-FLOAT)
(%i3) is(np_min(A) >= 0 and np_max(A) <= 9);
(%o3)                          true
```

See also: `np_rand`, `np_seed`

### Function: np_choice (a, n) / np_choice (a, n, replace)

Sample `n` elements from a 1D ndarray. By default samples with replacement.

**Parameters:**
- `a` — 1D ndarray to sample from
- `n` — number of samples
- `replace` — (optional) `true` for with replacement (default), `false` for without replacement

#### Examples

```maxima
(%i1) np_seed(42)$
(%i2) A : np_linspace(1, 5, 5)$
(%i3) np_to_list(np_choice(A, 3));
(%o3)                    [4.0, 2.0, 1.0]
(%i4) B : np_choice(A, 3, false)$
(%i5) np_size(B);
(%o5)                          3
```

See also: `np_shuffle`, `np_seed`, `np_rand`

### Function: np_shuffle (a)

In-place Fisher-Yates shuffle of a 1D ndarray. Modifies `a` in place and returns it.

**Parameters:** `a` — 1D ndarray

**Returns:** `a` (shuffled in place)

#### Examples

```maxima
(%i1) np_seed(42)$
(%i2) A : np_linspace(1, 5, 5)$
(%i3) np_shuffle(A)$
(%i4) np_to_list(A);
(%o4)               [4.0, 1.0, 2.0, 5.0, 3.0]
```

See also: `np_choice`, `np_seed`
