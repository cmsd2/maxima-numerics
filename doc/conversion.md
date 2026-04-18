## Conversion

### Function: ndarray (a) / ndarray (a, complex)

Convert a Maxima matrix or list to an ndarray.

If `a` is a Maxima matrix, it is converted element-by-element into a `double-float` ndarray with column-major layout. If `a` is a Maxima list, it creates a 1D ndarray unless a shape argument is provided. Pass `complex` as the last argument to create a `complex-double-float` array.

If `a` is already an ndarray, it is returned unchanged.

Calling forms:

- `ndarray(matrix)` -- convert a Maxima matrix to a 2D ndarray
- `ndarray(matrix, complex)` -- convert to a complex 2D ndarray
- `ndarray(list)` -- convert a flat Maxima list to a 1D ndarray
- `ndarray(list, shape)` -- convert a list and reshape to the given dimensions
- `ndarray(list, shape, complex)` -- convert a list to a complex ndarray with shape

The shape argument is a Maxima list of dimensions, e.g. `[2, 3]`. The total number of elements in the list must match the product of the dimensions.

Elements are filled in row-major order from the list, matching the convention that `ndarray([1,2,3,4], [2,2])` produces a matrix with first row `[1, 2]` and second row `[3, 4]`.

#### Examples

```maxima
(%i1) load("numerics")$
(%i2) A : ndarray(matrix([1, 2], [3, 4]));
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) np_ref(A, 0, 0);
(%o3)                          1.0
(%i4) B : ndarray([1, 2, 3, 4, 5, 6], [2, 3]);
(%o4)            ndarray([2, 3], DOUBLE-FLOAT)
(%i5) np_shape(B);
(%o5)                        [2, 3]
(%i6) C : ndarray([10, 20, 30]);
(%o6)            ndarray([3], DOUBLE-FLOAT)
(%i7) D : ndarray(matrix([1+%i, 2-3*%i], [4, 5+%i]), complex);
(%o7)            ndarray([2, 2], COMPLEX-DOUBLE-FLOAT)
```

See also: `np_to_matrix`, `np_to_list`, `ndarray_p`

### Function: ndarray_p (x)

Test whether `x` is an ndarray.

Returns `true` if `x` is an ndarray handle, `false` otherwise.

#### Examples

```maxima
(%i1) ndarray_p(np_zeros([2, 2]));
(%o1)                         true
(%i2) ndarray_p(42);
(%o2)                        false
(%i3) ndarray_p(matrix([1, 2]));
(%o3)                        false
```

See also: `ndarray`

### Function: np_to_matrix (a)

Convert a 2D ndarray back to a Maxima matrix.

The ndarray must be 2-dimensional. For 1D ndarrays, use `np_to_list` instead, or reshape to 2D first with `np_reshape`. Complex elements are returned in Maxima form (`a + b*%i`).

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(A);
(%o2)         matrix([1.0, 2.0], [3.0, 4.0])
(%i3) B : np_eye(3);
(%o3)            ndarray([3, 3], DOUBLE-FLOAT)
(%i4) np_to_matrix(B);
(%o4)  matrix([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
```

See also: `ndarray`, `np_to_list`, `np_reshape`

### Function: np_to_list (a)

Flatten an ndarray to a Maxima list.

Returns all elements as a flat Maxima list in row-major order (rows first, matching NumPy convention). Works for ndarrays of any dimensionality. Complex elements are returned in Maxima form (`a + b*%i`).

Round-tripping is consistent: `ndarray(list, shape)` fills in row-major order, and `np_to_list` returns elements in row-major order.

#### Examples

```maxima
(%i1) A : np_arange(4);
(%o1)            ndarray([4], DOUBLE-FLOAT)
(%i2) np_to_list(A);
(%o2)              [0.0, 1.0, 2.0, 3.0]
(%i3) B : ndarray(matrix([1, 2], [3, 4]));
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_to_list(B);
(%o4)              [1.0, 2.0, 3.0, 4.0]
```

See also: `ndarray`, `np_to_matrix`, `np_flatten`
