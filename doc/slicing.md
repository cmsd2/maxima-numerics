## Slicing

All slicing and indexing operations preserve the dtype of the input array.

### Function: np_ref (a, i)

Access a single element of an ndarray by index.

Indices are 0-based. For 1D arrays, pass one index. For 2D arrays, pass row and column indices.

Calling forms:

- `np_ref(a, i)` -- element at position `i` in a 1D array
- `np_ref(a, i, j)` -- element at row `i`, column `j` in a 2D array

#### Examples

```maxima
(%i1) A : np_arange(5);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_ref(A, 0);
(%o2)                          0.0
(%i3) np_ref(A, 3);
(%o3)                          3.0
(%i4) B : ndarray(matrix([10, 20], [30, 40]));
(%o4)            ndarray([2, 2], DOUBLE-FLOAT)
(%i5) np_ref(B, 1, 0);
(%o5)                         30.0
```

See also: `np_set`, `np_row`, `np_col`

### Function: np_set (a, i, j, val)

Set a single element of an ndarray (mutating).

Indices are 0-based. The last argument is the value to set, preceding arguments are indices. The array is modified in place and returned.

Calling forms:

- `np_set(a, i, val)` -- set element at position `i` in a 1D array
- `np_set(a, i, j, val)` -- set element at row `i`, column `j` in a 2D array

#### Examples

```maxima
(%i1) A : np_zeros([2, 2]);
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_set(A, 0, 0, 42.0);
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) np_ref(A, 0, 0);
(%o3)                         42.0
(%i4) B : np_zeros(3);
(%o4)            ndarray([3], DOUBLE-FLOAT)
(%i5) np_set(B, 1, 99.0);
(%o5)            ndarray([3], DOUBLE-FLOAT)
(%i6) np_to_list(B);
(%o6)                 [0.0, 99.0, 0.0]
```

See also: `np_ref`

### Function: np_row (a, i)

Extract a row from a 2D ndarray as a 1D ndarray.

The row index `i` is 0-based. Returns a new 1D ndarray containing a copy of the row.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2, 3], [4, 5, 6]));
(%o1)            ndarray([2, 3], DOUBLE-FLOAT)
(%i2) np_to_list(np_row(A, 0));
(%o2)                  [1.0, 2.0, 3.0]
(%i3) np_to_list(np_row(A, 1));
(%o3)                  [4.0, 5.0, 6.0]
```

See also: `np_col`, `np_slice`, `np_ref`

### Function: np_col (a, j)

Extract a column from a 2D ndarray as a 1D ndarray.

The column index `j` is 0-based. Returns a new 1D ndarray containing a copy of the column.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2, 3], [4, 5, 6]));
(%o1)            ndarray([2, 3], DOUBLE-FLOAT)
(%i2) np_to_list(np_col(A, 0));
(%o2)                     [1.0, 4.0]
(%i3) np_to_list(np_col(A, 2));
(%o3)                     [3.0, 6.0]
```

See also: `np_row`, `np_slice`, `np_ref`

### Function: np_slice (a, spec0, spec1, ..., specN)

Extract a sub-array from an N-dimensional ndarray.

One spec argument is required per dimension. Each spec can be:

- `all` — select the entire dimension
- `[start, end]` — half-open range `[start, end)`, 0-based, end exclusive
- An integer — select a single index, **collapsing** (removing) that dimension from the result

Negative indices count from the end of the dimension (`-1` is the last element).

Calling forms:

- `np_slice(a, rows, cols)` — 2D sub-matrix (unchanged from earlier versions)
- `np_slice(a, all, all, [0, 1])` — 3D slice selecting a channel from an image
- `np_slice(a, 0, all)` — select row 0, returning a 1D result
- `np_slice(a, [2, 5])` — slice a 1D array

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2, 3], [4, 5, 6], [7, 8, 9]));
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) /* Range slicing */
      B : np_slice(A, [0, 2], [1, 3]);
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) np_to_matrix(B);
(%o3)            matrix([2.0, 3.0], [5.0, 6.0])
(%i4) /* Use 'all' to select entire rows */
      C : np_slice(A, all, [0, 2]);
(%o4)            ndarray([3, 2], DOUBLE-FLOAT)
(%i5) /* Scalar index collapses a dimension */
      D : np_slice(A, 0, all);
(%o5)            ndarray([3], DOUBLE-FLOAT)
(%i6) np_to_list(D);
(%o6)                  [1.0, 2.0, 3.0]
(%i7) /* All indices scalar: returns a number */
      np_slice(A, 1, 2);
(%o7)                          6.0
```

3D slicing (e.g. extracting a colour channel from an image):

```maxima
(%i1) img : np_reshape(np_arange(24), [2, 3, 4]);
(%o1)            ndarray([2, 3, 4], DOUBLE-FLOAT)
(%i2) /* Extract channel 0 as a 2D array */
      ch0 : np_slice(img, all, all, 0);
(%o2)            ndarray([2, 3], DOUBLE-FLOAT)
(%i3) np_shape(ch0);
(%o3)                        [2, 3]
(%i4) /* Keep channel dimension with a range */
      ch0_3d : np_slice(img, all, all, [0, 1]);
(%o4)            ndarray([2, 3, 1], DOUBLE-FLOAT)
```

See also: `np_row`, `np_col`, `np_ref`, `np_reshape`

### Function: np_reshape (a, shape)

Reshape an ndarray to new dimensions.

The total number of elements must remain the same. Returns a new ndarray with the specified shape. The `shape` argument is a Maxima list of dimensions.

#### Examples

```maxima
(%i1) A : np_arange(6);
(%o1)            ndarray([6], DOUBLE-FLOAT)
(%i2) B : np_reshape(A, [2, 3]);
(%o2)            ndarray([2, 3], DOUBLE-FLOAT)
(%i3) np_shape(B);
(%o3)                        [2, 3]
(%i4) C : np_reshape(A, [3, 2]);
(%o4)            ndarray([3, 2], DOUBLE-FLOAT)
```

See also: `np_flatten`, `np_shape`

### Function: np_flatten (a)

Flatten an ndarray to 1D.

Returns a new 1D ndarray containing all elements in storage order (column-major).

#### Examples

```maxima
(%i1) A : np_ones([2, 3]);
(%o1)            ndarray([2, 3], DOUBLE-FLOAT)
(%i2) B : np_flatten(A);
(%o2)            ndarray([6], DOUBLE-FLOAT)
(%i3) np_size(B);
(%o3)                           6
(%i4) np_shape(B);
(%o4)                          [6]
```

See also: `np_reshape`, `np_to_list`

### Function: np_hstack (a, b)

Concatenate two 2D ndarrays horizontally (along columns).

Both arrays must have the same number of rows. The result has the combined number of columns. Promotes to complex if either input is complex.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) B : ndarray(matrix([5], [6]));
(%o2)            ndarray([2, 1], DOUBLE-FLOAT)
(%i3) C : np_hstack(A, B);
(%o3)            ndarray([2, 3], DOUBLE-FLOAT)
(%i4) np_to_matrix(C);
(%o4)      matrix([1.0, 2.0, 5.0], [3.0, 4.0, 6.0])
```

See also: `np_vstack`

### Function: np_vstack (a, b)

Concatenate two 2D ndarrays vertically (along rows).

Both arrays must have the same number of columns. The result has the combined number of rows. Promotes to complex if either input is complex.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) B : ndarray(matrix([5, 6]));
(%o2)            ndarray([1, 2], DOUBLE-FLOAT)
(%i3) C : np_vstack(A, B);
(%o3)            ndarray([3, 2], DOUBLE-FLOAT)
(%i4) np_to_matrix(C);
(%o4)  matrix([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
```

See also: `np_hstack`

### Function: np_shape (a)

Return the shape of an ndarray as a Maxima list.

For a 1D array, returns a list with one element. For a 2D array, returns `[rows, cols]`.

#### Examples

```maxima
(%i1) np_shape(np_zeros([3, 4]));
(%o1)                        [3, 4]
(%i2) np_shape(np_arange(10));
(%o2)                         [10]
(%i3) np_shape(np_eye(5));
(%o3)                        [5, 5]
```

See also: `np_size`, `np_dtype`, `np_reshape`

### Function: np_size (a)

Return the total number of elements in an ndarray.

#### Examples

```maxima
(%i1) np_size(np_zeros([3, 4]));
(%o1)                          12
(%i2) np_size(np_arange(10));
(%o2)                          10
(%i3) np_size(np_eye(5));
(%o3)                          25
```

See also: `np_shape`, `np_dtype`

### Function: np_dtype (a)

Return the element type of an ndarray as a string.

Returns `"DOUBLE-FLOAT"` for real arrays or `"COMPLEX-DOUBLE-FLOAT"` for complex arrays.

#### Examples

```maxima
(%i1) np_dtype(np_zeros([2, 2]));
(%o1)                    DOUBLE-FLOAT
```

See also: `np_shape`, `np_size`
