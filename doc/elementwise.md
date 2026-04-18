## Element-wise Operations

### Function: np_add (a, b)

Element-wise addition.

Supports ndarray + ndarray (same shape), ndarray + scalar, and scalar + ndarray. Returns a new ndarray.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_add(A, A));
(%o2)         matrix([2.0, 4.0], [6.0, 8.0])
(%i3) np_to_matrix(np_add(A, 10));
(%o3)      matrix([11.0, 12.0], [13.0, 14.0])
(%i4) np_to_matrix(np_add(5, A));
(%o4)        matrix([6.0, 7.0], [8.0, 9.0])
```

See also: `np_sub`, `np_scale`

### Function: np_sub (a, b)

Element-wise subtraction.

Supports ndarray - ndarray, ndarray - scalar, and scalar - ndarray. Returns a new ndarray.

#### Examples

```maxima
(%i1) A : ndarray(matrix([10, 20], [30, 40]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) B : ndarray(matrix([1, 2], [3, 4]));
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) np_to_matrix(np_sub(A, B));
(%o3)       matrix([9.0, 18.0], [27.0, 36.0])
(%i4) np_to_matrix(np_sub(A, 5));
(%o4)      matrix([5.0, 15.0], [25.0, 35.0])
```

See also: `np_add`, `np_neg`

### Function: np_mul (a, b)

Element-wise (Hadamard) product.

This is NOT matrix multiplication. Each element of `a` is multiplied by the corresponding element of `b`. For matrix multiplication, use `np_matmul`.

Supports ndarray * ndarray, ndarray * scalar, and scalar * ndarray.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_mul(A, A));
(%o2)        matrix([1.0, 4.0], [9.0, 16.0])
(%i3) np_to_matrix(np_mul(A, 3));
(%o3)       matrix([3.0, 6.0], [9.0, 12.0])
```

See also: `np_matmul`, `np_div`, `np_scale`

### Function: np_div (a, b)

Element-wise division.

Supports ndarray / ndarray, ndarray / scalar, and scalar / ndarray. Returns a new ndarray.

#### Examples

```maxima
(%i1) A : ndarray(matrix([10, 20], [30, 40]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_div(A, 10));
(%o2)         matrix([1.0, 2.0], [3.0, 4.0])
(%i3) B : ndarray(matrix([2, 4], [5, 8]));
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_to_matrix(np_div(A, B));
(%o4)         matrix([5.0, 5.0], [6.0, 5.0])
```

See also: `np_mul`

### Function: np_pow (a, p)

Element-wise exponentiation.

Raises each element of `a` to the power `p`. If `p` is an ndarray, the operation is applied element-wise between `a` and `p`.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_pow(A, 2));
(%o2)        matrix([1.0, 4.0], [9.0, 16.0])
(%i3) np_to_matrix(np_pow(A, 0.5));
(%o3)  matrix([1.0, 1.414..], [1.732.., 2.0])
```

See also: `np_sqrt`, `np_exp`, `np_log`

### Function: np_sqrt (a)

Element-wise square root.

Returns a new ndarray where each element is the square root of the corresponding element in `a`.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 4], [9, 16]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_sqrt(A));
(%o2)         matrix([1.0, 2.0], [3.0, 4.0])
```

See also: `np_pow`, `np_exp`

### Function: np_exp (a)

Element-wise exponential (e^x).

Returns a new ndarray where each element is `e` raised to the power of the corresponding element in `a`.

#### Examples

```maxima
(%i1) A : ndarray(matrix([0, 1], [2, 3]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_ref(np_exp(A), 0, 0);
(%o2)                          1.0
(%i3) B : np_log(np_exp(A));
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_ref(B, 1, 1);
(%o4)                          3.0
```

See also: `np_log`, `np_expm`

### Function: np_log (a)

Element-wise natural logarithm.

Returns a new ndarray where each element is the natural log of the corresponding element in `a`. Elements must be positive.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, %e], [%e^2, %e^3]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_log(A));
(%o2)         matrix([0.0, 1.0], [2.0, 3.0])
```

See also: `np_exp`

### Function: np_sin (a)

Element-wise sine.

Returns a new ndarray where each element is the sine of the corresponding element in `a` (radians).

#### Examples

```maxima
(%i1) A : ndarray(matrix([0, %pi/2], [%pi, 3*%pi/2]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_ref(np_sin(A), 0, 0);
(%o2)                          0.0
(%i3) np_ref(np_sin(A), 0, 1);
(%o3)                          1.0
```

See also: `np_cos`, `np_tan`

### Function: np_cos (a)

Element-wise cosine.

Returns a new ndarray where each element is the cosine of the corresponding element in `a` (radians).

#### Examples

```maxima
(%i1) A : ndarray(matrix([0, %pi/2], [%pi, 2*%pi]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_ref(np_cos(A), 0, 0);
(%o2)                          1.0
```

See also: `np_sin`, `np_tan`

### Function: np_tan (a)

Element-wise tangent.

Returns a new ndarray where each element is the tangent of the corresponding element in `a` (radians).

#### Examples

```maxima
(%i1) A : ndarray(matrix([0, %pi/4]));
(%o1)            ndarray([1, 2], DOUBLE-FLOAT)
(%i2) np_ref(np_tan(A), 0, 0);
(%o2)                          0.0
```

See also: `np_sin`, `np_cos`

### Function: np_abs (a)

Element-wise absolute value.

Returns a new ndarray where each element is the absolute value of the corresponding element in `a`.

#### Examples

```maxima
(%i1) A : ndarray(matrix([-1, 2], [-3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_abs(A));
(%o2)         matrix([1.0, 2.0], [3.0, 4.0])
```

See also: `np_neg`

### Function: np_neg (a)

Element-wise negation.

Returns a new ndarray where each element is negated (multiplied by -1).

#### Examples

```maxima
(%i1) A : np_ones([2, 2]);
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_neg(A));
(%o2)     matrix([-1.0, -1.0], [-1.0, -1.0])
```

See also: `np_abs`, `np_sub`

### Function: np_scale (alpha, a)

Multiply every element of an ndarray by a scalar.

The scalar `alpha` is coerced to double-float. Returns a new ndarray.

#### Examples

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_to_matrix(np_scale(3, A));
(%o2)       matrix([3.0, 6.0], [9.0, 12.0])
(%i3) np_to_matrix(np_scale(0.5, A));
(%o3)         matrix([0.5, 1.0], [1.5, 2.0])
```

See also: `np_mul`, `np_add`

### Function: np_map (f, a)

Apply a user-defined function element-wise to an ndarray.

Evaluates `f(x)` for each element `x` in `a` and returns a new ndarray of the same shape. The function `f` must accept a single numeric argument and return a numeric result.

**Performance:** If `f` has been translated with `translate(f)`, `np_map` automatically uses the fast compiled path (calling the CL function directly). Otherwise it falls back to the Maxima evaluator, which is convenient but much slower for large arrays.

Calling forms:

- `np_map(f, A)` -- apply `f` element-wise

#### Examples

```maxima
(%i1) f(x) := x^2 + 1$
(%i2) A : np_arange(5);
(%o2)            ndarray([5], DOUBLE-FLOAT)
(%i3) np_to_list(np_map(f, A));
(%o3)           [1.0, 2.0, 5.0, 10.0, 17.0]
(%i4) translate(f)$
(%i5) np_to_list(np_map(f, A));
(%o5)           [1.0, 2.0, 5.0, 10.0, 17.0]
```

For best performance on large arrays, translate the function first:

```maxima
(%i1) g(x) := exp(-x^2)$
(%i2) translate(g)$
(%i3) A : np_linspace(-3, 3, 10000)$
(%i4) B : np_map(g, A)$
```

See also: `np_map2`, `np_sqrt`, `np_exp`

### Function: np_map2 (f, a, b)

Apply a binary function element-wise to two ndarrays.

Evaluates `f(x, y)` for corresponding elements of `a` and `b`. Both arrays must have the same shape. Returns a new ndarray.

Like `np_map`, uses the fast compiled path if `f` has been translated.

#### Examples

```maxima
(%i1) f(x, y) := x^2 + y^2$
(%i2) A : np_arange(3);
(%o2)            ndarray([3], DOUBLE-FLOAT)
(%i3) B : np_ones(3);
(%o3)            ndarray([3], DOUBLE-FLOAT)
(%i4) np_to_list(np_map2(f, A, B));
(%o4)                 [1.0, 2.0, 5.0]
```

See also: `np_map`, `np_add`, `np_mul`

### Function: np_where (condition) / np_where (condition, x, y)

Conditional selection.

**Form 1: `np_where(condition)`** — returns a Maxima list of index arrays indicating where the condition ndarray has nonzero elements.

- For 1D input: returns `[indices]` (a list containing one 1D ndarray).
- For 2D input: returns `[row_indices, col_indices]` (two 1D ndarrays).

**Form 2: `np_where(condition, x, y)`** — element-wise selection. Returns a new ndarray: takes from `x` where condition is nonzero, from `y` where condition is zero. The arguments `x` and `y` can be ndarrays (same shape as condition) or scalars.

#### Examples

```maxima
(%i1) /* Form 1: find nonzero indices */
      A : ndarray([0, 1, 0, 3, 0], [5]);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_to_list(np_where(A)[1]);
(%o2)                     [1.0, 3.0]
(%i3) /* Form 1: 2D */
      B : ndarray(matrix([1, 0], [0, 4]));
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) [rows, cols] : np_where(B);
(%o4)              [ndarray, ndarray]
(%i5) np_to_list(rows);
(%o5)                     [0.0, 1.0]
(%i6) /* Form 2: select from x or y */
      cond : ndarray([1, 0, 1], [3]);
(%o6)            ndarray([3], DOUBLE-FLOAT)
(%i7) np_to_list(np_where(cond, ndarray([10,20,30],[3]),
                                ndarray([100,200,300],[3])));
(%o7)               [10.0, 200.0, 30.0]
(%i8) /* Form 2: scalar broadcasting */
      np_to_list(np_where(np_greater(A, 2), A, 0));
(%o8)              [0.0, 0.0, 0.0, 3.0, 0.0]
```

See also: `np_greater`, `np_extract`, `np_map`

### Function: np_greater (a, b)

Element-wise greater-than comparison.

Returns a new ndarray with 1.0 where `a > b` and 0.0 elsewhere. Supports ndarray + ndarray, ndarray + scalar, and scalar + ndarray.

#### Examples

```maxima
(%i1) A : ndarray([1, 5, 3, 7, 2], [5]);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_to_list(np_greater(A, 3));
(%o2)           [0.0, 1.0, 0.0, 1.0, 0.0]
(%i3) B : ndarray([2, 4, 3], [3]);
(%o3)            ndarray([3], DOUBLE-FLOAT)
(%i4) np_to_list(np_greater(ndarray([1,5,3],[3]), B));
(%o4)                  [0.0, 1.0, 0.0]
```

See also: `np_less`, `np_greater_equal`, `np_equal`

### Function: np_greater_equal (a, b)

Element-wise greater-than-or-equal comparison. Returns 1.0/0.0 ndarray.

See also: `np_greater`, `np_less_equal`

### Function: np_less (a, b)

Element-wise less-than comparison. Returns 1.0/0.0 ndarray.

See also: `np_greater`, `np_less_equal`

### Function: np_less_equal (a, b)

Element-wise less-than-or-equal comparison. Returns 1.0/0.0 ndarray.

See also: `np_less`, `np_greater_equal`

### Function: np_equal (a, b)

Element-wise equality comparison. Returns 1.0/0.0 ndarray.

See also: `np_not_equal`

### Function: np_not_equal (a, b)

Element-wise not-equal comparison. Returns 1.0/0.0 ndarray.

See also: `np_equal`

### Function: np_logical_and (a, b)

Element-wise logical AND. Nonzero is true. Returns 1.0/0.0 ndarray.

#### Examples

```maxima
(%i1) A : ndarray([1, 0, 1, 0], [4]);
(%o1)            ndarray([4], DOUBLE-FLOAT)
(%i2) B : ndarray([1, 1, 0, 0], [4]);
(%o2)            ndarray([4], DOUBLE-FLOAT)
(%i3) np_to_list(np_logical_and(A, B));
(%o3)              [1.0, 0.0, 0.0, 0.0]
```

See also: `np_logical_or`, `np_logical_not`

### Function: np_logical_or (a, b)

Element-wise logical OR. Nonzero is true. Returns 1.0/0.0 ndarray.

See also: `np_logical_and`, `np_logical_not`

### Function: np_logical_not (a)

Element-wise logical NOT. Nonzero becomes 0.0, zero becomes 1.0.

See also: `np_logical_and`, `np_logical_or`

### Function: np_test (f, a)

Apply a predicate function element-wise, returning a 1.0/0.0 mask ndarray.

The function `f` can be:
- A named function: `np_test(is_positive, A)` — fast if translated
- A lambda: `np_test(lambda([x], is(x > 3)), A)` — slow, uses Maxima evaluator

The result of `f(x)` is converted to 1.0 (truthy) or 0.0 (falsy). Numbers, booleans, and Maxima relational expressions are all handled.

**Performance:** Named functions with `translate(f)` are fastest. Lambda expressions use the Maxima evaluator and are much slower for large arrays. For simple comparisons, prefer `np_greater`, `np_less`, etc.

#### Examples

```maxima
(%i1) gt3(x) := if x > 3 then 1 else 0$
(%i2) A : ndarray([1, 5, 3, 7, 2], [5]);
(%o2)            ndarray([5], DOUBLE-FLOAT)
(%i3) np_to_list(np_test(gt3, A));
(%o3)           [0.0, 1.0, 0.0, 1.0, 0.0]
(%i4) /* Lambda form */
      np_to_list(np_test(lambda([x], is(x > 3)), A));
(%o4)           [0.0, 1.0, 0.0, 1.0, 0.0]
```

See also: `np_greater`, `np_extract`, `np_map`

### Function: np_extract (mask, a)

Extract elements where mask is nonzero (boolean indexing).

Returns a 1D ndarray containing the elements of `a` where the corresponding element in `mask` is nonzero. Elements are taken in row-major order. Returns an empty Maxima list `[]` if no elements match.

This is the equivalent of NumPy's `A[mask]`.

#### Examples

```maxima
(%i1) A : ndarray([10, 20, 30, 40, 50], [5]);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) mask : ndarray([1, 0, 1, 0, 1], [5]);
(%o2)            ndarray([5], DOUBLE-FLOAT)
(%i3) np_to_list(np_extract(mask, A));
(%o3)               [10.0, 30.0, 50.0]
(%i4) /* With comparison-generated mask: A[A > 25] */
      np_to_list(np_extract(np_greater(A, 25), A));
(%o4)               [30.0, 40.0, 50.0]
```

See also: `np_where`, `np_greater`, `np_test`
