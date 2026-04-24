# Package numerics

## Introduction to numerics

The `numerics` package provides NumPy-like numerical computing for Maxima. It wraps BLAS and LAPACK via the [magicl](https://github.com/quil-lang/magicl) library, giving Maxima users access to fast dense linear algebra, element-wise array operations, and statistical aggregations.

**Requirements:** SBCL (Steel Bank Common Lisp). Other Lisp implementations are not supported.

### Key concepts

**ndarray** is the central data type -- an opaque handle wrapping a dense numeric array. Unlike Maxima matrices, ndarrays:

- Use **0-based indexing** (like NumPy, C, and most programming languages)
- Store elements as **double-float** (64-bit IEEE 754)
- Use **column-major** memory layout for BLAS compatibility
- Are **mutable** -- functions like `np_set` modify the array in place

### Quick start

```maxima
(%i1) load("numerics")$
(%i2) A : ndarray(matrix([1, 2], [3, 4]));
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) B : np_inv(A);
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_to_matrix(np_matmul(A, B));
(%o4)            matrix([1.0, 0.0], [0.0, 1.0])
```

### Creating arrays

```maxima
(%i1) np_zeros([3, 3]);
(%o1)            ndarray([3, 3], DOUBLE-FLOAT)
(%i2) np_ones([2, 4]);
(%o2)            ndarray([2, 4], DOUBLE-FLOAT)
(%i3) np_arange(5);
(%o3)            ndarray([5], DOUBLE-FLOAT)
(%i4) np_linspace(0, 1, 5);
(%o4)            ndarray([5], DOUBLE-FLOAT)
```

### Element access

```maxima
(%i1) A : ndarray(matrix([10, 20, 30], [40, 50, 60]));
(%o1)            ndarray([2, 3], DOUBLE-FLOAT)
(%i2) np_ref(A, 0, 2);
(%o2)                          30.0
(%i3) np_set(A, 1, 1, 99.0);
(%o3)            ndarray([2, 3], DOUBLE-FLOAT)
(%i4) np_ref(A, 1, 1);
(%o4)                          99.0
```

### Element-wise operations

All element-wise functions support ndarray-ndarray and ndarray-scalar combinations:

```maxima
(%i1) A : ndarray(matrix([1, 2], [3, 4]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_add(A, 10);
(%o2)            ndarray([2, 2], DOUBLE-FLOAT)
(%i3) np_mul(A, A);
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_sqrt(A);
(%o4)            ndarray([2, 2], DOUBLE-FLOAT)
```

### Linear algebra

```maxima
(%i1) A : ndarray(matrix([2, 1], [5, 3]));
(%o1)            ndarray([2, 2], DOUBLE-FLOAT)
(%i2) np_det(A);
(%o2)                          1.0
(%i3) [U, S, Vt] : np_svd(A);
(%o3)        [ndarray, ndarray, ndarray]
(%i4) [Q, R] : np_qr(A);
(%o4)              [ndarray, ndarray]
```

### Converting back to Maxima

```maxima
(%i1) A : np_linspace(0, 1, 5);
(%o1)            ndarray([5], DOUBLE-FLOAT)
(%i2) np_to_list(A);
(%o2)           [0.0, 0.25, 0.5, 0.75, 1.0]
(%i3) M : ndarray(matrix([1, 2], [3, 4]));
(%o3)            ndarray([2, 2], DOUBLE-FLOAT)
(%i4) np_to_matrix(M);
(%o4)         matrix([1.0, 2.0], [3.0, 4.0])
```

## Definitions for numerics

<!-- include: conversion.md -->
<!-- include: constructors.md -->
<!-- include: slicing.md -->
<!-- include: elementwise.md -->
<!-- include: signal.md -->
<!-- include: image.md -->
<!-- include: aggregation.md -->
<!-- include: optimize.md -->
<!-- include: integrate.md -->
<!-- include: learn.md -->
<!-- include: linalg.md -->
