# Numerical Computing in Maxima: Architecture

## Overview

This document describes the architecture of the `numerics` package, which brings
NumPy-like numerical computing to Maxima. The design has three layers:

| Layer | Library | Role | Data shape |
|-------|---------|------|------------|
| 1 | magicl | Dense numerics, linear algebra | Row/col-major tensors |
| 2 | Arrow | Columnar interop, zero-copy exchange | Columnar buffers |
| 3 | DuckDB | SQL analytics, file I/O | Parquet/CSV on disk |

Target: **SBCL only**. magicl and CFFI require SBCL (or CCL); GCL is explicitly
unsupported. This matches the direction the Maxima community is heading — a 2022
mailing list thread was titled "Trying to accommodate GCL is keeping us from
making progress."

## Prior Art in Maxima

### What exists

- **LAPACK via f2cl**: Maxima ships LAPACK/BLAS translated from Fortran to Common
  Lisp via Raymond Toy's f2cl compiler (`share/lapack/`). Covers `dgeev`, `dgesvd`,
  `dgesv`, `dgeqrf`, `dgemm`, plus complex variants.
- **Marshalling functions**: `lapack-lispify-matrix` / `lapack-maxify-matrix` in
  `share/lapack/eigensys.lisp` convert between Maxima's nested-list matrices and
  column-major `double-float` arrays.
- **AMatrix**: `share/amatrix/` provides a struct-based matrix with submatrix views.
- **FFT**: `share/fftpack5/fftpack5-interface.lisp` uses a converter-factory pattern.
- **Custom display**: `displa-def` + `dim-$typename` for custom rendering
  (see `inference_result` in `share/stats/`, `amatrix` in `share/amatrix/`).
- **numericalio**: CSV/binary I/O via `read_matrix`, `read_array`, etc.

### What doesn't exist

- No magicl integration (mentioned once on mailing list in 2018, never pursued)
- No Apache Arrow or DuckDB — zero prior discussion
- No opaque handle type for foreign-backed numerical data
- No zero-copy data exchange between numerical libraries

### The marshalling problem

Raymond Toy (2014, 2020): "We still have to marshall all of maxima's arrays and
I think the conversion overhead would eat any gains from using native BLAS."

This is the central problem our design solves. The **opaque handle type** keeps
data in foreign memory across operations; only explicit `np_to_matrix()` or
`np_to_list()` materializes back to Maxima form.

---

## Package Structure

```
maxima-numerics/
├── manifest.toml                    # mxpm package metadata
├── numerics.mac                     # Maxima entry point
├── rtest_numerics.mac               # Maxima-level tests
├── doc/
│   ├── numerics.md                  # Package documentation
│   └── design/
│       └── architecture.md          # This file
│
├── lisp/                            # Common Lisp implementation
│   ├── packages.lisp                # CL package definitions
│   ├── numerics.asd                 # ASDF system definition
│   │
│   ├── core/                        # Layer 1: magicl wrapper
│   │   ├── handle.lisp              # ndarray handle type + GC
│   │   ├── display.lisp             # Custom Maxima display
│   │   ├── convert.lisp             # Maxima matrix <-> ndarray
│   │   ├── constructors.lisp        # zeros, ones, eye, rand, etc.
│   │   ├── linalg.lisp              # inv, det, svd, eig, solve, etc.
│   │   ├── elementwise.lisp         # +, -, *, /, exp, log, sin, etc.
│   │   ├── slicing.lisp             # Indexing, slicing, reshaping
│   │   ├── aggregation.lisp         # sum, mean, min, max, std, etc.
│   │   └── util.lisp                # Shared helpers
│   │
│   ├── arrow/                       # Layer 2: Arrow integration
│   │   ├── schema.lisp              # ArrowSchema CFFI struct
│   │   ├── array.lisp               # ArrowArray CFFI struct
│   │   ├── table.lisp               # Table type (named columns)
│   │   ├── bridge.lisp              # ndarray <-> Arrow zero-copy
│   │   └── io.lisp                  # Parquet/CSV/Feather I/O
│   │
│   └── duckdb/                      # Layer 3: DuckDB integration
│       ├── connection.lisp          # Connection management
│       ├── query.lisp               # SQL execution -> Arrow results
│       └── bridge.lisp              # DuckDB result -> Table -> ndarray
│
├── setup/
│   └── install-deps.sh              # Dependency installation script
│
└── .github/workflows/
    ├── docs.yml
    └── pages.yml
```

### ASDF System Definition

```lisp
;; lisp/numerics.asd
(defsystem "numerics"
  :description "NumPy-like numerical computing for Maxima"
  :version "0.1.0"
  :license "MIT"
  :depends-on ("magicl"           ; Core tensor + BLAS/LAPACK
               "cffi"             ; Foreign function interface
               "trivial-garbage"  ; Weak references + finalizers
               "static-vectors"   ; Pinned CL arrays for FFI
               "alexandria")      ; Utilities
  :serial t
  :components
  ((:file "packages")
   (:module "core"
    :serial t
    :components
    ((:file "util")
     (:file "handle")
     (:file "display")
     (:file "convert")
     (:file "constructors")
     (:file "linalg")
     (:file "elementwise")
     (:file "slicing")
     (:file "aggregation")))
   (:module "arrow"
    :serial t
    :components
    ((:file "schema")
     (:file "array")
     (:file "table")
     (:file "bridge")
     (:file "io")))
   (:module "duckdb"
    :serial t
    :components
    ((:file "connection")
     (:file "query")
     (:file "bridge")))))

;; Core-only system (no Arrow/DuckDB, fewer native deps)
(defsystem "numerics/core"
  :description "Core ndarray operations only"
  :depends-on ("magicl" "trivial-garbage" "alexandria")
  :serial t
  :components
  ((:file "packages")
   (:module "core"
    :serial t
    :components
    ((:file "util")
     (:file "handle")
     (:file "display")
     (:file "convert")
     (:file "constructors")
     (:file "linalg")
     (:file "elementwise")
     (:file "slicing")
     (:file "aggregation")))))
```

### CL Package Definitions

```lisp
;; lisp/packages.lisp
(defpackage #:numerics
  (:use #:cl)
  (:export
   ;; Handle type
   #:ndarray #:ndarray-tensor #:ndarray-id #:ndarray-dtype
   #:ndarray-shape #:ndarray-p #:make-ndarray
   ;; Table type
   #:table #:table-columns #:table-column-names #:table-p #:make-table
   ;; Arrow bridge
   #:ndarray-to-arrow-array #:arrow-to-ndarray
   ;; DuckDB
   #:duckdb-connection #:open-duckdb #:close-duckdb
   #:duckdb-query-to-table))
```

### Maxima Entry Point

```maxima
/* numerics.mac */
/* Verify SBCL */
if not is(build_info()@lisp_name = "SBCL") then
  error("numerics requires SBCL; current Lisp is", build_info()@lisp_name)$

/* Load the Lisp implementation via ASDF */
:lisp (require "asdf")
:lisp (let ((here (maxima::maxima-load-pathname-directory)))
        (pushnew here asdf:*central-registry* :test #'equal)
        (pushnew (merge-pathnames "lisp/" here)
                 asdf:*central-registry* :test #'equal))
:lisp (asdf:load-system "numerics/core")

/* Load doc index for ? and ?? help */
load("numerics-index.lisp")$
```

The `:lisp` forms execute Common Lisp directly from within a `.mac` file. ASDF
and Quicklisp must be available in the SBCL image. On SBCL-based Maxima
distributions with Quicklisp installed, `(require "asdf")` is a no-op and
Quicklisp's ASDF integration automatically resolves dependencies from
`~/quicklisp/`.

---

## Layer 1: The ndarray Handle Type

This is the central design element. An `ndarray` wraps a `magicl:tensor` and
presents it as a Maxima value. Operations take handles and return handles, so
data stays in contiguous foreign memory between operations.

### Lisp Implementation

```lisp
;; lisp/core/handle.lisp
(in-package #:numerics)

(defvar *ndarray-counter* 0)

(defstruct (ndarray
            (:constructor %make-ndarray)
            (:print-function print-ndarray))
  "An opaque handle wrapping a magicl tensor."
  (id     0   :type fixnum :read-only t)
  (tensor nil :type (or null magicl:tensor) :read-only nil)
  (dtype  :double-float :type keyword :read-only t))

(defun print-ndarray (obj stream depth)
  (declare (ignore depth))
  (format stream "#<ndarray ~A ~A ~A>"
          (ndarray-id obj)
          (ndarray-dtype obj)
          (when (ndarray-tensor obj)
            (magicl:shape (ndarray-tensor obj)))))

(defun make-ndarray (tensor &key (dtype :double-float))
  "Create an ndarray handle wrapping TENSOR.
   Registers a GC finalizer to release foreign storage."
  (let* ((id (incf *ndarray-counter*))
         (handle (%make-ndarray :id id :tensor tensor :dtype dtype)))
    ;; Register finalizer. The closure captures the tensor, NOT
    ;; the handle, to avoid preventing GC of the handle.
    (let ((t-ref tensor))
      (trivial-garbage:finalize handle
        (lambda ()
          (when t-ref (setf t-ref nil)))))
    handle))
```

### Maxima-level Representation

At the Maxima level, an ndarray is the S-expression:

```
(($ndarray simp) <lisp-ndarray-struct>)
```

The `simp` flag tells the display system the form is simplified. The Lisp struct
is stored as the second element, opaque to Maxima's simplifier.

```lisp
;; In maxima package
(in-package #:maxima)

(defun $ndarray_p (x)
  "Predicate: is X an ndarray handle?"
  (and (listp x)
       (listp (car x))
       (eq (caar x) '$ndarray)
       (typep (cadr x) 'numerics:ndarray)))

(defun numerics-unwrap (x)
  "Extract the Lisp ndarray struct from a Maxima ndarray expression."
  (unless ($ndarray_p x)
    (merror "Expected an ndarray, got: ~M" x))
  (cadr x))

(defun numerics-wrap (handle)
  "Wrap a Lisp ndarray struct into a Maxima expression."
  `(($ndarray simp) ,handle))
```

Every `$np_*` function calls `numerics-unwrap` on inputs and `numerics-wrap`
on outputs. This is the single chokepoint for type safety.

### Custom Display

Following the pattern from `inference_result` and `amatrix`:

```lisp
;; lisp/core/display.lisp
(in-package #:maxima)

(displa-def $ndarray dim-$ndarray)

(defun dim-$ndarray (form result)
  "Display an ndarray. Small 2D arrays show contents; large ones show summary."
  (let* ((handle (cadr form))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (dtype (numerics:ndarray-dtype handle))
         (id (numerics:ndarray-id handle)))
    (if (and (= (length shape) 2)
             (<= (first shape) 6)
             (<= (second shape) 6))
        ;; Small: show as matrix with header
        (let ((mat (numerics-to-maxima-matrix handle)))
          (dimension-function
           `((mprogn) ,(format nil "ndarray(~A, ~A)" shape dtype) ,mat)
           result))
        ;; Large or >2D: compact summary
        (dimension-function
         `((mprogn)
           ,(format nil "ndarray(id=~D, shape=~A, dtype=~A)" id shape dtype))
         result))))

;; TeX output
(defprop $ndarray tex-ndarray tex)

(defun tex-ndarray (x l r)
  (let* ((handle (cadr x))
         (shape (magicl:shape (numerics:ndarray-tensor handle)))
         (dtype (numerics:ndarray-dtype handle)))
    (append l
            (list (format nil "\\text{ndarray}(~A, ~A)" shape dtype))
            r)))
```

### Memory Management

SBCL's GC does not track foreign allocations. magicl tensors backed by the
C allocator use `static-vectors` — CL arrays pinned in memory with stable C
pointers. When the ndarray handle becomes unreachable:

1. `trivial-garbage:finalize` fires the cleanup closure
2. The closure drops its reference to the tensor
3. magicl's own finalizer on the static-vector frees the foreign memory

The closure must **not** capture the handle itself (only the tensor), otherwise
the handle can never be collected.

For Arrow-imported data, the finalizer calls the Arrow `release` callback
instead, which hands lifetime control back to the Arrow producer.

---

## Layer 1 API: Core NumPy Operations

### Conversion Functions

```lisp
;; lisp/core/convert.lisp
(in-package #:maxima)

(defun $ndarray (a &rest options)
  "Convert a Maxima matrix or list to an ndarray.
   ndarray(matrix([1,2],[3,4]))           => 2x2 double
   ndarray(matrix([1,2],[3,4]), complex)  => 2x2 complex
   ndarray([1,2,3,4], [2,2])             => reshape to 2x2"
  (cond
    (($ndarray_p a) a)  ; already an ndarray
    (($matrixp a)
     (let ((dtype (if (member '$complex options) :complex-double-float
                      :double-float)))
       (numerics-wrap (maxima-matrix-to-ndarray a dtype))))
    (($listp a)
     (let ((shape-arg (car options)))
       (numerics-wrap (maxima-list-to-ndarray a shape-arg :double-float))))
    (t (merror "ndarray: expected a matrix or list, got: ~M" a))))

(defun maxima-matrix-to-ndarray (mat dtype)
  "Convert (($matrix) ((mlist) ...) ...) to an ndarray.
   Uses column-major layout for BLAS compatibility."
  (multiple-value-bind (nrow ncol)
      (maxima-matrix-dims mat)
    (let* ((element-type (ecase dtype
                           (:double-float 'double-float)
                           (:complex-double-float '(complex double-float))))
           (tensor (magicl:empty (list nrow ncol)
                                 :type element-type
                                 :layout :column-major)))
      (let ((r 0))
        (dolist (row (cdr mat))
          (let ((c 0))
            (dolist (col (cdr row))
              (setf (magicl:tref tensor r c)
                    (coerce ($float col) element-type))
              (incf c)))
          (incf r)))
      (numerics:make-ndarray tensor :dtype dtype))))

(defun $np_to_matrix (x)
  "Convert an ndarray back to a Maxima matrix."
  (let* ((handle (numerics-unwrap x))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor)))
    (unless (= (length shape) 2)
      (merror "np_to_matrix: ndarray must be 2D, got shape ~A" shape))
    (numerics-to-maxima-matrix handle)))

(defun numerics-to-maxima-matrix (handle)
  "Internal: ndarray handle -> Maxima matrix S-expression."
  (let* ((tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (nrow (first shape))
         (ncol (second shape)))
    (let (rows)
      (dotimes (r nrow)
        (let (cols)
          (dotimes (c ncol)
            (let ((v (magicl:tref tensor r c)))
              (push (if (complexp v)
                        (add (realpart v) (mul '$%i (imagpart v)))
                        v)
                    cols)))
          (push `((mlist simp) ,@(nreverse cols)) rows)))
      `(($matrix simp) ,@(nreverse rows)))))

(defun $np_to_list (x)
  "Flatten an ndarray to a Maxima list."
  (let* ((handle (numerics-unwrap x))
         (tensor (numerics:ndarray-tensor handle))
         (storage (magicl:storage tensor)))
    `((mlist simp) ,@(coerce storage 'list))))
```

### Constructor Functions

| Maxima function | Signature | Description |
|---|---|---|
| `np_zeros(shape)` | `np_zeros([m,n])` or `np_zeros(n)` | Zero-filled |
| `np_ones(shape)` | `np_ones([m,n])` | Ones-filled |
| `np_eye(n)` | `np_eye(n)` or `np_eye(m,n)` | Identity matrix |
| `np_rand(shape)` | `np_rand([m,n])` | Uniform random [0,1) |
| `np_randn(shape)` | `np_randn([m,n])` | Standard normal |
| `np_arange(n)` | `np_arange(n)` | 0..n-1 as 1D |
| `np_linspace(a,b,n)` | `np_linspace(0, 1, 100)` | n evenly spaced points |
| `np_full(shape, val)` | `np_full([3,3], 7.0)` | Constant-filled |
| `np_empty(shape)` | `np_empty([m,n])` | Uninitialized (fast) |
| `np_diag(list)` | `np_diag([1,2,3])` | Diagonal matrix |
| `np_copy(A)` | `np_copy(A)` | Deep copy |

```lisp
;; lisp/core/constructors.lisp
(in-package #:maxima)

(defun $np_zeros (shape)
  (numerics-wrap
   (numerics:make-ndarray
    (magicl:zeros (maxima-shape-to-list shape)
                  :type 'double-float :layout :column-major))))

(defun $np_eye (n &optional m)
  (let ((shape (if m (list n m) (list n n))))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:eye shape :type 'double-float :layout :column-major)))))

(defun $np_rand (shape)
  (numerics-wrap
   (numerics:make-ndarray
    (magicl:rand (maxima-shape-to-list shape)
                 :type 'double-float :layout :column-major))))

(defun $np_linspace (start stop n)
  (let* ((start-f (coerce ($float start) 'double-float))
         (stop-f  (coerce ($float stop)  'double-float))
         (n-int   (truncate n))
         (tensor  (magicl:empty (list n-int) :type 'double-float)))
    (dotimes (i n-int)
      (setf (magicl:tref tensor i)
            (+ start-f (* (/ (float i 1.0d0) (float (1- n-int) 1.0d0))
                          (- stop-f start-f)))))
    (numerics-wrap (numerics:make-ndarray tensor))))

;; Helper
(defun maxima-shape-to-list (shape)
  (cond
    ((integerp shape) (list shape))
    (($listp shape) (cdr shape))
    (t (merror "Invalid shape: ~M" shape))))
```

### Linear Algebra

| Maxima function | Description |
|---|---|
| `np_matmul(A, B)` | Matrix multiplication (BLAS dgemm) |
| `np_inv(A)` | Matrix inverse |
| `np_det(A)` | Determinant (returns scalar) |
| `np_solve(A, b)` | Solve Ax = b |
| `np_svd(A)` | SVD, returns `[U, S, Vt]` |
| `np_eig(A)` | Eigendecomposition, returns `[eigenvalues, eigenvectors]` |
| `np_qr(A)` | QR decomposition, returns `[Q, R]` |
| `np_lu(A)` | LU decomposition, returns `[L, U, P]` |
| `np_norm(A)` | Matrix/vector norm |
| `np_rank(A)` | Numerical rank |
| `np_trace(A)` | Matrix trace (returns scalar) |
| `np_transpose(A)` | Transpose |
| `np_conj(A)` | Conjugate transpose |
| `np_expm(A)` | Matrix exponential |
| `np_lstsq(A, b)` | Least-squares solution |
| `np_cholesky(A)` | Cholesky decomposition |
| `np_pinv(A)` | Pseudo-inverse |

```lisp
;; lisp/core/linalg.lisp
(in-package #:maxima)

(defun $np_matmul (a b)
  "Matrix multiply: np_matmul(A, B)"
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (tb (numerics:ndarray-tensor (numerics-unwrap b))))
    (numerics-wrap (numerics:make-ndarray (magicl:@ ta tb)))))

(defun $np_inv (a)
  "Matrix inverse: np_inv(A)"
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (handler-case
        (numerics-wrap (numerics:make-ndarray (magicl:inv ta)))
      (error (e)
        (merror "np_inv: singular matrix or error: ~A" e)))))

(defun $np_det (a)
  "Determinant: np_det(A) => scalar"
  (magicl:det (numerics:ndarray-tensor (numerics-unwrap a))))

(defun $np_solve (a b)
  "Solve Ax = b: np_solve(A, b)"
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
        (tb (numerics:ndarray-tensor (numerics-unwrap b))))
    (handler-case
        (numerics-wrap (numerics:make-ndarray (magicl:linear-solve ta tb)))
      (error (e)
        (merror "np_solve: ~A" e)))))

(defun $np_svd (a)
  "SVD: np_svd(A) => [U, S, Vt]"
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (multiple-value-bind (u sigma vt) (magicl:svd ta)
      `((mlist simp)
        ,(numerics-wrap (numerics:make-ndarray u))
        ,(numerics-wrap (numerics:make-ndarray sigma))
        ,(numerics-wrap (numerics:make-ndarray vt))))))

(defun $np_eig (a)
  "Eigendecomposition: np_eig(A) => [eigenvalues, eigenvectors]"
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (multiple-value-bind (vals vecs) (magicl:eig ta)
      `((mlist simp)
        ,(numerics-wrap (numerics:make-ndarray vals))
        ,(numerics-wrap (numerics:make-ndarray vecs))))))

(defun $np_transpose (a)
  (numerics-wrap
   (numerics:make-ndarray
    (magicl:transpose (numerics:ndarray-tensor (numerics-unwrap a))))))
```

### Element-wise Operations

| Maxima function | Description |
|---|---|
| `np_add(A, B)` | Element-wise addition (ndarray or scalar) |
| `np_sub(A, B)` | Element-wise subtraction |
| `np_mul(A, B)` | Element-wise (Hadamard) product |
| `np_div(A, B)` | Element-wise division |
| `np_pow(A, p)` | Element-wise power |
| `np_sqrt(A)` | Element-wise square root |
| `np_exp(A)` | Element-wise exp |
| `np_log(A)` | Element-wise natural log |
| `np_sin(A)` / `np_cos(A)` / `np_tan(A)` | Trig functions |
| `np_abs(A)` | Absolute value |
| `np_neg(A)` | Negation |
| `np_scale(alpha, A)` | Scalar multiplication |

```lisp
;; lisp/core/elementwise.lisp
(in-package #:maxima)

(defun $np_add (a b)
  "Element-wise addition: np_add(A, B) or np_add(A, 3.0)"
  (numerics-binary-op a b #'magicl:.+))

(defun $np_mul (a b)
  "Element-wise (Hadamard) product. NOT matrix multiply."
  (numerics-binary-op a b #'magicl:.*))

(defun numerics-binary-op (a b op)
  "Apply a magicl element-wise binary op."
  (cond
    ;; Both ndarrays
    ((and ($ndarray_p a) ($ndarray_p b))
     (let ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
           (tb (numerics:ndarray-tensor (numerics-unwrap b))))
       (numerics-wrap (numerics:make-ndarray (funcall op ta tb)))))
    ;; ndarray + scalar
    (($ndarray_p a)
     (let ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
           (s  (coerce ($float b) 'double-float)))
       (numerics-wrap
        (numerics:make-ndarray
         (magicl:map! (lambda (x) (+ x s))
                      (magicl:deep-copy-tensor ta))))))
    ;; scalar + ndarray
    (($ndarray_p b)
     (let ((tb (numerics:ndarray-tensor (numerics-unwrap b)))
           (s  (coerce ($float a) 'double-float)))
       (numerics-wrap
        (numerics:make-ndarray
         (magicl:map! (lambda (x) (+ s x))
                      (magicl:deep-copy-tensor tb))))))
    (t (merror "Expected at least one ndarray argument"))))

(defun $np_exp (a)
  "Element-wise exponential."
  (numerics-unary-op a #'exp))

(defun numerics-unary-op (a fn)
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a)))
         (result (magicl:deep-copy-tensor tensor)))
    (magicl:map! fn result)
    (numerics-wrap (numerics:make-ndarray result))))
```

### Slicing and Indexing

| Maxima function | Signature | Description |
|---|---|---|
| `np_ref(A, i, j)` | `np_ref(A, 2, 3)` | Single element (0-indexed) |
| `np_set(A, i, j, val)` | `np_set(A, 0, 0, 5.0)` | Set element (mutating) |
| `np_row(A, i)` | `np_row(A, 0)` | Extract row as 1D ndarray |
| `np_col(A, j)` | `np_col(A, 0)` | Extract column as 1D ndarray |
| `np_slice(A, rows, cols)` | `np_slice(A, [0,2], [1,3])` | Sub-matrix |
| `np_reshape(A, shape)` | `np_reshape(A, [2,6])` | Reshape |
| `np_flatten(A)` | `np_flatten(A)` | Flatten to 1D |
| `np_hstack(A, B)` | `np_hstack(A, B)` | Horizontal concat |
| `np_vstack(A, B)` | `np_vstack(A, B)` | Vertical concat |
| `np_shape(A)` | `np_shape(A)` | Shape as Maxima list |
| `np_size(A)` | `np_size(A)` | Total element count |
| `np_dtype(A)` | `np_dtype(A)` | Element type as string |

```lisp
;; lisp/core/slicing.lisp
(in-package #:maxima)

(defun $np_ref (a &rest indices)
  "Single element access (0-indexed): np_ref(A, i, j)"
  (apply #'magicl:tref
         (numerics:ndarray-tensor (numerics-unwrap a))
         indices))

(defun $np_shape (a)
  "Shape as Maxima list: np_shape(A) => [3, 4]"
  `((mlist simp) ,@(magicl:shape
                     (numerics:ndarray-tensor (numerics-unwrap a)))))

(defun $np_reshape (a new-shape)
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a)))
         (shape-list (maxima-shape-to-list new-shape)))
    (unless (= (reduce #'* shape-list) (magicl:size tensor))
      (merror "np_reshape: total size ~D does not match new shape ~A"
              (magicl:size tensor) shape-list))
    (numerics-wrap
     (numerics:make-ndarray (magicl:reshape tensor shape-list)))))
```

### Aggregation Functions

| Maxima function | Description |
|---|---|
| `np_sum(A)` / `np_sum(A, axis)` | Sum (total or along axis) |
| `np_mean(A)` / `np_mean(A, axis)` | Mean |
| `np_min(A)` | Minimum element |
| `np_max(A)` | Maximum element |
| `np_argmin(A)` | Index of minimum |
| `np_argmax(A)` | Index of maximum |
| `np_std(A)` | Standard deviation |
| `np_var(A)` | Variance |
| `np_cumsum(A)` | Cumulative sum |
| `np_dot(a, b)` | Dot product (1D vectors) |

```lisp
;; lisp/core/aggregation.lisp
(in-package #:maxima)

(defun $np_sum (a &optional axis)
  "Sum of elements. np_sum(A) => scalar; np_sum(A, 0) => column sums."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        ;; Total sum: return scalar
        (let ((sum 0.0d0))
          (map nil (lambda (x) (incf sum x)) (magicl:storage tensor))
          sum)
        ;; Along axis: return ndarray
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape)))
          (ecase axis
            (0 (let ((result (magicl:zeros (list ncol) :type 'double-float)))
                 (dotimes (j ncol)
                   (let ((s 0.0d0))
                     (dotimes (i nrow) (incf s (magicl:tref tensor i j)))
                     (setf (magicl:tref result j) s)))
                 (numerics-wrap (numerics:make-ndarray result))))
            (1 (let ((result (magicl:zeros (list nrow) :type 'double-float)))
                 (dotimes (i nrow)
                   (let ((s 0.0d0))
                     (dotimes (j ncol) (incf s (magicl:tref tensor i j)))
                     (setf (magicl:tref result i) s)))
                 (numerics-wrap (numerics:make-ndarray result)))))))))
```

---

## Layer 2: Arrow Integration

### Why Arrow

Arrow is not an alternative to magicl — it's a **memory format** for zero-copy
interop with other tools (polars, DuckDB, Python, R). The key use cases:

1. Receive DuckDB query results without copying
2. Exchange columnar data with external processes
3. Read/write Parquet, Feather, CSV via Arrow-native I/O

### Arrow C Data Interface

Rather than linking to the large `libarrow` C++ library, we implement just the
two C structs from the Arrow C Data Interface. This minimizes native dependencies.

```lisp
;; lisp/arrow/schema.lisp
(in-package #:numerics)

(cffi:defcstruct arrow-schema
  (format :string)
  (name :string)
  (metadata :pointer)
  (flags :int64)
  (n-children :int64)
  (children :pointer)        ; ArrowSchema**
  (dictionary :pointer)      ; ArrowSchema*
  (release :pointer)         ; void (*)(ArrowSchema*)
  (private-data :pointer))

;; lisp/arrow/array.lisp
(cffi:defcstruct arrow-array
  (length :int64)
  (null-count :int64)
  (offset :int64)
  (n-buffers :int64)
  (n-children :int64)
  (buffers :pointer)         ; const void**
  (children :pointer)        ; ArrowArray**
  (dictionary :pointer)      ; ArrowArray*
  (release :pointer)         ; void (*)(ArrowArray*)
  (private-data :pointer))
```

Release callbacks manage lifetime:

```lisp
(cffi:defcallback arrow-release-schema :void ((schema :pointer))
  (let ((fmt (cffi:foreign-slot-value schema '(:struct arrow-schema) 'format)))
    (unless (cffi:null-pointer-p fmt)
      (cffi:foreign-string-free fmt)))
  (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'release)
        (cffi:null-pointer)))

(cffi:defcallback arrow-release-array :void ((array :pointer))
  (let ((pdata (cffi:foreign-slot-value array '(:struct arrow-array) 'private-data)))
    (unless (cffi:null-pointer-p pdata)
      ;; Release ref-counted static-vector or other backing store
      ))
  (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'release)
        (cffi:null-pointer)))
```

### Table Type

A `table` is a list of named 1D ndarray columns:

```lisp
;; lisp/arrow/table.lisp
(in-package #:numerics)

(defstruct (table (:constructor %make-table))
  "A columnar table: named 1D ndarray columns."
  (column-names nil :type list)
  (columns nil :type list)    ; list of ndarray handles
  (nrows 0 :type fixnum))

(defun make-table (names columns)
  (assert (= (length names) (length columns)))
  (let ((nrows (if columns
                   (first (magicl:shape (ndarray-tensor (first columns))))
                   0)))
    (%make-table :column-names names :columns columns :nrows nrows)))
```

Maxima-level table API:

| Maxima function | Description |
|---|---|
| `np_table(names, columns)` | Create table from named columns |
| `np_table_column(T, name)` | Extract column as ndarray |
| `np_table_to_ndarray(T)` | Stack numeric columns into 2D ndarray |
| `np_ndarray_to_table(A, names)` | Split 2D ndarray into named columns |
| `np_table_shape(T)` | Returns `[nrows, ncols]` |
| `np_table_names(T)` | Returns column name list |
| `np_table_head(T, n)` | First n rows |

### Zero-copy Bridge: ndarray <-> Arrow

Both magicl (with `static-vectors`) and Arrow Float64 columns store data as
contiguous `double-float` buffers. The bridge exploits this:

```lisp
;; lisp/arrow/bridge.lisp
(in-package #:numerics)

(defun ndarray-to-arrow-array (handle)
  "Export ndarray as ArrowArray + ArrowSchema (zero-copy for float64).
   Returns (values arrow-array-ptr arrow-schema-ptr)."
  (let* ((tensor (ndarray-tensor handle))
         (storage (magicl:storage tensor))
         (n (magicl:size tensor)))
    ;; Ensure storage is a static-vector (pinned, stable C pointer)
    (let* ((static-vec (if (static-vectors:static-vector-p storage)
                           storage
                           (let ((sv (static-vectors:make-static-vector
                                      n :element-type 'double-float)))
                             (replace sv storage)
                             sv)))
           (data-ptr (static-vectors:static-vector-pointer static-vec)))
      ;; Build ArrowSchema
      (let ((schema (cffi:foreign-alloc '(:struct arrow-schema))))
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'format)
              (cffi:foreign-string-alloc "g"))   ; "g" = float64
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'name)
              (cffi:foreign-string-alloc ""))
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'flags) 0)
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'n-children) 0)
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'release)
              (cffi:callback arrow-release-schema))
        ;; Build ArrowArray
        (let ((array (cffi:foreign-alloc '(:struct arrow-array))))
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'length) n)
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'null-count) 0)
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'offset) 0)
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'n-buffers) 2)
          ;; buffers: [null_bitmap (null), data_ptr]
          (let ((buffers (cffi:foreign-alloc :pointer :count 2)))
            (setf (cffi:mem-aref buffers :pointer 0) (cffi:null-pointer))
            (setf (cffi:mem-aref buffers :pointer 1) data-ptr)
            (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'buffers)
                  buffers))
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'n-children) 0)
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'release)
                (cffi:callback arrow-release-array))
          (values array schema))))))

(defun arrow-to-ndarray (arrow-array-ptr arrow-schema-ptr)
  "Import ArrowArray as ndarray (zero-copy for float64).
   Caller transfers ownership; release callback fires on GC."
  (let* ((format-str (cffi:foreign-slot-value
                      arrow-schema-ptr '(:struct arrow-schema) 'format))
         (n (cffi:foreign-slot-value
             arrow-array-ptr '(:struct arrow-array) 'length))
         (buffers (cffi:foreign-slot-value
                   arrow-array-ptr '(:struct arrow-array) 'buffers))
         (data-ptr (cffi:mem-aref buffers :pointer 1)))
    (unless (string= format-str "g")
      (error "arrow-to-ndarray: only float64 ('g') supported, got ~S"
             format-str))
    ;; Wrap the Arrow buffer as a CL array -> magicl tensor storage
    (let* ((storage (cffi:foreign-array-to-lisp data-ptr (list n)
                                                 :element-type 'double-float))
           (tensor (magicl:from-storage storage (list n)))
           (handle (make-ndarray tensor)))
      ;; Finalizer calls Arrow release callbacks
      (let ((aptr arrow-array-ptr) (sptr arrow-schema-ptr))
        (trivial-garbage:finalize handle
          (lambda ()
            (let ((rel (cffi:foreign-slot-value
                        aptr '(:struct arrow-array) 'release)))
              (unless (cffi:null-pointer-p rel)
                (cffi:foreign-funcall-pointer rel () :pointer aptr :void)))
            (let ((rel (cffi:foreign-slot-value
                        sptr '(:struct arrow-schema) 'release)))
              (unless (cffi:null-pointer-p rel)
                (cffi:foreign-funcall-pointer rel () :pointer sptr :void))))))
      handle)))
```

**How zero-copy works:**

- **ndarray -> Arrow**: `static-vectors:static-vector-pointer` gives the raw C
  pointer to the magicl tensor's storage. We set it as `ArrowArray.buffers[1]`.
  No `memcpy`.
- **Arrow -> ndarray**: `cffi:foreign-array-to-lisp` wraps the Arrow data pointer
  as a CL array, which becomes the magicl tensor's storage. No `memcpy`.

In both directions, lifetime is managed through release callbacks / GC finalizers.

### I/O Functions

| Maxima function | Description |
|---|---|
| `np_read_csv(path)` | CSV -> Table |
| `np_read_parquet(path)` | Parquet -> Table |
| `np_write_csv(T, path)` | Table -> CSV |
| `np_write_parquet(T, path)` | Table -> Parquet |

These are implemented via DuckDB (Layer 3) rather than linking libarrow directly,
keeping the native dependency count to one (`libduckdb`).

---

## Layer 3: DuckDB Integration

### Why DuckDB

DuckDB is an in-process analytical SQL engine that reads Parquet, CSV, Arrow,
and its own format. It returns results as Arrow by default. For Maxima this gives:

- SQL over local files without a server
- Joins, groupbys, window functions
- A stable, compact C API (~30 functions for the common path)

### CFFI Bindings

```lisp
;; lisp/duckdb/connection.lisp
(in-package #:numerics)

(cffi:define-foreign-library libduckdb
  (:darwin "libduckdb.dylib")
  (:unix "libduckdb.so")
  (t (:default "libduckdb")))

(cffi:defcfun "duckdb_open" :int
  (path :string) (out-database :pointer))
(cffi:defcfun "duckdb_connect" :int
  (database :pointer) (out-connection :pointer))
(cffi:defcfun "duckdb_disconnect" :void
  (connection :pointer))
(cffi:defcfun "duckdb_close" :void
  (database :pointer))
(cffi:defcfun "duckdb_query_arrow" :int
  (connection :pointer) (query :string) (out-result :pointer))
(cffi:defcfun "duckdb_query_arrow_schema" :int
  (result :pointer) (out-schema :pointer))
(cffi:defcfun "duckdb_query_arrow_array" :int
  (result :pointer) (out-array :pointer))
(cffi:defcfun "duckdb_destroy_arrow" :void
  (result :pointer))
```

### Connection Management

```lisp
(defstruct (duckdb-connection (:constructor %make-duckdb-connection))
  (database-ptr (cffi:null-pointer) :type cffi:foreign-pointer)
  (connection-ptr (cffi:null-pointer) :type cffi:foreign-pointer)
  (path nil :type (or null string)))

(defun open-duckdb (&optional path)
  "Open a DuckDB database. NIL for in-memory."
  (cffi:use-foreign-library 'libduckdb)
  (let ((db-ptr (cffi:foreign-alloc :pointer))
        (conn-ptr (cffi:foreign-alloc :pointer)))
    (let ((state (duckdb-open (or path (cffi:null-pointer)) db-ptr)))
      (unless (zerop state)
        (cffi:foreign-free db-ptr)
        (cffi:foreign-free conn-ptr)
        (error "duckdb_open failed")))
    (let ((state (duckdb-connect (cffi:mem-ref db-ptr :pointer) conn-ptr)))
      (unless (zerop state)
        (duckdb-close db-ptr)
        (cffi:foreign-free db-ptr)
        (cffi:foreign-free conn-ptr)
        (error "duckdb_connect failed")))
    (let ((conn (%make-duckdb-connection
                 :database-ptr db-ptr :connection-ptr conn-ptr :path path)))
      ;; Register finalizer
      (let ((d db-ptr) (c conn-ptr))
        (trivial-garbage:finalize conn
          (lambda ()
            (duckdb-disconnect c)
            (duckdb-close d)
            (cffi:foreign-free c)
            (cffi:foreign-free d))))
      conn)))

(defun close-duckdb (conn)
  (duckdb-disconnect (duckdb-connection-connection-ptr conn))
  (duckdb-close (duckdb-connection-database-ptr conn))
  (trivial-garbage:cancel-finalization conn))
```

### Maxima-level DuckDB API

| Maxima function | Description |
|---|---|
| `duckdb_open()` | Open in-memory database, return handle |
| `duckdb_open("/path")` | Open persistent database |
| `duckdb_close(conn)` | Close connection |
| `duckdb_query(conn, sql)` | Execute SQL, return Table |
| `duckdb_query_ndarray(conn, sql)` | SQL -> 2D ndarray directly |
| `duckdb_read_csv(path)` | Shortcut: CSV -> Table |
| `duckdb_read_parquet(path)` | Shortcut: Parquet -> Table |
| `duckdb_sql(conn, stmt)` | Execute DDL/DML (no result) |

```lisp
;; lisp/duckdb/query.lisp
(in-package #:maxima)

(defun $duckdb_open (&optional path)
  (let ((conn (numerics:open-duckdb (when path ($sconcat path)))))
    `(($duckdb_connection simp) ,conn)))

(defun $duckdb_query (conn-expr sql)
  "Execute SQL and return results as a Table."
  (let* ((conn (duckdb-unwrap conn-expr))
         (sql-str ($sconcat sql)))
    (cffi:with-foreign-object (result-ptr :pointer)
      (let ((state (numerics:duckdb-query-arrow
                    (cffi:mem-ref
                     (numerics:duckdb-connection-connection-ptr conn) :pointer)
                    sql-str result-ptr)))
        (unless (zerop state)
          (merror "duckdb_query failed: ~A" sql-str))
        (unwind-protect
            (let ((table (numerics:arrow-result-to-table
                          (cffi:mem-ref result-ptr :pointer))))
              (numerics-wrap-table table))
          (numerics:duckdb-destroy-arrow result-ptr))))))

(defun $duckdb_read_parquet (path)
  "Read Parquet into Table via transient DuckDB connection."
  (let ((conn (numerics:open-duckdb nil))
        (sql (format nil "SELECT * FROM read_parquet('~A')" ($sconcat path))))
    (unwind-protect
        ($duckdb_query `(($duckdb_connection simp) ,conn) sql)
      (numerics:close-duckdb conn))))
```

### DuckDB -> Table Bridge

```lisp
;; lisp/duckdb/bridge.lisp
(in-package #:numerics)

(defun arrow-result-to-table (arrow-result-ptr)
  "Convert DuckDB Arrow result to Table by iterating record batches."
  (cffi:with-foreign-object (schema-ptr :pointer)
    (duckdb-query-arrow-schema arrow-result-ptr schema-ptr)
    (let* ((schema (cffi:mem-ref schema-ptr :pointer))
           (n-cols (cffi:foreign-slot-value
                    schema '(:struct arrow-schema) 'n-children))
           (col-names (loop for i below n-cols
                            collect (get-arrow-child-name schema i)))
           (col-accumulators (make-list n-cols :initial-element nil)))
      ;; Iterate chunks
      (loop
        (cffi:with-foreign-object (array-ptr :pointer)
          (let ((state (duckdb-query-arrow-array arrow-result-ptr array-ptr)))
            (unless (zerop state) (return))
            (let ((chunk (cffi:mem-ref array-ptr :pointer)))
              (when (cffi:null-pointer-p chunk) (return))
              (dotimes (i n-cols)
                (let ((col-array (get-arrow-child-array chunk i))
                      (col-schema (get-arrow-child-schema schema i)))
                  (push (arrow-to-ndarray col-array col-schema)
                        (nth i col-accumulators))))))))
      ;; Concatenate chunks per column
      (let ((final-columns
              (loop for acc in col-accumulators
                    collect (if (= 1 (length acc))
                                (first acc)
                                (concatenate-ndarrays (nreverse acc))))))
        (make-table col-names final-columns)))))
```

---

## Zero-Copy Data Flow

```
                    ZERO COPY                    ZERO COPY
  DuckDB result ─────────────> Arrow column ─────────────> ndarray (magicl)
  (C heap)                     (C heap ptr)                (static-vector ptr)
       │                            │                            │
       │  duckdb_query_arrow_array  │  arrow-to-ndarray          │  BLAS/LAPACK
       │  returns ArrowArray with   │  wraps pointer as           │  operates directly
       │  buffer pointer to         │  magicl:from-storage        │  on this contiguous
       │  column data               │  (no memcpy)                │  buffer
```

**Where copies happen (and why they're acceptable):**

1. **Maxima matrix -> ndarray** (`$ndarray`): Always copies. Maxima matrices are
   nested lists of arbitrary-precision numbers; coercion to `double-float` and
   contiguous layout is unavoidable. One-time boundary cost.

2. **ndarray -> Maxima matrix** (`$np_to_matrix`): Always copies. The user
   explicitly opts into this materialization.

3. **Non-float64 Arrow columns**: Integer or string columns are copied/converted.
   Only `float64` gets zero-copy.

4. **magicl operations**: `magicl:inv`, `magicl:@`, etc. allocate new storage for
   results. This is intrinsic (you need space for the output). The key point is
   that **chaining operations never round-trips through Maxima**:
   `np_inv(np_matmul(A, B))` stays in foreign memory the whole time.

---

## Error Handling

All errors surface to Maxima via `merror`, catchable with `errcatch()`.

### Error Categories

**Type errors:**
```lisp
(unless ($ndarray_p x)
  (merror "Expected an ndarray, got: ~M" x))
```

**Dimension mismatch:**
```lisp
(unless (= (second sha) (first shb))
  (merror "np_matmul: incompatible shapes ~A and ~A" sha shb))
```

**Singular matrix:**
```lisp
(handler-case
    (numerics-wrap (numerics:make-ndarray (magicl:inv ta)))
  (error (e)
    (merror "np_inv: singular matrix or error: ~A" e)))
```

**Non-numeric data:**
```lisp
(handler-case
    (coerce ($float col) 'double-float)
  (error ()
    (merror "ndarray: element ~M is not numeric" col)))
```

**DuckDB SQL errors:**
```lisp
(unless (zerop state)
  (merror "duckdb_query: SQL error: ~A" (duckdb-result-error result-ptr)))
```

### Maxima-side usage

```maxima
result : errcatch(np_inv(singular_matrix));
if result = [] then print("Matrix was singular");
```

---

## Testing Strategy

### Maxima-level tests (`rtest_numerics.mac`)

Run via `batch("rtest_numerics.mac", test)` or `mxpm test`.

```maxima
/* Load */
(load("numerics"), true);
true;

/* Constructors */
(A : np_zeros([2,3]), np_shape(A));
[2, 3];

(np_ref(np_eye(3), 0, 0));
1.0;

/* Round-trip */
(M : matrix([1,2],[3,4]),
 A : ndarray(M),
 R : np_to_matrix(A),
 is(R = matrix([1.0, 2.0], [3.0, 4.0])));
true;

/* matmul: I * A = A */
(A : ndarray(matrix([1,2],[3,4])),
 I : np_eye(2),
 B : np_matmul(I, A),
 is(np_to_matrix(B) = np_to_matrix(A)));
true;

/* inv: A * inv(A) ~ I */
(A : ndarray(matrix([1,2],[3,4])),
 Ainv : np_inv(A),
 prod : np_matmul(A, Ainv),
 M : np_to_matrix(prod),
 is(abs(M[1,1] - 1.0) < 1e-10 and abs(M[1,2]) < 1e-10
    and abs(M[2,1]) < 1e-10 and abs(M[2,2] - 1.0) < 1e-10));
true;

/* det */
(A : ndarray(matrix([1,2],[3,4])),
 is(abs(np_det(A) - (-2.0)) < 1e-10));
true;

/* SVD returns three components */
(A : ndarray(matrix([1,0],[0,2],[0,0])),
 [U, S, Vt] : np_svd(A),
 is(length(np_shape(U)) = 2));
true;

/* Element-wise */
(np_ref(np_sqrt(ndarray(matrix([1,4],[9,16]))), 1, 1));
4.0;

/* Aggregation */
(np_sum(ndarray(matrix([1,2],[3,4]))));
10.0;

/* Error handling */
(errcatch(np_inv(ndarray(matrix([1,0],[0,0])))));
[];
```

### Lisp-level tests (fiveam)

For low-level handle lifecycle, GC, and zero-copy bridge testing:

```lisp
(def-suite core-tests)
(in-suite core-tests)

(test handle-creation-and-gc
  (let ((h (numerics:make-ndarray
            (magicl:zeros '(10 10) :type 'double-float))))
    (is (equal (magicl:shape (numerics:ndarray-tensor h)) '(10 10))))
  (trivial-garbage:gc :full t))

(test zero-copy-arrow-roundtrip
  (let* ((tensor (magicl:from-list '(1.0d0 2.0d0 3.0d0 4.0d0) '(4)
                                    :type 'double-float))
         (handle (numerics:make-ndarray tensor))
         (arrow-pair (multiple-value-list
                      (numerics:ndarray-to-arrow-array handle)))
         (roundtripped (numerics:arrow-to-ndarray
                        (first arrow-pair) (second arrow-pair))))
    (is (= 4 (magicl:size (numerics:ndarray-tensor roundtripped))))
    (is (= 1.0d0 (magicl:tref (numerics:ndarray-tensor roundtripped) 0)))))
```

### Test matrix

| Level | Tool | Covers |
|---|---|---|
| Maxima integration | `rtest_numerics.mac` | All `$np_*` functions, error messages, round-trips |
| Lisp unit | fiveam | Handle lifecycle, GC finalizers, zero-copy bridge |
| Layer 2 Arrow | fiveam | Schema/Array construction, release callbacks |
| Layer 3 DuckDB | fiveam | Connection open/close, SQL, Parquet reads |
| Performance | Manual benchmarks | Overhead vs. raw magicl, vs. Maxima matrices |

---

## Build and Dependency Management

### System requirements

1. **SBCL** (>= 2.0)
2. **Quicklisp** — for magicl and transitive CL dependencies
3. **BLAS/LAPACK** — macOS: Accelerate.framework (bundled); Linux: `liblapack-dev libblas-dev`
4. **libduckdb** (optional, Layer 3) — `brew install duckdb` on macOS

### Installation flow

```bash
# 1. System deps
brew install sbcl duckdb           # macOS
# or: apt install sbcl liblapack-dev libblas-dev  # Linux

# 2. Quicklisp (one-time)
curl -O https://beta.quicklisp.org/quicklisp.lisp
sbcl --load quicklisp.lisp \
     --eval '(quicklisp-quicklisp:install :path "~/quicklisp/")' --quit

# 3. Pre-install magicl (one-time)
sbcl --eval '(ql:quickload :magicl)' --quit

# 4. Install the Maxima package
mxpm install numerics

# 5. Use
maxima
(%i1) load("numerics");
(%i2) A : ndarray(matrix([1,2],[3,4]));
(%i3) np_inv(A);
```

### Loading mechanism

When `load("numerics")` runs in Maxima:

1. `(require "asdf")` — ensures ASDF is available
2. Push the package's `lisp/` directory onto `asdf:*central-registry*`
3. `(asdf:load-system "numerics/core")` — loads core module; Quicklisp's ASDF
   integration auto-resolves `magicl`, `cffi`, `trivial-garbage`

Optional subsystems:
```maxima
load("numerics");                     /* core only */
:lisp (asdf:load-system "numerics")   /* full: core + Arrow + DuckDB */

/* Or from Maxima: */
numerics_load_duckdb();               /* convenience function */
```

---

## Usage Examples

### Basic linear algebra

```maxima
load("numerics")$

A : ndarray(matrix([4, 7], [2, 6]));
Ainv : np_inv(A);
np_to_matrix(np_matmul(A, Ainv));
  => matrix([1.0, 0.0], [0.0, 1.0])

[vals, vecs] : np_eig(A);
np_to_list(vals);
  => [8.3166..., 1.6833...]
```

### Large-scale computation

```maxima
load("numerics")$

/* 1000x1000 -- stays in foreign memory */
A : np_rand([1000, 1000]);
B : np_rand([1000, 1000]);

/* BLAS dgemm, ~0.1s for 1000x1000 */
C : np_matmul(A, B);

/* Chained ops, no marshalling overhead */
result : np_sum(np_exp(np_scale(0.01, C)));
```

### Data loading with DuckDB

```maxima
load("numerics")$
numerics_load_duckdb()$

T : duckdb_read_parquet("/data/measurements.parquet");

x : np_table_column(T, "temperature");
y : np_table_column(T, "pressure");

/* Least-squares fit */
ones : np_ones([np_size(x), 1]);
X : np_hstack(np_reshape(x, [np_size(x), 1]), ones);
Y : np_reshape(y, [np_size(y), 1]);
coeffs : np_lstsq(X, Y);
np_to_list(coeffs);
  => [slope, intercept]
```

### SQL analytics to numerical computation

```maxima
load("numerics")$
numerics_load_duckdb()$

conn : duckdb_open()$
T : duckdb_query(conn,
  "SELECT date, price, volume
   FROM read_csv('/data/stocks.csv')
   WHERE symbol = 'AAPL'
   ORDER BY date");

prices : np_table_column(T, "price");
returns : np_div(np_sub(np_slice(prices, [1, -1]),
                         np_slice(prices, [0, -2])),
                  np_slice(prices, [0, -2]));
print("Mean return:", np_mean(returns))$
print("Std return:", np_std(returns))$

duckdb_close(conn)$
```

---

## Implementation Phasing

### Phase 1 — MVP: Layer 1 Core

- Handle type with display and GC
- Conversion functions (ndarray, np_to_matrix, np_to_list)
- Constructors (zeros, ones, eye, rand)
- Linear algebra (matmul, inv, det, solve, svd, eig, transpose)
- Element-wise ops (add, sub, mul, scale, exp, sqrt)
- Aggregations (sum, mean, min, max)
- Shape inspection and reshape
- Maxima-level tests

### Phase 2 — Arrow Bridge

- CFFI struct definitions for ArrowSchema/ArrowArray
- Zero-copy ndarray-to-Arrow and Arrow-to-ndarray
- Table type with column access

### Phase 3 — DuckDB Integration

- CFFI bindings to libduckdb
- Connection management with finalizers
- SQL query -> Table via Arrow
- Convenience I/O (read_parquet, read_csv)

### Phase 4 — Polish

- Full documentation (numerics.md with function headings for ? / ?? help)
- Performance benchmarks vs. raw Maxima matrices
- Error message refinement
- Complex number support throughout
- Higher-dimensional tensor support (beyond 2D)

---

## Reference: Key Files in Maxima

These files in `../maxima/` informed the design patterns used here:

| File | Pattern used |
|------|-------------|
| `share/lapack/eigensys.lisp` | Maxima matrix marshalling (`lapack-lispify-matrix` / `lapack-maxify-matrix`) |
| `share/stats/inference_result.lisp` | Custom type display via `displa-def` + `dim-$typename` |
| `share/amatrix/amatrix.lisp` | Struct-based matrix with custom display, subscript assignment |
| `share/fftpack5/fftpack5-interface.lisp` | Converter-factory pattern for CL vector <-> Maxima form |
| `src/numerical/f2cl-lib.lisp` | Fortran type mappings, column-major indexing |
