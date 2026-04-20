;;; util.lisp — Shared helpers for maxima-numerics

(in-package #:maxima)

(defun maxima-shape-to-list (shape)
  "Convert a Maxima shape argument to a CL list of integers.
   Accepts an integer (for 1D) or a Maxima list like [2,3]."
  (cond
    ((integerp shape) (list shape))
    (($listp shape) (mapcar (lambda (x) (truncate x)) (cdr shape)))
    (t (merror "Invalid shape: ~M" shape))))

(defun maxima-matrix-dims (mat)
  "Return (values nrow ncol) for a Maxima matrix."
  (let* ((rows (cdr mat))
         (nrow (length rows))
         (ncol (length (cdar rows))))
    (values nrow ncol)))

(defun numerics-flat-array (tensor)
  "Return a flat 1D displaced array view of a tensor's backing storage.
   Works with any dimensionality — always returns a 1D array for iteration."
  (let ((arr (magicl:lisp-array tensor)))
    (if (= 1 (length (array-dimensions arr)))
        arr
        (make-array (array-total-size arr)
                    :element-type (array-element-type arr)
                    :displaced-to arr))))

;;; LAPACK float-trap masking
;;; SBCL traps on overflow by default, but LAPACK routines (e.g. dgeev) may
;;; produce intermediate overflows that are harmless.  magicl's with-blapack
;;; masks :divide-by-zero and :invalid but NOT :overflow, so we provide our
;;; own wrapper that also masks :overflow.

#+sbcl
(defmacro numerics-with-lapack (&body body)
  `(sb-int:with-float-traps-masked (:overflow :divide-by-zero :invalid)
     ,@body))

#-sbcl
(defmacro numerics-with-lapack (&body body)
  `(magicl:with-blapack ,@body))

;;; Dtype helpers for complex number support

(defun numerics-element-type (dtype)
  "Map an ndarray dtype keyword to a CL type specifier."
  (ecase dtype
    (:double-float 'double-float)
    (:complex-double-float '(complex double-float))))

(defun numerics-result-dtype (a-dtype b-dtype)
  "Compute output dtype from two input dtypes. Complex wins."
  (if (or (eq a-dtype :complex-double-float)
          (eq b-dtype :complex-double-float))
      :complex-double-float
      :double-float))

(defun numerics-parse-dtype (dtype-arg)
  "Parse an optional dtype argument from Maxima.
   nil or $float64 => :double-float
   $complex or $complex128 => :complex-double-float"
  (cond
    ((null dtype-arg) :double-float)
    ((member dtype-arg '($complex $complex128)) :complex-double-float)
    ((member dtype-arg '($float64 $real)) :double-float)
    (t (merror "Unknown dtype: ~M" dtype-arg))))

(defun maxima-to-lisp-number (x dtype)
  "Convert a Maxima number to a CL number of the given dtype.
   For :double-float, calls $float and coerces.
   For :complex-double-float, handles Maxima complex forms (a + b*%i)."
  (let ((element-type (numerics-element-type dtype)))
    (cond
      ;; Already a CL number of the right type
      ((typep x element-type) x)
      ;; CL complex number
      ((complexp x)
       (coerce x element-type))
      ;; Try $float first — works for simple real numbers
      (t
       (let* ((re (coerce ($float ($realpart x)) 'double-float))
              (im (coerce ($float ($imagpart x)) 'double-float)))
         (if (eq dtype :complex-double-float)
             (complex re im)
             (if (zerop im)
                 re
                 (merror "Cannot convert complex value ~M to real" x))))))))

(defun lisp-to-maxima-number (v)
  "Convert a CL number to Maxima representation.
   Real numbers pass through; complex numbers with zero imaginary part
   return just the real part; otherwise become a + b*%i."
  (if (complexp v)
      (if (zerop (imagpart v))
          (realpart v)
          (add (realpart v) (mul '$%i (imagpart v))))
      v))

(defun numerics-require-real (a op-name)
  "Signal an error if the ndarray is complex. Used by comparison/sort/min/max."
  (when (and ($ndarray_p a)
             (eq (numerics:ndarray-dtype (numerics-unwrap a))
                 :complex-double-float))
    (merror "~A: not supported for complex ndarrays" op-name)))
