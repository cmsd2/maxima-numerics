;;; util.lisp — Shared helpers for maxima-numerics

(in-package #:maxima)

(defun maxima-shape-to-list (shape)
  "Convert a Maxima shape argument to a CL list of integers.
   Accepts an integer (for 1D) or a Maxima list like [2,3]."
  (let ((dims (cond
                ((integerp shape) (list shape))
                (($listp shape) (mapcar (lambda (x) (truncate x)) (cdr shape)))
                (t (merror "Invalid shape: ~M" shape)))))
    ;; magicl rejects zero-length dimensions, so say so here
    (unless (every #'plusp dims)
      (merror "Invalid shape ~M: every dimension must be at least 1 (zero-length arrays are not supported)"
              shape))
    dims))

(defmacro with-quiet-rat (&body body)
  "Run BODY with ratprint bound to false.  diff and jacobian of max or min
   with float constants compare terms through rat, which otherwise prints
   a \"rat: replaced\" line for every float."
  `(let (($ratprint nil))
     ,@body))

(defun maxima-matrix-dims (mat)
  "Return (values nrow ncol) for a Maxima matrix."
  (let* ((rows (cdr mat))
         (nrow (length rows))
         (ncol (length (cdar rows))))
    (values nrow ncol)))

(defun numerics-tensor-storage (tensor)
  "Return the actual backing simple-array of a magicl tensor.
   Unlike magicl:lisp-array (which copies), this returns the live storage."
  (slot-value tensor 'magicl::storage))

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

(defun numerics-row-major-elements (tensor)
  "Elements of TENSOR as a list in row-major (C) order, whatever its storage
   layout.  The last index varies fastest, as in np_to_list."
  (let* ((shape (magicl:shape tensor))
         (rank (length shape))
         (idx (make-list rank :initial-element 0))
         (out nil))
    (dotimes (k (magicl:size tensor) (nreverse out))
      (push (apply #'magicl:tref tensor idx) out)
      (loop for d from (1- rank) downto 0
            do (incf (nth d idx))
               (if (< (nth d idx) (nth d shape))
                   (return)
                   (setf (nth d idx) 0))))))

(defun numerics-reshape-row-major (tensor shape &optional (et (magicl:element-type tensor)))
  "A new column-major tensor of SHAPE and element type ET holding TENSOR's
   elements in row-major order, as NumPy's reshape does.  magicl:reshape
   follows storage order instead, which is column-major here."
  (magicl:from-list (mapcar (lambda (x) (coerce x et)) (numerics-row-major-elements tensor))
                    shape
                    :type et
                    :layout :column-major
                    :input-layout :row-major))

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

(defun numerics-callable-p (f)
  "True if F is a Maxima function name (symbol) or lambda expression.
   Used to distinguish callable arguments from symbolic expressions."
  (or (symbolp f) (numerics-lambda-p f)))

(defun numerics-require-1d (a op-name)
  "Signal an error unless the ndarray is one-dimensional."
  (let ((shape (magicl:shape (numerics:ndarray-tensor (numerics-unwrap a)))))
    (unless (= (length shape) 1)
      (merror "~A: expected a 1D ndarray, got shape ~A" op-name shape))))

(defun numerics-require-real (a op-name)
  "Signal an error if the ndarray is complex. Used by comparison/sort/min/max."
  (when (and ($ndarray_p a)
             (eq (numerics:ndarray-dtype (numerics-unwrap a))
                 :complex-double-float))
    (merror "~A: not supported for complex ndarrays" op-name)))
