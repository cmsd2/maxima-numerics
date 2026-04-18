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
