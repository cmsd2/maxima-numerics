;;; display.lisp — Custom Maxima display for ndarray

(in-package #:maxima)

(displa-def $ndarray dim-$ndarray)

(defun dim-$ndarray (form result)
  "Display an ndarray. Small 2D arrays show contents; large ones show summary.
   Falls back to generic display if the argument is not an ndarray struct
   (e.g. when displaying unevaluated expressions)."
  (let ((handle (cadr form)))
    (if (typep handle 'numerics:ndarray)
        (let* ((tensor (numerics:ndarray-tensor handle))
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
               result)))
        ;; Not an ndarray struct -- display as generic function call
        (dimension-function form result))))

;; TeX output
(defprop $ndarray tex-ndarray tex)

(defun tex-ndarray (x l r)
  (let ((handle (cadr x)))
    (if (typep handle 'numerics:ndarray)
        (let* ((shape (magicl:shape (numerics:ndarray-tensor handle)))
               (dtype (numerics:ndarray-dtype handle)))
          (append l
                  (list (format nil "\\text{ndarray}(~A, ~A)" shape dtype))
                  r))
        ;; Fallback for symbolic/unevaluated forms
        (append l (list "\\text{ndarray}(\\ldots)") r))))
