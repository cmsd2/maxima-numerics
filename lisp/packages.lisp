;;; packages.lisp — CL package definitions for maxima-numerics

(defpackage #:numerics
  (:use #:cl)
  (:export
   ;; Handle type
   #:ndarray
   #:ndarray-tensor
   #:ndarray-id
   #:ndarray-dtype
   #:ndarray-p
   #:make-ndarray
   ;; Table type (Layer 2)
   #:table
   #:table-columns
   #:table-column-names
   #:table-nrows
   #:table-p
   #:make-table
   ;; Arrow bridge (Layer 2)
   #:ndarray-to-arrow-array
   #:arrow-to-ndarray
   #:concatenate-ndarrays))
