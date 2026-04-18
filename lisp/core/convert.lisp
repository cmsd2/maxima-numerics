;;; convert.lisp — Conversion between Maxima matrices/lists and ndarrays

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
     (let* ((dtype (if (member '$complex options) :complex-double-float
                       :double-float))
            (shape-arg (find-if (lambda (o) (not (member o '($complex $complex128 $float64 $real))))
                                options)))
       (numerics-wrap (maxima-list-to-ndarray a shape-arg dtype))))
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
                    (maxima-to-lisp-number col dtype))
              (incf c)))
          (incf r)))
      (numerics:make-ndarray tensor :dtype dtype))))

(defun maxima-list-to-ndarray (lst shape-arg dtype)
  "Convert a Maxima list to an ndarray with optional reshape."
  (let* ((elements (cdr lst))
         (n (length elements))
         (shape (if shape-arg
                    (maxima-shape-to-list shape-arg)
                    (list n)))
         (total (reduce #'* shape))
         (element-type (ecase dtype
                         (:double-float 'double-float)
                         (:complex-double-float '(complex double-float))))
         (tensor (magicl:empty shape :type element-type
                                     :layout :column-major)))
    (unless (= n total)
      (merror "ndarray: list has ~D elements but shape ~A requires ~D"
              n shape total))
    ;; Fill in row-major order (as the user provides), magicl stores column-major
    (let ((i 0))
      (dolist (el elements)
        (let ((indices (if (= (length shape) 1)
                          (list i)
                          ;; row-major index to subscripts
                          (let* ((ncol (second shape))
                                 (r (floor i ncol))
                                 (c (mod i ncol)))
                            (list r c)))))
          (apply #'(setf magicl:tref)
                 (maxima-to-lisp-number el dtype)
                 tensor indices))
        (incf i)))
    (numerics:make-ndarray tensor :dtype dtype)))

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
  "Flatten an ndarray to a Maxima list in row-major order."
  (let* ((handle (numerics-unwrap x))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (n (magicl:size tensor)))
    (if (= (length shape) 1)
        ;; 1D: just dump elements in order
        (let ((vals nil))
          (dotimes (i n)
            (push (lisp-to-maxima-number (magicl:tref tensor i)) vals))
          `((mlist simp) ,@(nreverse vals)))
        ;; 2D+: iterate in row-major order
        (let* ((nrow (first shape))
               (ncol (second shape))
               (vals nil))
          (dotimes (i nrow)
            (dotimes (j ncol)
              (push (lisp-to-maxima-number (magicl:tref tensor i j)) vals)))
          `((mlist simp) ,@(nreverse vals))))))
