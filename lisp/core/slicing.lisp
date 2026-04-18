;;; slicing.lisp — Indexing, slicing, reshaping, and shape operations

(in-package #:maxima)

(defun $np_ref (a &rest indices)
  "Single element access (0-indexed): np_ref(A, i, j)"
  (apply #'magicl:tref
         (numerics:ndarray-tensor (numerics-unwrap a))
         indices))

(defun $np_set (a &rest args)
  "Set single element (mutating, 0-indexed): np_set(A, i, j, val)
   Last argument is the value, preceding arguments are indices."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (val (coerce ($float (car (last args))) 'double-float))
         (indices (butlast args)))
    (apply #'(setf magicl:tref) val tensor indices)
    a))

(defun $np_row (a i)
  "Extract row i as a 1D ndarray (0-indexed): np_row(A, i)"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (ncol (second shape))
         (result (magicl:empty (list ncol) :type 'double-float)))
    (dotimes (j ncol)
      (setf (magicl:tref result j) (magicl:tref tensor i j)))
    (numerics-wrap (numerics:make-ndarray result))))

(defun $np_col (a j)
  "Extract column j as a 1D ndarray (0-indexed): np_col(A, j)"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (nrow (first shape))
         (result (magicl:empty (list nrow) :type 'double-float)))
    (dotimes (i nrow)
      (setf (magicl:tref result i) (magicl:tref tensor i j)))
    (numerics-wrap (numerics:make-ndarray result))))

(defun $np_slice (a rows cols)
  "Extract a sub-matrix: np_slice(A, [row_start, row_end], [col_start, col_end])
   Ranges are 0-indexed, end is exclusive. Negative indices count from end."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (nrow (first shape))
         (ncol (second shape))
         (r-start (normalize-index (nth 1 rows) nrow))
         (r-end   (normalize-index (nth 2 rows) nrow))
         (c-start (normalize-index (nth 1 cols) ncol))
         (c-end   (normalize-index (nth 2 cols) ncol))
         (out-rows (- r-end r-start))
         (out-cols (- c-end c-start)))
    (when (or (<= out-rows 0) (<= out-cols 0))
      (merror "np_slice: empty slice"))
    (let ((result (magicl:empty (list out-rows out-cols)
                                :type 'double-float :layout :column-major)))
      (dotimes (i out-rows)
        (dotimes (j out-cols)
          (setf (magicl:tref result i j)
                (magicl:tref tensor (+ r-start i) (+ c-start j)))))
      (numerics-wrap (numerics:make-ndarray result)))))

(defun normalize-index (idx size)
  "Normalize a possibly-negative index. Negative indices count from end."
  (if (< idx 0) (+ size idx) idx))

(defun $np_reshape (a new-shape)
  "Reshape ndarray: np_reshape(A, [m,n])"
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a)))
         (shape-list (maxima-shape-to-list new-shape)))
    (unless (= (reduce #'* shape-list) (magicl:size tensor))
      (merror "np_reshape: total size ~D does not match new shape ~A"
              (magicl:size tensor) shape-list))
    (numerics-wrap
     (numerics:make-ndarray (magicl:reshape tensor shape-list)))))

(defun $np_flatten (a)
  "Flatten ndarray to 1D: np_flatten(A)"
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a)))
         (n (magicl:size tensor)))
    (numerics-wrap
     (numerics:make-ndarray (magicl:reshape tensor (list n))))))

(defun $np_hstack (a b)
  "Horizontal concatenation: np_hstack(A, B)
   Concatenates along columns (axis 1)."
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (tb (numerics:ndarray-tensor (numerics-unwrap b)))
         (sha (magicl:shape ta))
         (shb (magicl:shape tb))
         (nrow (first sha))
         (ncol-a (second sha))
         (ncol-b (second shb)))
    (unless (= nrow (first shb))
      (merror "np_hstack: row counts differ: ~D vs ~D" nrow (first shb)))
    (let ((result (magicl:empty (list nrow (+ ncol-a ncol-b))
                                :type 'double-float :layout :column-major)))
      (dotimes (i nrow)
        (dotimes (j ncol-a)
          (setf (magicl:tref result i j) (magicl:tref ta i j)))
        (dotimes (j ncol-b)
          (setf (magicl:tref result i (+ ncol-a j)) (magicl:tref tb i j))))
      (numerics-wrap (numerics:make-ndarray result)))))

(defun $np_vstack (a b)
  "Vertical concatenation: np_vstack(A, B)
   Concatenates along rows (axis 0)."
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (tb (numerics:ndarray-tensor (numerics-unwrap b)))
         (sha (magicl:shape ta))
         (shb (magicl:shape tb))
         (nrow-a (first sha))
         (nrow-b (first shb))
         (ncol (second sha)))
    (unless (= ncol (second shb))
      (merror "np_vstack: column counts differ: ~D vs ~D" ncol (second shb)))
    (let ((result (magicl:empty (list (+ nrow-a nrow-b) ncol)
                                :type 'double-float :layout :column-major)))
      (dotimes (i nrow-a)
        (dotimes (j ncol)
          (setf (magicl:tref result i j) (magicl:tref ta i j))))
      (dotimes (i nrow-b)
        (dotimes (j ncol)
          (setf (magicl:tref result (+ nrow-a i) j) (magicl:tref tb i j))))
      (numerics-wrap (numerics:make-ndarray result)))))

(defun $np_shape (a)
  "Shape as Maxima list: np_shape(A) => [3, 4]"
  `((mlist simp) ,@(magicl:shape
                     (numerics:ndarray-tensor (numerics-unwrap a)))))

(defun $np_size (a)
  "Total element count: np_size(A) => 12"
  (magicl:size (numerics:ndarray-tensor (numerics-unwrap a))))

(defun $np_dtype (a)
  "Element type as string: np_dtype(A)"
  (let ((dtype (numerics:ndarray-dtype (numerics-unwrap a))))
    (symbol-name dtype)))
