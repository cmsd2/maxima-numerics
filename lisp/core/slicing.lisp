;;; slicing.lisp — Indexing, slicing, reshaping, and shape operations

(in-package #:maxima)

(defun $np_ref (a &rest indices)
  "Single element access (0-indexed): np_ref(A, i, j)"
  (lisp-to-maxima-number
   (apply #'magicl:tref
          (numerics:ndarray-tensor (numerics-unwrap a))
          indices)))

(defun $np_set (a &rest args)
  "Set single element (mutating, 0-indexed): np_set(A, i, j, val)
   Last argument is the value, preceding arguments are indices."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle))
         (val (maxima-to-lisp-number (car (last args)) dtype))
         (indices (butlast args)))
    (apply #'(setf magicl:tref) val tensor indices)
    a))

(defun $np_row (a i)
  "Extract row i as a 1D ndarray (0-indexed): np_row(A, i)"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle))
         (et (numerics-element-type dtype))
         (shape (magicl:shape tensor))
         (ncol (second shape))
         (result (magicl:empty (list ncol) :type et)))
    (dotimes (j ncol)
      (setf (magicl:tref result j) (magicl:tref tensor i j)))
    (numerics-wrap (numerics:make-ndarray result :dtype dtype))))

(defun $np_col (a j)
  "Extract column j as a 1D ndarray (0-indexed): np_col(A, j)"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle))
         (et (numerics-element-type dtype))
         (shape (magicl:shape tensor))
         (nrow (first shape))
         (result (magicl:empty (list nrow) :type et)))
    (dotimes (i nrow)
      (setf (magicl:tref result i) (magicl:tref tensor i j)))
    (numerics-wrap (numerics:make-ndarray result :dtype dtype))))

(defun normalize-index (idx size)
  "Normalize a possibly-negative index. Negative indices count from end."
  (if (< idx 0) (+ size idx) idx))

(defun compute-column-major-strides (shape)
  "Compute column-major strides for SHAPE.
   stride[k] = product(shape[0] ... shape[k-1]), stride[0] = 1."
  (let ((strides (make-array (length shape) :element-type 'fixnum))
        (acc 1))
    (loop for k from 0 below (length shape)
          do (setf (aref strides k) acc)
             (setf acc (* acc (nth k shape))))
    strides))

(defun parse-slice-spec (spec dim-size)
  "Parse a single dimension spec for np_slice.
   Returns (:range start end) or (:index idx)."
  (cond
    ;; all => entire dimension
    ((eq spec '$all)
     (list :range 0 dim-size))
    ;; Maxima list [start, end] => range
    ((and (consp spec) (consp (car spec)) (eq (caar spec) 'mlist))
     (let* ((elts (cdr spec))
            (start (normalize-index (first elts) dim-size))
            (end   (normalize-index (second elts) dim-size)))
       (when (>= start end)
         (merror "np_slice: empty range [~D, ~D) in dimension of size ~D"
                 start end dim-size))
       (list :range start end)))
    ;; Bare integer => scalar index (collapses dimension)
    ((integerp spec)
     (let ((idx (normalize-index spec dim-size)))
       (when (or (< idx 0) (>= idx dim-size))
         (merror "np_slice: index ~D out of bounds for dimension of size ~D"
                 spec dim-size))
       (list :index idx)))
    (t (merror "np_slice: invalid spec (expected all, [start,end], or integer)"))))

(defun nd-slice-copy (src dst src-shape src-strides dst-shape dst-strides parsed)
  "Copy elements from SRC to DST flat arrays according to PARSED specs.
   PARSED is a simple-vector of (:range start end) or (:index idx)."
  (declare (type (simple-array double-float (*)) src dst))
  (let* ((src-ndim (length src-shape))
         (out-ndim (length dst-shape))
         (fixed-offset 0)
         ;; Per output-dim: source stride and start offset
         (out-src-stride (make-array out-ndim :element-type 'fixnum))
         (out-start      (make-array out-ndim :element-type 'fixnum))
         (out-dim 0))
    (declare (type fixnum fixed-offset out-dim))
    ;; Build mapping
    (loop for k from 0 below src-ndim
          for spec = (aref parsed k)
          do (ecase (first spec)
               (:index
                (incf fixed-offset (* (second spec) (aref src-strides k))))
               (:range
                (setf (aref out-src-stride out-dim) (aref src-strides k))
                (setf (aref out-start out-dim) (second spec))
                (incf out-dim))))
    ;; Fast path: bulk copy when innermost output dim maps to contiguous source memory
    ;; (i.e. output dim 0 has source stride 1 — the column-major innermost stride)
    (if (and (> out-ndim 0) (= (aref out-src-stride 0) 1))
        ;; Bulk-copy inner dimension
        (let ((inner-len (first dst-shape))
              (inner-src-start (aref out-start 0))
              (outer-size (if (> out-ndim 1)
                              (reduce #'* (cdr dst-shape))
                              1))
              (counter (make-array (max (1- out-ndim) 1)
                                   :element-type 'fixnum :initial-element 0))
              (outer-ndim (1- out-ndim)))
          (declare (type fixnum inner-len inner-src-start outer-size))
          (dotimes (outer-flat outer-size)
            ;; Compute source base for outer dims
            (let ((src-base fixed-offset))
              (declare (type fixnum src-base))
              (dotimes (d outer-ndim)
                (incf src-base (* (+ (aref out-start (1+ d))
                                     (aref counter d))
                                  (aref out-src-stride (1+ d)))))
              ;; Bulk copy inner run
              (let ((dst-base (* outer-flat inner-len))
                    (src-off (+ src-base inner-src-start)))
                (replace dst src
                         :start1 dst-base :end1 (+ dst-base inner-len)
                         :start2 src-off  :end2 (+ src-off inner-len))))
            ;; Increment outer odometer
            (when (> outer-ndim 0)
              (loop for d from 0 below outer-ndim
                    do (incf (aref counter d))
                       (when (< (aref counter d) (nth (1+ d) dst-shape))
                         (return))
                       (setf (aref counter d) 0)))))
        ;; Generic per-element path
        (let ((out-size (reduce #'* dst-shape :initial-value 1))
              (counter (make-array out-ndim :element-type 'fixnum
                                            :initial-element 0)))
          (dotimes (flat-out out-size)
            (let ((src-idx fixed-offset))
              (declare (type fixnum src-idx))
              (dotimes (d out-ndim)
                (incf src-idx (* (+ (aref out-start d) (aref counter d))
                                 (aref out-src-stride d))))
              (setf (aref dst flat-out) (aref src src-idx)))
            ;; Increment odometer
            (loop for d from 0 below out-ndim
                  do (incf (aref counter d))
                     (when (< (aref counter d) (nth d dst-shape))
                       (return))
                     (setf (aref counter d) 0)))))))

(defun $np_slice (a &rest specs)
  "N-dimensional slicing: np_slice(A, spec0, spec1, ..., specN-1)
   Each spec can be:
     all       -- select entire dimension
     [s, e]    -- range [start, end), 0-indexed, negative indices supported
     integer   -- select single index, collapsing that dimension
   Ranges are 0-indexed with exclusive end. Negative indices count from end."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype  (numerics:ndarray-dtype handle))
         (et     (numerics-element-type dtype))
         (src-shape   (magicl:shape tensor))
         (src-ndim    (length src-shape))
         (src-storage (numerics-tensor-storage tensor)))
    ;; Validate spec count
    (unless (= (length specs) src-ndim)
      (merror "np_slice: tensor has ~D dimensions but ~D specs given"
              src-ndim (length specs)))
    ;; Parse specs
    (let* ((parsed (make-array src-ndim))
           (src-strides (compute-column-major-strides src-shape)))
      (loop for k from 0 below src-ndim
            do (setf (aref parsed k)
                     (parse-slice-spec (nth k specs) (nth k src-shape))))
      ;; Compute output shape
      (let ((out-shape
             (loop for k from 0 below src-ndim
                   for spec = (aref parsed k)
                   when (eq (first spec) :range)
                     collect (- (third spec) (second spec)))))
        ;; All dimensions collapsed => return scalar
        (when (null out-shape)
          (let ((flat-idx 0))
            (loop for k from 0 below src-ndim
                  do (incf flat-idx (* (second (aref parsed k))
                                       (aref src-strides k))))
            (return-from $np_slice
              (lisp-to-maxima-number (aref src-storage flat-idx)))))
        ;; Allocate output and copy
        (let* ((result (magicl:empty out-shape :type et :layout :column-major))
               (dst-storage (numerics-tensor-storage result))
               (dst-strides (compute-column-major-strides out-shape)))
          (nd-slice-copy src-storage dst-storage
                         src-shape src-strides
                         out-shape dst-strides
                         parsed)
          (numerics-wrap (numerics:make-ndarray result :dtype dtype)))))))

(defun $np_reshape (a new-shape)
  "Reshape ndarray: np_reshape(A, [m,n])"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle))
         (shape-list (maxima-shape-to-list new-shape)))
    (unless (= (reduce #'* shape-list) (magicl:size tensor))
      (merror "np_reshape: total size ~D does not match new shape ~A"
              (magicl:size tensor) shape-list))
    (numerics-wrap
     (numerics:make-ndarray (magicl:reshape tensor shape-list) :dtype dtype))))

(defun $np_flatten (a)
  "Flatten ndarray to 1D: np_flatten(A)"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle))
         (n (magicl:size tensor)))
    (numerics-wrap
     (numerics:make-ndarray (magicl:reshape tensor (list n)) :dtype dtype))))

(defun numerics-ensure-2d (tensor)
  "If tensor is 1D of length n, reshape to (n, 1). Otherwise return as-is."
  (if (= (length (magicl:shape tensor)) 1)
      (magicl:reshape tensor (list (first (magicl:shape tensor)) 1))
      tensor))

(defun numerics-hstack-two (ta tb dtype)
  "Horizontally concatenate two 2D tensors."
  (let* ((ta (numerics-ensure-2d ta))
         (tb (numerics-ensure-2d tb))
         (et (numerics-element-type dtype))
         (nrow (first (magicl:shape ta)))
         (ncol-a (second (magicl:shape ta)))
         (ncol-b (second (magicl:shape tb))))
    (unless (= nrow (first (magicl:shape tb)))
      (merror "np_hstack: row counts differ: ~D vs ~D" nrow (first (magicl:shape tb))))
    (let ((result (magicl:empty (list nrow (+ ncol-a ncol-b))
                                :type et :layout :column-major)))
      (dotimes (i nrow)
        (dotimes (j ncol-a)
          (setf (magicl:tref result i j) (magicl:tref ta i j)))
        (dotimes (j ncol-b)
          (setf (magicl:tref result i (+ ncol-a j)) (magicl:tref tb i j))))
      result)))

(defun numerics-stack-args (args name)
  "Normalize stack arguments: accept np_hstack(A,B,...) or np_hstack([A,B,...]).
   Returns a list of Maxima ndarray expressions."
  (let ((items (if (and (= (length args) 1)
                        (consp (first args))
                        (consp (car (first args)))
                        (eq (caar (first args)) 'mlist))
                   ;; Single Maxima list argument: extract elements
                   (cdr (first args))
                   ;; Variadic arguments
                   args)))
    (when (< (length items) 2)
      (merror "~A: need at least 2 arrays" name))
    items))

(defun $np_hstack (&rest args)
  "Horizontal concatenation: np_hstack(A, B, ...) or np_hstack([A, B, ...])
   Concatenates along columns (axis 1). Accepts 2 or more ndarrays.
   1D arrays are treated as column vectors."
  (let* ((items (numerics-stack-args args "np_hstack"))
         (handles (mapcar #'numerics-unwrap items))
         (dtype (reduce #'numerics-result-dtype
                        (mapcar #'numerics:ndarray-dtype handles)))
         (tensors (mapcar #'numerics:ndarray-tensor handles))
         (result (reduce (lambda (acc t2) (numerics-hstack-two acc t2 dtype))
                         (cdr tensors)
                         :initial-value (numerics-ensure-2d (car tensors)))))
    (numerics-wrap (numerics:make-ndarray result :dtype dtype))))

(defun numerics-vstack-two (ta tb dtype)
  "Vertically concatenate two 2D tensors."
  (let* ((ta (numerics-ensure-2d ta))
         (tb (numerics-ensure-2d tb))
         (et (numerics-element-type dtype))
         (nrow-a (first (magicl:shape ta)))
         (nrow-b (first (magicl:shape tb)))
         (ncol (second (magicl:shape ta))))
    (unless (= ncol (second (magicl:shape tb)))
      (merror "np_vstack: column counts differ: ~D vs ~D" ncol (second (magicl:shape tb))))
    (let ((result (magicl:empty (list (+ nrow-a nrow-b) ncol)
                                :type et :layout :column-major)))
      (dotimes (i nrow-a)
        (dotimes (j ncol)
          (setf (magicl:tref result i j) (magicl:tref ta i j))))
      (dotimes (i nrow-b)
        (dotimes (j ncol)
          (setf (magicl:tref result (+ nrow-a i) j) (magicl:tref tb i j))))
      result)))

(defun $np_vstack (&rest args)
  "Vertical concatenation: np_vstack(A, B, ...) or np_vstack([A, B, ...])
   Concatenates along rows (axis 0). Accepts 2 or more ndarrays.
   1D arrays are treated as column vectors."
  (let* ((items (numerics-stack-args args "np_vstack"))
         (handles (mapcar #'numerics-unwrap items))
         (dtype (reduce #'numerics-result-dtype
                        (mapcar #'numerics:ndarray-dtype handles)))
         (tensors (mapcar #'numerics:ndarray-tensor handles))
         (result (reduce (lambda (acc t2) (numerics-vstack-two acc t2 dtype))
                         (cdr tensors)
                         :initial-value (numerics-ensure-2d (car tensors)))))
    (numerics-wrap (numerics:make-ndarray result :dtype dtype))))

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
