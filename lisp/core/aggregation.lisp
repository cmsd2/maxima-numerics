;;; aggregation.lisp — Aggregation functions

(in-package #:maxima)

(defun $np_sum (a &optional axis)
  "Sum of elements. np_sum(A) => scalar; np_sum(A, 0) => column sums."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        ;; Total sum: return scalar
        (let ((flat (numerics-flat-array tensor))
              (sum 0.0d0))
          (map nil (lambda (x) (incf sum x)) flat)
          sum)
        ;; Along axis: return ndarray
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape)))
          (ecase axis
            (0 ;; Sum across rows => 1D of length ncol
             (let ((result (magicl:zeros (list ncol) :type 'double-float)))
               (dotimes (j ncol)
                 (let ((s 0.0d0))
                   (dotimes (i nrow) (incf s (magicl:tref tensor i j)))
                   (setf (magicl:tref result j) s)))
               (numerics-wrap (numerics:make-ndarray result))))
            (1 ;; Sum across columns => 1D of length nrow
             (let ((result (magicl:zeros (list nrow) :type 'double-float)))
               (dotimes (i nrow)
                 (let ((s 0.0d0))
                   (dotimes (j ncol) (incf s (magicl:tref tensor i j)))
                   (setf (magicl:tref result i) s)))
               (numerics-wrap (numerics:make-ndarray result)))))))))

(defun $np_mean (a &optional axis)
  "Mean of elements. np_mean(A) => scalar; np_mean(A, 0) => column means."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        (/ ($np_sum a) (coerce (magicl:size tensor) 'double-float))
        (let* ((shape (magicl:shape tensor))
               (sum-result (numerics-unwrap ($np_sum a axis)))
               (sum-tensor (numerics:ndarray-tensor sum-result))
               (result (magicl:deep-copy-tensor sum-tensor))
               (divisor (coerce (nth axis shape) 'double-float)))
          (magicl:map! (lambda (x) (/ x divisor)) result)
          (numerics-wrap (numerics:make-ndarray result))))))

(defun $np_min (a &optional axis)
  "Minimum element. np_min(A) => scalar; np_min(A, 0) => column mins."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        (reduce #'min (numerics-flat-array tensor))
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape)))
          (ecase axis
            (0 ;; Min across rows => 1D of length ncol
             (let ((result (magicl:empty (list ncol) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (j ncol)
                 (let ((m (magicl:tref tensor 0 j)))
                   (loop for i from 1 below nrow
                         do (setf m (min m (magicl:tref tensor i j))))
                   (setf (magicl:tref result j) m)))
               (numerics-wrap (numerics:make-ndarray result))))
            (1 ;; Min across columns => 1D of length nrow
             (let ((result (magicl:empty (list nrow) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (i nrow)
                 (let ((m (magicl:tref tensor i 0)))
                   (loop for j from 1 below ncol
                         do (setf m (min m (magicl:tref tensor i j))))
                   (setf (magicl:tref result i) m)))
               (numerics-wrap (numerics:make-ndarray result)))))))))


(defun $np_max (a &optional axis)
  "Maximum element. np_max(A) => scalar; np_max(A, 0) => column maxes."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        (reduce #'max (numerics-flat-array tensor))
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape)))
          (ecase axis
            (0 ;; Max across rows => 1D of length ncol
             (let ((result (magicl:empty (list ncol) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (j ncol)
                 (let ((m (magicl:tref tensor 0 j)))
                   (loop for i from 1 below nrow
                         do (setf m (max m (magicl:tref tensor i j))))
                   (setf (magicl:tref result j) m)))
               (numerics-wrap (numerics:make-ndarray result))))
            (1 ;; Max across columns => 1D of length nrow
             (let ((result (magicl:empty (list nrow) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (i nrow)
                 (let ((m (magicl:tref tensor i 0)))
                   (loop for j from 1 below ncol
                         do (setf m (max m (magicl:tref tensor i j))))
                   (setf (magicl:tref result i) m)))
               (numerics-wrap (numerics:make-ndarray result)))))))))


(defun $np_argmin (a &optional axis)
  "Index of minimum element. np_argmin(A) => integer; np_argmin(A, 0) => 1D ndarray."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        ;; Scalar: flat index
        (let* ((flat (numerics-flat-array tensor))
               (min-val (aref flat 0))
               (min-idx 0))
          (loop for i from 1 below (length flat)
                do (when (< (aref flat i) min-val)
                     (setf min-val (aref flat i)
                           min-idx i)))
          min-idx)
        ;; Along axis: return 1D ndarray of indices
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape)))
          (ecase axis
            (0 ;; Argmin across rows => index per column
             (let ((result (magicl:empty (list ncol) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (j ncol)
                 (let ((min-val (magicl:tref tensor 0 j))
                       (min-idx 0))
                   (loop for i from 1 below nrow
                         do (when (< (magicl:tref tensor i j) min-val)
                              (setf min-val (magicl:tref tensor i j)
                                    min-idx i)))
                   (setf (magicl:tref result j) (coerce min-idx 'double-float))))
               (numerics-wrap (numerics:make-ndarray result))))
            (1 ;; Argmin across columns => index per row
             (let ((result (magicl:empty (list nrow) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (i nrow)
                 (let ((min-val (magicl:tref tensor i 0))
                       (min-idx 0))
                   (loop for j from 1 below ncol
                         do (when (< (magicl:tref tensor i j) min-val)
                              (setf min-val (magicl:tref tensor i j)
                                    min-idx j)))
                   (setf (magicl:tref result i) (coerce min-idx 'double-float))))
               (numerics-wrap (numerics:make-ndarray result)))))))))

(defun $np_argmax (a &optional axis)
  "Index of maximum element. np_argmax(A) => integer; np_argmax(A, 0) => 1D ndarray."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        ;; Scalar: flat index
        (let* ((flat (numerics-flat-array tensor))
               (max-val (aref flat 0))
               (max-idx 0))
          (loop for i from 1 below (length flat)
                do (when (> (aref flat i) max-val)
                     (setf max-val (aref flat i)
                           max-idx i)))
          max-idx)
        ;; Along axis: return 1D ndarray of indices
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape)))
          (ecase axis
            (0 ;; Argmax across rows => index per column
             (let ((result (magicl:empty (list ncol) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (j ncol)
                 (let ((max-val (magicl:tref tensor 0 j))
                       (max-idx 0))
                   (loop for i from 1 below nrow
                         do (when (> (magicl:tref tensor i j) max-val)
                              (setf max-val (magicl:tref tensor i j)
                                    max-idx i)))
                   (setf (magicl:tref result j) (coerce max-idx 'double-float))))
               (numerics-wrap (numerics:make-ndarray result))))
            (1 ;; Argmax across columns => index per row
             (let ((result (magicl:empty (list nrow) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (i nrow)
                 (let ((max-val (magicl:tref tensor i 0))
                       (max-idx 0))
                   (loop for j from 1 below ncol
                         do (when (> (magicl:tref tensor i j) max-val)
                              (setf max-val (magicl:tref tensor i j)
                                    max-idx j)))
                   (setf (magicl:tref result i) (coerce max-idx 'double-float))))
               (numerics-wrap (numerics:make-ndarray result)))))))))

(defun $np_var (a &optional axis)
  "Variance. np_var(A) => scalar; np_var(A, 0) => column variances."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        ;; Scalar variance
        (let* ((flat (numerics-flat-array tensor))
               (n (length flat))
               (mean (/ (reduce #'+ flat) (coerce n 'double-float)))
               (sum-sq 0.0d0))
          (map nil (lambda (x)
                     (let ((d (- x mean)))
                       (incf sum-sq (* d d))))
               flat)
          (/ sum-sq (coerce n 'double-float)))
        ;; Along axis
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape)))
          (ecase axis
            (0 ;; Variance across rows => 1D of length ncol
             (let ((result (magicl:empty (list ncol) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (j ncol)
                 (let ((mean 0.0d0))
                   (dotimes (i nrow) (incf mean (magicl:tref tensor i j)))
                   (setf mean (/ mean (coerce nrow 'double-float)))
                   (let ((sum-sq 0.0d0))
                     (dotimes (i nrow)
                       (let ((d (- (magicl:tref tensor i j) mean)))
                         (incf sum-sq (* d d))))
                     (setf (magicl:tref result j)
                           (/ sum-sq (coerce nrow 'double-float))))))
               (numerics-wrap (numerics:make-ndarray result))))
            (1 ;; Variance across columns => 1D of length nrow
             (let ((result (magicl:empty (list nrow) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (i nrow)
                 (let ((mean 0.0d0))
                   (dotimes (j ncol) (incf mean (magicl:tref tensor i j)))
                   (setf mean (/ mean (coerce ncol 'double-float)))
                   (let ((sum-sq 0.0d0))
                     (dotimes (j ncol)
                       (let ((d (- (magicl:tref tensor i j) mean)))
                         (incf sum-sq (* d d))))
                     (setf (magicl:tref result i)
                           (/ sum-sq (coerce ncol 'double-float))))))
               (numerics-wrap (numerics:make-ndarray result)))))))))

(defun $np_std (a &optional axis)
  "Standard deviation. np_std(A) => scalar; np_std(A, 0) => column std devs."
  (if (null axis)
      (sqrt ($np_var a))
      ;; Apply sqrt element-wise to the variance result
      (let* ((var-result (numerics-unwrap ($np_var a axis)))
             (var-tensor (numerics:ndarray-tensor var-result))
             (result (magicl:deep-copy-tensor var-tensor)))
        (magicl:map! #'sqrt result)
        (numerics-wrap (numerics:make-ndarray result)))))

(defun $np_cumsum (a)
  "Cumulative sum (1D): np_cumsum(A)"
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a)))
         (n (magicl:size tensor))
         (acc 0.0d0)
         (vals (loop for i below n
                     do (incf acc (magicl:tref tensor i))
                     collect acc)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:from-list vals (list n) :type 'double-float)))))

(defun $np_dot (a b)
  "Dot product of two 1D vectors: np_dot(a, b) => scalar"
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (tb (numerics:ndarray-tensor (numerics-unwrap b)))
         (sa (numerics-flat-array ta))
         (sb (numerics-flat-array tb)))
    (unless (= (length sa) (length sb))
      (merror "np_dot: vectors must have same length, got ~D and ~D"
              (length sa) (length sb)))
    (let ((sum 0.0d0))
      (dotimes (i (length sa))
        (incf sum (* (aref sa i) (aref sb i))))
      sum)))

(defun $np_sort (a &optional axis)
  "Sort elements. np_sort(A) => sorted 1D; np_sort(A, 0) => sort columns;
   np_sort(A, 1) => sort rows. Returns a new ndarray (stable ascending sort)."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        ;; No axis: flatten, sort, return 1D
        (let* ((flat (numerics-flat-array tensor))
               (n (length flat))
               (sorted (make-array n :element-type 'double-float)))
          (replace sorted flat)
          (setf sorted (stable-sort sorted #'<))
          (numerics-wrap
           (numerics:make-ndarray
            (magicl:from-list (coerce sorted 'list) (list n)
                              :type 'double-float))))
        ;; Axis: sort along axis, result has same shape
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape))
               (result (magicl:deep-copy-tensor tensor)))
          (ecase axis
            (0 ;; Sort each column (sort within rows for each col index)
             (dotimes (j ncol)
               (let ((col (make-array nrow :element-type 'double-float)))
                 (dotimes (i nrow) (setf (aref col i) (magicl:tref tensor i j)))
                 (setf col (stable-sort col #'<))
                 (dotimes (i nrow) (setf (magicl:tref result i j) (aref col i))))))
            (1 ;; Sort each row
             (dotimes (i nrow)
               (let ((row (make-array ncol :element-type 'double-float)))
                 (dotimes (j ncol) (setf (aref row j) (magicl:tref tensor i j)))
                 (setf row (stable-sort row #'<))
                 (dotimes (j ncol) (setf (magicl:tref result i j) (aref row j)))))))
          (numerics-wrap (numerics:make-ndarray result))))))

(defun $np_argsort (a &optional axis)
  "Indices that sort elements. np_argsort(A) => 1D indices;
   np_argsort(A, 0) => row indices per column; np_argsort(A, 1) => col indices per row.
   Returns ndarray of indices as double-float (stable ascending sort)."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (if (null axis)
        ;; No axis: flatten, argsort, return 1D
        (let* ((flat (numerics-flat-array tensor))
               (n (length flat))
               (indices (make-array n :element-type 'fixnum)))
          (dotimes (i n) (setf (aref indices i) i))
          (setf indices (stable-sort indices
                                     (lambda (a b) (< (aref flat a) (aref flat b)))))
          (let ((result (magicl:empty (list n) :type 'double-float
                                               :layout :column-major)))
            (dotimes (i n)
              (setf (magicl:tref result i) (coerce (aref indices i) 'double-float)))
            (numerics-wrap (numerics:make-ndarray result))))
        ;; Axis: argsort along axis, result has same shape
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape))
               (result (magicl:empty shape :type 'double-float
                                           :layout :column-major)))
          (ecase axis
            (0 ;; For each column, return row indices that sort it
             (dotimes (j ncol)
               (let ((indices (make-array nrow :element-type 'fixnum)))
                 (dotimes (i nrow) (setf (aref indices i) i))
                 (setf indices
                       (stable-sort indices
                                    (lambda (a b)
                                      (< (magicl:tref tensor a j)
                                         (magicl:tref tensor b j)))))
                 (dotimes (i nrow)
                   (setf (magicl:tref result i j)
                         (coerce (aref indices i) 'double-float))))))
            (1 ;; For each row, return col indices that sort it
             (dotimes (i nrow)
               (let ((indices (make-array ncol :element-type 'fixnum)))
                 (dotimes (j ncol) (setf (aref indices j) j))
                 (setf indices
                       (stable-sort indices
                                    (lambda (a b)
                                      (< (magicl:tref tensor i a)
                                         (magicl:tref tensor i b)))))
                 (dotimes (j ncol)
                   (setf (magicl:tref result i j)
                         (coerce (aref indices j) 'double-float)))))))
          (numerics-wrap (numerics:make-ndarray result))))))
