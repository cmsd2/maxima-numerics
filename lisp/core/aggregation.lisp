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

(defun $np_min (a)
  "Minimum element: np_min(A) => scalar"
  (let ((flat (numerics-flat-array (numerics:ndarray-tensor (numerics-unwrap a)))))
    (reduce #'min flat)))

(defun $np_max (a)
  "Maximum element: np_max(A) => scalar"
  (let ((flat (numerics-flat-array (numerics:ndarray-tensor (numerics-unwrap a)))))
    (reduce #'max flat)))

(defun $np_argmin (a)
  "Index of minimum element (in storage order): np_argmin(A) => integer"
  (let* ((flat (numerics-flat-array (numerics:ndarray-tensor (numerics-unwrap a))))
         (min-val (aref flat 0))
         (min-idx 0))
    (loop for i from 1 below (length flat)
          do (when (< (aref flat i) min-val)
               (setf min-val (aref flat i)
                     min-idx i)))
    min-idx))

(defun $np_argmax (a)
  "Index of maximum element (in storage order): np_argmax(A) => integer"
  (let* ((flat (numerics-flat-array (numerics:ndarray-tensor (numerics-unwrap a))))
         (max-val (aref flat 0))
         (max-idx 0))
    (loop for i from 1 below (length flat)
          do (when (> (aref flat i) max-val)
               (setf max-val (aref flat i)
                     max-idx i)))
    max-idx))

(defun $np_std (a)
  "Standard deviation: np_std(A) => scalar"
  (sqrt ($np_var a)))

(defun $np_var (a)
  "Variance: np_var(A) => scalar"
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a)))
         (flat (numerics-flat-array tensor))
         (n (length flat))
         (mean (/ (reduce #'+ flat) (coerce n 'double-float)))
         (sum-sq 0.0d0))
    (map nil (lambda (x)
               (let ((d (- x mean)))
                 (incf sum-sq (* d d))))
         flat)
    (/ sum-sq (coerce n 'double-float))))

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
