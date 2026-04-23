;;; aggregation.lisp — Aggregation functions

(in-package #:maxima)

(defun $np_sum (a &optional axis)
  "Sum of elements. np_sum(A) => scalar; np_sum(A, 0) => column sums."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle))
         (et (numerics-element-type dtype)))
    (if (null axis)
        ;; Total sum: return scalar
        (let ((flat (numerics-flat-array tensor))
              (sum (coerce 0 et)))
          (map nil (lambda (x) (incf sum x)) flat)
          (lisp-to-maxima-number sum))
        ;; Along axis: return ndarray
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape)))
          (ecase axis
            (0 ;; Sum across rows => 1D of length ncol
             (let ((result (magicl:zeros (list ncol) :type et)))
               (dotimes (j ncol)
                 (let ((s (coerce 0 et)))
                   (dotimes (i nrow) (incf s (magicl:tref tensor i j)))
                   (setf (magicl:tref result j) s)))
               (numerics-wrap (numerics:make-ndarray result :dtype dtype))))
            (1 ;; Sum across columns => 1D of length nrow
             (let ((result (magicl:zeros (list nrow) :type et)))
               (dotimes (i nrow)
                 (let ((s (coerce 0 et)))
                   (dotimes (j ncol) (incf s (magicl:tref tensor i j)))
                   (setf (magicl:tref result i) s)))
               (numerics-wrap (numerics:make-ndarray result :dtype dtype)))))))))

(defun $np_mean (a &optional axis)
  "Mean of elements. np_mean(A) => scalar; np_mean(A, 0) => column means."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle)))
    (if (null axis)
        (let ((s ($np_sum a))
              (n (coerce (magicl:size tensor) 'double-float)))
          (lisp-to-maxima-number
           (/ (maxima-to-lisp-number s dtype) n)))
        (let* ((shape (magicl:shape tensor))
               (sum-result (numerics-unwrap ($np_sum a axis)))
               (sum-tensor (numerics:ndarray-tensor sum-result))
               (result (magicl:deep-copy-tensor sum-tensor))
               (divisor (coerce (nth axis shape) 'double-float)))
          (magicl:map! (lambda (x) (/ x divisor)) result)
          (numerics-wrap (numerics:make-ndarray result :dtype dtype))))))

(defun $np_min (a &optional axis)
  "Minimum element. np_min(A) => scalar; np_min(A, 0) => column mins."
  (numerics-require-real a "np_min")
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
  (numerics-require-real a "np_max")
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
  (numerics-require-real a "np_argmin")
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
  (numerics-require-real a "np_argmax")
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
  "Variance. np_var(A) => scalar; np_var(A, 0) => column variances.
   For complex arrays, computes |x - mean|^2. Result is always real."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle)))
    (if (null axis)
        ;; Scalar variance
        (let* ((flat (numerics-flat-array tensor))
               (n (length flat))
               (mean (/ (reduce #'+ flat) (coerce n 'double-float)))
               (sum-sq 0.0d0))
          (map nil (lambda (x)
                     (let ((d (- x mean)))
                       (incf sum-sq (expt (abs d) 2))))
               flat)
          (/ sum-sq (coerce n 'double-float)))
        ;; Along axis — result always real
        (let* ((shape (magicl:shape tensor))
               (nrow (first shape))
               (ncol (second shape)))
          (ecase axis
            (0 ;; Variance across rows => 1D of length ncol
             (let ((result (magicl:empty (list ncol) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (j ncol)
                 (let ((mean (coerce 0 (numerics-element-type
                                        (numerics:ndarray-dtype handle)))))
                   (dotimes (i nrow) (incf mean (magicl:tref tensor i j)))
                   (setf mean (/ mean (coerce nrow 'double-float)))
                   (let ((sum-sq 0.0d0))
                     (dotimes (i nrow)
                       (let ((d (- (magicl:tref tensor i j) mean)))
                         (incf sum-sq (expt (abs d) 2))))
                     (setf (magicl:tref result j)
                           (/ sum-sq (coerce nrow 'double-float))))))
               (numerics-wrap (numerics:make-ndarray result))))
            (1 ;; Variance across columns => 1D of length nrow
             (let ((result (magicl:empty (list nrow) :type 'double-float
                                                     :layout :column-major)))
               (dotimes (i nrow)
                 (let ((mean (coerce 0 (numerics-element-type
                                        (numerics:ndarray-dtype handle)))))
                   (dotimes (j ncol) (incf mean (magicl:tref tensor i j)))
                   (setf mean (/ mean (coerce ncol 'double-float)))
                   (let ((sum-sq 0.0d0))
                     (dotimes (j ncol)
                       (let ((d (- (magicl:tref tensor i j) mean)))
                         (incf sum-sq (expt (abs d) 2))))
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
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle))
         (et (numerics-element-type dtype))
         (n (magicl:size tensor))
         (acc (coerce 0 et))
         (vals (loop for i below n
                     do (incf acc (magicl:tref tensor i))
                     collect acc)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:from-list vals (list n) :type et)
      :dtype dtype))))

(defun $np_dot (a b)
  "Dot product of two 1D vectors: np_dot(a, b) => scalar"
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (tb (numerics:ndarray-tensor (numerics-unwrap b)))
         (sa (numerics-flat-array ta))
         (sb (numerics-flat-array tb)))
    (unless (= (length sa) (length sb))
      (merror "np_dot: vectors must have same length, got ~D and ~D"
              (length sa) (length sb)))
    (let ((sum 0))
      (dotimes (i (length sa))
        (incf sum (* (aref sa i) (aref sb i))))
      (lisp-to-maxima-number sum))))

(defun $np_sort (a &optional axis)
  "Sort elements. np_sort(A) => sorted 1D; np_sort(A, 0) => sort columns;
   np_sort(A, 1) => sort rows. Returns a new ndarray (stable ascending sort)."
  (numerics-require-real a "np_sort")
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

(defun $np_trapz (y &optional x)
  "Trapezoidal integration of 1D data.
   np_trapz(y) — assumes unit spacing (dx=1).
   np_trapz(y, x) — uses x values for spacing."
  (let* ((hy (numerics-unwrap y))
         (ty (numerics:ndarray-tensor hy))
         (n (magicl:size ty)))
    (when (< n 2)
      (merror "np_trapz: need at least 2 points, got ~D" n))
    (if (null x)
        ;; Unit spacing
        (let ((sum 0.0d0))
          (loop for i from 0 below (1- n)
                do (incf sum (* 0.5d0
                                (+ (magicl:tref ty i)
                                   (magicl:tref ty (1+ i))))))
          sum)
        ;; Variable spacing
        (let* ((hx (numerics-unwrap x))
               (tx (numerics:ndarray-tensor hx))
               (nx (magicl:size tx))
               (sum 0.0d0))
          (unless (= n nx)
            (merror "np_trapz: x and y must have same length, got ~D and ~D" nx n))
          (loop for i from 0 below (1- n)
                for dx = (- (magicl:tref tx (1+ i)) (magicl:tref tx i))
                do (incf sum (* 0.5d0 dx
                                (+ (magicl:tref ty i)
                                   (magicl:tref ty (1+ i))))))
          sum))))

(defun $np_diff (a)
  "First-order finite differences of a 1D ndarray: np_diff(a)
   Returns an ndarray of length n-1 where result[i] = a[i+1] - a[i]."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle))
         (et (numerics-element-type dtype))
         (n (magicl:size tensor)))
    (when (< n 2)
      (merror "np_diff: need at least 2 elements, got ~D" n))
    (let* ((m (1- n))
           (result (magicl:empty (list m) :type et :layout :column-major)))
      (dotimes (i m)
        (setf (magicl:tref result i)
              (- (magicl:tref tensor (1+ i)) (magicl:tref tensor i))))
      (numerics-wrap (numerics:make-ndarray result :dtype dtype)))))

(defun $np_discount (rewards gamma)
  "Discounted cumulative returns (1D): np_discount(rewards, gamma).
   G[i] = r[i] + gamma*G[i+1] (reverse cumsum with decay).
   Used in reinforcement learning for computing returns from a reward sequence."
  (let* ((handle (numerics-unwrap rewards))
         (tensor (numerics:ndarray-tensor handle))
         (n (magicl:size tensor))
         (g (coerce ($float gamma) 'double-float))
         (result (magicl:empty (list n) :type 'double-float :layout :column-major))
         (acc 0.0d0))
    (loop for i from (1- n) downto 0
          do (setf acc (+ (magicl:tref tensor i) (* g acc)))
             (setf (magicl:tref result i) acc))
    (numerics-wrap (numerics:make-ndarray result))))

(defun compute-cov-tensor (tensor)
  "Internal: compute sample covariance matrix from a 2D magicl tensor.
   Returns the raw p x p magicl tensor (no ndarray wrapping)."
  (let* ((shape (magicl:shape tensor))
         (n (first shape))
         (p (second shape)))
    (when (< n 2)
      (merror "np_cov: need at least 2 observations, got ~D" n))
    ;; Compute column means
    (let ((means (make-array p :element-type 'double-float :initial-element 0.0d0)))
      (dotimes (j p)
        (dotimes (i n)
          (incf (aref means j) (magicl:tref tensor i j)))
        (setf (aref means j) (/ (aref means j) (coerce n 'double-float))))
      ;; Compute covariance matrix (exploit symmetry)
      (let ((result (magicl:zeros (list p p) :type 'double-float
                                              :layout :column-major))
            (denom (coerce (1- n) 'double-float)))
        (dotimes (j1 p)
          (loop for j2 from j1 below p do
            (let ((s 0.0d0))
              (dotimes (i n)
                (incf s (* (- (magicl:tref tensor i j1) (aref means j1))
                           (- (magicl:tref tensor i j2) (aref means j2)))))
              (setf s (/ s denom))
              (setf (magicl:tref result j1 j2) s)
              (setf (magicl:tref result j2 j1) s))))
        result))))

(defun $np_cov (a)
  "Sample covariance matrix of a 2D ndarray (columns = variables).
   np_cov(A) => p x p matrix where A is n x p.
   Uses sample covariance (divides by n-1)."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle)))
    (numerics-wrap (numerics:make-ndarray (compute-cov-tensor tensor)))))

(defun $np_corrcoef (a)
  "Pearson correlation matrix of a 2D ndarray (columns = variables).
   np_corrcoef(A) => p x p matrix where A is n x p."
  (let* ((handle (numerics-unwrap a))
         (cov-tensor (compute-cov-tensor (numerics:ndarray-tensor handle)))
         (p (first (magicl:shape cov-tensor)))
         (result (magicl:zeros (list p p) :type 'double-float
                                           :layout :column-major)))
    ;; corr[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])
    ;; Exploit symmetry
    (dotimes (i p)
      (setf (magicl:tref result i i) 1.0d0)
      (loop for j from (1+ i) below p do
        (let ((denom (sqrt (* (magicl:tref cov-tensor i i)
                              (magicl:tref cov-tensor j j)))))
          (let ((r (if (zerop denom) 0.0d0
                       (/ (magicl:tref cov-tensor i j) denom))))
            (setf (magicl:tref result i j) r)
            (setf (magicl:tref result j i) r)))))
    (numerics-wrap (numerics:make-ndarray result))))

(defun $np_argsort (a &optional axis)
  "Indices that sort elements. np_argsort(A) => 1D indices;
   np_argsort(A, 0) => row indices per column; np_argsort(A, 1) => col indices per row.
   Returns ndarray of indices as double-float (stable ascending sort)."
  (numerics-require-real a "np_argsort")
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
