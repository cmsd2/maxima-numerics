;;; constructors.lisp — ndarray constructor functions

(in-package #:maxima)

(defun $np_zeros (shape &optional dtype-arg)
  "Create a zero-filled ndarray: np_zeros([m,n]) or np_zeros(n)
   Optional dtype: np_zeros([m,n], complex)"
  (let* ((dtype (numerics-parse-dtype dtype-arg))
         (et (numerics-element-type dtype)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:zeros (maxima-shape-to-list shape)
                    :type et :layout :column-major)
      :dtype dtype))))

(defun $np_ones (shape &optional dtype-arg)
  "Create a ones-filled ndarray: np_ones([m,n]) or np_ones(n)
   Optional dtype: np_ones([m,n], complex)"
  (let* ((dtype (numerics-parse-dtype dtype-arg))
         (et (numerics-element-type dtype)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:ones (maxima-shape-to-list shape)
                   :type et :layout :column-major)
      :dtype dtype))))

(defun $np_eye (n &optional m dtype-arg)
  "Create an identity matrix: np_eye(n) or np_eye(n,m)
   Optional dtype: np_eye(n, n, complex)"
  (let* ((dtype (numerics-parse-dtype dtype-arg))
         (et (numerics-element-type dtype))
         (shape (if m (list n m) (list n n))))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:eye shape :type et :layout :column-major)
      :dtype dtype))))

(defun $np_rand (shape)
  "Create a uniform random ndarray [0,1): np_rand([m,n])"
  (numerics-wrap
   (numerics:make-ndarray
    (magicl:rand (maxima-shape-to-list shape)
                 :type 'double-float :layout :column-major))))

(defun $np_randn (shape)
  "Create a standard normal random ndarray: np_randn([m,n])
   Uses Box-Muller transform via magicl:random-normal."
  (let* ((dims (maxima-shape-to-list shape))
         (tensor (magicl:random-normal dims :type 'double-float)))
    (numerics-wrap (numerics:make-ndarray tensor))))

(defun $np_arange (&rest args)
  "Create a 1D ndarray of evenly spaced values.
   np_arange(stop) — values 0, 1, ..., stop-1
   np_arange(start, stop) — values start, start+1, ..., stop-1
   np_arange(start, stop, step) — values start, start+step, ..."
  (let (start stop step)
    (ecase (length args)
      (1 (setf start 0.0d0
               stop  (coerce ($float (first args)) 'double-float)
               step  1.0d0))
      (2 (setf start (coerce ($float (first args)) 'double-float)
               stop  (coerce ($float (second args)) 'double-float)
               step  1.0d0))
      (3 (setf start (coerce ($float (first args)) 'double-float)
               stop  (coerce ($float (second args)) 'double-float)
               step  (coerce ($float (third args)) 'double-float))))
    (when (zerop step)
      (merror "np_arange: step must be non-zero"))
    (let* ((n (max 0 (ceiling (/ (- stop start) step))))
           (vals (loop for i below n
                       collect (+ start (* (coerce i 'double-float) step)))))
      (if (zerop n)
          (numerics-wrap
           (numerics:make-ndarray
            (magicl:empty (list 0) :type 'double-float :layout :column-major)))
          (numerics-wrap
           (numerics:make-ndarray
            (magicl:from-list vals (list n) :type 'double-float)))))))

(defun $np_linspace (start stop n)
  "Create n evenly spaced points [start, stop]: np_linspace(0, 1, 100)"
  (let* ((start-f (coerce ($float start) 'double-float))
         (stop-f  (coerce ($float stop)  'double-float))
         (n-int   (truncate n))
         (vals (if (= n-int 1)
                   (list start-f)
                   (loop for i below n-int
                         collect (+ start-f
                                    (* (/ (float i 1.0d0) (float (1- n-int) 1.0d0))
                                       (- stop-f start-f)))))))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:from-list vals (list n-int) :type 'double-float)))))

(defun $np_full (shape val &optional dtype-arg)
  "Create a constant-filled ndarray: np_full([3,3], 7.0)
   Optional dtype: np_full([2,2], 1+2*%i, complex)"
  (let* ((dtype (numerics-parse-dtype dtype-arg))
         (et (numerics-element-type dtype))
         (dims (maxima-shape-to-list shape))
         (v (maxima-to-lisp-number val dtype))
         (total (reduce #'* dims))
         (vals (make-list total :initial-element v)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:from-list vals dims :type et :layout :column-major)
      :dtype dtype))))

(defun $np_empty (shape &optional dtype-arg)
  "Create an uninitialized ndarray (fast): np_empty([m,n])
   Optional dtype: np_empty([m,n], complex)"
  (let* ((dtype (numerics-parse-dtype dtype-arg))
         (et (numerics-element-type dtype)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:empty (maxima-shape-to-list shape)
                    :type et :layout :column-major)
      :dtype dtype))))

(defun $np_diag (lst &optional dtype-arg)
  "Create a diagonal matrix from a Maxima list: np_diag([1,2,3])
   Optional dtype: np_diag([1+%i, 2], complex)"
  (unless ($listp lst)
    (merror "np_diag: expected a list, got: ~M" lst))
  (let* ((dtype (numerics-parse-dtype dtype-arg))
         (et (numerics-element-type dtype))
         (elements (cdr lst))
         (vals (mapcar (lambda (el) (maxima-to-lisp-number el dtype)) elements)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:from-diag vals :type et)
      :dtype dtype))))

(defun $np_logspace (start stop n)
  "Create n logarithmically spaced points from 10^start to 10^stop:
   np_logspace(0, 3, 50) => 50 points from 1 to 1000"
  (let* ((start-f (coerce ($float start) 'double-float))
         (stop-f  (coerce ($float stop)  'double-float))
         (n-int   (truncate n))
         (vals (if (= n-int 1)
                   (list (expt 10.0d0 start-f))
                   (loop for i below n-int
                         for t-val = (+ start-f
                                        (* (/ (float i 1.0d0)
                                              (float (1- n-int) 1.0d0))
                                           (- stop-f start-f)))
                         collect (expt 10.0d0 t-val)))))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:from-list vals (list n-int) :type 'double-float)))))

(defun $np_copy (a)
  "Deep copy an ndarray: np_copy(A)"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle)))
    (numerics-wrap
     (numerics:make-ndarray (magicl:deep-copy-tensor tensor) :dtype dtype))))

;;; --- Random utilities ---

(defun $np_seed (n)
  "Set the random seed for reproducibility: np_seed(42).
   Affects np_rand, np_randn, np_randint, np_choice, np_shuffle."
  #+sbcl (setf *random-state* (sb-ext:seed-random-state (truncate n)))
  #-sbcl (merror "np_seed: only supported on SBCL")
  '$done)

(defun $np_randint (lo hi shape)
  "Create an ndarray of random integers in [lo, hi): np_randint(0, 10, [3,3]).
   Values are stored as double-float."
  (let* ((lo-int (truncate lo))
         (hi-int (truncate hi))
         (range (- hi-int lo-int))
         (dims (maxima-shape-to-list shape))
         (total (reduce #'* dims))
         (vals (loop repeat total
                     collect (coerce (+ lo-int (random range)) 'double-float))))
    (when (<= range 0)
      (merror "np_randint: hi (~D) must be greater than lo (~D)" hi-int lo-int))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:from-list vals dims :type 'double-float :layout :column-major)))))

(defun $np_choice (a n &optional replace-arg)
  "Sample n elements from a 1D ndarray.
   np_choice(a, 5)        — 5 samples with replacement (default)
   np_choice(a, 5, true)  — 5 samples with replacement
   np_choice(a, 5, false) — 5 samples without replacement"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (flat (numerics-flat-array tensor))
         (m (length flat))
         (k (truncate n))
         (with-replacement (or (null replace-arg)
                               (eq replace-arg t)
                               (and (numberp replace-arg) (not (zerop replace-arg))))))
    (when (and (not with-replacement) (> k m))
      (merror "np_choice: cannot sample ~D from array of size ~D without replacement" k m))
    (let ((vals (if with-replacement
                    (loop repeat k collect (aref flat (random m)))
                    ;; Fisher-Yates partial shuffle on index copy
                    (let ((idx (make-array m :element-type 'fixnum)))
                      (dotimes (i m) (setf (aref idx i) i))
                      (loop for i from 0 below k
                            for j = (+ i (random (- m i)))
                            do (rotatef (aref idx i) (aref idx j))
                            collect (aref flat (aref idx i)))))))
      (numerics-wrap
       (numerics:make-ndarray
        (magicl:from-list vals (list k) :type 'double-float :layout :column-major))))))

(defun $np_shuffle (a)
  "In-place Fisher-Yates shuffle of a 1D ndarray: np_shuffle(A).
   Modifies A in place and returns it."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (storage (numerics-tensor-storage tensor))
         (n (length storage)))
    (loop for i from (1- n) downto 1
          for j = (random (1+ i))
          do (rotatef (aref storage i) (aref storage j)))
    a))
