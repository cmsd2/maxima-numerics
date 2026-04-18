;;; constructors.lisp — ndarray constructor functions

(in-package #:maxima)

(defun $np_zeros (shape)
  "Create a zero-filled ndarray: np_zeros([m,n]) or np_zeros(n)"
  (numerics-wrap
   (numerics:make-ndarray
    (magicl:zeros (maxima-shape-to-list shape)
                  :type 'double-float :layout :column-major))))

(defun $np_ones (shape)
  "Create a ones-filled ndarray: np_ones([m,n]) or np_ones(n)"
  (numerics-wrap
   (numerics:make-ndarray
    (magicl:ones (maxima-shape-to-list shape)
                 :type 'double-float :layout :column-major))))

(defun $np_eye (n &optional m)
  "Create an identity matrix: np_eye(n) or np_eye(n,m)"
  (let ((shape (if m (list n m) (list n n))))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:eye shape :type 'double-float :layout :column-major)))))

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

(defun $np_full (shape val)
  "Create a constant-filled ndarray: np_full([3,3], 7.0)"
  (let* ((dims (maxima-shape-to-list shape))
         (v (coerce ($float val) 'double-float))
         (total (reduce #'* dims))
         (vals (make-list total :initial-element v)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:from-list vals dims :type 'double-float :layout :column-major)))))

(defun $np_empty (shape)
  "Create an uninitialized ndarray (fast): np_empty([m,n])"
  (numerics-wrap
   (numerics:make-ndarray
    (magicl:empty (maxima-shape-to-list shape)
                  :type 'double-float :layout :column-major))))

(defun $np_diag (lst)
  "Create a diagonal matrix from a Maxima list: np_diag([1,2,3])"
  (unless ($listp lst)
    (merror "np_diag: expected a list, got: ~M" lst))
  (let* ((elements (cdr lst))
         (n (length elements))
         (vals (mapcar (lambda (el) (coerce ($float el) 'double-float)) elements)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:from-diag vals :type 'double-float)))))

(defun $np_copy (a)
  "Deep copy an ndarray: np_copy(A)"
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (dtype (numerics:ndarray-dtype handle)))
    (numerics-wrap
     (numerics:make-ndarray (magicl:deep-copy-tensor tensor) :dtype dtype))))
