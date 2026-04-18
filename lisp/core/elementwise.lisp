;;; elementwise.lisp — Element-wise operations

(in-package #:maxima)

;;; Helpers

(defun numerics-binary-op (a b op)
  "Apply a magicl element-wise binary op.
   Supports ndarray+ndarray, ndarray+scalar, and scalar+ndarray."
  (cond
    ;; Both ndarrays
    ((and ($ndarray_p a) ($ndarray_p b))
     (let ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
           (tb (numerics:ndarray-tensor (numerics-unwrap b))))
       (numerics-wrap (numerics:make-ndarray (funcall op ta tb)))))
    ;; ndarray + scalar
    (($ndarray_p a)
     (let ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
           (s  (coerce ($float b) 'double-float)))
       (numerics-wrap
        (numerics:make-ndarray
         (magicl:map! (lambda (x) (funcall op x s))
                      (magicl:deep-copy-tensor ta))))))
    ;; scalar + ndarray
    (($ndarray_p b)
     (let ((tb (numerics:ndarray-tensor (numerics-unwrap b)))
           (s  (coerce ($float a) 'double-float)))
       (numerics-wrap
        (numerics:make-ndarray
         (magicl:map! (lambda (x) (funcall op s x))
                      (magicl:deep-copy-tensor tb))))))
    (t (merror "Expected at least one ndarray argument"))))

(defun numerics-binary-op-magicl (a b magicl-op scalar-op)
  "Binary op using magicl operations for ndarray+ndarray and scalar fallback."
  (cond
    ((and ($ndarray_p a) ($ndarray_p b))
     (let ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
           (tb (numerics:ndarray-tensor (numerics-unwrap b))))
       (numerics-wrap (numerics:make-ndarray (funcall magicl-op ta tb)))))
    (($ndarray_p a)
     (let ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
           (s  (coerce ($float b) 'double-float)))
       (numerics-wrap
        (numerics:make-ndarray
         (magicl:map! (lambda (x) (funcall scalar-op x s))
                      (magicl:deep-copy-tensor ta))))))
    (($ndarray_p b)
     (let ((tb (numerics:ndarray-tensor (numerics-unwrap b)))
           (s  (coerce ($float a) 'double-float)))
       (numerics-wrap
        (numerics:make-ndarray
         (magicl:map! (lambda (x) (funcall scalar-op s x))
                      (magicl:deep-copy-tensor tb))))))
    (t (merror "Expected at least one ndarray argument"))))

(defun numerics-unary-op (a fn)
  "Apply a unary function element-wise, returning a new ndarray."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a)))
         (result (magicl:deep-copy-tensor tensor)))
    (magicl:map! fn result)
    (numerics-wrap (numerics:make-ndarray result))))

;;; Binary operations

(defun $np_add (a b)
  "Element-wise addition: np_add(A, B) or np_add(A, 3.0)"
  (numerics-binary-op-magicl a b #'magicl:.+ #'+))

(defun $np_sub (a b)
  "Element-wise subtraction: np_sub(A, B)"
  (numerics-binary-op-magicl a b #'magicl:.- #'-))

(defun $np_mul (a b)
  "Element-wise (Hadamard) product. NOT matrix multiply."
  (numerics-binary-op-magicl a b #'magicl:.* #'*))

(defun $np_div (a b)
  "Element-wise division: np_div(A, B)"
  (numerics-binary-op-magicl a b #'magicl:./ #'/))

(defun $np_pow (a p)
  "Element-wise power: np_pow(A, p)"
  (if ($ndarray_p p)
      (numerics-binary-op a p #'expt)
      (let ((pf (coerce ($float p) 'double-float)))
        (numerics-unary-op a (lambda (x) (expt x pf))))))

;;; Unary operations

(defun $np_sqrt (a)
  "Element-wise square root: np_sqrt(A)"
  (numerics-unary-op a #'cl:sqrt))

(defun $np_exp (a)
  "Element-wise exponential: np_exp(A)"
  (numerics-unary-op a #'cl:exp))

(defun $np_log (a)
  "Element-wise natural log: np_log(A)"
  (numerics-unary-op a #'cl:log))

(defun $np_sin (a)
  "Element-wise sine: np_sin(A)"
  (numerics-unary-op a #'cl:sin))

(defun $np_cos (a)
  "Element-wise cosine: np_cos(A)"
  (numerics-unary-op a #'cl:cos))

(defun $np_tan (a)
  "Element-wise tangent: np_tan(A)"
  (numerics-unary-op a #'cl:tan))

(defun $np_abs (a)
  "Element-wise absolute value: np_abs(A)"
  (numerics-unary-op a #'cl:abs))

(defun $np_neg (a)
  "Element-wise negation: np_neg(A)"
  (numerics-unary-op a #'cl:-))

;;; Mapping user functions

(defun $np_map (f a)
  "Apply a function element-wise: np_map(f, A).
   If f has been translate()'d, uses the fast compiled path.
   Otherwise falls back to the Maxima evaluator (slow)."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (n (magicl:size tensor))
         (fname (if (symbolp f) f
                    (merror "np_map: expected a function name, got: ~M" f))))
    ;; Check if the function has been translated (has a CL function)
    (if (and (get fname 'translated)
             (fboundp fname))
        ;; Fast path: call the CL function directly via magicl:map!
        (let* ((cl-fn (symbol-function fname))
               (result (magicl:deep-copy-tensor tensor)))
          (magicl:map! (lambda (x) (coerce ($float (funcall cl-fn x)) 'double-float))
                       result)
          (numerics-wrap (numerics:make-ndarray result)))
        ;; Slow path: call through Maxima evaluator
        (let ((result (magicl:empty shape :type 'double-float
                                          :layout :column-major)))
          (if (= (length shape) 1)
              (dotimes (i n)
                (let* ((x (magicl:tref tensor i))
                       (y (mfuncall fname x)))
                  (setf (magicl:tref result i)
                        (coerce ($float y) 'double-float))))
              (let ((nrow (first shape))
                    (ncol (second shape)))
                (dotimes (i nrow)
                  (dotimes (j ncol)
                    (let* ((x (magicl:tref tensor i j))
                           (y (mfuncall fname x)))
                      (setf (magicl:tref result i j)
                            (coerce ($float y) 'double-float)))))))
          (numerics-wrap (numerics:make-ndarray result))))))

(defun $np_map2 (f a b)
  "Apply a binary function element-wise: np_map2(f, A, B).
   Both arrays must have the same shape.
   If f has been translate()'d, uses the fast compiled path."
  (let* ((ha (numerics-unwrap a))
         (hb (numerics-unwrap b))
         (ta (numerics:ndarray-tensor ha))
         (tb (numerics:ndarray-tensor hb))
         (shape (magicl:shape ta))
         (fname (if (symbolp f) f
                    (merror "np_map2: expected a function name, got: ~M" f))))
    (unless (equal shape (magicl:shape tb))
      (merror "np_map2: shape mismatch: ~A vs ~A" shape (magicl:shape tb)))
    (let ((result (magicl:empty shape :type 'double-float
                                      :layout :column-major)))
      (if (and (get fname 'translated) (fboundp fname))
          ;; Fast path
          (let ((cl-fn (symbol-function fname)))
            (if (= (length shape) 1)
                (dotimes (i (magicl:size ta))
                  (setf (magicl:tref result i)
                        (coerce ($float (funcall cl-fn
                                                 (magicl:tref ta i)
                                                 (magicl:tref tb i)))
                                'double-float)))
                (let ((nrow (first shape))
                      (ncol (second shape)))
                  (dotimes (i nrow)
                    (dotimes (j ncol)
                      (setf (magicl:tref result i j)
                            (coerce ($float (funcall cl-fn
                                                     (magicl:tref ta i j)
                                                     (magicl:tref tb i j)))
                                    'double-float)))))))
          ;; Slow path
          (if (= (length shape) 1)
              (dotimes (i (magicl:size ta))
                (setf (magicl:tref result i)
                      (coerce ($float (mfuncall fname
                                                (magicl:tref ta i)
                                                (magicl:tref tb i)))
                              'double-float)))
              (let ((nrow (first shape))
                    (ncol (second shape)))
                (dotimes (i nrow)
                  (dotimes (j ncol)
                    (setf (magicl:tref result i j)
                          (coerce ($float (mfuncall fname
                                                    (magicl:tref ta i j)
                                                    (magicl:tref tb i j)))
                                  'double-float)))))))
      (numerics-wrap (numerics:make-ndarray result)))))

;;; Scalar multiplication

(defun $np_scale (alpha a)
  "Scalar multiplication: np_scale(alpha, A)"
  (let* ((s (coerce ($float alpha) 'double-float))
         (tensor (numerics:ndarray-tensor (numerics-unwrap a))))
    (numerics-wrap
     (numerics:make-ndarray (magicl:scale tensor s)))))

;;; Conditional selection

(defun numerics-where-indices (tensor shape)
  "Return index arrays for nonzero elements. For 1D returns a single index list;
   for 2D returns (row-indices col-indices) as two lists. Iterates row-major."
  (if (= (length shape) 1)
      ;; 1D: collect flat indices
      (let ((idxs '()))
        (dotimes (i (first shape))
          (when (/= 0.0d0 (magicl:tref tensor i))
            (push (coerce i 'double-float) idxs)))
        (list (nreverse idxs)))
      ;; 2D: collect (row, col) pairs in row-major order
      (let ((rows '()) (cols '())
            (nrow (first shape))
            (ncol (second shape)))
        (dotimes (i nrow)
          (dotimes (j ncol)
            (when (/= 0.0d0 (magicl:tref tensor i j))
              (push (coerce i 'double-float) rows)
              (push (coerce j 'double-float) cols))))
        (list (nreverse rows) (nreverse cols)))))

(defun numerics-list-to-1d-ndarray (vals)
  "Convert a list of double-floats to a 1D ndarray. Returns empty Maxima list for no values."
  (let ((n (length vals)))
    (if (zerop n)
        '((mlist simp))
        (numerics-wrap
         (numerics:make-ndarray
          (magicl:from-list vals (list n) :type 'double-float))))))

(defun $np_where (&rest args)
  "Conditional selection.
   np_where(condition) => list of index arrays where condition is nonzero.
   np_where(condition, x, y) => ndarray selecting from x where true, y where false."
  (ecase (length args)
    (1 ;; np_where(condition) => index arrays
     (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap (first args))))
            (shape (magicl:shape tensor))
            (idx-lists (numerics-where-indices tensor shape)))
       ;; Return Maxima list of 1D ndarrays
       `((mlist simp) ,@(mapcar #'numerics-list-to-1d-ndarray idx-lists))))
    (3 ;; np_where(condition, x, y) => element-wise select
     (let* ((tc (numerics:ndarray-tensor (numerics-unwrap (first args))))
            (tx (numerics:ndarray-tensor (numerics-unwrap (second args))))
            (ty (numerics:ndarray-tensor (numerics-unwrap (third args))))
            (shape (magicl:shape tc))
            (result (magicl:empty shape :type 'double-float
                                        :layout :column-major)))
       (if (= (length shape) 1)
           (dotimes (i (first shape))
             (setf (magicl:tref result i)
                   (if (/= 0.0d0 (magicl:tref tc i))
                       (magicl:tref tx i)
                       (magicl:tref ty i))))
           (let ((nrow (first shape))
                 (ncol (second shape)))
             (dotimes (i nrow)
               (dotimes (j ncol)
                 (setf (magicl:tref result i j)
                       (if (/= 0.0d0 (magicl:tref tc i j))
                           (magicl:tref tx i j)
                           (magicl:tref ty i j)))))))
       (numerics-wrap (numerics:make-ndarray result))))))
