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
