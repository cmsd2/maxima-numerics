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
   np_where(condition, x, y) => ndarray selecting from x where true, y where false.
   In the 3-arg form, x and y can be ndarrays or scalars."
  (ecase (length args)
    (1 ;; np_where(condition) => index arrays
     (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap (first args))))
            (shape (magicl:shape tensor))
            (idx-lists (numerics-where-indices tensor shape)))
       ;; Return Maxima list of 1D ndarrays
       `((mlist simp) ,@(mapcar #'numerics-list-to-1d-ndarray idx-lists))))
    (3 ;; np_where(condition, x, y) => element-wise select
     (let* ((tc (numerics:ndarray-tensor (numerics-unwrap (first args))))
            (shape (magicl:shape tc))
            (x-nd-p ($ndarray_p (second args)))
            (y-nd-p ($ndarray_p (third args)))
            (tx (when x-nd-p (numerics:ndarray-tensor (numerics-unwrap (second args)))))
            (ty (when y-nd-p (numerics:ndarray-tensor (numerics-unwrap (third args)))))
            (sx (unless x-nd-p (coerce ($float (second args)) 'double-float)))
            (sy (unless y-nd-p (coerce ($float (third args)) 'double-float)))
            (result (magicl:empty shape :type 'double-float
                                        :layout :column-major)))
       (flet ((get-x (&rest idx) (if x-nd-p (apply #'magicl:tref tx idx) sx))
              (get-y (&rest idx) (if y-nd-p (apply #'magicl:tref ty idx) sy)))
         (if (= (length shape) 1)
             (dotimes (i (first shape))
               (setf (magicl:tref result i)
                     (if (/= 0.0d0 (magicl:tref tc i))
                         (get-x i)
                         (get-y i))))
             (let ((nrow (first shape))
                   (ncol (second shape)))
               (dotimes (i nrow)
                 (dotimes (j ncol)
                   (setf (magicl:tref result i j)
                         (if (/= 0.0d0 (magicl:tref tc i j))
                             (get-x i j)
                             (get-y i j))))))))
       (numerics-wrap (numerics:make-ndarray result))))))

;;; Comparison operations — return 1.0/0.0 ndarrays

(defun numerics-comparison-op (a b cmp-fn)
  "Element-wise comparison returning 1.0d0/0.0d0 ndarray.
   CMP-FN takes two double-floats and returns a generalized boolean.
   Supports ndarray+ndarray, ndarray+scalar, scalar+ndarray."
  (flet ((bool-val (x y) (if (funcall cmp-fn x y) 1.0d0 0.0d0)))
    (cond
      ((and ($ndarray_p a) ($ndarray_p b))
       (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
              (tb (numerics:ndarray-tensor (numerics-unwrap b)))
              (shape (magicl:shape ta))
              (result (magicl:empty shape :type 'double-float
                                          :layout :column-major)))
         (if (= (length shape) 1)
             (dotimes (i (first shape))
               (setf (magicl:tref result i)
                     (bool-val (magicl:tref ta i) (magicl:tref tb i))))
             (let ((nrow (first shape)) (ncol (second shape)))
               (dotimes (i nrow)
                 (dotimes (j ncol)
                   (setf (magicl:tref result i j)
                         (bool-val (magicl:tref ta i j) (magicl:tref tb i j)))))))
         (numerics-wrap (numerics:make-ndarray result))))
      (($ndarray_p a)
       (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
              (s (coerce ($float b) 'double-float))
              (result (magicl:deep-copy-tensor ta)))
         (magicl:map! (lambda (x) (bool-val x s)) result)
         (numerics-wrap (numerics:make-ndarray result))))
      (($ndarray_p b)
       (let* ((tb (numerics:ndarray-tensor (numerics-unwrap b)))
              (s (coerce ($float a) 'double-float))
              (result (magicl:deep-copy-tensor tb)))
         (magicl:map! (lambda (x) (bool-val s x)) result)
         (numerics-wrap (numerics:make-ndarray result))))
      (t (merror "Expected at least one ndarray argument")))))

(defun $np_greater (a b)
  "Element-wise greater-than: np_greater(A, B) => 1.0 where A > B, else 0.0"
  (numerics-comparison-op a b #'>))

(defun $np_greater_equal (a b)
  "Element-wise greater-or-equal: np_greater_equal(A, B)"
  (numerics-comparison-op a b #'>=))

(defun $np_less (a b)
  "Element-wise less-than: np_less(A, B)"
  (numerics-comparison-op a b #'<))

(defun $np_less_equal (a b)
  "Element-wise less-or-equal: np_less_equal(A, B)"
  (numerics-comparison-op a b #'<=))

(defun $np_equal (a b)
  "Element-wise equality: np_equal(A, B)"
  (numerics-comparison-op a b #'=))

(defun $np_not_equal (a b)
  "Element-wise not-equal: np_not_equal(A, B)"
  (numerics-comparison-op a b #'/=))

;;; Logical operations on mask ndarrays

(defun $np_logical_and (a b)
  "Element-wise logical AND: nonzero is true. Returns 1.0/0.0 ndarray."
  (numerics-comparison-op a b
    (lambda (x y) (and (/= 0.0d0 x) (/= 0.0d0 y)))))

(defun $np_logical_or (a b)
  "Element-wise logical OR: nonzero is true. Returns 1.0/0.0 ndarray."
  (numerics-comparison-op a b
    (lambda (x y) (or (/= 0.0d0 x) (/= 0.0d0 y)))))

(defun $np_logical_not (a)
  "Element-wise logical NOT: nonzero => 0.0, zero => 1.0."
  (numerics-unary-op a (lambda (x) (if (= 0.0d0 x) 1.0d0 0.0d0))))

;;; Predicate testing

(defun numerics-to-mask (val)
  "Convert a Maxima value to 1.0d0 (truthy) or 0.0d0 (falsy).
   Handles numbers, CL booleans, and Maxima relational expressions."
  (cond
    ((numberp val) (if (zerop val) 0.0d0 1.0d0))
    ((eq val t) 1.0d0)
    ((null val) 0.0d0)
    ;; Maxima relational expression — evaluate as predicate
    ((and (consp val) (consp (car val))
          (member (caar val)
                  '(mgreaterp mlessp mgeqp mleqp mnotequal)
                  :test #'eq))
     (if (mevalp val) 1.0d0 0.0d0))
    ;; Maxima $equal
    ((and (consp val) (consp (car val)) (eq (caar val) '$equal))
     (if (mevalp val) 1.0d0 0.0d0))
    (t 0.0d0)))

(defun $np_test (f a)
  "Apply a predicate element-wise, returning a 1.0/0.0 mask ndarray.
   np_test(f, A) where f is a function name or lambda([x], ...).
   Named functions use the fast compiled path if translated.
   Lambda expressions use the Maxima evaluator (slow)."
  (let* ((handle (numerics-unwrap a))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (result (magicl:empty shape :type 'double-float
                                     :layout :column-major)))
    (cond
      ;; Named function (symbol)
      ((symbolp f)
       (if (and (get f 'translated) (fboundp f))
           ;; Fast path: compiled CL function
           (let ((cl-fn (symbol-function f)))
             (if (= (length shape) 1)
                 (dotimes (i (first shape))
                   (setf (magicl:tref result i)
                         (numerics-to-mask (funcall cl-fn (magicl:tref tensor i)))))
                 (let ((nrow (first shape)) (ncol (second shape)))
                   (dotimes (i nrow)
                     (dotimes (j ncol)
                       (setf (magicl:tref result i j)
                             (numerics-to-mask
                              (funcall cl-fn (magicl:tref tensor i j)))))))))
           ;; Slow path: Maxima evaluator
           (if (= (length shape) 1)
               (dotimes (i (first shape))
                 (setf (magicl:tref result i)
                       (numerics-to-mask (mfuncall f (magicl:tref tensor i)))))
               (let ((nrow (first shape)) (ncol (second shape)))
                 (dotimes (i nrow)
                   (dotimes (j ncol)
                     (setf (magicl:tref result i j)
                           (numerics-to-mask
                            (mfuncall f (magicl:tref tensor i j))))))))))
      ;; Lambda expression (LAMBDA without $ prefix in Maxima internals)
      ((and (consp f) (consp (car f))
            (member (caar f) '($lambda lambda) :test #'eq))
       (if (= (length shape) 1)
           (dotimes (i (first shape))
             (setf (magicl:tref result i)
                   (numerics-to-mask
                    (mlambda f (list (magicl:tref tensor i)) nil nil nil))))
           (let ((nrow (first shape)) (ncol (second shape)))
             (dotimes (i nrow)
               (dotimes (j ncol)
                 (setf (magicl:tref result i j)
                       (numerics-to-mask
                        (mlambda f (list (magicl:tref tensor i j))
                                 nil nil nil))))))))
      (t (merror "np_test: expected a function name or lambda, got: ~M" f)))
    (numerics-wrap (numerics:make-ndarray result))))

;;; Boolean indexing

(defun $np_extract (mask a)
  "Extract elements from A where MASK is nonzero. Returns a 1D ndarray.
   Both arrays must have the same shape. Elements are taken in row-major order."
  (let* ((tm (numerics:ndarray-tensor (numerics-unwrap mask)))
         (ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (shape (magicl:shape tm))
         (vals '()))
    (if (= (length shape) 1)
        (dotimes (i (first shape))
          (when (/= 0.0d0 (magicl:tref tm i))
            (push (magicl:tref ta i) vals)))
        (let ((nrow (first shape)) (ncol (second shape)))
          (dotimes (i nrow)
            (dotimes (j ncol)
              (when (/= 0.0d0 (magicl:tref tm i j))
                (push (magicl:tref ta i j) vals))))))
    (let ((result-list (nreverse vals)))
      (if (null result-list)
          '((mlist simp))
          (numerics-wrap
           (numerics:make-ndarray
            (magicl:from-list result-list (list (length result-list))
                             :type 'double-float)))))))
