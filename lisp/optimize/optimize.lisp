;;; optimize.lisp — Numerical optimization (L-BFGS wrapper)
;;;
;;; Supports two calling conventions:
;;;   np_minimize(f_func, grad_func, x0 [, tol, max_iter])  -- function mode
;;;   np_minimize(expr, vars, x0 [, tol, max_iter])          -- expression mode

(in-package #:maxima)

;; lbfgs is loaded by numerics-loader.lisp before this ASDF system compiles.

(defun np-minimize-call (f arg)
  "Call a Maxima function F with a single argument ARG.
   F can be a symbol (function name) or a lambda expression.
   Uses mapply for correct dispatch of both cases."
  (if (symbolp f)
      (mfuncall f arg)
      (mapply f (list arg) f)))

(defun np-minimize-core (eval-f eval-g x0-storage shape eps max-iter)
  "L-BFGS core optimization loop.
   EVAL-F: (function ndarray-storage -> double-float) evaluates objective.
   EVAL-G: (function ndarray-storage -> double-float-array) evaluates gradient.
   X0-STORAGE: initial point as a double-float simple-array.
   SHAPE: original ndarray shape (preserved in output).
   Returns [x_opt_wrapped, f_opt, converged]."
  (let* ((n (reduce #'* shape))
         (m (min 25 n))
         (nwork (+ (* n (+ (* 2 m) 1)) (* 2 m)))
         (x (make-array n :element-type 'double-float :initial-element 0.0d0))
         (g (make-array n :element-type 'double-float :initial-element 0.0d0))
         (w (make-array nwork :element-type 'double-float :initial-element 0.0d0))
         (diag (make-array n :element-type 'double-float :initial-element 0.0d0))
         (scache (make-array n :element-type 'double-float :initial-element 0.0d0))
         (iprint (make-array 2 :element-type 'f2cl-lib:integer4
                               :initial-contents '(0 0)))
         (diagco f2cl-lib:%false%)
         (xtol +flonum-epsilon+)
         (iflag 0)
         (f-val 0.0d0)
         (eps-f (coerce ($float eps) 'double-float))
         (max-iter-i (truncate max-iter)))
    ;; Copy initial point
    (replace x x0-storage)
    ;; Initialize lbfgs common block
    (common-lisp-user::/blockdata-lb2/)
    ;; Optimization loop
    (dotimes (iter max-iter-i)
      ;; Evaluate objective
      (setf f-val (funcall eval-f x))
      ;; Evaluate gradient
      (let ((g-arr (funcall eval-g x)))
        (replace g g-arr))
      ;; Call lbfgs core
      (multiple-value-bind (var-0 var-1 var-2 var-3 var-4 var-5 var-6
                            var-7 var-8 var-9 var-10 var-11 var-12)
          (common-lisp-user::lbfgs n m x f-val g diagco diag
                                   iprint eps-f xtol w iflag scache)
        (declare (ignore var-0 var-1 var-2 var-3 var-4 var-5 var-6
                         var-7 var-8 var-9 var-10 var-12))
        (setf iflag var-11)
        (when (eql iflag 0) (return))
        (when (< iflag 0)
          (merror "np_minimize: L-BFGS failed (iflag=~A). ~
                   Try different initial point or check gradient." iflag))))
    ;; Build result
    (let* ((best (if (eql iflag 0) scache x))
           (result-tensor (magicl:empty shape :type 'double-float
                                              :layout :column-major))
           (result-storage (numerics-tensor-storage result-tensor)))
      (replace result-storage best)
      (let* ((result-handle (numerics:make-ndarray result-tensor))
             (result-wrapped (numerics-wrap result-handle))
             (converged (if (eql iflag 0) t nil)))
        `((mlist) ,result-wrapped ,f-val ,converged)))))

(defun np-minimize-function-mode (f-func g-func x0 eps max-iter)
  "Function mode: f and grad are Maxima callables taking ndarrays."
  (let* ((handle (numerics-unwrap x0))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (x0-storage (numerics-tensor-storage tensor)))
    (numerics-require-real x0 "np_minimize")
    ;; Pre-allocate callback ndarray (reused each iteration)
    (let* ((cb-tensor (magicl:empty shape :type 'double-float
                                          :layout :column-major))
           (cb-handle (numerics:make-ndarray cb-tensor))
           (cb-wrapped (numerics-wrap cb-handle))
           (cb-storage (numerics-tensor-storage cb-tensor)))
      (np-minimize-core
       ;; eval-f: copy x into callback ndarray, call f
       (lambda (x-arr)
         (replace cb-storage x-arr)
         (coerce ($float (np-minimize-call f-func cb-wrapped)) 'double-float))
       ;; eval-g: copy x into callback ndarray, call grad, extract storage
       (lambda (x-arr)
         (replace cb-storage x-arr)
         (let* ((g-result (np-minimize-call g-func cb-wrapped))
                (g-handle (numerics-unwrap g-result))
                (g-tensor (numerics:ndarray-tensor g-handle)))
           (numerics-tensor-storage g-tensor)))
       x0-storage shape eps max-iter))))

(defun np-minimize-expression-mode (expr vars x0 eps max-iter)
  "Expression mode: expr is a symbolic expression, vars is [x1, x2, ...]."
  (unless (and (listp vars) (eq (caar vars) 'mlist))
    (merror "np_minimize: second argument must be a variable list [x1, x2, ...], got: ~M" vars))
  (let* ((handle (numerics-unwrap x0))
         (tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (x0-storage (numerics-tensor-storage tensor))
         (var-list (cdr vars))
         (n (length var-list)))
    (numerics-require-real x0 "np_minimize")
    (unless (= n (magicl:size tensor))
      (merror "np_minimize: number of variables (~A) must match x0 size (~A)"
              n (magicl:size tensor)))
    ;; Compute gradient symbolically and compile both
    (let* ((grad-exprs (cons '(mlist)
                             (loop for v in var-list
                                   collect (meval `(($diff) ,expr ,v)))))
           (f-compiled (compile nil (coerce-float-fun expr vars)))
           (g-compiled (compile nil (coerce-float-fun grad-exprs vars))))
      (np-minimize-core
       ;; eval-f: extract values, call compiled objective
       (lambda (x-arr)
         (let ((x-list (coerce x-arr 'list)))
           (cl:float (apply f-compiled x-list) 1d0)))
       ;; eval-g: extract values, call compiled gradient, return storage
       (lambda (x-arr)
         (let* ((x-list (coerce x-arr 'list))
                (g-vals (apply g-compiled x-list))
                (result (make-array n :element-type 'double-float)))
           (loop for v in (cdr g-vals) for i from 0
                 do (setf (aref result i) (cl:float v 1d0)))
           result))
       x0-storage shape eps max-iter))))

(defun $np_minimize (&rest args)
  "Minimize a scalar function using L-BFGS.

   Function mode: np_minimize(f, grad, x0 [, tol, max_iter])
     f(x) and grad(x) take ndarrays, return scalar and ndarray respectively.

   Expression mode: np_minimize(expr, [x1, x2, ...], x0 [, tol, max_iter])
     expr is a symbolic expression; gradient is computed automatically via diff().

   Returns [x_opt, f_opt, converged]."
  (unless (>= (length args) 3)
    (merror "np_minimize: expected at least 3 arguments"))
  (let ((first-arg (first args))
        (eps (or (fourth args) 1.0d-8))
        (max-iter (or (fifth args) 200)))
    (if (numerics-callable-p first-arg)
        ;; Function mode: np_minimize(f, grad, x0 [, tol, max_iter])
        (np-minimize-function-mode first-arg (second args) (third args)
                                   eps max-iter)
        ;; Expression mode: np_minimize(expr, vars, x0 [, tol, max_iter])
        (np-minimize-expression-mode first-arg (second args) (third args)
                                     eps max-iter))))
