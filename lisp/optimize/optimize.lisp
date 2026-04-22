;;; optimize.lisp — Numerical optimization (L-BFGS wrapper)

(in-package #:maxima)

;; lbfgs is loaded by numerics-loader.lisp before this ASDF system compiles.

(defun np-minimize-call (f arg)
  "Call a Maxima function F with a single argument ARG.
   F can be a symbol (function name) or a lambda expression.
   Uses mapply for correct dispatch of both cases."
  (if (symbolp f)
      (mfuncall f arg)
      (mapply f (list arg) f)))

(defun $np_minimize (f-func g-func x0
                     &optional (eps 1.0d-8) (max-iter 200))
  "Minimize a scalar function using L-BFGS: np_minimize(f, grad, x0).
   f(x)    — function taking an ndarray, returning a scalar.
   grad(x) — function taking an ndarray, returning an ndarray (gradient).
   x0      — initial ndarray (real). Shape is preserved in the result.
   Optional: tolerance (default 1e-8), max_iter (default 200).
   Returns [x_opt, f_opt, converged]."
  (let* ((handle (numerics-unwrap x0)))
    (numerics-require-real x0 "np_minimize")
    (let* ((tensor (numerics:ndarray-tensor handle))
           (shape (magicl:shape tensor))
           (n (magicl:size tensor))  ; total elements (flatten for lbfgs)
           (m (min 25 n))            ; number of BFGS corrections
           (nwork (+ (* n (+ (* 2 m) 1)) (* 2 m)))
           ;; Work arrays for lbfgs core
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
      ;; Copy x0's storage into the work array x
      (let ((src (numerics-tensor-storage tensor)))
        (replace x src))
      ;; Pre-allocate a single ndarray for passing to callbacks.
      ;; Each iteration we copy lbfgs's x into this ndarray's storage.
      ;; Always use the original shape so user callbacks see the expected shape.
      (let* ((cb-tensor (magicl:empty shape :type 'double-float
                                            :layout :column-major))
             (cb-handle (numerics:make-ndarray cb-tensor))
             (cb-wrapped (numerics-wrap cb-handle))
             (cb-storage (numerics-tensor-storage cb-tensor)))
        ;; Initialize lbfgs common block
        (common-lisp-user::/blockdata-lb2/)
        ;; Optimization loop
        (dotimes (iter max-iter-i)
          ;; Copy current x into callback ndarray
          (replace cb-storage x)
          ;; Evaluate objective function
          (let ((f-result (np-minimize-call f-func cb-wrapped)))
            (setf f-val (coerce ($float f-result) 'double-float)))
          ;; Evaluate gradient function
          (let* ((g-result (np-minimize-call g-func cb-wrapped))
                 (g-handle (numerics-unwrap g-result))
                 (g-tensor (numerics:ndarray-tensor g-handle))
                 (g-src (numerics-tensor-storage g-tensor)))
            (replace g g-src))
          ;; Call lbfgs core
          (multiple-value-bind (var-0 var-1 var-2 var-3 var-4 var-5 var-6
                                var-7 var-8 var-9 var-10 var-11 var-12)
              (common-lisp-user::lbfgs n m x f-val g diagco diag
                                       iprint eps-f xtol w iflag scache)
            (declare (ignore var-0 var-1 var-2 var-3 var-4 var-5 var-6
                             var-7 var-8 var-9 var-10 var-12))
            (setf iflag var-11)
            (when (eql iflag 0) (return))    ; converged
            (when (< iflag 0)                ; error
              (merror "np_minimize: L-BFGS failed (iflag=~A). ~
                       Try different initial point or check gradient." iflag)))))
      ;; Build result: copy best x into a new tensor with x0's original shape
      (let* ((best (if (eql iflag 0) scache x))
             (result-tensor (magicl:empty shape :type 'double-float
                                                :layout :column-major))
             (result-storage (numerics-tensor-storage result-tensor)))
        (replace result-storage best)
        (let* ((result-handle (numerics:make-ndarray result-tensor))
               (result-wrapped (numerics-wrap result-handle))
               (converged (if (eql iflag 0) t nil)))
          `((mlist) ,result-wrapped ,f-val ,converged))))))
