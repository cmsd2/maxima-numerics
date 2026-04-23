;;; gradient.lisp — Symbolic-to-numeric gradient bridge
;;;
;;; np_compile_gradient(expr, vars) takes a symbolic loss expression and
;;; a list of variables, computes the gradient via diff(), and returns
;;; [f_func, grad_func] — two Maxima-callable function symbols compatible
;;; with np_minimize.

(in-package #:maxima)

(defvar *np-gradient-counter* 0)

(defun np-gradient-extract-values (x-ndarray)
  "Extract a list of double-float values from an ndarray argument."
  (let* ((handle (numerics-unwrap x-ndarray))
         (tensor (numerics:ndarray-tensor handle))
         (storage (numerics-tensor-storage tensor)))
    (coerce storage 'list)))

(defun $np_compile_gradient (expr vars)
  "Compile a symbolic loss expression and its gradient into numeric functions.
   np_compile_gradient(expr, [x1, x2, ...]) => [f_func, grad_func]
   where f_func(x) takes an ndarray and returns a scalar,
   and grad_func(x) takes an ndarray and returns an ndarray.
   Both are compatible with np_minimize."
  (unless (and (listp vars) (eq (caar vars) 'mlist))
    (merror "np_compile_gradient: vars must be a list of variables, got: ~M" vars))
  (let* ((var-list (cdr vars))
         (n (length var-list)))
    (when (zerop n)
      (merror "np_compile_gradient: vars list must not be empty"))
    (let* (;; Compute symbolic gradient: [diff(expr,v1), diff(expr,v2), ...]
           (grad-exprs (cons '(mlist)
                             (loop for v in var-list
                                   collect (meval `(($diff) ,expr ,v)))))
           ;; Compile to efficient Lisp closures
           (f-compiled (compile nil (coerce-float-fun expr vars)))
           (g-compiled (compile nil (coerce-float-fun grad-exprs vars)))
           (id (incf *np-gradient-counter*))
           ;; Create unique Maxima-callable function symbols
           (f-name (intern (format nil "$NP_COMPILED_F~A" id) :maxima))
           (g-name (intern (format nil "$NP_COMPILED_G~A" id) :maxima)))
      ;; Define objective: ndarray -> scalar
      (setf (symbol-function f-name)
            (lambda (x-ndarray)
              (let ((x-list (np-gradient-extract-values x-ndarray)))
                (apply f-compiled x-list))))
      ;; Define gradient: ndarray -> ndarray
      (setf (symbol-function g-name)
            (lambda (x-ndarray)
              (let* ((x-list (np-gradient-extract-values x-ndarray))
                     (g-vals (apply g-compiled x-list))
                     ;; g-vals is a Maxima mlist: ((mlist) v1 v2 ...)
                     (result (magicl:empty (list n) :type 'double-float
                                                    :layout :column-major))
                     (result-storage (numerics-tensor-storage result)))
                (loop for v in (cdr g-vals) for i from 0
                      do (setf (aref result-storage i) (cl:float v 1d0)))
                (numerics-wrap (numerics:make-ndarray result)))))
      ;; Return function symbols — np_minimize uses mfuncall for symbols
      `((mlist) ,f-name ,g-name))))
