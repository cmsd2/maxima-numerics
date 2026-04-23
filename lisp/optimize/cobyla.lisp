;;; cobyla.lisp — Constrained optimization via COBYLA
;;;
;;; np_minimize_cobyla(f, vars, x0, constraints [, rhobeg, rhoend, maxfun])
;;; Wraps share/cobyla/ COBYLA for derivative-free constrained optimization.

(in-package #:maxima)

;; cobyla is loaded by numerics-optimize-loader.lisp before this ASDF system.

(defun np-cobyla-normalize-constraints (constraints)
  "Normalize constraints to the form h(x) >= 0.
   >= becomes lhs-rhs, <= becomes rhs-lhs, = becomes both directions."
  (let (result)
    (dolist (c (cdr constraints))
      (let ((op ($op c)))
        (cond ((string-equal op ">=")
               (push (sub ($lhs c) ($rhs c)) result))
              ((string-equal op "<=")
               (push (sub ($rhs c) ($lhs c)) result))
              ((string-equal op "=")
               (push (sub ($lhs c) ($rhs c)) result)
               (push (sub ($rhs c) ($lhs c)) result))
              (t
               (merror "np_minimize_cobyla: constraint must use =, <=, or >=, got: ~M" op)))))
    (list* '(mlist) (nreverse result))))

(defun np-cobyla-extract-x0 (x0 n name)
  "Extract initial values as a double-float array from ndarray or Maxima list."
  (cond
    (($ndarray_p x0)
     (let* ((handle (numerics-unwrap x0))
            (tensor (numerics:ndarray-tensor handle))
            (storage (numerics-tensor-storage tensor)))
       (unless (= (length storage) n)
         (merror "~A: x0 length (~A) does not match number of variables (~A)"
                 name (length storage) n))
       (make-array n :element-type 'double-float
                     :initial-contents (coerce storage 'list))))
    ((and (listp x0) (eq (caar x0) 'mlist))
     (unless (= (length (cdr x0)) n)
       (merror "~A: x0 length (~A) does not match number of variables (~A)"
               name (length (cdr x0)) n))
     (make-array n :element-type 'double-float
                   :initial-contents
                   (mapcar (lambda (v) (coerce ($float v) 'double-float))
                           (cdr x0))))
    (t (merror "~A: x0 must be an ndarray or list, got: ~M" name x0))))

(defun $np_minimize_cobyla (f vars x0 constraints
                            &optional (rhobeg 1.0d0) (rhoend 1.0d-6) (maxfun 1000))
  "Minimize f subject to constraints using COBYLA (derivative-free).
   np_minimize_cobyla(f, vars, x0, constraints [, rhobeg, rhoend, maxfun])
     f           - objective expression in vars
     vars        - [x1, x2, ...]
     x0          - initial guess (ndarray or list)
     constraints - [expr >= 0, expr <= val, expr = val, ...]
     rhobeg      - initial trust region radius (default 1.0)
     rhoend      - final accuracy (default 1e-6)
     maxfun      - max function evaluations (default 1000)
   Returns: [x_opt, f_opt, n_evals, info] where x_opt is a 1D ndarray.
   info: 0 = success, 1 = maxfun reached, 2 = rounding errors, -1 = constraints violated."
  ;; Validate inputs
  (unless (and (listp vars) (eq (caar vars) 'mlist))
    (merror "np_minimize_cobyla: vars must be a list of variables, got: ~M" vars))
  (unless (and (listp constraints) (eq (caar constraints) 'mlist))
    (merror "np_minimize_cobyla: constraints must be a list, got: ~M" constraints))

  (let* ((var-list (cdr vars))
         (n (length var-list))
         (normalized (np-cobyla-normalize-constraints constraints))
         (m (length (cdr normalized)))
         (x (np-cobyla-extract-x0 x0 n "np_minimize_cobyla"))
         (rhobeg-f (coerce ($float rhobeg) 'double-float))
         (rhoend-f (coerce ($float rhoend) 'double-float))
         (maxfun-i (truncate maxfun))
         ;; Work arrays (sizing from cobyla.f)
         (w (make-array (+ (* n (+ (* 3 n) (* 2 m) 11)) (* 4 m) 6 6)
                        :element-type 'double-float))
         (iact (make-array (1+ m) :element-type 'f2cl-lib::integer4))
         ;; Compile objective and constraints
         (fv (coerce-float-fun f vars))
         (cv (coerce-float-fun normalized vars))
         ;; Set up CALCFC callback
         (*calcfc*
          (lambda (nn mm xval cval)
            (declare (fixnum nn mm)
                     (type (cl:array cl:double-float (*)) xval cval))
            (let* ((x-list (coerce xval 'list))
                   (fval (apply fv x-list))
                   (cvals (apply cv x-list)))
              (unless (floatp fval)
                (merror "np_minimize_cobyla: objective did not evaluate to a number at ~M"
                        (list* '(mlist) x-list)))
              (replace cval cvals :start2 1)
              (values nn mm nil fval nil)))))
    ;; Call COBYLA
    (multiple-value-bind (v0 v1 v2 v3 v4 v5 neval v6 v7 ierr)
        (cobyla:cobyla n m x rhobeg-f rhoend-f 0 maxfun-i w iact 0)
      (declare (ignore v0 v1 v2 v3 v4 v5 v6 v7))
      ;; Build result: x_opt as 1D ndarray
      (let* ((result-tensor (magicl:empty (list n) :type 'double-float
                                                   :layout :column-major))
             (result-storage (numerics-tensor-storage result-tensor)))
        (replace result-storage x)
        (let* ((x-list (coerce x 'list))
               (f-opt (apply fv x-list))
               (max-cv (aref w (1+ m)))
               (info (if (> max-cv rhoend-f) -1 ierr)))
          `((mlist)
            ,(numerics-wrap (numerics:make-ndarray result-tensor))
            ,f-opt
            ,neval
            ,info))))))
