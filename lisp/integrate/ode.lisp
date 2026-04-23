;;; odepack.lisp — Numeric ODE integration via ODEPACK/DLSODE
;;;
;;; Supports two calling conventions:
;;;   np_odeint([f1,...], [t,y1,...], y0, tspan [, rtol, atol, method])  -- expression
;;;   np_odeint(f_func, y0, tspan [, rtol, atol, method])               -- function
;;;
;;; Wraps share/odepack/ DLSODE, collecting trajectory output as a 2D ndarray.

(in-package #:maxima)

(defun np-odeint-extract-values (arg name)
  "Extract a list of double-float values from an ndarray or Maxima list.
   NAME is used for error messages."
  (cond
    (($ndarray_p arg)
     (let* ((handle (numerics-unwrap arg))
            (tensor (numerics:ndarray-tensor handle)))
       (coerce (numerics-tensor-storage tensor) 'list)))
    ((and (listp arg) (eq (caar arg) 'mlist))
     (mapcar (lambda (v) (coerce ($float v) 'double-float)) (cdr arg)))
    (t (merror "~A: expected ndarray or list, got: ~M" name arg))))

(defun np-odeint-parse-method (method)
  "Parse method keyword to DLSODE mf value.
   adams -> 10 (non-stiff), bdf -> 22 (stiff, internal Jacobian)."
  (cond
    ((null method) 10)
    ((eq method '$adams) 10)
    ((eq method '$bdf) 22)
    (t (merror "np_odeint: method must be adams or bdf, got: ~M" method))))

(defun np-odeint-core (fex-fn jex-fn neq y0-list t-list mf rtol atol)
  "DLSODE stepping core.
   FEX-FN: (function (neq-arg tt y ydot) -> ...) computes dy/dt.
   JEX-FN: (function (neq-arg tt y ml mu pd nrpd) -> ...) Jacobian (or nil).
   Returns a wrapped 2D ndarray [n_times, 1+neq]."
  (let* ((n-times (length t-list))
         (lrw (ecase mf
                (10 (+ 20 (* 16 neq)))
                ((21 22) (+ 22 (* 9 neq) (* neq neq)))))
         (liw (ecase mf
                (10 20)
                ((21 22) (+ 20 neq))))
         (rwork (make-array lrw :element-type 'double-float
                                :initial-element 0.0d0))
         (iwork (make-array liw :element-type 'f2cl-lib:integer4
                                :initial-element 0))
         (y-array (make-array neq :element-type 'double-float
                                  :initial-contents y0-list))
         (tt (first t-list))
         (istate 1)
         (ncols (1+ neq))
         (result (magicl:empty (list n-times ncols) :type 'double-float
                                                     :layout :column-major))
         (result-storage (numerics-tensor-storage result))
         (rtol-arr (make-array 1 :element-type 'double-float
                                 :initial-element rtol))
         (atol-arr (make-array 1 :element-type 'double-float
                                 :initial-element atol))
         (neq-arr (make-array 1 :element-type 'f2cl-lib:integer4
                                :initial-element neq)))
    ;; Step through each output time
    (loop for idx from 0 below n-times
          for tout in t-list
          do
             (when (/= tt tout)
               (multiple-value-bind (v0 v1 v2 ret-tout v4 v5 v6 v7 v8
                                     ret-istate)
                   (odepack:dlsode fex-fn neq-arr y-array tt tout
                                   1 rtol-arr atol-arr 1 istate
                                   0 rwork lrw iwork liw jex-fn mf)
                 (declare (ignore v0 v1 v2 v4 v5 v6 v7 v8))
                 (setf tt ret-tout)
                 (setf istate ret-istate)
                 (when (minusp istate)
                   (merror "np_odeint: DLSODE error (istate=~A): ~A"
                           istate
                           (ecase istate
                             (-1 "excess work done (try smaller step or different method)")
                             (-2 "excess accuracy requested (tolerances too small)")
                             (-3 "illegal input detected")
                             (-4 "repeated error test failures")
                             (-5 "repeated convergence failures (try bdf method)")
                             (-6 "error weight became zero"))))))
             ;; Store [t, y1, ..., yn] in result tensor (column-major)
             (setf (aref result-storage idx) tout)
             (loop for j from 0 below neq
                   do (setf (aref result-storage (+ idx (* (1+ j) n-times)))
                            (aref y-array j))))
    ;; Return wrapped 2D ndarray
    (numerics-wrap (numerics:make-ndarray result))))

(defun np-odeint-expression-mode (f vars y0 tspan rtol-arg atol-arg method)
  "Expression mode: f is an mlist of RHS expressions, vars is [t, y1, ...]."
  (unless (and (listp f) (eq (caar f) 'mlist))
    (merror "np_odeint: f must be a list of RHS expressions, got: ~M" f))
  (unless (and (listp vars) (eq (caar vars) 'mlist))
    (merror "np_odeint: vars must be a list [t, y1, ...], got: ~M" vars))
  (let* ((mf (np-odeint-parse-method method))
         (neq (length (cdr f)))
         (rtol (coerce ($float rtol-arg) 'double-float))
         (atol (coerce ($float atol-arg) 'double-float))
         (y0-list (np-odeint-extract-values y0 "np_odeint (y0)"))
         (t-list (np-odeint-extract-values tspan "np_odeint (tspan)")))
    (unless (= neq (length y0-list))
      (merror "np_odeint: number of equations (~A) must match length of y0 (~A)"
              neq (length y0-list)))
    (unless (>= neq 1)
      (merror "np_odeint: need at least 1 equation"))
    (unless (>= (length t-list) 1)
      (merror "np_odeint: tspan must have at least 1 time point"))
    (unless (= (1+ neq) (length (cdr vars)))
      (merror "np_odeint: vars should be [t, y1, ..., yN] with ~A dependent variables"
              neq))
    ;; Compile RHS
    (let* ((ff (compile nil (coerce-float-fun f vars)))
           (fjac (when (= mf 21)
                   (compile nil
                            (coerce-float-fun
                             (meval `(($jacobian) ,f
                                      ,(list* '(mlist) (cddr vars))))
                             vars)))))
      (flet ((fex (neq-arg tt-val y ydot)
               (declare (type double-float tt-val)
                        (type (cl:array double-float (*)) y ydot)
                        (ignore neq-arg))
               (let* ((y-list (coerce y 'list))
                      (yval (cl:apply ff tt-val y-list)))
                 (replace ydot (cdr yval))))
             (jex (neq-arg tt-val y ml mu pd nrpd)
               (declare (type f2cl-lib:integer4 ml mu nrpd)
                        (type double-float tt-val)
                        (type (cl:array double-float (*)) y)
                        (type (cl:array double-float *) pd)
                        (ignore neq-arg ml mu))
               (when fjac
                 (let* ((y-list (coerce y 'list))
                        (j (cl:apply fjac tt-val y-list))
                        (row 1))
                   (dolist (r (cdr j))
                     (let ((col 1))
                       (dolist (c (cdr r))
                         (setf (f2cl-lib:fref pd (row col) ((1 nrpd) (1)))
                               c)
                         (incf col)))
                     (incf row))))))
        (np-odeint-core #'fex #'jex neq y0-list t-list mf rtol atol)))))

(defun np-odeint-function-mode (f-func y0 tspan rtol-arg atol-arg method)
  "Function mode: f_func(t, y) takes scalar t and 1D ndarray y, returns Maxima list."
  (let* ((mf (np-odeint-parse-method method))
         (y0-list (np-odeint-extract-values y0 "np_odeint (y0)"))
         (t-list (np-odeint-extract-values tspan "np_odeint (tspan)"))
         (neq (length y0-list))
         (rtol (coerce ($float rtol-arg) 'double-float))
         (atol (coerce ($float atol-arg) 'double-float)))
    (unless (>= neq 1)
      (merror "np_odeint: need at least 1 equation (y0 has 0 elements)"))
    (unless (>= (length t-list) 1)
      (merror "np_odeint: tspan must have at least 1 time point"))
    ;; Pre-allocate callback ndarray for y
    (let* ((cb-tensor (magicl:empty (list neq) :type 'double-float
                                               :layout :column-major))
           (cb-handle (numerics:make-ndarray cb-tensor))
           (cb-wrapped (numerics-wrap cb-handle))
           (cb-storage (numerics-tensor-storage cb-tensor)))
      (flet ((fex (neq-arg tt-val y ydot)
               (declare (type double-float tt-val)
                        (type (cl:array double-float (*)) y ydot)
                        (ignore neq-arg))
               ;; Copy y into callback ndarray
               (replace cb-storage y)
               ;; Call user function: f(t, y_ndarray) -> Maxima list
               (let ((result (if (symbolp f-func)
                                 (mfuncall f-func tt-val cb-wrapped)
                                 (mapply f-func (list tt-val cb-wrapped) f-func))))
                 ;; result should be a Maxima list of derivatives
                 (let ((vals (if (and (consp result) (eq (caar result) 'mlist))
                                 (cdr result)
                                 ;; Maybe the function returns differently
                                 (merror "np_odeint: f(t,y) must return a list of derivatives, got: ~M" result))))
                   (loop for v in vals for i from 0
                         do (setf (aref ydot i)
                                  (coerce ($float v) 'double-float))))))
             (jex (neq-arg tt-val y ml mu pd nrpd)
               (declare (ignore neq-arg tt-val y ml mu pd nrpd))))
        (np-odeint-core #'fex #'jex neq y0-list t-list mf rtol atol)))))

(defun $np_odeint (&rest args)
  "Integrate ODE system dy/dt = f(t, y) using DLSODE.

   Expression mode: np_odeint([f1,...], [t,y1,...], y0, tspan [, rtol, atol, method])
     f is a list of RHS expressions; vars names the variables.

   Function mode: np_odeint(f, y0, tspan [, rtol, atol, method])
     f(t, y) takes scalar t and 1D ndarray y, returns a list of derivatives.

   Returns: 2D ndarray [n_times, 1+neq], column 0 = time."
  (unless (>= (length args) 3)
    (merror "np_odeint: expected at least 3 arguments"))
  (let ((first-arg (first args)))
    (if (numerics-callable-p first-arg)
        ;; Function mode: np_odeint(f, y0, tspan [, rtol, atol, method])
        (np-odeint-function-mode first-arg
                                 (second args) (third args)
                                 (or (fourth args) 1.0d-8)
                                 (or (fifth args) 1.0d-8)
                                 (sixth args))
        ;; Expression mode: np_odeint(f, vars, y0, tspan [, rtol, atol, method])
        (progn
          (unless (>= (length args) 4)
            (merror "np_odeint: expression mode requires at least 4 arguments"))
          (np-odeint-expression-mode first-arg
                                     (second args) (third args) (fourth args)
                                     (or (fifth args) 1.0d-8)
                                     (or (sixth args) 1.0d-8)
                                     (seventh args))))))
