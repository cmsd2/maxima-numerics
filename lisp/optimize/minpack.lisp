;;; minpack.lisp — Root finding and nonlinear least squares via MINPACK
;;;
;;; Supports two calling conventions:
;;;   np_fsolve([f1,...], [x1,...], x0 [, jacobian, tol])  -- expression mode
;;;   np_fsolve(f_func, x0 [, tolerance])                  -- function mode
;;;
;;;   np_lsq_nonlinear([f1,...], [x1,...], x0 [, jacobian, tol])  -- expression
;;;   np_lsq_nonlinear(f_func, x0 [, tolerance])                  -- function
;;;
;;; Wraps share/minpack/ HYBRD1/HYBRJ1 and LMDIF1/LMDER1.

(in-package #:maxima)

;; minpack is loaded by numerics-optimize-loader.lisp before this ASDF system.

(defun np-minpack-extract-x0 (x0 n name)
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

(defun np-minpack-extract-x0-flexible (x0 name)
  "Extract initial values, inferring n from the input."
  (cond
    (($ndarray_p x0)
     (let* ((handle (numerics-unwrap x0))
            (tensor (numerics:ndarray-tensor handle))
            (storage (numerics-tensor-storage tensor))
            (n (length storage)))
       (values (make-array n :element-type 'double-float
                             :initial-contents (coerce storage 'list))
               n)))
    ((and (listp x0) (eq (caar x0) 'mlist))
     (let ((n (length (cdr x0))))
       (values (make-array n :element-type 'double-float
                             :initial-contents
                             (mapcar (lambda (v) (coerce ($float v) 'double-float))
                                     (cdr x0)))
               n)))
    (t (merror "~A: x0 must be an ndarray or list, got: ~M" name x0))))

(defun np-minpack-wrap-result (x n fvec m info)
  "Wrap MINPACK result as [x_opt_ndarray, residual_norm, info]."
  (let* ((result-tensor (magicl:empty (list n) :type 'double-float
                                               :layout :column-major))
         (result-storage (numerics-tensor-storage result-tensor)))
    (replace result-storage x)
    `((mlist)
      ,(numerics-wrap (numerics:make-ndarray result-tensor))
      ,(minpack:enorm m fvec)
      ,info)))

(defun np-minpack-call-fn (f-func x-ndarray)
  "Call a Maxima function with a single ndarray argument.
   Returns a list of double-float values extracted from the Maxima list result."
  (let ((result (np-minimize-call f-func x-ndarray)))
    (unless (and (consp result) (eq (caar result) 'mlist))
      (merror "function must return a list, got: ~M" result))
    (mapcar (lambda (z) (coerce ($float z) 'double-float)) (cdr result))))

;;; ======================== np_fsolve ========================

(defun np-fsolve-expression-mode (fcns vars x0 jacobian-arg tolerance)
  "Expression mode: fcns and vars are Maxima lists of expressions/variables."
  (unless (and (listp fcns) (eq (caar fcns) 'mlist))
    (merror "np_fsolve: fcns must be a list of expressions, got: ~M" fcns))
  (unless (and (listp vars) (eq (caar vars) 'mlist))
    (merror "np_fsolve: vars must be a list of variables, got: ~M" vars))
  (unless (= (length (cdr fcns)) (length (cdr vars)))
    (merror "np_fsolve: number of equations (~A) must equal number of variables (~A)"
            (length (cdr fcns)) (length (cdr vars))))

  (let* ((n (length (cdr vars)))
         (x (np-minpack-extract-x0 x0 n "np_fsolve"))
         (tol (coerce ($float tolerance) 'double-float))
         (fvec (make-array n :element-type 'double-float))
         (info 0)
         (use-jacobian (eq jacobian-arg t))
         (fv (coerce-float-fun fcns vars)))

    (if use-jacobian
        ;; With Jacobian: use HYBRJ1
        (let* ((fjac-expr (mfuncall '$jacobian fcns vars))
               (fj (coerce-float-fun fjac-expr vars))
               (fjac (make-array (* n n) :element-type 'double-float))
               (ldfjac n)
               (lwa (truncate (* n (+ n 13)) 2))
               (wa (make-array lwa :element-type 'double-float)))
          (labels ((fcn-and-jac (nn x fvec fjac ldfjac iflag)
                     (declare (type f2cl-lib:integer4 nn ldfjac iflag)
                              (type (cl:array double-float (*)) x fvec fjac))
                     (let ((x-list (subseq (coerce x 'list) 0 nn)))
                       (ecase iflag
                         (1
                          (let ((val (apply #'funcall fv x-list)))
                            (replace fvec (mapcar (lambda (z) (cl:float z 1d0))
                                                  (cdr val)))))
                         (2
                          (let ((j (apply #'funcall fj x-list)))
                            (let ((row-index 0))
                              (dolist (row (cdr j))
                                (let ((col-index 0))
                                  (dolist (col (cdr row))
                                    (setf (aref fjac (+ row-index (* ldfjac col-index)))
                                          (cl:float col 1d0))
                                    (incf col-index)))
                                (incf row-index)))))))
                     (values nn nil nil nil ldfjac iflag)))
            (multiple-value-bind (v0 v1 v2 v3 v4 v5 v6 ret-info)
                (minpack:hybrj1 #'fcn-and-jac n x fvec fjac ldfjac
                                tol info wa lwa)
              (declare (ignore v0 v1 v2 v3 v4 v5 v6))
              (np-minpack-wrap-result x n fvec n ret-info))))

        ;; Without Jacobian: use HYBRD1
        (let* ((lwa (truncate (* n (+ (* 3 n) 13)) 2))
               (wa (make-array lwa :element-type 'double-float)))
          (labels ((fval (nn x fvec iflag)
                     (declare (type f2cl-lib:integer4 nn iflag)
                              (type (cl:array double-float (*)) x fvec))
                     (let* ((x-list (subseq (coerce x 'list) 0 nn))
                            (val (apply #'funcall fv x-list)))
                       (replace fvec (mapcar (lambda (z) (cl:float z 1d0))
                                             (cdr val))))
                     (values nn nil nil iflag)))
            (multiple-value-bind (v0 v1 v2 v3 v4 ret-info)
                (minpack:hybrd1 #'fval n x fvec tol info wa lwa)
              (declare (ignore v0 v1 v2 v3 v4))
              (np-minpack-wrap-result x n fvec n ret-info)))))))

(defun np-fsolve-function-mode (f-func x0 tolerance)
  "Function mode: f(x) takes 1D ndarray, returns Maxima list. Uses HYBRD1."
  (multiple-value-bind (x n) (np-minpack-extract-x0-flexible x0 "np_fsolve")
    (let* ((tol (coerce ($float tolerance) 'double-float))
           (fvec (make-array n :element-type 'double-float))
           (info 0)
           ;; Pre-allocate callback ndarray
           (cb-tensor (magicl:empty (list n) :type 'double-float
                                             :layout :column-major))
           (cb-handle (numerics:make-ndarray cb-tensor))
           (cb-wrapped (numerics-wrap cb-handle))
           (cb-storage (numerics-tensor-storage cb-tensor))
           (lwa (truncate (* n (+ (* 3 n) 13)) 2))
           (wa (make-array lwa :element-type 'double-float)))
      (labels ((fval (nn x fvec iflag)
                 (declare (type f2cl-lib:integer4 nn iflag)
                          (type (cl:array double-float (*)) x fvec))
                 (replace cb-storage x :end1 nn :end2 nn)
                 (let ((vals (np-minpack-call-fn f-func cb-wrapped)))
                   (loop for v in vals for i from 0
                         do (setf (aref fvec i) v)))
                 (values nn nil nil iflag)))
        (multiple-value-bind (v0 v1 v2 v3 v4 ret-info)
            (minpack:hybrd1 #'fval n x fvec tol info wa lwa)
          (declare (ignore v0 v1 v2 v3 v4))
          (np-minpack-wrap-result x n fvec n ret-info))))))

(defun $np_fsolve (&rest args)
  "Solve system of n nonlinear equations in n unknowns.

   Expression mode: np_fsolve([f1,...], [x1,...], x0 [, jacobian, tolerance])
   Function mode:   np_fsolve(f, x0 [, tolerance])
     f(x) takes a 1D ndarray, returns a Maxima list of n values.

   Returns: [x_opt, residual_norm, info]."
  (unless (>= (length args) 2)
    (merror "np_fsolve: expected at least 2 arguments"))
  (if (numerics-callable-p (first args))
      ;; Function mode: np_fsolve(f, x0 [, tolerance])
      (np-fsolve-function-mode (first args) (second args)
                               (or (third args)
                                   #.(sqrt double-float-epsilon)))
      ;; Expression mode: np_fsolve(fcns, vars, x0 [, jacobian, tol])
      (progn
        (unless (>= (length args) 3)
          (merror "np_fsolve: expression mode requires at least 3 arguments"))
        (np-fsolve-expression-mode (first args) (second args) (third args)
                                   (if (>= (length args) 4)
                                       (fourth args) t)
                                   (if (>= (length args) 5)
                                       (fifth args)
                                       #.(sqrt double-float-epsilon))))))

;;; ======================== np_lsq_nonlinear ========================

(defun np-lsq-expression-mode (fcns vars x0 jacobian-arg tolerance)
  "Expression mode: fcns and vars are Maxima lists."
  (unless (and (listp fcns) (eq (caar fcns) 'mlist))
    (merror "np_lsq_nonlinear: fcns must be a list of expressions, got: ~M" fcns))
  (unless (and (listp vars) (eq (caar vars) 'mlist))
    (merror "np_lsq_nonlinear: vars must be a list of variables, got: ~M" vars))

  (let* ((n (length (cdr vars)))
         (m (length (cdr fcns))))
    (unless (>= m n)
      (merror "np_lsq_nonlinear: need at least as many equations (~A) as variables (~A)"
              m n))

    (let* ((x (np-minpack-extract-x0 x0 n "np_lsq_nonlinear"))
           (tol (coerce ($float tolerance) 'double-float))
           (fvec (make-array m :element-type 'double-float))
           (info 0)
           (ipvt (make-array n :element-type 'f2cl-lib:integer4))
           (use-jacobian (eq jacobian-arg t))
           (fv (coerce-float-fun fcns vars)))

      (if use-jacobian
          ;; With Jacobian: use LMDER1
          (let* ((fjac-expr (meval `(($jacobian) ,fcns ,vars)))
                 (fj (coerce-float-fun fjac-expr vars))
                 (fjac (make-array (* m n) :element-type 'double-float))
                 (ldfjac m)
                 (lwa (+ m (* 5 n)))
                 (wa (make-array lwa :element-type 'double-float)))
            (labels ((fcn-and-jac (mm nn x fvec fjac ldfjac iflag)
                       (declare (type f2cl-lib:integer4 mm nn ldfjac iflag)
                                (type (cl:array double-float (*)) x fvec fjac))
                       (let ((x-list (subseq (coerce x 'list) 0 nn)))
                         (ecase iflag
                           (1
                            (let ((val (apply #'funcall fv x-list)))
                              (replace fvec (mapcar (lambda (z) (cl:float z 1d0))
                                                    (cdr val)))))
                           (2
                            (let ((j (apply #'funcall fj x-list)))
                              (let ((row-index 0))
                                (dolist (row (cdr j))
                                  (let ((col-index 0))
                                    (dolist (col (cdr row))
                                      (setf (aref fjac (+ row-index (* ldfjac col-index)))
                                            (cl:float col 1d0))
                                      (incf col-index)))
                                  (incf row-index)))))))
                       (values mm nn nil nil nil ldfjac iflag)))
              (multiple-value-bind (v0 v1 v2 v3 v4 v5 v6 v7 ret-info)
                  (minpack:lmder1 #'fcn-and-jac m n x fvec fjac ldfjac
                                  tol info ipvt wa lwa)
                (declare (ignore v0 v1 v2 v3 v4 v5 v6 v7))
                (np-minpack-wrap-result x n fvec m ret-info))))

          ;; Without Jacobian: use LMDIF1
          (let* ((lwa (+ m (* 5 n) (* m n)))
                 (wa (make-array lwa :element-type 'double-float)))
            (labels ((fval (mm nn x fvec iflag)
                       (declare (type f2cl-lib:integer4 mm nn iflag)
                                (type (cl:array double-float (*)) x fvec))
                       (let* ((x-list (subseq (coerce x 'list) 0 nn))
                              (val (apply #'funcall fv x-list)))
                         (replace fvec (mapcar (lambda (z) (cl:float z 1d0))
                                               (cdr val))))
                       (values mm nn nil nil iflag)))
              (multiple-value-bind (v0 v1 v2 v3 v4 v5 ret-info)
                  (minpack:lmdif1 #'fval m n x fvec tol info ipvt wa lwa)
                (declare (ignore v0 v1 v2 v3 v4 v5))
                (np-minpack-wrap-result x n fvec m ret-info))))))))

(defun np-lsq-function-mode (f-func x0 tolerance)
  "Function mode: f(x) takes 1D ndarray, returns Maxima list of m residuals.
   Uses LMDIF1 (finite differences). m is determined from first call."
  (multiple-value-bind (x n) (np-minpack-extract-x0-flexible x0 "np_lsq_nonlinear")
    (let* ((tol (coerce ($float tolerance) 'double-float))
           ;; Pre-allocate callback ndarray
           (cb-tensor (magicl:empty (list n) :type 'double-float
                                             :layout :column-major))
           (cb-handle (numerics:make-ndarray cb-tensor))
           (cb-wrapped (numerics-wrap cb-handle))
           (cb-storage (numerics-tensor-storage cb-tensor)))
      ;; Call f once to determine m
      (replace cb-storage x)
      (let* ((first-result (np-minpack-call-fn f-func cb-wrapped))
             (m (length first-result)))
        (unless (>= m n)
          (merror "np_lsq_nonlinear: need at least as many residuals (~A) as variables (~A)"
                  m n))
        (let* ((fvec (make-array m :element-type 'double-float))
               (info 0)
               (ipvt (make-array n :element-type 'f2cl-lib:integer4))
               (lwa (+ m (* 5 n) (* m n)))
               (wa (make-array lwa :element-type 'double-float))
               (first-call t))
          (labels ((fval (mm nn x fvec iflag)
                     (declare (type f2cl-lib:integer4 mm nn iflag)
                              (type (cl:array double-float (*)) x fvec))
                     (if first-call
                         ;; Reuse cached result from the probe call
                         (progn
                           (loop for v in first-result for i from 0
                                 do (setf (aref fvec i) v))
                           (setf first-call nil))
                         ;; Normal call
                         (progn
                           (replace cb-storage x :end1 nn :end2 nn)
                           (let ((vals (np-minpack-call-fn f-func cb-wrapped)))
                             (loop for v in vals for i from 0
                                   do (setf (aref fvec i) v)))))
                     (values mm nn nil nil iflag)))
            (multiple-value-bind (v0 v1 v2 v3 v4 v5 ret-info)
                (minpack:lmdif1 #'fval m n x fvec tol info ipvt wa lwa)
              (declare (ignore v0 v1 v2 v3 v4 v5))
              (np-minpack-wrap-result x n fvec m ret-info))))))))

(defun $np_lsq_nonlinear (&rest args)
  "Nonlinear least squares: minimize sum(fi^2).

   Expression mode: np_lsq_nonlinear([f1,...], [x1,...], x0 [, jacobian, tol])
   Function mode:   np_lsq_nonlinear(f, x0 [, tolerance])
     f(x) takes a 1D ndarray, returns a Maxima list of m residuals (m >= n).

   Returns: [x_opt, residual_norm, info]."
  (unless (>= (length args) 2)
    (merror "np_lsq_nonlinear: expected at least 2 arguments"))
  (if (numerics-callable-p (first args))
      ;; Function mode: np_lsq_nonlinear(f, x0 [, tolerance])
      (np-lsq-function-mode (first args) (second args)
                             (or (third args)
                                 #.(sqrt double-float-epsilon)))
      ;; Expression mode: np_lsq_nonlinear(fcns, vars, x0 [, jacobian, tol])
      (progn
        (unless (>= (length args) 3)
          (merror "np_lsq_nonlinear: expression mode requires at least 3 arguments"))
        (np-lsq-expression-mode (first args) (second args) (third args)
                                (if (>= (length args) 4)
                                    (fourth args) t)
                                (if (>= (length args) 5)
                                    (fifth args)
                                    #.(sqrt double-float-epsilon))))))
