;;; compile_grid.lisp — Fast batch evaluation of a Maxima expression
;;;                     over a Cartesian product of variable ranges.
;;;
;;; compile_grid(expr, [[var, lo, hi, n], ...]) compiles EXPR into a
;;; tight CL lambda (via Maxima's plot-internal coerce-float-fun),
;;; samples it over the listed grid, and returns the result as a
;;; magicl-backed double-float ndarray.
;;;
;;; This is the foundation for reactive-views Component 1 (see
;;; ax-plots/doc/design/reactive-views.md): the perf budget for live
;;; animation and interactive views depends on getting per-sample
;;; cost from ~100 µs (subst + float) to ~100 ns (compiled lambda).
;;;
;;; The fast path uses coerce-float-fun from plot.lisp:443.  When
;;; coerce-float-fun refuses an expression (some special functions,
;;; symbolic conditionals, …) we fall back to subst + float per cell
;;; so the call still produces a correct ndarray, just slower.
;;;
;;; Supports 1D, 2D, and 3D grids.  Higher dimensions error; can be
;;; added later — the loop nest just gets longer.

(in-package #:maxima)

;;; A double-float NaN.  Computed once at load time with float traps
;;; masked so it doesn't blow up the build.  Used by the fallback
;;; path to mark non-evaluable cells.

(defparameter *compile-grid-nan*
  #+sbcl (sb-int:with-float-traps-masked (:invalid :divide-by-zero)
           (the double-float (/ 0d0 0d0)))
  #-sbcl (the double-float (/ 0d0 0d0)))

;;; ----------------------------------------------------------------
;;; Spec parsing

(defun compile-grid-spec-p (spec)
  "Recognise ((mlist) var lo hi n) — Maxima list of 4 elements."
  (and (listp spec)
       (consp (car spec))
       (eq (caar spec) 'mlist)
       (= 4 (length (cdr spec)))))

(defun compile-grid-parse-spec (spec)
  "Validate one [var, lo, hi, n] sublist; return (var lo hi n) as a
   Lisp tuple with lo and hi already coerced to double-float and n
   to a fixnum."
  (unless (compile-grid-spec-p spec)
    (merror "compile_grid: each spec must be [var, lo, hi, n]; got ~M" spec))
  (destructuring-bind (mlist-head var lo hi n) spec
    (declare (ignore mlist-head))
    (unless (symbolp var)
      (merror "compile_grid: spec variable must be a symbol; got ~M" var))
    (let ((lo-d (handler-case (coerce ($float lo) 'double-float)
                  (error () (merror "compile_grid: lo must be a number; got ~M" lo))))
          (hi-d (handler-case (coerce ($float hi) 'double-float)
                  (error () (merror "compile_grid: hi must be a number; got ~M" hi))))
          (n-i  (cond
                  ((and (integerp n) (>= n 1)) n)
                  (t (merror "compile_grid: n must be an integer >= 1; got ~M" n)))))
      (list var lo-d hi-d n-i))))

(defun compile-grid-parse-specs (specs)
  "SPECS is a Maxima list of [var, lo, hi, n] sublists.  Returns a
   Lisp list of (var lo hi n) tuples in input order."
  (unless (and (listp specs)
               (consp (car specs))
               (eq (caar specs) 'mlist))
    (merror "compile_grid: specs must be a list of [var, lo, hi, n] sublists; got ~M"
            specs))
  (mapcar #'compile-grid-parse-spec (cdr specs)))

;;; ----------------------------------------------------------------
;;; Axis values

(defun compile-grid-axis-values (spec)
  "(var lo hi n) → (simple-array double-float (n)) of linspace
   values from LO to HI inclusive at both endpoints.  N=1 gives a
   one-element array containing LO."
  (declare (optimize (speed 3) (safety 1)))
  (destructuring-bind (var lo hi n) spec
    (declare (ignore var)
             (type double-float lo hi)
             (type (and fixnum (integer 1)) n))
    (let ((arr (make-array n :element-type 'double-float)))
      (declare (type (simple-array double-float (*)) arr))
      (cond
        ((= n 1) (setf (aref arr 0) lo))
        (t (let ((h (/ (- hi lo) (coerce (1- n) 'double-float))))
             (declare (type double-float h))
             (dotimes (i n)
               (setf (aref arr i) (+ lo (* (coerce i 'double-float) h)))))))
      arr)))

;;; ----------------------------------------------------------------
;;; Compilation

(defun compile-grid-try-compile (expr vars)
  "Use coerce-float-fun (plot.lisp:443) to translate EXPR into a CL
   lambda accepting VARS as double-floats and returning a float.
   Returns the compiled function on success, NIL when Maxima's
   translator refuses (we fall back to subst+float in that case)."
  (handler-case
      (let* ((mvars (cons '(mlist) vars))
             (fun   (coerce-float-fun expr mvars "compile_grid")))
        fun)
    (error () nil)))

;;; ----------------------------------------------------------------
;;; Hot loops — one per supported dimensionality.
;;;
;;; Bypass MAGICL:TREF and write to the backing simple-array
;;; directly using column-major offset arithmetic.  This skips the
;;; tref dispatch and the runtime layout check, which dominate at
;;; small per-cell costs.  Measured ~3× faster on sin(k*x), with
;;; the gap growing for cheaper expressions and shrinking for more
;;; expensive ones (where the lambda call dominates anyway).

(defun compile-grid-fill-1d (tensor fn axes)
  (declare (optimize (speed 3) (safety 1)))
  (let ((vs0     (first axes))
        (storage (numerics-tensor-storage tensor)))
    (declare (type (simple-array double-float (*)) vs0 storage))
    (dotimes (i (length vs0))
      (setf (aref storage i)
            (the double-float (funcall fn (aref vs0 i)))))))

(defun compile-grid-fill-2d (tensor fn axes)
  (declare (optimize (speed 3) (safety 1)))
  (let* ((vs0     (first axes))
         (vs1     (second axes))
         (storage (numerics-tensor-storage tensor))
         (n0      (length vs0)))
    (declare (type (simple-array double-float (*)) vs0 vs1 storage)
             (type fixnum n0))
    (dotimes (j (length vs1))
      (let ((col-base (the fixnum (* j n0)))
            (xj       (aref vs1 j)))
        (declare (type double-float xj))
        (dotimes (i n0)
          (setf (aref storage (the fixnum (+ i col-base)))
                (the double-float (funcall fn (aref vs0 i) xj))))))))

(defun compile-grid-fill-3d (tensor fn axes)
  (declare (optimize (speed 3) (safety 1)))
  (let* ((vs0     (first axes))
         (vs1     (second axes))
         (vs2     (third axes))
         (storage (numerics-tensor-storage tensor))
         (n0      (length vs0))
         (n1      (length vs1))
         (slab    (the fixnum (* n0 n1))))
    (declare (type (simple-array double-float (*)) vs0 vs1 vs2 storage)
             (type fixnum n0 n1 slab))
    (dotimes (k (length vs2))
      (let ((slab-base (the fixnum (* k slab)))
            (zk        (aref vs2 k)))
        (declare (type double-float zk))
        (dotimes (j n1)
          (let ((col-base (the fixnum (+ slab-base (* j n0))))
                (yj       (aref vs1 j)))
            (declare (type double-float yj))
            (dotimes (i n0)
              (setf (aref storage (the fixnum (+ i col-base)))
                    (the double-float
                         (funcall fn (aref vs0 i) yj zk))))))))))

;;; ----------------------------------------------------------------
;;; Fallback path: per-cell subst + float
;;;
;;; Used when coerce-float-fun rejects the expression.  Same shape
;;; as the fast path; just much slower.  Bind a sentinel for any
;;; cell that doesn't evaluate to a finite number — Plotly handles
;;; NaN by gapping the line / surface.

(defun compile-grid-eval-cell (expr vars vals)
  "Substitute VALS for VARS in EXPR and float.  Returns a
   double-float (NaN if the substituted form doesn't reduce to a
   finite real number)."
  (handler-case
      (let* ((substed expr))
        (mapc (lambda (v val)
                (setf substed
                      (maxima-substitute (coerce val 'double-float) v substed)))
              vars vals)
        (let ((floated (mfuncall '$float substed)))
          (if (and (numberp floated) (realp floated))
              (coerce floated 'double-float)
              ;; Symbolic result, complex result, etc. → NaN
              *compile-grid-nan*)))
    (error () *compile-grid-nan*)))

(defun compile-grid-fill-fallback (tensor expr vars axes)
  "Per-cell subst+float.  Slower than the compiled path but works for
   expressions coerce-float-fun refuses.  Reuses the same direct-
   storage strategy as the fast path; the cost dominates in
   compile-grid-eval-cell anyway."
  (let ((dim     (length axes))
        (storage (numerics-tensor-storage tensor)))
    (declare (type (simple-array double-float (*)) storage))
    (case dim
      (1 (let ((vs0 (first axes)))
           (declare (type (simple-array double-float (*)) vs0))
           (dotimes (i (length vs0))
             (setf (aref storage i)
                   (compile-grid-eval-cell expr vars (list (aref vs0 i)))))))
      (2 (let* ((vs0 (first axes)) (vs1 (second axes))
                (n0 (length vs0)))
           (declare (type (simple-array double-float (*)) vs0 vs1))
           (dotimes (j (length vs1))
             (let ((col-base (* j n0)))
               (dotimes (i n0)
                 (setf (aref storage (+ i col-base))
                       (compile-grid-eval-cell
                         expr vars (list (aref vs0 i) (aref vs1 j)))))))))
      (3 (let* ((vs0 (first axes))
                (vs1 (second axes))
                (vs2 (third axes))
                (n0  (length vs0))
                (n1  (length vs1))
                (slab (* n0 n1)))
           (declare (type (simple-array double-float (*)) vs0 vs1 vs2))
           (dotimes (k (length vs2))
             (let ((slab-base (* k slab)))
               (dotimes (j n1)
                 (let ((col-base (+ slab-base (* j n0))))
                   (dotimes (i n0)
                     (let ((cell (compile-grid-eval-cell
                                   expr vars
                                   (list (aref vs0 i)
                                         (aref vs1 j)
                                         (aref vs2 k)))))
                       (setf (aref storage (+ i col-base)) cell)))))))))
      (t (merror "compile_grid: only 1D, 2D, 3D grids supported in v0.1")))))

;;; ----------------------------------------------------------------
;;; Top-level entry point

(defun $compile_grid (expr specs)
  "compile_grid(expr, [[var, lo, hi, n], ...])

   Compile EXPR into a tight Lisp loop and sample it over the
   Cartesian product of the given variable ranges.  Returns an
   ndarray of shape (n1, n2, ...) — axis i corresponds to spec i.

   When coerce-float-fun cannot compile EXPR (some special
   functions, symbolic conditionals, user-defined Maxima functions
   without an mexpr property), falls back to per-cell subst+float.
   The result shape is the same; non-evaluable cells become NaN.

   Supports up to 3 grid dimensions in this version."
  (let* ((spec-list (compile-grid-parse-specs specs))
         (vars      (mapcar #'first spec-list))
         (shape     (mapcar #'fourth spec-list))
         (axes      (mapcar #'compile-grid-axis-values spec-list))
         (dim       (length shape))
         (tensor    (magicl:zeros shape
                                  :type 'double-float
                                  :layout :column-major))
         (fn        (compile-grid-try-compile expr vars)))
    (when (> dim 3)
      (merror "compile_grid: only 1D, 2D, 3D grids supported in v0.1; got ~A dims"
              dim))
    (cond
      (fn (case dim
            (1 (compile-grid-fill-1d tensor fn axes))
            (2 (compile-grid-fill-2d tensor fn axes))
            (3 (compile-grid-fill-3d tensor fn axes))))
      (t (compile-grid-fill-fallback tensor expr vars axes)))
    (numerics-wrap (numerics:make-ndarray tensor))))
