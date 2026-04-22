;;; numerics-optimize-loader.lisp — Bootstrap and load the numerics/optimize ASDF system
;;; This file is loaded as Common Lisp from numerics-optimize.mac via load().
;;; Requires numerics-loader.lisp to have been loaded first (Quicklisp bootstrapped).

(in-package :maxima)

;; Derive the project directory from this file's location
(let* ((here (make-pathname :directory (pathname-directory *load-truename*))))
  (pushnew here asdf:*central-registry* :test #'equal)
  (pushnew (merge-pathnames "lisp/" here)
           asdf:*central-registry* :test #'equal))

;; Load lbfgs (Maxima's L-BFGS optimizer) — needed before ASDF compile.
;; Some lbfgs copies reference `flonum-epsilon` (without earmuffs) which
;; isn't visible at compile time. Declare it so SBCL's compiler sees it,
;; and selectively muffle the warning if it still appears (e.g. from
;; an unprecompiled copy in ~/.maxima/).
(defvar flonum-epsilon double-float-epsilon)
(unless (fboundp 'common-lisp-user::lbfgs)
  (handler-bind
      ((warning
        (lambda (w)
          (when (search "FLONUM-EPSILON" (princ-to-string w))
            (muffle-warning w)))))
    ($load "lbfgs")))

;; Load the optimize system via Quicklisp
(funcall (intern "QUICKLOAD" :ql) "numerics/optimize" :silent t)
