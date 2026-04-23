;;; numerics-integrate-loader.lisp — Bootstrap and load the numerics/integrate ASDF system
;;; This file is loaded as Common Lisp from numerics-integrate.mac via load().
;;; Requires numerics-loader.lisp to have been loaded first (Quicklisp bootstrapped).

(in-package :maxima)

;; Derive the project directory from this file's location
(let* ((here (make-pathname :directory (pathname-directory *load-truename*))))
  (pushnew here asdf:*central-registry* :test #'equal)
  (pushnew (merge-pathnames "lisp/" here)
           asdf:*central-registry* :test #'equal))

;; Load ODEPACK (Maxima's adaptive ODE solver) — needed before ASDF compile.
;; Use "dlsode" which loads the full odepack system + interface.
(unless (find-package :odepack)
  ($load "dlsode"))

;; Load the integrate system via Quicklisp
(funcall (intern "QUICKLOAD" :ql) "numerics/integrate" :silent t)
