;;; numerics-learn-loader.lisp — Bootstrap and load the numerics/learn ASDF system
;;; This file is loaded as Common Lisp from numerics-learn.mac via load().
;;; Requires numerics-loader.lisp to have been loaded first (Quicklisp bootstrapped).

(in-package :maxima)

;; Derive the project directory from this file's location
(let* ((here (make-pathname :directory (pathname-directory *load-truename*))))
  (pushnew here asdf:*central-registry* :test #'equal)
  (pushnew (merge-pathnames "lisp/" here)
           asdf:*central-registry* :test #'equal))

;; Load the learn system via Quicklisp
(funcall (intern "QUICKLOAD" :ql) "numerics/learn" :silent t)
