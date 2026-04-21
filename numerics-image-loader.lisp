;;; numerics-image-loader.lisp — Bootstrap and load the numerics/image ASDF system
;;; This file is loaded as Common Lisp from numerics-image.mac via load().
;;; Requires numerics-loader.lisp to have been loaded first (Quicklisp bootstrapped).

(in-package :maxima)

;; Derive the project directory from this file's location
(let* ((here (make-pathname :directory (pathname-directory *load-truename*))))
  (pushnew here asdf:*central-registry* :test #'equal)
  (pushnew (merge-pathnames "lisp/" here)
           asdf:*central-registry* :test #'equal))

;; Capture the source directory *before* ASDF compilation relocates paths.
;; image.lisp uses this to find bundled assets (data/mandrill.png).
(defvar *numerics-image-dir*
  (make-pathname :directory (pathname-directory *load-truename*)))

;; Load the image system via Quicklisp (resolves opticl + dependencies)
(funcall (intern "QUICKLOAD" :ql) "numerics/image" :silent t)
