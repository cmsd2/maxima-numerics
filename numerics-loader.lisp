;;; numerics-loader.lisp — Bootstrap Quicklisp and load the numerics ASDF system
;;; This file is loaded as Common Lisp from numerics.mac via load().

(in-package :maxima)

;; Bootstrap Quicklisp if available but not loaded
(unless (find-package :quicklisp)
  (let ((ql-init (merge-pathnames "quicklisp/setup.lisp"
                                  (user-homedir-pathname))))
    (if (probe-file ql-init)
        (load ql-init)
        (error "Quicklisp not found. Install it:~%~
                curl -O https://beta.quicklisp.org/quicklisp.lisp~%~
                sbcl --load quicklisp.lisp ~
                --eval '(quicklisp-quicklisp:install)' --quit"))))

;; Derive the project directory from this file's location
(let* ((here (make-pathname :directory (pathname-directory *load-truename*))))
  (pushnew here asdf:*central-registry* :test #'equal)
  (pushnew (merge-pathnames "lisp/" here)
           asdf:*central-registry* :test #'equal))

;; Load fftpack5 (Maxima's mixed-radix FFT) — needed before ASDF compile
(unless (find-package :fftpack5)
  ($load "fftpack5"))

;; Load the core system via Quicklisp (resolves magicl + dependencies)
(funcall (intern "QUICKLOAD" :ql) "numerics/core" :silent t)

;; Load doc index for ? and ?? help (if available)
(let* ((here (make-pathname :directory (pathname-directory *load-truename*)))
       (idx (merge-pathnames "numerics-index.lisp" here)))
  (when (probe-file idx)
    ($load (namestring idx))))
