;;; capture_output.lisp — test helper: run a Maxima function of no
;;; arguments and return everything it printed, on standard output and
;;; on error output (where Lisp warnings go), as one string.

(in-package #:maxima)

(defun $captured_all_output (f)
  (let ((s (make-string-output-stream)))
    (let ((*standard-output* s)
          (*error-output* s))
      (mfuncall f))
    (get-output-stream-string s)))
