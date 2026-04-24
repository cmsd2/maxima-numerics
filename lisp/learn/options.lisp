;;; options.lisp — Shared utilities for numerics/learn
;;;
;;; Keyword option parsing, Maxima function dispatch, and boolean normalization.

(in-package #:maxima)

(defun parse-learn-options (args defaults)
  "Parse keyword=value pairs from &rest ARGS against DEFAULTS alist.
   DEFAULTS is an alist of (maxima-keyword . default-value).
   Maxima parses `n_samples=50` as `((mequal) $n_samples 50)`.
   Returns a fresh alist with parsed values merged over defaults."
  (let ((result (copy-alist defaults)))
    (dolist (arg args result)
      (when (and (consp arg) (consp (car arg)) (eq (caar arg) 'mequal))
        (let* ((key (second arg))
               (val (third arg))
               (pair (assoc key result)))
          (if pair
              (setf (cdr pair) val)
              (merror "Unknown option: ~M" key)))))))

(defun learn-opt (options key)
  "Get parsed option value by Maxima keyword symbol."
  (cdr (assoc key options)))

(defun np-learn-call (f &rest args)
  "Call Maxima function F with ARGS.
   F can be a symbol (function name) or a lambda expression.
   Uses mfuncall for symbols, mapply for lambdas."
  (if (symbolp f)
      (apply #'mfuncall f args)
      (mapply f args f)))

(defun np-learn-done-p (val)
  "Normalize a Maxima boolean to CL boolean.
   true/t/1 → T; false/nil/0/NIL → NIL.
   Handles Maxima's various truth representations."
  (cond
    ((eq val t) t)
    ((eq val nil) nil)
    ((eq val '$true) t)
    ((eq val '$false) nil)
    ((and (numberp val) (not (zerop val))) t)
    ((and (numberp val) (zerop val)) nil)
    (t nil)))
