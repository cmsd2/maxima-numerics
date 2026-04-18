;;; handle.lisp — The ndarray opaque handle type

(in-package #:numerics)

(defvar *ndarray-counter* 0)

(defstruct (ndarray
            (:constructor %make-ndarray)
            (:print-function print-ndarray))
  "An opaque handle wrapping a magicl tensor."
  (id     0   :type fixnum :read-only t)
  (tensor nil :read-only nil)
  (dtype  :double-float :type keyword :read-only t))

(defun print-ndarray (obj stream depth)
  (declare (ignore depth))
  (format stream "#<ndarray ~A ~A ~A>"
          (ndarray-id obj)
          (ndarray-dtype obj)
          (when (ndarray-tensor obj)
            (magicl:shape (ndarray-tensor obj)))))

(defun make-ndarray (tensor &key (dtype :double-float))
  "Create an ndarray handle wrapping TENSOR.
   Registers a GC finalizer to release foreign storage."
  (let* ((id (incf *ndarray-counter*))
         (handle (%make-ndarray :id id :tensor tensor :dtype dtype)))
    ;; Register finalizer. The closure captures the tensor, NOT
    ;; the handle, to avoid preventing GC of the handle.
    (let ((t-ref tensor))
      (trivial-garbage:finalize handle
        (lambda ()
          (when t-ref (setf t-ref nil)))))
    handle))

;;; Maxima-level helpers (in maxima package)

(in-package #:maxima)

(defun $ndarray_p (x)
  "Predicate: is X an ndarray handle?"
  (and (listp x)
       (listp (car x))
       (eq (caar x) '$ndarray)
       (typep (cadr x) 'numerics:ndarray)))

(defun numerics-unwrap (x)
  "Extract the Lisp ndarray struct from a Maxima ndarray expression."
  (unless ($ndarray_p x)
    (merror "Expected an ndarray, got: ~M" x))
  (cadr x))

(defun numerics-wrap (handle)
  "Wrap a Lisp ndarray struct into a Maxima expression."
  `(($ndarray simp) ,handle))
