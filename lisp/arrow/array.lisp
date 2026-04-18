;;; array.lisp — ArrowArray CFFI struct definition

(in-package #:numerics)

(cffi:defcstruct arrow-array
  (length :int64)
  (null-count :int64)
  (offset :int64)
  (n-buffers :int64)
  (n-children :int64)
  (buffers :pointer)         ; const void**
  (children :pointer)        ; ArrowArray**
  (dictionary :pointer)      ; ArrowArray*
  (release :pointer)         ; void (*)(ArrowArray*)
  (private-data :pointer))

(cffi:defcallback arrow-release-array :void ((array :pointer))
  "Release callback for ArrowArray. Releases backing storage."
  (let ((pdata (cffi:foreign-slot-value array '(:struct arrow-array) 'private-data)))
    (declare (ignore pdata))
    ;; Backing storage is managed by static-vectors / magicl GC;
    ;; the release callback just marks the array as released.
    )
  (let ((buffers (cffi:foreign-slot-value array '(:struct arrow-array) 'buffers)))
    (unless (cffi:null-pointer-p buffers)
      (cffi:foreign-free buffers)))
  (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'release)
        (cffi:null-pointer)))
