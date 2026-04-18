;;; schema.lisp — ArrowSchema CFFI struct definition

(in-package #:numerics)

(cffi:defcstruct arrow-schema
  (format :string)
  (name :string)
  (metadata :pointer)
  (flags :int64)
  (n-children :int64)
  (children :pointer)        ; ArrowSchema**
  (dictionary :pointer)      ; ArrowSchema*
  (release :pointer)         ; void (*)(ArrowSchema*)
  (private-data :pointer))

(cffi:defcallback arrow-release-schema :void ((schema :pointer))
  "Release callback for ArrowSchema. Frees the format string and marks released."
  (let ((fmt (cffi:foreign-slot-value schema '(:struct arrow-schema) 'format)))
    (unless (cffi:null-pointer-p fmt)
      (cffi:foreign-string-free fmt)))
  (let ((name (cffi:foreign-slot-value schema '(:struct arrow-schema) 'name)))
    (unless (cffi:null-pointer-p name)
      (cffi:foreign-string-free name)))
  (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'release)
        (cffi:null-pointer)))
