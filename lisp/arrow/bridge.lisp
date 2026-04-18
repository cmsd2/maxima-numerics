;;; bridge.lisp — Zero-copy ndarray <-> Arrow bridge

(in-package #:numerics)

(defun ndarray-to-arrow-array (handle)
  "Export ndarray as ArrowArray + ArrowSchema for float64.
   Copies data into a static-vector for stable C pointer.
   Returns (values arrow-array-ptr arrow-schema-ptr)."
  (let* ((tensor (ndarray-tensor handle))
         (n (magicl:size tensor))
         ;; Copy data into a static-vector (pinned, stable C pointer)
         (static-vec (static-vectors:make-static-vector
                      n :element-type 'double-float))
         (flat (maxima::numerics-flat-array tensor)))
    (replace static-vec flat)
    (let ((data-ptr (static-vectors:static-vector-pointer static-vec)))
      ;; Build ArrowSchema
      (let ((schema (cffi:foreign-alloc '(:struct arrow-schema))))
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'format)
              (cffi:foreign-string-alloc "g"))   ; "g" = float64
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'name)
              (cffi:foreign-string-alloc ""))
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'metadata)
              (cffi:null-pointer))
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'flags) 0)
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'n-children) 0)
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'children)
              (cffi:null-pointer))
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'dictionary)
              (cffi:null-pointer))
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'release)
              (cffi:callback arrow-release-schema))
        (setf (cffi:foreign-slot-value schema '(:struct arrow-schema) 'private-data)
              (cffi:null-pointer))
        ;; Build ArrowArray
        (let ((array (cffi:foreign-alloc '(:struct arrow-array))))
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'length) n)
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'null-count) 0)
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'offset) 0)
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'n-buffers) 2)
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'n-children) 0)
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'children)
                (cffi:null-pointer))
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'dictionary)
                (cffi:null-pointer))
          ;; buffers: [null_bitmap (null), data_ptr]
          (let ((buffers (cffi:foreign-alloc :pointer :count 2)))
            (setf (cffi:mem-aref buffers :pointer 0) (cffi:null-pointer))
            (setf (cffi:mem-aref buffers :pointer 1) data-ptr)
            (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'buffers)
                  buffers))
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'release)
                (cffi:callback arrow-release-array))
          (setf (cffi:foreign-slot-value array '(:struct arrow-array) 'private-data)
                (cffi:null-pointer))
          (values array schema))))))

(defun arrow-to-ndarray (arrow-array-ptr arrow-schema-ptr)
  "Import ArrowArray as ndarray (copies data for float64).
   Caller transfers ownership; release callback fires on GC."
  (let* ((format-str (cffi:foreign-slot-value
                      arrow-schema-ptr '(:struct arrow-schema) 'format))
         (n (cffi:foreign-slot-value
             arrow-array-ptr '(:struct arrow-array) 'length))
         (buffers (cffi:foreign-slot-value
                   arrow-array-ptr '(:struct arrow-array) 'buffers))
         (data-ptr (cffi:mem-aref buffers :pointer 1)))
    (unless (string= format-str "g")
      (error "arrow-to-ndarray: only float64 ('g') supported, got ~S"
             format-str))
    ;; Copy Arrow buffer into a CL array, then build magicl tensor
    (let* ((data (cffi:foreign-array-to-lisp data-ptr (list n)
                                              :element-type 'double-float))
           (tensor (magicl:from-list (coerce data 'list) (list n)
                                     :type 'double-float))
           (handle (make-ndarray tensor)))
      ;; Finalizer calls Arrow release callbacks
      (let ((aptr arrow-array-ptr) (sptr arrow-schema-ptr))
        (trivial-garbage:finalize handle
          (lambda ()
            (let ((rel (cffi:foreign-slot-value
                        aptr '(:struct arrow-array) 'release)))
              (unless (cffi:null-pointer-p rel)
                (cffi:foreign-funcall-pointer rel () :pointer aptr :void)))
            (let ((rel (cffi:foreign-slot-value
                        sptr '(:struct arrow-schema) 'release)))
              (unless (cffi:null-pointer-p rel)
                (cffi:foreign-funcall-pointer rel () :pointer sptr :void))))))
      handle)))

(defun concatenate-ndarrays (handles)
  "Concatenate a list of 1D ndarray handles into a single 1D ndarray."
  (let* ((all-vals nil))
    (dolist (h handles)
      (let* ((tensor (ndarray-tensor h))
             (n (magicl:size tensor)))
        (dotimes (i n)
          (push (magicl:tref tensor i) all-vals))))
    (let* ((vals (nreverse all-vals))
           (total (length vals)))
      (make-ndarray
       (magicl:from-list vals (list total) :type 'double-float)))))
