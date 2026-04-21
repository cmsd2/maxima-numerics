;;; image.lisp — Image I/O and opticl interop for maxima-numerics

(in-package #:maxima)

;;; *numerics-image-dir* is set by numerics-image-loader.lisp (which is loaded
;;; directly via Maxima's load(), so *load-truename* points to the source tree).
;;; We only declare it here so the compiler knows the variable exists.
(defvar *numerics-image-dir*)

;;; Cache for np_mandrill
(defvar *mandrill-cache* nil)

;;; ============================================================
;;; Conversion helpers
;;; ============================================================

(defun image-array-to-ndarray (img)
  "Convert an opticl image (CL array of unsigned-byte 8) to an ndarray.
   Supports grayscale (h w) and color (h w c) images.
   Returns a double-float ndarray with values 0.0 to 255.0."
  (let* ((dims (array-dimensions img))
         (ndim (length dims)))
    (cond
      ;; Grayscale: (h w)
      ((= ndim 2)
       (let* ((h (first dims))
              (w (second dims))
              (tensor (magicl:zeros (list h w) :type 'double-float
                                                :layout :column-major))
              (out (numerics-tensor-storage tensor)))
         (declare (type (simple-array double-float (*)) out))
         ;; Column-major: flat index = row + col * h
         (dotimes (j w)
           (let ((col-off (* j h)))
             (dotimes (i h)
               (setf (aref out (+ i col-off))
                     (coerce (aref img i j) 'double-float)))))
         (numerics-wrap (numerics:make-ndarray tensor))))
      ;; Color: (h w c) — typically c = 3 (RGB) or 4 (RGBA, alpha dropped)
      ((= ndim 3)
       (let* ((h (first dims))
              (w (second dims))
              (c (min (third dims) 3))  ; drop alpha if RGBA
              (tensor (magicl:zeros (list h w c) :type 'double-float
                                                  :layout :column-major))
              (out (numerics-tensor-storage tensor)))
         (declare (type (simple-array double-float (*)) out))
         ;; 3D column-major: flat index = i + j*h + k*h*w
         (let ((hw (* h w)))
           (dotimes (k c)
             (let ((plane-off (* k hw)))
               (dotimes (j w)
                 (let ((col-off (+ (* j h) plane-off)))
                   (dotimes (i h)
                     (setf (aref out (+ i col-off))
                           (coerce (aref img i j k) 'double-float))))))))
         (numerics-wrap (numerics:make-ndarray tensor))))
      (t (merror "image-array-to-ndarray: unsupported image dimensions ~A" dims)))))

(defun ndarray-to-image-array (handle)
  "Convert an ndarray to an opticl image (CL array of unsigned-byte 8).
   Values are clamped to [0, 255] and rounded."
  (let* ((tensor (numerics:ndarray-tensor handle))
         (shape (magicl:shape tensor))
         (ndim (length shape))
         (src (numerics-tensor-storage tensor)))
    (declare (type (simple-array double-float (*)) src))
    (flet ((clamp-u8 (x)
             (max 0 (min 255 (round (the double-float x))))))
      (cond
        ;; 2D grayscale
        ((= ndim 2)
         (let* ((h (first shape))
                (w (second shape))
                (img (opticl:make-8-bit-gray-image h w)))
           (dotimes (j w)
             (let ((col-off (* j h)))
               (dotimes (i h)
                 (setf (aref img i j)
                       (clamp-u8 (aref src (+ i col-off)))))))
           img))
        ;; 3D color [h, w, 3]
        ((and (= ndim 3) (= (third shape) 3))
         (let* ((h (first shape))
                (w (second shape))
                (hw (* h w))
                (img (opticl:make-8-bit-rgb-image h w)))
           (dotimes (j w)
             (dotimes (i h)
               (dotimes (k 3)
                 (setf (aref img i j k)
                       (clamp-u8 (aref src (+ i (* j h) (* k hw))))))))
           img))
        (t (merror "ndarray-to-image-array: expected 2D or 3D (h,w,3) array, got shape ~A"
                    shape))))))

;;; ============================================================
;;; Public Maxima functions
;;; ============================================================

(defun $np_read_image (path)
  "Read an image file and return an ndarray.
   np_read_image(path) — path is a string.
   Returns shape [h,w] for grayscale or [h,w,3] for color.
   Pixel values are double-float in range 0.0 to 255.0."
  (let* ((path-str (if (stringp path) path
                       (coerce (mstring path) 'string)))
         (img (opticl:read-image-file path-str)))
    (image-array-to-ndarray img)))

(defun $np_write_image (a path)
  "Write an ndarray to an image file.
   np_write_image(A, path) — format inferred from file extension.
   Values are clamped to [0, 255]."
  (let* ((handle (numerics-unwrap a))
         (path-str (if (stringp path) path
                       (coerce (mstring path) 'string)))
         (img (ndarray-to-image-array handle)))
    (opticl:write-image-file path-str img)
    '$done))

(defun $np_to_image (a)
  "Convert an ndarray to a raw opticl image array (unsigned-byte 8).
   np_to_image(A) — returns a CL array that can be passed to opticl functions.
   Values are clamped to [0, 255]."
  (let* ((handle (numerics-unwrap a))
         (img (ndarray-to-image-array handle)))
    ;; Wrap in a Maxima noun form so it survives Maxima's evaluator
    `(($opticl_image simp) ,img)))

(defun $np_from_image (img-expr)
  "Convert an opticl image array to an ndarray.
   np_from_image(img) — accepts a wrapped ($opticl_image ...) or raw CL array."
  (let ((img (cond
               ;; Wrapped form from np_to_image
               ((and (listp img-expr)
                     (listp (car img-expr))
                     (eq (caar img-expr) '$opticl_image))
                (cadr img-expr))
               ;; Raw CL array (from :lisp calls)
               ((arrayp img-expr)
                img-expr)
               (t (merror "np_from_image: expected an opticl image, got: ~M" img-expr)))))
    (image-array-to-ndarray img)))

(defun $np_imshow (a)
  "Display an ndarray as an image in the notebook.
   np_imshow(A) — writes a temporary PNG and prints the path for the renderer."
  (let* ((handle (numerics-unwrap a))
         (temp-dir (namestring
                    (uiop:ensure-directory-pathname
                     (or (and (boundp '$maxima_tempdir)
                              (stringp $maxima_tempdir)
                              (not (string= $maxima_tempdir ""))
                              $maxima_tempdir)
                         (namestring (uiop:temporary-directory))))))
         (path (format nil "~Anp_imshow_~A_~A.png"
                       temp-dir
                       (numerics:ndarray-id handle)
                       (get-universal-time)))
         (img (ndarray-to-image-array handle)))
    (opticl:write-image-file path img)
    (format t "~A~%" path)
    '$done))

(defun $np_mandrill ()
  "Return the bundled 512x512 mandrill test image as an ndarray [512,512,3].
   The image is cached after the first call."
  (or *mandrill-cache*
      (setf *mandrill-cache*
            ($np_read_image
             (namestring
              (merge-pathnames
               (make-pathname :directory '(:relative "data")
                              :name "mandrill" :type "png")
               *numerics-image-dir*))))))
