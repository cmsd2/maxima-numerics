;;; signal.lisp — Signal processing operations (FFT, convolution)

(in-package #:maxima)

;; fftpack5 (Maxima's mixed-radix FFT) is loaded by numerics-loader.lisp
;; before this ASDF system compiles.

(defun numerics-ensure-1d (handle op-name)
  "Signal an error if the ndarray is not 1D."
  (let ((shape (magicl:shape (numerics:ndarray-tensor handle))))
    (unless (= (length shape) 1)
      (merror "~A: expected 1D array, got shape ~A" op-name shape))))

(defun $np_fft (a)
  "Compute the FFT of a 1D ndarray: np_fft(A).
   Uses the standard convention: Y[k] = sum(x[n] * exp(-2*pi*i*k*n/N)).
   Returns a complex 1D ndarray of the same length.
   Accepts any length (most efficient for lengths of the form 2^r * 3^s * 5^t)."
  (let* ((handle (numerics-unwrap a)))
    (numerics-ensure-1d handle "np_fft")
    (let* ((tensor-in (numerics:ndarray-tensor handle))
           (n (first (magicl:shape tensor-in)))
           (nf (coerce n 'double-float))
           ;; Allocate result tensor and get its live backing storage
           (result (magicl:empty (list n) :type '(complex double-float)
                                          :layout :column-major))
           (z (numerics-tensor-storage result))
           (src (numerics-tensor-storage tensor-in)))
      ;; Copy input into result's backing storage
      (if (eq (numerics:ndarray-dtype handle) :complex-double-float)
          (replace z src)
          (dotimes (i n)
            (setf (aref z i) (complex (the double-float (aref src i)) 0d0))))
      ;; In-place FFT (fftpack5 convention includes 1/N scaling)
      (fftpack5:cfft z)
      ;; Remove 1/N scaling to match standard/NumPy convention
      (dotimes (i n)
        (setf (aref z i) (* nf (aref z i))))
      (numerics-wrap (numerics:make-ndarray result :dtype :complex-double-float)))))

(defun $np_ifft (a)
  "Compute the inverse FFT of a 1D ndarray: np_ifft(A).
   Uses the standard convention: x[n] = (1/N) * sum(Y[k] * exp(+2*pi*i*k*n/N)).
   Returns a complex 1D ndarray of the same length."
  (let* ((handle (numerics-unwrap a)))
    (numerics-ensure-1d handle "np_ifft")
    (let* ((tensor-in (numerics:ndarray-tensor handle))
           (n (first (magicl:shape tensor-in)))
           (nf (coerce n 'double-float))
           (result (magicl:empty (list n) :type '(complex double-float)
                                          :layout :column-major))
           (z (numerics-tensor-storage result))
           (src (numerics-tensor-storage tensor-in)))
      ;; Copy input into result's backing storage
      (if (eq (numerics:ndarray-dtype handle) :complex-double-float)
          (replace z src)
          (dotimes (i n)
            (setf (aref z i) (complex (the double-float (aref src i)) 0d0))))
      ;; In-place inverse FFT (fftpack5 has no scaling on inverse)
      (fftpack5:inverse-cfft z)
      ;; Apply 1/N scaling for standard/NumPy convention
      (dotimes (i n)
        (setf (aref z i) (/ (aref z i) nf)))
      (numerics-wrap (numerics:make-ndarray result :dtype :complex-double-float)))))

(defun numerics-require-2d (handle op-name)
  "Signal an error if the ndarray is not 2D."
  (let ((shape (magicl:shape (numerics:ndarray-tensor handle))))
    (unless (= (length shape) 2)
      (merror "~A: expected 2D array, got shape ~A" op-name shape))))

(defun $np_convolve2d (a kernel)
  "2D convolution: np_convolve2d(A, kernel).
   Both inputs must be 2D real ndarrays. Single-channel only.
   Returns a 2D ndarray of shape (h - kh + 1, w - kw + 1) (valid mode).
   Uses flat storage with manual column-major indexing for performance."
  (let* ((ha (numerics-unwrap a))
         (hk (numerics-unwrap kernel)))
    (numerics-require-2d ha "np_convolve2d")
    (numerics-require-2d hk "np_convolve2d")
    (numerics-require-real a "np_convolve2d")
    (numerics-require-real kernel "np_convolve2d")
    (let* ((ta (numerics:ndarray-tensor ha))
           (tk (numerics:ndarray-tensor hk))
           (shape-a (magicl:shape ta))
           (shape-k (magicl:shape tk))
           (h  (first shape-a))
           (w  (second shape-a))
           (kh (first shape-k))
           (kw (second shape-k))
           (oh (- h kh -1))
           (ow (- w kw -1)))
      (when (or (<= oh 0) (<= ow 0))
        (merror "np_convolve2d: kernel (~Ax~A) larger than input (~Ax~A)"
                kh kw h w))
      (let* ((result (magicl:zeros (list oh ow) :type 'double-float
                                                 :layout :column-major))
             (fa  (numerics-tensor-storage ta))
             (fk  (numerics-tensor-storage tk))
             (out (numerics-tensor-storage result)))
        (declare (type (simple-array double-float (*)) fa fk out))
        ;; Column-major flat index: element (row, col) of shape (nrow, ncol)
        ;; is at flat offset row + col * nrow.
        ;; Direct convolution O(oh * ow * kh * kw)
        (dotimes (j ow)
          (dotimes (i oh)
            (let ((sum 0d0))
              (declare (type double-float sum))
              (dotimes (kj kw)
                (let ((a-col-off (* (+ j kj) h))
                      (k-col-off (* kj kh)))
                  (dotimes (ki kh)
                    (incf sum (* (aref fa (+ (+ i ki) a-col-off))
                                 (aref fk (+ ki k-col-off)))))))
              (setf (aref out (+ i (* j oh))) sum))))
        (numerics-wrap (numerics:make-ndarray result))))))

(defun $np_convolve (a b)
  "1D convolution: np_convolve(A, B).
   Returns a 1D ndarray of length len(A) + len(B) - 1.
   Both inputs must be 1D real ndarrays."
  (let* ((ha (numerics-unwrap a))
         (hb (numerics-unwrap b)))
    (numerics-ensure-1d ha "np_convolve")
    (numerics-ensure-1d hb "np_convolve")
    (numerics-require-real a "np_convolve")
    (numerics-require-real b "np_convolve")
    (let* ((ta (numerics:ndarray-tensor ha))
           (tb (numerics:ndarray-tensor hb))
           (na (first (magicl:shape ta)))
           (nb (first (magicl:shape tb)))
           (out-len (+ na nb -1))
           (fa (numerics-tensor-storage ta))
           (fb (numerics-tensor-storage tb))
           (result (magicl:zeros (list out-len) :type 'double-float
                                                :layout :column-major))
           (out (numerics-tensor-storage result)))
      ;; Direct convolution: O(na * nb)
      (dotimes (i na)
        (let ((ai (aref fa i)))
          (dotimes (j nb)
            (incf (aref out (+ i j)) (* ai (aref fb j))))))
      (numerics-wrap (numerics:make-ndarray result)))))
