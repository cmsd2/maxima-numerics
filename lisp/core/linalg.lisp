;;; linalg.lisp — Linear algebra operations

(in-package #:maxima)

(defun $np_matmul (a b)
  "Matrix multiply: np_matmul(A, B)"
  (let* ((ha (numerics-unwrap a))
         (hb (numerics-unwrap b))
         (ta (numerics:ndarray-tensor ha))
         (tb (numerics:ndarray-tensor hb))
         (dtype (numerics-result-dtype
                 (numerics:ndarray-dtype ha)
                 (numerics:ndarray-dtype hb))))
    (numerics-wrap (numerics:make-ndarray (numerics-with-lapack (magicl:@ ta tb)) :dtype dtype))))

(defun $np_inv (a)
  "Matrix inverse: np_inv(A)"
  (let* ((ha (numerics-unwrap a))
         (ta (numerics:ndarray-tensor ha))
         (dtype (numerics:ndarray-dtype ha)))
    (handler-case
        (numerics-wrap (numerics:make-ndarray (numerics-with-lapack (magicl:inv ta)) :dtype dtype))
      (error (e)
        (merror "np_inv: singular matrix or error: ~A" e)))))

(defun $np_det (a)
  "Determinant: np_det(A) => scalar"
  (lisp-to-maxima-number
   (numerics-with-lapack (magicl:det (numerics:ndarray-tensor (numerics-unwrap a))))))

(defun $np_solve (a b)
  "Solve Ax = b: np_solve(A, b)"
  (let* ((ha (numerics-unwrap a))
         (hb (numerics-unwrap b))
         (ta (numerics:ndarray-tensor ha))
         (tb (numerics:ndarray-tensor hb))
         (dtype (numerics-result-dtype
                 (numerics:ndarray-dtype ha)
                 (numerics:ndarray-dtype hb))))
    (handler-case
        (numerics-wrap (numerics:make-ndarray (numerics-with-lapack (magicl:linear-solve ta tb)) :dtype dtype))
      (error (e)
        (merror "np_solve: ~A" e)))))

(defun $np_svd (a)
  "SVD: np_svd(A) => [U, S, Vt]
   Returns the economy (reduced) SVD so that A = U * diag(S) * Vt works
   directly for any m*n matrix.  U is m*k, S is length k, Vt is k*n
   where k = min(m, n).
   S is always real (double-float) even for complex inputs."
  (let* ((ha (numerics-unwrap a))
         (ta (numerics:ndarray-tensor ha))
         (dtype (numerics:ndarray-dtype ha)))
    (multiple-value-bind (u sigma vt) (numerics-with-lapack (magicl:svd ta :reduced t))
      ;; magicl returns sigma as a diagonal matrix; extract the diagonal
      ;; as a 1D vector of singular values (always real)
      (let* ((shape (magicl:shape sigma))
             (k (apply #'min shape))
             (s-vec (magicl:empty (list k) :type 'double-float
                                            :layout :column-major)))
        (dotimes (i k)
          (setf (magicl:tref s-vec i)
                (coerce (realpart (magicl:tref sigma i i)) 'double-float)))
        `((mlist simp)
          ,(numerics-wrap (numerics:make-ndarray u :dtype dtype))
          ,(numerics-wrap (numerics:make-ndarray s-vec :dtype :double-float))
          ,(numerics-wrap (numerics:make-ndarray vt :dtype dtype)))))))

;;; Internal helper: extract singular values as a CL array from magicl SVD sigma
(defun numerics-svd-values (sigma)
  "Extract singular values from magicl's diagonal sigma matrix as a simple CL array."
  (let* ((shape (magicl:shape sigma))
         (k (apply #'min shape))
         (arr (make-array k :element-type 'double-float)))
    (dotimes (i k)
      (setf (aref arr i) (coerce (realpart (magicl:tref sigma i i)) 'double-float)))
    arr))

(defun $np_eig (a)
  "Eigendecomposition: np_eig(A) => [eigenvalues, eigenvectors]
   Eigenvalues are returned as a 1D ndarray.
   Eigenvectors are returned as a 2D ndarray (columns are eigenvectors).
   Returns complex ndarrays when eigenvalues have non-negligible imaginary parts."
  (let* ((ha (numerics-unwrap a))
         (ta (numerics:ndarray-tensor ha))
         (input-complex (eq (numerics:ndarray-dtype ha) :complex-double-float)))
    (multiple-value-bind (vals vecs) (numerics-with-lapack (magicl:eig ta))
      ;; Use complex output if input was complex or eigenvalues have
      ;; non-negligible imaginary parts
      (let* ((n (length vals))
             (has-complex (or input-complex
                              (some (lambda (v)
                                      (and (complexp v)
                                           (> (abs (imagpart v))
                                              (* 1000 double-float-epsilon
                                                 (max 1.0d0 (abs v))))))
                                    vals)))
             (out-dtype (if has-complex :complex-double-float :double-float))
             (out-et (numerics-element-type out-dtype)))
        ;; Build eigenvalue vector
        (let ((v-vec (magicl:empty (list n) :type out-et :layout :column-major)))
          (loop for i from 0 below n
                for v in vals
                do (setf (magicl:tref v-vec i) (coerce v out-et)))
          ;; Build eigenvector matrix
          (let ((out-vecs
                  (if has-complex
                      ;; Keep complex eigenvectors as-is (coerce element type if needed)
                      (if (eq out-et (magicl:element-type vecs))
                          vecs
                          (let ((rv (magicl:empty (magicl:shape vecs)
                                                  :type out-et
                                                  :layout :column-major)))
                            (dotimes (i (first (magicl:shape vecs)))
                              (dotimes (j (second (magicl:shape vecs)))
                                (setf (magicl:tref rv i j)
                                      (coerce (magicl:tref vecs i j) out-et))))
                            rv))
                      ;; Extract real parts into a double-float matrix
                      (let ((rv (magicl:empty (magicl:shape vecs)
                                             :type 'double-float
                                             :layout :column-major)))
                        (dotimes (i (first (magicl:shape vecs)))
                          (dotimes (j (second (magicl:shape vecs)))
                            (setf (magicl:tref rv i j)
                                  (coerce (realpart (magicl:tref vecs i j))
                                          'double-float))))
                        rv))))
            `((mlist simp)
              ,(numerics-wrap (numerics:make-ndarray v-vec :dtype out-dtype))
              ,(numerics-wrap (numerics:make-ndarray out-vecs :dtype out-dtype)))))))))

(defun numerics-qr-householder (a-tensor)
  "QR decomposition via Householder reflections for double-float matrices.
   Works for any m*n matrix including wide (m < n) matrices that magicl's
   LAPACK wrapper cannot handle.  Returns (values Q R) where Q is m*m
   orthogonal and R is m*n upper trapezoidal."
  (let* ((m (magicl:nrows a-tensor))
         (n (magicl:ncols a-tensor))
         (p (min m n))
         (r (magicl:deep-copy-tensor a-tensor))
         (q (magicl:eye m :type 'double-float :layout :column-major)))
    (dotimes (k p)
      (let* ((len (- m k))
             (v (make-array len :element-type 'double-float
                                :initial-element 0.0d0)))
        ;; Extract sub-column R[k:m, k]
        (dotimes (i len) (setf (aref v i) (magicl:tref r (+ k i) k)))
        (let ((alpha (sqrt (loop for vi across v sum (* vi vi)))))
          (when (> alpha (* 100 double-float-epsilon))
            ;; v[0] += sign(v[0]) * alpha
            (if (>= (aref v 0) 0.0d0)
                (incf (aref v 0) alpha)
                (decf (aref v 0) alpha))
            ;; Normalize v
            (let ((nv (sqrt (loop for vi across v sum (* vi vi)))))
              (dotimes (i len) (setf (aref v i) (/ (aref v i) nv)))
              ;; H = I - 2*v*v^T; apply to R from left
              (dotimes (j n)
                (let ((d 0.0d0))
                  (dotimes (i len)
                    (incf d (* (aref v i) (magicl:tref r (+ k i) j))))
                  (setf d (* 2.0d0 d))
                  (dotimes (i len)
                    (decf (magicl:tref r (+ k i) j) (* (aref v i) d)))))
              ;; Apply to Q from right
              (dotimes (i m)
                (let ((d 0.0d0))
                  (dotimes (j len)
                    (incf d (* (magicl:tref q i (+ k j)) (aref v j))))
                  (setf d (* 2.0d0 d))
                  (dotimes (j len)
                    (decf (magicl:tref q i (+ k j)) (* d (aref v j)))))))))))
    ;; Force positive diagonal (match LAPACK/magicl convention)
    (dotimes (k p)
      (when (minusp (magicl:tref r k k))
        (dotimes (j n) (setf (magicl:tref r k j) (- (magicl:tref r k j))))
        (dotimes (i m) (setf (magicl:tref q i k) (- (magicl:tref q i k))))))
    (values q r)))

(defun $np_qr (a)
  "QR decomposition: np_qr(A) => [Q, R]
   Works for any m*n matrix.  Uses LAPACK for tall/square matrices (m >= n)
   and Householder reflections for wide matrices (m < n)."
  (let* ((ha (numerics-unwrap a))
         (ta (numerics:ndarray-tensor ha))
         (dtype (numerics:ndarray-dtype ha))
         (m (magicl:nrows ta))
         (n (magicl:ncols ta)))
    (multiple-value-bind (q r)
        (if (<= n m)
            (numerics-with-lapack (magicl:qr ta))
            (numerics-with-lapack (numerics-qr-householder ta)))
      `((mlist simp)
        ,(numerics-wrap (numerics:make-ndarray q :dtype dtype))
        ,(numerics-wrap (numerics:make-ndarray r :dtype dtype))))))

(defun $np_lu (a)
  "LU decomposition: np_lu(A) => [L, U, P]
   L is lower-triangular with unit diagonal.
   U is upper-triangular.
   P is a permutation matrix such that P * A = L * U."
  (let* ((ha (numerics-unwrap a))
         (ta (numerics:ndarray-tensor ha))
         (dtype (numerics:ndarray-dtype ha))
         (et (numerics-element-type dtype))
         (m (first (magicl:shape ta)))
         (n (second (magicl:shape ta)))
         (k (min m n)))
    (multiple-value-bind (lu-packed ipiv) (numerics-with-lapack (magicl:lu ta))
      ;; Extract L: lower triangle of lu-packed with 1s on diagonal
      (let ((l-mat (magicl:zeros (list m k) :type et
                                             :layout :column-major))
            (u-mat (magicl:zeros (list k n) :type et
                                             :layout :column-major))
            (p-mat (magicl:eye m :type et :layout :column-major)))
        ;; Fill L
        (dotimes (i m)
          (dotimes (j (min (1+ i) k))
            (if (= i j)
                (setf (magicl:tref l-mat i j) (coerce 1 et))
                (setf (magicl:tref l-mat i j) (magicl:tref lu-packed i j)))))
        ;; Fill U
        (dotimes (i k)
          (loop for j from i below n do
            (setf (magicl:tref u-mat i j) (magicl:tref lu-packed i j))))
        ;; Build P from ipiv (LAPACK pivot vector, 1-based row swap sequence)
        ;; ipiv is a magicl tensor of (signed-byte 32) values
        (let ((prows (magicl:size ipiv)))
          (dotimes (i prows)
            (let ((swap-row (1- (the fixnum
                                  (round (magicl:tref ipiv i))))))
              (unless (= i swap-row)
                ;; Swap rows i and swap-row in P
                (dotimes (c m)
                  (let ((tmp (magicl:tref p-mat i c)))
                    (setf (magicl:tref p-mat i c) (magicl:tref p-mat swap-row c))
                    (setf (magicl:tref p-mat swap-row c) tmp)))))))
        `((mlist simp)
          ,(numerics-wrap (numerics:make-ndarray l-mat :dtype dtype))
          ,(numerics-wrap (numerics:make-ndarray u-mat :dtype dtype))
          ,(numerics-wrap (numerics:make-ndarray p-mat :dtype dtype)))))))

(defun $np_norm (a &optional ord)
  "Matrix or vector norm: np_norm(A) or np_norm(A, ord) => scalar.
   Defaults: 2-norm for vectors, Frobenius for matrices.
   ord values: 1, 2, $inf (or $INF), $fro (or $FRO).
   Vector: 1 = sum(|x|), 2 = Euclidean, inf = max(|x|).
   Matrix: 1 = max col sum, 2 = spectral (largest singular value),
           inf = max row sum, fro = Frobenius."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a)))
         (shape (magicl:shape tensor)))
    (if (= (length shape) 1)
        ;; Vector norms
        (let ((n (first shape)))
          (cond
            ((or (null ord) (eql ord 2))
             ;; 2-norm (Euclidean)
             (let ((sum 0.0d0))
               (dotimes (i n) (let ((x (magicl:tref tensor i)))
                                (incf sum (expt (abs x) 2))))
               (sqrt sum)))
            ((eql ord 1)
             ;; 1-norm: sum of absolute values
             (let ((sum 0.0d0))
               (dotimes (i n) (incf sum (abs (magicl:tref tensor i))))
               sum))
            ((or (eql ord '$inf) (eql ord '$INF))
             ;; inf-norm: max absolute value
             (let ((mx 0.0d0))
               (dotimes (i n) (setf mx (max mx (abs (magicl:tref tensor i)))))
               mx))
            (t (merror "np_norm: unsupported ord for vectors: ~M" ord))))
        ;; Matrix norms
        (let ((nrow (first shape))
              (ncol (second shape)))
          (cond
            ((or (null ord) (eql ord '$fro) (eql ord '$FRO))
             ;; Frobenius norm
             (let ((sum 0.0d0))
               (dotimes (i nrow)
                 (dotimes (j ncol)
                   (let ((x (magicl:tref tensor i j)))
                     (incf sum (expt (abs x) 2)))))
               (sqrt sum)))
            ((eql ord 1)
             ;; 1-norm: max absolute column sum
             (let ((mx 0.0d0))
               (dotimes (j ncol)
                 (let ((col-sum 0.0d0))
                   (dotimes (i nrow) (incf col-sum (abs (magicl:tref tensor i j))))
                   (setf mx (max mx col-sum))))
               mx))
            ((eql ord 2)
             ;; 2-norm (spectral): largest singular value
             (let* ((sigma (nth-value 1 (numerics-with-lapack (magicl:svd tensor))))
                    (s-arr (numerics-svd-values sigma)))
               (reduce #'max s-arr)))
            ((or (eql ord '$inf) (eql ord '$INF))
             ;; inf-norm: max absolute row sum
             (let ((mx 0.0d0))
               (dotimes (i nrow)
                 (let ((row-sum 0.0d0))
                   (dotimes (j ncol) (incf row-sum (abs (magicl:tref tensor i j))))
                   (setf mx (max mx row-sum))))
               mx))
            (t (merror "np_norm: unsupported ord for matrices: ~M" ord)))))))

(defun $np_rank (a)
  "Numerical rank via SVD: np_rank(A)"
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (sigma (nth-value 1 (numerics-with-lapack (magicl:svd ta))))
         (s-arr (numerics-svd-values sigma))
         (tol (* (max (first (magicl:shape ta))
                      (second (magicl:shape ta)))
                 double-float-epsilon
                 (reduce #'max s-arr))))
    (count-if (lambda (x) (> x tol)) s-arr)))

(defun $np_trace (a)
  "Matrix trace: np_trace(A) => scalar"
  (lisp-to-maxima-number
   (magicl:trace (numerics:ndarray-tensor (numerics-unwrap a)))))

(defun $np_transpose (a)
  "Transpose: np_transpose(A)"
  (let* ((ha (numerics-unwrap a))
         (dtype (numerics:ndarray-dtype ha)))
    (numerics-wrap
     (numerics:make-ndarray
      (magicl:transpose (numerics:ndarray-tensor ha))
      :dtype dtype))))

(defun $np_conj (a)
  "Element-wise complex conjugate: np_conj(A).
   For real (double-float) arrays, returns a copy of A unchanged.
   For complex arrays, conjugates each element."
  (let* ((ha (numerics-unwrap a))
         (ta (numerics:ndarray-tensor ha))
         (dtype (numerics:ndarray-dtype ha)))
    (if (eq dtype :complex-double-float)
        ;; Actually conjugate each element
        (let ((result (magicl:deep-copy-tensor ta)))
          (magicl:map! #'conjugate result)
          (numerics-wrap (numerics:make-ndarray result :dtype dtype)))
        ;; Real: identity (just copy)
        (numerics-wrap
         (numerics:make-ndarray (magicl:deep-copy-tensor ta) :dtype dtype)))))

(defun $np_ctranspose (a)
  "Conjugate transpose (Hermitian transpose): np_ctranspose(A).
   For real arrays, equivalent to np_transpose."
  (let* ((ha (numerics-unwrap a))
         (ta (numerics:ndarray-tensor ha))
         (dtype (numerics:ndarray-dtype ha)))
    (numerics-wrap
     (numerics:make-ndarray (magicl:conjugate-transpose ta) :dtype dtype))))

;;; Matrix exponential via scaling-and-squaring with Pade approximation.
;;; Algorithm follows Higham (2005), "The Scaling and Squaring Method for
;;; the Matrix Exponential Revisited", SIAM J. Matrix Anal. Appl. 26(4).
;;; Implementation matches Julia's LinearAlgebra.exp! (stdlib/LinearAlgebra/src/dense.jl).

(defun expm-norm1 (m n)
  "Compute the 1-norm of an n*n matrix M (max absolute column sum)."
  (let ((mx 0.0d0))
    (dotimes (j n mx)
      (let ((col-sum 0.0d0))
        (dotimes (i n)
          (incf col-sum (abs (magicl:tref m i j))))
        (setf mx (max mx col-sum))))))

;; Pade coefficients b_k = (2m-k)!*m! / ((2m)!*k!*(m-k)!)
;; Stored as integer arrays (converted to double-float at use site).
;; From Higham (2005), Table 2.3 / Julia source.

(defparameter *expm-pade3*
  #(120d0 60d0 12d0 1d0))

(defparameter *expm-pade5*
  #(30240d0 15120d0 3360d0 420d0 30d0 1d0))

(defparameter *expm-pade7*
  #(17297280d0 8648640d0 1995840d0 277200d0 25200d0 1512d0 56d0 1d0))

(defparameter *expm-pade9*
  #(17643225600d0 8821612800d0 2075673600d0 302702400d0 30270240d0
    2162160d0 110880d0 3960d0 90d0 1d0))

(defparameter *expm-pade13*
  #(64764752532480000d0 32382376266240000d0 7771770303897600d0
    1187353796428800d0 129060195264000d0 10559470521600d0
    670442572800d0 33522128640d0 1323241920d0
    40840800d0 960960d0 16380d0 182d0 1d0))

;; Norm thresholds theta_m from Higham (2005), Table 2.3.
;; Maximum 1-norm for which Pade(m,m) achieves unit roundoff in IEEE double.
(defconstant +expm-theta3+  1.495585217958292d-2)
(defconstant +expm-theta5+  2.539398330063230d-1)
(defconstant +expm-theta7+  9.504178996162932d-1)
(defconstant +expm-theta9+  2.097847961257068d0)
(defconstant +expm-theta13+ 5.371920351148152d0)

(defun expm-pade-uv-small (a a2 eye c)
  "Compute U, V for Pade orders 3, 5, 7, 9.
   C is the coefficient vector, A2 = A*A, EYE = I."
  (let* ((p (length c))         ;; p = m+1 where m is the Pade order
         (u (magicl:scale eye (aref c 1)))    ;; U = c[1]*I  (odd: c1)
         (v (magicl:scale eye (aref c 0)))    ;; V = c[0]*I  (even: c0)
         (pow (magicl:deep-copy-tensor a2)))   ;; pow = A^2
    ;; Add terms: c[2]*A^2, c[3]*A^2 to V, U; then c[4]*A^4, c[5]*A^4; etc.
    (loop for k from 1 below (floor p 2) do
      (let ((even-idx (* 2 k))
            (odd-idx  (1+ (* 2 k))))
        (setf v (magicl:.+ v (magicl:scale pow (aref c even-idx))))
        (when (< odd-idx p)
          (setf u (magicl:.+ u (magicl:scale pow (aref c odd-idx)))))
        ;; Advance power: pow = pow * A2 (for next iteration)
        (when (< (1+ k) (floor p 2))
          (setf pow (magicl:@ pow a2)))))
    ;; U = A * U (multiply by A to get odd powers)
    (setf u (magicl:@ a u))
    (values u v)))

(defun expm-pade-uv-13 (a a2 a4 a6 eye c)
  "Compute U, V for Pade order 13 using Horner-like evaluation.
   Requires only 6 matrix multiplications total (A2, A4, A6 already computed).
   C is *expm-pade13*."
  ;; U = A * (A6*(c13*A6 + c11*A4 + c9*A2) + c7*A6 + c5*A4 + c3*A2 + c1*I)
  (let* ((u-inner (magicl:.+ (magicl:scale a6 (aref c 13))
                             (magicl:.+ (magicl:scale a4 (aref c 11))
                                        (magicl:scale a2 (aref c 9)))))
         (u-inner (magicl:.+ (magicl:@ a6 u-inner)
                             (magicl:.+ (magicl:scale a6 (aref c 7))
                                        (magicl:.+ (magicl:scale a4 (aref c 5))
                                                   (magicl:.+ (magicl:scale a2 (aref c 3))
                                                              (magicl:scale eye (aref c 1)))))))
         (u (magicl:@ a u-inner))
         ;; V = A6*(c12*A6 + c10*A4 + c8*A2) + c6*A6 + c4*A4 + c2*A2 + c0*I
         (v-inner (magicl:.+ (magicl:scale a6 (aref c 12))
                             (magicl:.+ (magicl:scale a4 (aref c 10))
                                        (magicl:scale a2 (aref c 8)))))
         (v (magicl:.+ (magicl:@ a6 v-inner)
                       (magicl:.+ (magicl:scale a6 (aref c 6))
                                  (magicl:.+ (magicl:scale a4 (aref c 4))
                                             (magicl:.+ (magicl:scale a2 (aref c 2))
                                                        (magicl:scale eye (aref c 0))))))))
    (values u v)))

(defun expm-pade (m)
  "Matrix exponential via scaling-and-squaring with adaptive Pade approximation.
   Selects among Pade orders {3,5,7,9,13} based on the 1-norm of M, following
   Higham (2005) and Julia's implementation."
  (let* ((n (first (magicl:shape m)))
         (na (expm-norm1 m n))
         (et (magicl:element-type m))
         (eye (magicl:eye n :type et :layout :column-major))
         (a2 (magicl:@ m m))
         (si 0)    ;; number of squarings
         u v)
    (cond
      ((<= na +expm-theta3+)
       (multiple-value-setq (u v) (expm-pade-uv-small m a2 eye *expm-pade3*)))
      ((<= na +expm-theta5+)
       (multiple-value-setq (u v) (expm-pade-uv-small m a2 eye *expm-pade5*)))
      ((<= na +expm-theta7+)
       (multiple-value-setq (u v) (expm-pade-uv-small m a2 eye *expm-pade7*)))
      ((<= na +expm-theta9+)
       (multiple-value-setq (u v) (expm-pade-uv-small m a2 eye *expm-pade9*)))
      (t
       ;; Scale so that ||A/2^si|| <= theta_13
       (let ((s (log (/ na +expm-theta13+) 2.0d0)))
         (when (> s 0.0d0)
           (setf si (ceiling s))
           (let ((scale (expt 2.0d0 (- si))))
             (setf m  (magicl:scale m scale))
             (setf a2 (magicl:scale a2 (* scale scale))))))
       (let ((a4 (magicl:@ a2 a2))
             (a6 (magicl:@ a2 (magicl:@ a2 a2))))
         ;; Recompute a6 properly: a6 = a2 * a4
         (setf a6 (magicl:@ a2 a4))
         (multiple-value-setq (u v) (expm-pade-uv-13 m a2 a4 a6 eye *expm-pade13*)))))
    ;; Compute (V - U)^{-1} (V + U)
    ;; Note: magicl:linear-solve only supports vector RHS, so we use inv.
    ;; V - U is well-conditioned by construction (Pade denominator).
    (let ((result (magicl:@ (magicl:inv (magicl:.- v u)) (magicl:.+ v u))))
      ;; Squaring phase
      (dotimes (i si)
        (setf result (magicl:@ result result)))
      result)))

(defun $np_expm (a)
  "Matrix exponential: np_expm(A)
   Computed via adaptive Pade approximation with scaling and squaring.
   Uses Pade orders {3,5,7,9,13} selected by matrix 1-norm (Higham 2005)."
  (let* ((ha (numerics-unwrap a))
         (ta (numerics:ndarray-tensor ha))
         (dtype (numerics:ndarray-dtype ha)))
    (handler-case
        (numerics-wrap (numerics:make-ndarray (expm-pade ta) :dtype dtype))
      (error (e)
        (merror "np_expm: ~A" e)))))

(defun $np_lstsq (a b)
  "Least-squares solution: np_lstsq(A, b) => [x, residuals, rank, S]
   Solves min ||Ax - b||_2 via SVD.
   Returns a Maxima list [x, residuals, rank, S] where:
   - x is the n-by-p solution ndarray
   - residuals is a 1D ndarray of ||Ax_j - b_j||^2 (empty list if m <= n or rank < n)
   - rank is the effective rank (integer)
   - S is a 1D ndarray of singular values"
  (let* ((ha (numerics-unwrap a))
         (hb (numerics-unwrap b))
         (ta (numerics:ndarray-tensor ha))
         (tb (numerics:ndarray-tensor hb))
         (dtype (numerics-result-dtype
                 (numerics:ndarray-dtype ha)
                 (numerics:ndarray-dtype hb)))
         (et (numerics-element-type dtype)))
    (multiple-value-bind (u sigma vt) (numerics-with-lapack (magicl:svd ta))
      (let* ((ut (magicl:conjugate-transpose u))
             (v (magicl:conjugate-transpose vt))
             (utb (magicl:@ ut tb))
             (s-arr (numerics-svd-values sigma))
             (m (first (magicl:shape ta)))
             (n (second (magicl:shape ta)))
             (tol (* (reduce #'max s-arr) double-float-epsilon
                     (max m n)))
             (rank (count-if (lambda (x) (> x tol)) s-arr))
             (k (length s-arr))
             (p (second (magicl:shape utb))))
        ;; Build n*p solution: first k rows scaled by S^{-1}, rest zero
        (let ((sinv-utb (magicl:zeros (list n p) :type et
                                                  :layout :column-major)))
          (dotimes (i (min k n))
            (let ((si (aref s-arr i)))
              (dotimes (j p)
                (setf (magicl:tref sinv-utb i j)
                      (if (> si tol)
                          (* (/ (coerce 1 et) si) (magicl:tref utb i j))
                          (coerce 0 et))))))
          (let* ((x-tensor (magicl:@ v sinv-utb))
                 (x-nd (numerics-wrap (numerics:make-ndarray x-tensor :dtype dtype)))
                 ;; Singular values as 1D ndarray (always real)
                 (s-vec (magicl:empty (list k) :type 'double-float
                                               :layout :column-major)))
            (dotimes (i k)
              (setf (magicl:tref s-vec i) (aref s-arr i)))
            (let ((s-nd (numerics-wrap (numerics:make-ndarray s-vec)))
                  ;; Residuals: only for overdetermined full-rank case
                  (residuals
                    (if (and (> m n) (= rank n))
                        ;; Compute ||Ax_j - b_j||^2 for each column j
                        (let ((res-vec (magicl:empty (list p) :type 'double-float
                                                              :layout :column-major))
                              (ax (magicl:@ ta x-tensor)))
                          (dotimes (j p)
                            (let ((sum-sq 0.0d0))
                              (dotimes (i m)
                                (let ((d (- (magicl:tref ax i j)
                                            (magicl:tref tb i j))))
                                  (incf sum-sq (expt (abs d) 2))))
                              (setf (magicl:tref res-vec j) sum-sq)))
                          (numerics-wrap (numerics:make-ndarray res-vec)))
                        ;; Empty list for under/exactly-determined or rank-deficient
                        '((mlist simp)))))
              `((mlist simp) ,x-nd ,residuals ,rank ,s-nd))))))))

(defun $np_pinv (a)
  "Moore-Penrose pseudo-inverse: np_pinv(A)
   Computed via SVD: A+ = V S+ U*  (conjugate transpose for complex)"
  (let* ((ha (numerics-unwrap a))
         (ta (numerics:ndarray-tensor ha))
         (dtype (numerics:ndarray-dtype ha))
         (et (numerics-element-type dtype)))
    (multiple-value-bind (u sigma vt) (numerics-with-lapack (magicl:svd ta))
      (let* ((v (magicl:conjugate-transpose vt))
             (ut (magicl:conjugate-transpose u))
             (s-arr (numerics-svd-values sigma))
             (tol (* (reduce #'max s-arr) double-float-epsilon
                     (max (first (magicl:shape ta))
                          (second (magicl:shape ta)))))
             (k (length s-arr))
             (m (first (magicl:shape ut))))
        ;; Build diagonal S+ matrix
        (let ((s-plus (magicl:zeros (list k m) :type et
                                               :layout :column-major)))
          (dotimes (i k)
            (let ((si (aref s-arr i)))
              (when (> si tol)
                (setf (magicl:tref s-plus i i) (/ (coerce 1 et) si)))))
          (numerics-wrap
           (numerics:make-ndarray (magicl:@ v s-plus ut) :dtype dtype)))))))
