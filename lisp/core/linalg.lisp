;;; linalg.lisp — Linear algebra operations

(in-package #:maxima)

(defun $np_matmul (a b)
  "Matrix multiply: np_matmul(A, B)"
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (tb (numerics:ndarray-tensor (numerics-unwrap b))))
    (numerics-wrap (numerics:make-ndarray (magicl:@ ta tb)))))

(defun $np_inv (a)
  "Matrix inverse: np_inv(A)"
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (handler-case
        (numerics-wrap (numerics:make-ndarray (magicl:inv ta)))
      (error (e)
        (merror "np_inv: singular matrix or error: ~A" e)))))

(defun $np_det (a)
  "Determinant: np_det(A) => scalar"
  (magicl:det (numerics:ndarray-tensor (numerics-unwrap a))))

(defun $np_solve (a b)
  "Solve Ax = b: np_solve(A, b)"
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
        (tb (numerics:ndarray-tensor (numerics-unwrap b))))
    (handler-case
        (numerics-wrap (numerics:make-ndarray (magicl:linear-solve ta tb)))
      (error (e)
        (merror "np_solve: ~A" e)))))

(defun $np_svd (a)
  "SVD: np_svd(A) => [U, S, Vt]
   S is returned as a 1D ndarray of singular values (not a diagonal matrix)."
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (multiple-value-bind (u sigma vt) (magicl:svd ta)
      ;; magicl returns sigma as a diagonal matrix; extract the diagonal
      ;; as a 1D vector of singular values
      (let* ((shape (magicl:shape sigma))
             (k (apply #'min shape))
             (s-vec (magicl:empty (list k) :type 'double-float
                                            :layout :column-major)))
        (dotimes (i k)
          (setf (magicl:tref s-vec i) (magicl:tref sigma i i)))
        `((mlist simp)
          ,(numerics-wrap (numerics:make-ndarray u))
          ,(numerics-wrap (numerics:make-ndarray s-vec))
          ,(numerics-wrap (numerics:make-ndarray vt)))))))

;;; Internal helper: extract singular values as a CL array from magicl SVD sigma
(defun numerics-svd-values (sigma)
  "Extract singular values from magicl's diagonal sigma matrix as a simple CL array."
  (let* ((shape (magicl:shape sigma))
         (k (apply #'min shape))
         (arr (make-array k :element-type 'double-float)))
    (dotimes (i k)
      (setf (aref arr i) (magicl:tref sigma i i)))
    arr))

(defun $np_eig (a)
  "Eigendecomposition: np_eig(A) => [eigenvalues, eigenvectors]
   Eigenvalues are returned as a 1D ndarray.
   Eigenvectors are returned as a 2D ndarray (columns are eigenvectors).
   For real-valued results the complex parts are dropped."
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (multiple-value-bind (vals vecs) (magicl:eig ta)
      ;; magicl returns vals as a plain CL list of (possibly complex) numbers
      (let* ((n (length vals))
             (v-vec (magicl:empty (list n) :type 'double-float
                                           :layout :column-major)))
        (loop for i from 0 below n
              for v in vals
              do (setf (magicl:tref v-vec i) (coerce (realpart v) 'double-float)))
        ;; magicl returns eigenvectors as complex-double-float matrix;
        ;; extract real parts into a double-float matrix for compatibility
        (let* ((shape (magicl:shape vecs))
               (real-vecs (magicl:empty shape :type 'double-float
                                              :layout :column-major)))
          (dotimes (i (first shape))
            (dotimes (j (second shape))
              (setf (magicl:tref real-vecs i j)
                    (coerce (realpart (magicl:tref vecs i j)) 'double-float))))
          `((mlist simp)
            ,(numerics-wrap (numerics:make-ndarray v-vec))
            ,(numerics-wrap (numerics:make-ndarray real-vecs))))))))

(defun $np_qr (a)
  "QR decomposition: np_qr(A) => [Q, R]"
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (multiple-value-bind (q r) (magicl:qr ta)
      `((mlist simp)
        ,(numerics-wrap (numerics:make-ndarray q))
        ,(numerics-wrap (numerics:make-ndarray r))))))

(defun $np_lu (a)
  "LU decomposition: np_lu(A) => [L, U, P]
   L is lower-triangular with unit diagonal.
   U is upper-triangular.
   P is a permutation matrix such that P * A = L * U."
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (m (first (magicl:shape ta)))
         (n (second (magicl:shape ta)))
         (k (min m n)))
    (multiple-value-bind (lu-packed ipiv) (magicl:lu ta)
      ;; Extract L: lower triangle of lu-packed with 1s on diagonal
      (let ((l-mat (magicl:zeros (list m k) :type 'double-float
                                             :layout :column-major))
            (u-mat (magicl:zeros (list k n) :type 'double-float
                                             :layout :column-major))
            (p-mat (magicl:eye m :type 'double-float :layout :column-major)))
        ;; Fill L
        (dotimes (i m)
          (dotimes (j (min (1+ i) k))
            (if (= i j)
                (setf (magicl:tref l-mat i j) 1.0d0)
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
          ,(numerics-wrap (numerics:make-ndarray l-mat))
          ,(numerics-wrap (numerics:make-ndarray u-mat))
          ,(numerics-wrap (numerics:make-ndarray p-mat)))))))

(defun $np_norm (a)
  "Matrix or vector norm: np_norm(A) => scalar.
   For vectors returns the 2-norm; for matrices returns the Frobenius norm."
  (let* ((tensor (numerics:ndarray-tensor (numerics-unwrap a)))
         (shape (magicl:shape tensor)))
    (if (= (length shape) 1)
        ;; Vector: compute 2-norm manually
        (let ((sum 0.0d0)
              (n (first shape)))
          (dotimes (i n)
            (let ((x (magicl:tref tensor i)))
              (incf sum (* x x))))
          (sqrt sum))
        ;; Matrix: Frobenius norm
        (let ((sum 0.0d0)
              (nrow (first shape))
              (ncol (second shape)))
          (dotimes (i nrow)
            (dotimes (j ncol)
              (let ((x (magicl:tref tensor i j)))
                (incf sum (* x x)))))
          (sqrt sum)))))

(defun $np_rank (a)
  "Numerical rank via SVD: np_rank(A)"
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (sigma (nth-value 1 (magicl:svd ta)))
         (s-arr (numerics-svd-values sigma))
         (tol (* (max (first (magicl:shape ta))
                      (second (magicl:shape ta)))
                 double-float-epsilon
                 (reduce #'max s-arr))))
    (count-if (lambda (x) (> x tol)) s-arr)))

(defun $np_trace (a)
  "Matrix trace: np_trace(A) => scalar"
  (magicl:trace (numerics:ndarray-tensor (numerics-unwrap a))))

(defun $np_transpose (a)
  "Transpose: np_transpose(A)"
  (numerics-wrap
   (numerics:make-ndarray
    (magicl:transpose (numerics:ndarray-tensor (numerics-unwrap a))))))

(defun $np_conj (a)
  "Conjugate transpose: np_conj(A)"
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (numerics-wrap
     (numerics:make-ndarray (magicl:conjugate-transpose ta)))))

;;; Matrix exponential via scaling-and-squaring with Pade approximation.
;;; Algorithm follows Higham (2005), "The Scaling and Squaring Method for
;;; the Matrix Exponential Revisited", SIAM J. Matrix Anal. Appl. 26(4).
;;; Implementation matches Julia's LinearAlgebra.exp! (stdlib/LinearAlgebra/src/dense.jl).

(defun expm-norm1 (m n)
  "Compute the 1-norm of an n×n matrix M (max absolute column sum)."
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
         (eye (magicl:eye n :type 'double-float :layout :column-major))
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
  (let ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (handler-case
        (numerics-wrap (numerics:make-ndarray (expm-pade ta)))
      (error (e)
        (merror "np_expm: ~A" e)))))

(defun $np_lstsq (a b)
  "Least-squares solution: np_lstsq(A, b)
   Solves min ||Ax - b||_2 via SVD."
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a)))
         (tb (numerics:ndarray-tensor (numerics-unwrap b))))
    (multiple-value-bind (u sigma vt) (magicl:svd ta)
      (let* ((ut (magicl:transpose u))
             (v (magicl:transpose vt))
             (utb (magicl:@ ut tb))
             (s-arr (numerics-svd-values sigma))
             (tol (* (reduce #'max s-arr) double-float-epsilon
                     (max (first (magicl:shape ta))
                          (second (magicl:shape ta)))))
             (k (length s-arr))
             (n (second (magicl:shape ta)))   ;; columns of A
             (p (second (magicl:shape utb)))) ;; columns of b
        ;; Build n×p result: first k rows scaled by S^{-1}, rest zero
        (let ((sinv-utb (magicl:zeros (list n p) :type 'double-float
                                                  :layout :column-major)))
          (dotimes (i (min k n))
            (let ((si (aref s-arr i)))
              (dotimes (j p)
                (setf (magicl:tref sinv-utb i j)
                      (if (> si tol)
                          (* (/ 1.0d0 si) (magicl:tref utb i j))
                          0.0d0)))))
          (numerics-wrap
           (numerics:make-ndarray (magicl:@ v sinv-utb))))))))

(defun $np_pinv (a)
  "Moore-Penrose pseudo-inverse: np_pinv(A)
   Computed via SVD: A+ = V S+ Ut"
  (let* ((ta (numerics:ndarray-tensor (numerics-unwrap a))))
    (multiple-value-bind (u sigma vt) (magicl:svd ta)
      (let* ((v (magicl:transpose vt))
             (ut (magicl:transpose u))
             (s-arr (numerics-svd-values sigma))
             (tol (* (reduce #'max s-arr) double-float-epsilon
                     (max (first (magicl:shape ta))
                          (second (magicl:shape ta)))))
             (k (length s-arr))
             (m (first (magicl:shape ut))))
        ;; Build diagonal S+ matrix
        (let ((s-plus (magicl:zeros (list k m) :type 'double-float
                                               :layout :column-major)))
          (dotimes (i k)
            (let ((si (aref s-arr i)))
              (when (> si tol)
                (setf (magicl:tref s-plus i i) (/ 1.0d0 si)))))
          (numerics-wrap
           (numerics:make-ndarray (magicl:@ v s-plus ut))))))))
