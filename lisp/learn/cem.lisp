;;; cem.lisp — Cross-Entropy Method optimization
;;;
;;; np_cem(cost_fn, n_params, ...) → [best_params, cost_history]

(in-package #:maxima)

(defvar *cem-defaults*
  `(($n_samples . 50)
    ($n_elites  . 10)
    ($n_gens    . 25)
    ($sigma0    . 1.0d0)
    ($sigma_min . 0.1d0)
    ($mu0       . nil))
  "Default options for np_cem.")

(defun $np_cem (cost-fn n-params &rest option-args)
  "Cross-Entropy Method for black-box optimization.
   cost_fn(params_ndarray) → scalar (lower is better).
   Returns [best_params_ndarray, cost_history_ndarray]."
  (unless (numerics-callable-p cost-fn)
    (merror "np_cem: first argument must be a callable function"))
  (let* ((opts (parse-learn-options option-args *cem-defaults*))
         (n-samples (truncate (learn-opt opts '$n_samples)))
         (n-elites  (truncate (learn-opt opts '$n_elites)))
         (n-gens    (truncate (learn-opt opts '$n_gens)))
         (sigma0    (coerce ($float (learn-opt opts '$sigma0)) 'double-float))
         (sigma-min (coerce ($float (learn-opt opts '$sigma_min)) 'double-float))
         (mu0-arg   (learn-opt opts '$mu0))
         (np        (truncate n-params))
         ;; Initialize mu
         (mu (make-array np :element-type 'double-float :initial-element 0.0d0))
         ;; Initialize sigma
         (sigma (make-array np :element-type 'double-float :initial-element sigma0))
         ;; Cost history (best cost per generation)
         (history (make-array n-gens :element-type 'double-float :initial-element 0.0d0))
         ;; Pre-allocate callback ndarray (reused for each cost evaluation)
         (cb-tensor (magicl:empty (list np) :type 'double-float :layout :column-major))
         (cb-handle (numerics:make-ndarray cb-tensor))
         (cb-wrapped (numerics-wrap cb-handle))
         (cb-storage (numerics-tensor-storage cb-tensor))
         ;; Sample storage: n_samples x np
         (samples (make-array (* n-samples np) :element-type 'double-float))
         ;; Cost storage
         (costs (make-array n-samples :element-type 'double-float))
         (indices (make-array n-samples :element-type 'fixnum))
         ;; Best tracking
         (best-cost most-positive-double-float)
         (best-params (make-array np :element-type 'double-float)))
    ;; Parse mu0 if provided
    (when mu0-arg
      (let* ((mu0-handle (numerics-unwrap mu0-arg))
             (mu0-tensor (numerics:ndarray-tensor mu0-handle))
             (mu0-storage (numerics-tensor-storage mu0-tensor)))
        (unless (= (length mu0-storage) np)
          (merror "np_cem: mu0 size (~D) must match n_params (~D)"
                  (length mu0-storage) np))
        (replace mu mu0-storage)))
    ;; Main CEM loop
    (dotimes (gen n-gens)
      ;; Initialize index array
      (dotimes (i n-samples) (setf (aref indices i) i))
      ;; Generate all random samples for this generation via magicl:random-normal.
      ;; This uses *random-state* (set by np_seed) for reproducibility.
      (let* ((noise-tensor (magicl:random-normal (list n-samples np)
                                                  :type 'double-float))
             (noise-storage (numerics-tensor-storage noise-tensor)))
        ;; Build samples: mu + sigma * noise, evaluate costs
        (dotimes (i n-samples)
          (let ((offset (* i np)))
            (dotimes (j np)
              ;; noise-tensor is column-major: element (i,j) at index j*n_samples+i
              (let* ((z (aref noise-storage (+ (* j n-samples) i)))
                     (val (+ (aref mu j) (* (aref sigma j) z))))
                (setf (aref samples (+ offset j)) val)
                (setf (aref cb-storage j) val)))
            ;; Evaluate cost with error handling
            (let ((cost (handler-case
                            (coerce ($float (np-learn-call cost-fn cb-wrapped))
                                    'double-float)
                          (error () most-positive-double-float))))
              (setf (aref costs i) cost)
              ;; Track overall best
              (when (< cost best-cost)
                (setf best-cost cost)
                (replace best-params cb-storage))))))
      ;; Sort indices by cost (ascending)
      (sort indices (lambda (a b) (< (aref costs a) (aref costs b))))
      ;; Record best cost this generation
      (setf (aref history gen) (aref costs (aref indices 0)))
      ;; Compute elite mean
      (let ((new-mu (make-array np :element-type 'double-float :initial-element 0.0d0))
            (inv-elites (/ 1.0d0 (coerce n-elites 'double-float))))
        (dotimes (e n-elites)
          (let ((offset (* (aref indices e) np)))
            (dotimes (j np)
              (incf (aref new-mu j) (aref samples (+ offset j))))))
        (dotimes (j np)
          (setf (aref new-mu j) (* (aref new-mu j) inv-elites)))
        ;; Compute elite std
        (let ((new-sigma (make-array np :element-type 'double-float :initial-element 0.0d0)))
          (dotimes (e n-elites)
            (let ((offset (* (aref indices e) np)))
              (dotimes (j np)
                (let ((diff (- (aref samples (+ offset j)) (aref new-mu j))))
                  (incf (aref new-sigma j) (* diff diff))))))
          (dotimes (j np)
            (setf (aref new-sigma j)
                  (max sigma-min (sqrt (* (aref new-sigma j) inv-elites)))))
          ;; Update mu and sigma
          (replace mu new-mu)
          (replace sigma new-sigma))))
    ;; Build result
    (let* ((result-tensor (magicl:empty (list np) :type 'double-float
                                                  :layout :column-major))
           (result-storage (numerics-tensor-storage result-tensor)))
      (replace result-storage best-params)
      (let* ((result-handle (numerics:make-ndarray result-tensor))
             (result-wrapped (numerics-wrap result-handle))
             (hist-tensor (magicl:from-list (coerce history 'list)
                                            (list n-gens)
                                            :type 'double-float
                                            :layout :column-major))
             (hist-handle (numerics:make-ndarray hist-tensor))
             (hist-wrapped (numerics-wrap hist-handle)))
        `((mlist) ,result-wrapped ,hist-wrapped)))))
