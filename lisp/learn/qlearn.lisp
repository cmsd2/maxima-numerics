;;; qlearn.lisp — Tabular Q-Learning
;;;
;;; np_qlearn(step_fn, n_states, n_actions, ...) → [Q, episode_rewards, episode_lengths]

(in-package #:maxima)

(defvar *qlearn-defaults*
  `(($n_episodes   . 500)
    ($alpha        . 0.1d0)
    ($discount     . 0.99d0)
    ($epsilon      . 1.0d0)
    ($epsilon_decay . 0.995d0)
    ($epsilon_min  . 0.01d0)
    ($max_steps    . 200)
    ($start_state  . 0))
  "Default options for np_qlearn.")

(defun $np_qlearn (step-fn n-states n-actions &rest option-args)
  "Tabular Q-Learning.
   step_fn(state, action) → [next_state, reward, finished]
   States and actions are integers (0-indexed).
   Returns [Q_ndarray, episode_rewards_ndarray, episode_lengths_ndarray]."
  (unless (numerics-callable-p step-fn)
    (merror "np_qlearn: step_fn must be a callable function"))
  (let* ((opts (parse-learn-options option-args *qlearn-defaults*))
         (ns (truncate n-states))
         (na (truncate n-actions))
         (n-episodes  (truncate (learn-opt opts '$n_episodes)))
         (alpha       (coerce ($float (learn-opt opts '$alpha)) 'double-float))
         (gamma       (coerce ($float (learn-opt opts '$discount)) 'double-float))
         (epsilon     (coerce ($float (learn-opt opts '$epsilon)) 'double-float))
         (eps-decay   (coerce ($float (learn-opt opts '$epsilon_decay)) 'double-float))
         (eps-min     (coerce ($float (learn-opt opts '$epsilon_min)) 'double-float))
         (max-steps   (truncate (learn-opt opts '$max_steps)))
         (s0-arg      (learn-opt opts '$start_state))
         (s0-callable (numerics-callable-p s0-arg))
         ;; Q-table as magicl tensor (n_states x n_actions), direct access
         (q-tensor (magicl:empty (list ns na) :type 'double-float :layout :column-major))
         ;; Episode tracking
         (ep-rewards (make-array n-episodes :element-type 'double-float :initial-element 0.0d0))
         (ep-lengths (make-array n-episodes :element-type 'fixnum :initial-element 0)))
    ;; Main training loop
    (dotimes (ep n-episodes)
      ;; Get initial state
      (let* ((state (if s0-callable
                        (truncate ($float (np-learn-call s0-arg)))
                        (truncate s0-arg)))
             (ep-reward 0.0d0)
             (ep-len 0))
        ;; Episode loop
        (dotimes (step max-steps)
          ;; Epsilon-greedy action selection (operates in CL, no Maxima overhead).
          ;; CL random uses *random-state*, controlled by np_seed for reproducibility.
          (let ((action (if (< (random 1.0d0) epsilon)
                            (random na)
                            ;; Greedy: argmax_a Q(s,a)
                            (let ((best-a 0)
                                  (best-q (magicl:tref q-tensor state 0)))
                              (loop for a from 1 below na
                                    for q-val = (magicl:tref q-tensor state a)
                                    when (> q-val best-q)
                                    do (setf best-a a best-q q-val))
                              best-a))))
            ;; Step environment (crosses into Maxima)
            (let* ((result (np-learn-call step-fn state action))
                   (next-state (truncate ($float (second result))))
                   (reward (coerce ($float (third result)) 'double-float))
                   (done-raw (fourth result))
                   (done (np-learn-done-p done-raw)))
              ;; Bellman update (operates on magicl tensor directly)
              (let* ((old-q (magicl:tref q-tensor state action))
                     (max-next-q (if done
                                     0.0d0
                                     (let ((best (magicl:tref q-tensor next-state 0)))
                                       (loop for a from 1 below na
                                             for q-val = (magicl:tref q-tensor next-state a)
                                             when (> q-val best)
                                             do (setf best q-val))
                                       best)))
                     (target (+ reward (* gamma max-next-q)))
                     (new-q (+ old-q (* alpha (- target old-q)))))
                (setf (magicl:tref q-tensor state action) new-q))
              ;; Accumulate episode stats
              (incf ep-reward reward)
              (incf ep-len)
              (setf state next-state)
              ;; Check termination
              (when done (return)))))
        ;; Record episode stats
        (setf (aref ep-rewards ep) ep-reward)
        (setf (aref ep-lengths ep) ep-len)
        ;; Decay epsilon
        (setf epsilon (max eps-min (* epsilon eps-decay)))))
    ;; Build result
    (let* ((q-handle (numerics:make-ndarray q-tensor))
           (q-wrapped (numerics-wrap q-handle))
           (rew-tensor (magicl:from-list (coerce ep-rewards 'list)
                                         (list n-episodes)
                                         :type 'double-float
                                         :layout :column-major))
           (rew-handle (numerics:make-ndarray rew-tensor))
           (rew-wrapped (numerics-wrap rew-handle))
           (len-tensor (magicl:from-list (mapcar (lambda (x) (coerce x 'double-float))
                                                 (coerce ep-lengths 'list))
                                         (list n-episodes)
                                         :type 'double-float
                                         :layout :column-major))
           (len-handle (numerics:make-ndarray len-tensor))
           (len-wrapped (numerics-wrap len-handle)))
      `((mlist) ,q-wrapped ,rew-wrapped ,len-wrapped))))
