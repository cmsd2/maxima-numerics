;;; rollout.lisp — Episode collection for reinforcement learning
;;;
;;; np_rollout(step_fn, policy_fn, s0, horizon) → [states, actions, rewards, length]

(in-package #:maxima)

(defun $np_rollout (step-fn policy-fn s0 horizon)
  "Collect an episode by running a policy in an environment.
   step_fn(state, action) → [next_state, reward, finished]
   policy_fn(state) → action
   Returns [states_list, actions_list, rewards_ndarray, actual_length]."
  (unless (numerics-callable-p step-fn)
    (merror "np_rollout: step_fn must be a callable function"))
  (unless (numerics-callable-p policy-fn)
    (merror "np_rollout: policy_fn must be a callable function"))
  (let* ((max-steps (truncate horizon))
         (states (list s0))
         (actions nil)
         (rewards nil)
         (state s0)
         (actual-len 0))
    (dotimes (t-step max-steps)
      ;; Get action from policy
      (let ((action (np-learn-call policy-fn state)))
        (push action actions)
        ;; Step the environment
        (let* ((result (np-learn-call step-fn state action))
               (next-state (if ($listp result) (second result) (merror "np_rollout: step_fn must return [next_state, reward, done]")))
               (reward (if ($listp result) (third result) 0))
               (done-raw (if ($listp result) (fourth result) nil)))
          (push next-state states)
          (push (coerce ($float reward) 'double-float) rewards)
          (incf actual-len)
          (setf state next-state)
          ;; Check termination
          (when (np-learn-done-p done-raw)
            (return)))))
    ;; Build results
    ;; States and actions as Maxima lists (may be non-numeric)
    (let* ((states-mlist (cons '(mlist) (nreverse states)))
           (actions-mlist (cons '(mlist) (nreverse actions)))
           ;; Rewards as ndarray
           (rewards-list (nreverse rewards))
           (rewards-tensor (magicl:from-list rewards-list
                                             (list actual-len)
                                             :type 'double-float
                                             :layout :column-major))
           (rewards-handle (numerics:make-ndarray rewards-tensor))
           (rewards-wrapped (numerics-wrap rewards-handle)))
      `((mlist) ,states-mlist ,actions-mlist ,rewards-wrapped ,actual-len))))
