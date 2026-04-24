# numerics-learn — RL Algorithm Shells

Generic reinforcement learning algorithm shells for Maxima. Depends only on
`numerics/core` — no external solver dependencies.

Load with:
```maxima
load("numerics")$
load("numerics-learn")$
```

## Functions

### `np_cem(cost_fn, n_params, ...)` — Cross-Entropy Method

Black-box optimization via iterative sampling and elite selection.

```maxima
sphere(x) := np_sum(np_pow(x, 2))$
[best, history] : np_cem(sphere, 2, n_samples=50, n_elites=10, n_gens=30)$
```

**Arguments:**
- `cost_fn` — callable `f(params_ndarray) → scalar` (lower is better)
- `n_params` — dimensionality of the parameter vector

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `n_samples` | 50 | Samples per generation |
| `n_elites` | 10 | Number of elite samples |
| `n_gens` | 25 | Number of generations |
| `sigma0` | 1.0 | Initial standard deviation |
| `sigma_min` | 0.1 | Minimum standard deviation (prevents collapse) |
| `mu0` | zeros | Initial mean (ndarray, or omit for zeros) |

**Returns:** `[best_params_ndarray, cost_history_ndarray]`

Failed cost evaluations receive +infinity (robustness via `handler-case`).

---

### `np_rollout(step_fn, policy_fn, s0, horizon)` — Episode Collection

Collects a single episode by running a policy in an environment.

```maxima
my_step(s, a) := [s + a, -s^2, is(abs(s) > 10)]$
my_policy(s) := -0.5 * s$
[states, actions, rewards, len] : np_rollout(my_step, my_policy, 1.0, 200)$
returns : np_discount(rewards, 0.99)$
```

**Arguments:**
- `step_fn(state, action) → [next_state, reward, finished]`
- `policy_fn(state) → action`
- `s0` — initial state (any Maxima value)
- `horizon` — maximum number of steps

**Returns:** `[states_list, actions_list, rewards_ndarray, actual_length]`

States and actions remain as Maxima lists (they may be non-numeric).
Rewards become an ndarray, ready for `np_discount`.

---

### `np_qlearn(step_fn, n_states, n_actions, ...)` — Tabular Q-Learning

Trains a Q-table via temporal-difference learning with epsilon-greedy exploration.

```maxima
grid_step(s, a) := block([...], [next_s, reward, is_done])$
[Q, ep_rewards, ep_lengths] : np_qlearn(grid_step, 25, 4, n_episodes=500)$
```

**Arguments:**
- `step_fn(state, action) → [next_state, reward, finished]` (states/actions are integers, 0-indexed)
- `n_states` — number of discrete states
- `n_actions` — number of discrete actions

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `n_episodes` | 500 | Training episodes |
| `alpha` | 0.1 | Learning rate |
| `discount` | 0.99 | Discount factor |
| `epsilon` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.995 | Multiplicative decay per episode |
| `epsilon_min` | 0.01 | Minimum exploration rate |
| `max_steps` | 200 | Maximum steps per episode |
| `start_state` | 0 | Initial state (integer or callable returning integer) |

**Returns:** `[Q_ndarray, episode_rewards_ndarray, episode_lengths_ndarray]`

The Q-table is an `n_states × n_actions` ndarray. Bellman updates operate
directly on the magicl tensor (no Maxima evaluator overhead). Only the
`step_fn` call crosses into Maxima.

## Option Syntax

Options use Maxima's `key=value` syntax:

```maxima
np_cem(my_cost, 3, n_samples=100, n_elites=20)
np_qlearn(my_step, 25, 4, epsilon=0.5, n_episodes=1000)
```

## Design Notes

**Why not `np_reinforce`?** REINFORCE has too many interacting callbacks
(policy mean, score function, noise injection, action clipping, initial state
sampler). The symbolic `diff()` step for the score function is the key Maxima
selling point — hiding it in a library removes the educational value. Instead,
use `np_rollout` + `np_discount` + a few lines of user code.
