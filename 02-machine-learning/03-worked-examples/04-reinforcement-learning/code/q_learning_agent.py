"""A tabular TD-learning agent: Q-learning (off-policy) and SARSA (on-policy), one class.

Companion module for:
  02-machine-learning/03-worked-examples/04-reinforcement-learning/01-reinforcement-learning-and-self-play.md

Both update rules share everything except one line -- what value they bootstrap the target
off of -- which is the whole point of implementing them side by side instead of as two
unrelated scripts. Both are quoted, verbatim in spirit, from Sutton & Barto (2018),
*Reinforcement Learning: An Introduction*, 2nd ed., Chapter 6 (NOTE-ML-15-2):

    Q-learning (off-policy), eq. 6.8:
        Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * max_a' Q(S_{t+1}, a') - Q(S_t, A_t)]

    SARSA (on-policy), eq. 6.7:
        Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

Q-learning bootstraps off the *best possible* action in the next state, regardless of what
the agent actually does next (off-policy: it learns about the greedy policy while behaving
epsilon-greedy). SARSA bootstraps off the action the agent *actually takes* next under its
current (epsilon-greedy) policy (on-policy: what it learns about is exactly what it does).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chess_rl_env import CornerCaptureEnv, N_ACTIONS


@dataclass
class TrainingHistory:
    """Per-episode series collected during training, for plotting and inspection."""

    episode_rewards: list[float] = field(default_factory=list)
    epsilon_values: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    episode_wins: list[bool] = field(default_factory=list)


class TabularTDAgent:
    """A dict-based Q-table: `Q[state]` is an 8-entry array, one value per compass action.

    `state` is whatever hashable object the environment hands back -- here, the
    `(king_square, target_square)` tuple from `CornerCaptureEnv.state`. New states are
    initialised to all-zero on first visit (an "optimistic-enough" start given every reward
    in this environment is <= 0 except the terminal capture).
    """

    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        update_rule: str = "q_learning",
        seed: int = 42,
    ) -> None:
        if update_rule not in ("q_learning", "sarsa"):
            raise ValueError(f"update_rule must be 'q_learning' or 'sarsa', got {update_rule!r}")
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_rule = update_rule
        self.rng = np.random.default_rng(seed)
        self.q: dict[tuple[int, int], np.ndarray] = {}

    def _row(self, state: tuple[int, int]) -> np.ndarray:
        """Return this state's 8-entry Q-row, creating it (all zeros) on first visit."""
        row = self.q.get(state)
        if row is None:
            row = np.zeros(self.n_actions, dtype=np.float64)
            self.q[state] = row
        return row

    def select_action(self, state: tuple[int, int], greedy: bool = False) -> int:
        """Epsilon-greedy: with probability epsilon, act randomly (explore); otherwise take
        the current best-known action (exploit). `greedy=True` forces pure exploitation --
        used at evaluation time, after training, when we want to see what the agent learned,
        not how it explored.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self._row(state)))

    def greedy_action(self, env: CornerCaptureEnv, rng: np.random.Generator) -> int:
        """Adapter matching `policies.Policy`'s call shape, so a trained agent can be scored
        by the exact same `evaluate_policy` harness as the two baseline policies.
        """
        del rng  # the trained agent's own rng is not used for greedy evaluation
        return self.select_action(env.state, greedy=True)

    def q_values(self, state: tuple[int, int]) -> np.ndarray:
        """Read-only lookup of a state's learned Q-row: a copy, so callers (plotting,
        inspection) can't accidentally mutate the live table. Unvisited states read as all
        zeros -- their true value, since this agent never visited them.
        """
        row = self.q.get(state)
        return np.zeros(self.n_actions, dtype=np.float64) if row is None else row.copy()

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        next_action: int | None,
        done: bool,
    ) -> float:
        """One TD update. Returns the TD error (the "surprise") for logging/inspection.

        `next_action` is only read by SARSA; Q-learning ignores it and looks up the max over
        the next state's whole row instead -- that one line is the entire off-policy vs
        on-policy difference between the two update rules.
        """
        row = self._row(state)
        if done:
            target = reward  # no successor state -- the episode ended right here
        elif self.update_rule == "q_learning":
            target = reward + self.gamma * float(np.max(self._row(next_state)))
        else:  # sarsa
            assert next_action is not None, "SARSA needs the actual next action"
            target = reward + self.gamma * float(self._row(next_state)[next_action])
        td_error = target - row[action]
        row[action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(
    agent: TabularTDAgent,
    env: CornerCaptureEnv,
    n_episodes: int,
    env_rng: np.random.Generator,
) -> TrainingHistory:
    """Run `n_episodes` of training, updating `agent` in place; return the per-episode
    history used for the learning-curve artefact.

    The loop always pre-selects the *next* action before updating -- irrelevant to
    Q-learning's target (it re-derives max_a' from the row directly) but exactly what SARSA
    needs, since SARSA's target is the Q-value of the specific action about to be taken.
    Sharing one loop for both update rules is deliberate: it makes the on-policy/off-policy
    difference visible as a one-line change in `update()`, not two divergent code paths.
    """
    history = TrainingHistory()
    for _ in range(n_episodes):
        state = env.reset(env_rng)
        action = agent.select_action(state)
        episode_reward = 0.0
        steps = 0
        won = False
        done = False
        while not done:
            result = env.step(action)
            episode_reward += result.reward
            steps += 1
            done = result.done
            won = result.won
            next_state = result.state
            next_action = None if done else agent.select_action(next_state)
            agent.update(state, action, result.reward, next_state, next_action, done)
            state, action = next_state, next_action if next_action is not None else action
        agent.decay_epsilon()
        history.episode_rewards.append(episode_reward)
        history.epsilon_values.append(agent.epsilon)
        history.episode_steps.append(steps)
        history.episode_wins.append(won)
    return history
