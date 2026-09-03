"""Fixed baseline policies for the Corner Capture environment, and an evaluation harness.

Companion module for:
  02-machine-learning/03-worked-examples/04-reinforcement-learning/01-reinforcement-learning-and-self-play.md

Two policies, deliberately dumb in different ways:

- `random_policy` -- zero thinking, uniform over the 8 compass directions. About half its
  attempts are illegal (off the 4x4 arena, or into a square the Rook attacks) and eat the
  -2 penalty.
- `greedy_taker_policy` -- reimplements, for this environment, the idea behind
  `kaggle_chess/the_taker.py`: a materially-greedy bot with no lookahead. Here "material" is
  replaced by "distance to the only capturable piece": always step in whichever compass
  direction shortens the King-move (Chebyshev) distance to the target the most. It never
  consults the environment's real legal-move check, so it can walk straight at a square the
  Rook defends and get rejected every single time it tries -- proximity is not the same
  thing as a legal, winning move.

Both policies share the `Policy` calling convention `(env, rng) -> action:int` so
`evaluate_policy` below can score a hand-coded policy or a trained agent's greedy action
selection with the exact same harness.
"""
from __future__ import annotations

from typing import Callable, Protocol

import chess
import numpy as np

from chess_rl_env import CornerCaptureEnv, DIRECTIONS, N_ACTIONS, chebyshev_distance


class Policy(Protocol):
    def __call__(self, env: CornerCaptureEnv, rng: np.random.Generator) -> int: ...


def random_policy(env: CornerCaptureEnv, rng: np.random.Generator) -> int:
    """Pick one of the 8 compass directions uniformly at random.

    Does not call `env.legal_directions()` at all -- it is the "no policy" baseline every
    other approach (heuristic or learned) has to beat.
    """
    del env  # unused: random_policy ignores the state entirely
    return int(rng.integers(0, N_ACTIONS))


def greedy_taker_policy(env: CornerCaptureEnv, rng: np.random.Generator) -> int:
    """Always step toward the target by shortest King-move distance -- no rules awareness.

    For each of the 8 compass directions, compute where it would land (clipped to the real
    8x8 board only -- this policy does not even know about the 4x4 arena boundary) and its
    Chebyshev distance to the target square; return the direction with the smallest
    resulting distance. Ties keep the first (N, NE, E, SE, S, SW, W, NW) direction found, so
    the policy is deterministic given a state.

    This is the "material eval with no search" idea from `kaggle_chess/the_taker.py`,
    reimplemented from scratch for a King-vs-target task: minimize distance now, and never
    ask whether the move is legal.
    """
    del rng  # unused: greedy_taker_policy is a deterministic function of state
    king_sq, target_sq = env.king_sq, env.target_sq
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    best_action = 0
    best_distance: int | None = None
    for action, (df, dr) in enumerate(DIRECTIONS):
        f, r = kf + df, kr + dr
        if not (0 <= f < 8 and 0 <= r < 8):
            continue  # off the real board -- even a naive bot won't try this
        dest = chess.square(f, r)
        distance = chebyshev_distance(dest, target_sq)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_action = action
    return best_action


def evaluate_policy(
    policy: Policy,
    env: CornerCaptureEnv,
    rng: np.random.Generator,
    n_episodes: int,
) -> dict[str, float]:
    """Run `policy` for `n_episodes` fresh episodes; return win rate, average reward, average
    steps-to-done. `policy` may be one of the two functions above, or `agent.greedy_action`
    from a trained `TabularTDAgent` (q_learning_agent.py) -- both share the same call shape.
    """
    total_reward = 0.0
    total_steps = 0
    wins = 0
    for _ in range(n_episodes):
        env.reset(rng)
        done = False
        episode_reward = 0.0
        steps = 0
        won = False
        while not done:
            action = policy(env, rng)
            result = env.step(action)
            episode_reward += result.reward
            steps += 1
            done = result.done
            won = result.won
        total_reward += episode_reward
        total_steps += steps
        wins += int(won)
    return {
        "win_rate": wins / n_episodes,
        "avg_reward": total_reward / n_episodes,
        "avg_steps": total_steps / n_episodes,
    }
