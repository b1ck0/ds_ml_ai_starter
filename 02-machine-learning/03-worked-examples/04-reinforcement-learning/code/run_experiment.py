"""Train Q-learning and SARSA on the Corner Capture chess environment, evaluate both
against random and greedy-taker baselines, and generate this chapter's three artefacts.

Companion module for:
  02-machine-learning/03-worked-examples/04-reinforcement-learning/01-reinforcement-learning-and-self-play.md

Run it from the repo root with the pinned `.venv-ml` environment
(chess==1.11.2, numpy==2.5.2, matplotlib==3.11.1 -- NOTE-ML-15-1):

    .venv-ml/Scripts/python.exe \
        "02-machine-learning/03-worked-examples/04-reinforcement-learning/code/run_experiment.py"

Everything is seeded (SEED = 42) so the printed run log and the three PNGs in
`../artefacts/` reproduce exactly on any machine running the pinned versions.
"""
from __future__ import annotations

from pathlib import Path

import chess
import matplotlib

matplotlib.use("Agg")  # headless: this script only ever saves PNGs, never opens a window
import matplotlib.pyplot as plt
import numpy as np

from chess_rl_env import FILES, RANKS, DIRECTIONS, CornerCaptureEnv, N_ACTIONS, REGION_SQUARES
from policies import evaluate_policy, greedy_taker_policy, random_policy
from q_learning_agent import TabularTDAgent, TrainingHistory, train

ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"
SEED = 42
N_TRAIN_EPISODES = 3000
N_EVAL_EPISODES = 500
ROLLING_WINDOW = 50


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    """Simple moving average over `window` episodes -- per-episode reward is noisy (each
    episode has a different random King/Rook start), so the plotted curve needs smoothing
    to show the underlying trend instead of a scribble.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def log_progress(label: str, history: TrainingHistory) -> None:
    """Print reward and epsilon at six evenly-spaced checkpoints through training, so the
    run log itself (not just the plot) shows reward rising and epsilon decaying."""
    n = len(history.episode_rewards)
    checkpoints = sorted({max(1, n // 6), n // 3, n // 2, (2 * n) // 3, (5 * n) // 6, n})
    for cp in checkpoints:
        window = history.episode_rewards[max(0, cp - 100):cp]
        print(
            f"  {label:<10} episode {cp:5d}/{n}  "
            f"avg_reward(last 100)={np.mean(window):+7.2f}  "
            f"epsilon={history.epsilon_values[cp - 1]:.3f}"
        )


def plot_reward_curve(q_history: TrainingHistory, sarsa_history: TrainingHistory) -> Path:
    """Artefact 1: reward-per-episode learning curve (both algorithms) plus epsilon decay,
    stacked so the two are visibly correlated in time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    q_roll = rolling_mean(q_history.episode_rewards, ROLLING_WINDOW)
    sarsa_roll = rolling_mean(sarsa_history.episode_rewards, ROLLING_WINDOW)
    ax1.plot(np.arange(len(q_roll)) + ROLLING_WINDOW, q_roll,
              label="Q-learning (off-policy)", color="#1f77b4")
    ax1.plot(np.arange(len(sarsa_roll)) + ROLLING_WINDOW, sarsa_roll,
              label="SARSA (on-policy)", color="#ff7f0e")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_ylabel(f"reward / episode\n({ROLLING_WINDOW}-ep rolling mean)")
    ax1.set_title("Learning curve -- reward rises as the agent learns to route around the "
                   "Rook's attack lines")
    ax1.legend(loc="lower right")

    ax2.plot(q_history.epsilon_values, color="#2ca02c")
    ax2.set_ylabel("epsilon\n(explore probability)")
    ax2.set_xlabel("training episode")
    ax2.set_title("epsilon decay -- exploration fades as the agent commits to what it learned")

    fig.tight_layout()
    out = ARTEFACTS_DIR / "01_reward_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_learned_policy(agent: TabularTDAgent, target_sq: int) -> Path:
    """Artefact 2: the learned Q-table, visualised for one fixed target square -- an arrow
    per King square showing argmax_a Q(s,a), coloured by max_a Q(s,a) (the learned state
    value). This is the "inspect the learned Q-values" artefact the spec asks for: instead
    of dumping 1,920 raw numbers, show the policy they imply.
    """
    tf, tr = chess.square_file(target_sq), chess.square_rank(target_sq)

    xs, ys, us, vs, values = [], [], [], [], []
    for sq in REGION_SQUARES:
        if sq == target_sq:
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        row = agent.q_values((sq, target_sq))
        best_action = int(np.argmax(row))
        df, dr = DIRECTIONS[best_action]
        xs.append(f)
        ys.append(r)
        us.append(df * 0.6)
        vs.append(dr * 0.6)
        values.append(float(np.max(row)))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    scatter = ax.scatter(xs, ys, c=values, cmap="viridis", s=900, marker="s",
                          edgecolors="black", zorder=2)
    ax.quiver(xs, ys, us, vs, color="white", angles="xy", scale_units="xy", scale=1,
              width=0.012, zorder=3)
    ax.scatter([tf], [tr], marker="*", s=1000, color="red", edgecolors="black",
               zorder=4, label=f"target Rook ({chess.square_name(target_sq)})")
    ax.set_xticks(range(FILES))
    ax.set_xticklabels(["a", "b", "c", "d"])
    ax.set_yticks(range(RANKS))
    ax.set_yticklabels(["1", "2", "3", "4"])
    ax.set_xlim(-0.7, FILES - 0.3)
    ax.set_ylim(-0.7, RANKS - 0.3)
    ax.set_xlabel("file")
    ax.set_ylabel("rank")
    ax.set_title(
        f"Learned greedy policy, target fixed at {chess.square_name(target_sq)}\n"
        "arrow = argmax_a Q(s,a)  |  colour = max_a Q(s,a) (learned state value)",
        pad=14,
    )
    fig.colorbar(scatter, ax=ax, label="max_a Q(s,a)", shrink=0.85)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    fig.tight_layout()
    out = ARTEFACTS_DIR / "02_learned_policy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_evaluation_comparison(results: dict[str, dict[str, float]]) -> Path:
    """Artefact 3: win rate / avg reward / avg steps, all four policies side by side --
    the "evaluated vs. random/greedy" evidence the spec's outline section 6 asks for.
    """
    names = list(results.keys())
    colors = ["#7f7f7f", "#d62728", "#1f77b4", "#ff7f0e"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].bar(names, [results[n]["win_rate"] for n in names], color=colors)
    axes[0].set_title("win rate")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(names, [results[n]["avg_reward"] for n in names], color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("avg. reward / episode")

    axes[2].bar(names, [results[n]["avg_steps"] for n in names], color=colors)
    axes[2].set_title("avg. steps to done")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Q-learning / SARSA vs. baselines ({N_EVAL_EPISODES} eval episodes each, "
                  "epsilon=0)")
    fig.tight_layout()
    out = ARTEFACTS_DIR / "03_evaluation_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    env = CornerCaptureEnv()

    n_states = len(REGION_SQUARES) * (len(REGION_SQUARES) - 1)
    print(
        f"State space: {len(REGION_SQUARES)} King squares x {len(REGION_SQUARES) - 1} "
        f"remaining Rook squares = {n_states} states x {N_ACTIONS} actions = "
        f"{n_states * N_ACTIONS} (state, action) table entries"
    )

    print(f"\n=== Training Q-learning (off-policy), {N_TRAIN_EPISODES} episodes ===")
    q_agent = TabularTDAgent(update_rule="q_learning", seed=SEED)
    q_history = train(q_agent, env, N_TRAIN_EPISODES, np.random.default_rng(SEED))
    log_progress("Q-learning", q_history)

    print(f"\n=== Training SARSA (on-policy), {N_TRAIN_EPISODES} episodes ===")
    sarsa_agent = TabularTDAgent(update_rule="sarsa", seed=SEED)
    sarsa_history = train(sarsa_agent, env, N_TRAIN_EPISODES, np.random.default_rng(SEED))
    log_progress("SARSA", sarsa_history)

    print(f"\n=== Evaluating (greedy, epsilon=0), {N_EVAL_EPISODES} episodes each ===")
    results = {
        "random": evaluate_policy(
            random_policy, env, np.random.default_rng(SEED + 10), N_EVAL_EPISODES),
        "greedy-taker": evaluate_policy(
            greedy_taker_policy, env, np.random.default_rng(SEED + 11), N_EVAL_EPISODES),
        "Q-learning": evaluate_policy(
            q_agent.greedy_action, env, np.random.default_rng(SEED + 12), N_EVAL_EPISODES),
        "SARSA": evaluate_policy(
            sarsa_agent.greedy_action, env, np.random.default_rng(SEED + 13), N_EVAL_EPISODES),
    }
    for name, stats in results.items():
        print(
            f"  {name:<14} win_rate={stats['win_rate']:.3f}  "
            f"avg_reward={stats['avg_reward']:+7.2f}  avg_steps={stats['avg_steps']:.2f}"
        )

    assert results["Q-learning"]["win_rate"] > results["random"]["win_rate"], \
        "Q-learning must beat the random baseline's win rate"
    assert results["Q-learning"]["avg_reward"] > results["greedy-taker"]["avg_reward"], \
        "Q-learning must beat the greedy-taker baseline's average reward"

    p1 = plot_reward_curve(q_history, sarsa_history)
    p2 = plot_learned_policy(q_agent, target_sq=chess.D4)
    p3 = plot_evaluation_comparison(results)
    print(f"\nArtefacts written:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
