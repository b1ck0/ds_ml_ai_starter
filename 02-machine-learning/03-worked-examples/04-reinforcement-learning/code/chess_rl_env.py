"""A deliberately tiny, genuinely-chess RL environment: "Corner Capture."

Companion module for:
  02-machine-learning/03-worked-examples/04-reinforcement-learning/01-reinforcement-learning-and-self-play.md

The task: a lone White King, confined to a 4x4 corner of a real chess board (files a-d,
ranks 1-4, 16 squares), must capture a stationary, undefended Black Rook placed
somewhere else in that same corner. A second Black King sits inertly on h8 purely so the
position is a legal chess position for python-chess's rules engine (see `_build_board`) —
it never moves and never matters strategically.

Every move the agent's King makes is validated by `python-chess`'s real legal-move
generator (`chess.Board.legal_moves`), so illegal moves — including moving the King onto
a square the Rook attacks along its rank or file ("moving into check," illegal in real
chess even though nothing is nominally "Black's turn" here) — are rejected exactly as a
real chess engine would reject them. This is what makes the "greedy taker" baseline
(policies.py) genuinely worse than the learned agent: it ignores the Rook's attack lines
and repeatedly tries illegal squares, while Q-learning discovers to route around them
from experience alone, no rules encoded.

State space: 16 King squares x 15 remaining Rook squares = 240 (state, ...) pairs, well
under the ~10^3-10^4 dict-Q-table ceiling this chapter's spec requires (NOTE-ML-15-1).

Reward shaping (the "material eval" idea, borrowed conceptually from the reward-shaping
practice described in SPEC-ML-15 and the conventional piece values in NOTE-ML-15-4):
  - -1  per ordinary move (time pressure: fewer moves is better technique)
  - -2  per illegal/no-op attempt (worse than a wasted legal move -- discourages
        repeatedly bumping into an attacked square)
  - +REWARD_SCALE * piece_value(target) on capture, episode ends (WIN)
  - episode also ends (timeout, no bonus) after `max_steps` with no capture

Dependencies (pinned, NOTE-ML-15-1):
    chess==1.11.2
    numpy==2.5.2
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

# --------------------------------------------------------------------------------------
# The playing field: files a-d (0-3), ranks 1-4 (0-3) -- 16 squares in chess's own
# a1..d4 corner. FILES/RANKS are chess.py's 0-indexed file/rank numbers, not chess move
# counts.
# --------------------------------------------------------------------------------------
FILES = 4
RANKS = 4
REGION_SQUARES: list[int] = [chess.square(f, r) for r in range(RANKS) for f in range(FILES)]

# The 8 compass directions a King can move -- this chapter's fixed, 8-action action
# space. Index order matters: it is the order the Q-table's 8 columns are printed in.
DIRECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
]
DIRECTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
N_ACTIONS = len(DIRECTIONS)

# The conventional piece-value heuristic (NOTE-ML-15-4; pawn=1, knight=bishop=3, rook=5,
# queen=9 -- a human-designed convention, not a law of chess). Only ROOK is used as the
# target piece in this chapter's environment, but the table is kept in full so the reward
# function reads as "material value x a scale factor," not a magic number.
PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
REWARD_SCALE = 10  # capturing the rook (value 5) is worth +50
STEP_PENALTY = -1.0
ILLEGAL_PENALTY = -2.0

INERT_BLACK_KING_SQUARE = chess.H8  # far outside the 4x4 corner; never moves, never matters


def _build_board(king_sq: int, target_sq: int, target_piece: chess.PieceType) -> chess.Board:
    """A real `chess.Board`, cleared, with exactly three pieces on it.

    The inert Black King on h8 exists only so the position is a legal chess position
    (python-chess's legality checker wants both sides to have a king); it is never
    adjacent to anything in the 4x4 corner and never gets a turn -- White always moves,
    which `CornerCaptureEnv.step` enforces by resetting `board.turn` after every push.
    """
    board = chess.Board(None)  # empty board, no starting position
    board.set_piece_at(king_sq, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(target_sq, chess.Piece(target_piece, chess.BLACK))
    board.set_piece_at(INERT_BLACK_KING_SQUARE, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    return board


@dataclass
class StepResult:
    state: tuple[int, int]
    reward: float
    done: bool
    won: bool
    legal: bool


class CornerCaptureEnv:
    """A lone White King hunts a stationary Black Rook in a 4x4 corner of a real board.

    State: (king_square, target_square) -- both `python-chess` square indices (0-63),
    always drawn from the 16-square REGION_SQUARES corner.
    Action: an int 0-7 indexing DIRECTIONS (a compass direction for the King to try).
    """

    def __init__(self, target_piece: chess.PieceType = chess.ROOK, max_steps: int = 20):
        self.target_piece = target_piece
        self.max_steps = max_steps
        self.king_sq: int = REGION_SQUARES[0]
        self.target_sq: int = REGION_SQUARES[-1]
        self.board: chess.Board = _build_board(self.king_sq, self.target_sq, self.target_piece)
        self.steps_taken = 0

    # -- episode lifecycle ---------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> tuple[int, int]:
        """Start a new episode: King and target each get a fresh, distinct random
        square drawn from the 16-square corner."""
        king_sq, target_sq = rng.choice(REGION_SQUARES, size=2, replace=False)
        self.king_sq, self.target_sq = int(king_sq), int(target_sq)
        self.board = _build_board(self.king_sq, self.target_sq, self.target_piece)
        self.steps_taken = 0
        return self.state

    @property
    def state(self) -> tuple[int, int]:
        return (self.king_sq, self.target_sq)

    def legal_directions(self) -> dict[int, int]:
        """Which of the 8 compass directions are legal RIGHT NOW, and where each lands.

        Delegates entirely to `chess.Board.legal_moves` -- this is the one place real
        chess rules (including "can't move into check" from the Rook's rank/file) enter
        the environment. Directions that would land outside the 4x4 corner are pruned
        even if python-chess would call them legal on the full 8x8 board -- this is the
        region constraint that keeps the state space at 16x15 instead of 64x63.
        """
        legal: dict[int, int] = {}
        kf, kr = chess.square_file(self.king_sq), chess.square_rank(self.king_sq)
        for i, (df, dr) in enumerate(DIRECTIONS):
            f, r = kf + df, kr + dr
            if not (0 <= f < 8 and 0 <= r < 8):
                continue  # off the 8x8 board entirely
            dest = chess.square(f, r)
            if dest not in REGION_SQUARES:
                continue  # off the 4x4 arena this chapter's task confines itself to
            move = chess.Move(self.king_sq, dest)
            if move in self.board.legal_moves:
                legal[i] = dest
        return legal

    def step(self, action: int) -> StepResult:
        assert self.steps_taken < self.max_steps, "step() called after episode ended"
        self.steps_taken += 1
        legal = self.legal_directions()

        if action not in legal:
            # Illegal move (off the arena, or moving into a square the Rook attacks):
            # a no-op, penalised harder than a normal step so the agent has a reason
            # to learn "don't try that square again from here."
            timed_out = self.steps_taken >= self.max_steps
            return StepResult(self.state, ILLEGAL_PENALTY, timed_out, False, legal=False)

        dest = legal[action]
        move = chess.Move(self.king_sq, dest)
        is_capture = self.board.is_capture(move)
        self.board.push(move)
        self.board.turn = chess.WHITE  # Black never gets a turn in this single-agent task
        self.king_sq = dest

        if is_capture:
            reward = REWARD_SCALE * PIECE_VALUES[self.target_piece]
            return StepResult(self.state, reward, True, True, legal=True)

        timed_out = self.steps_taken >= self.max_steps
        return StepResult(self.state, STEP_PENALTY, timed_out, False, legal=True)

    def render(self) -> str:
        return str(self.board)


def chebyshev_distance(a_sq: int, b_sq: int) -> int:
    """King-move distance between two squares -- the minimum number of King steps to
    go from one to the other on an empty board, ignoring legality."""
    af, ar = chess.square_file(a_sq), chess.square_rank(a_sq)
    bf, br = chess.square_file(b_sq), chess.square_rank(b_sq)
    return max(abs(af - bf), abs(ar - br))
