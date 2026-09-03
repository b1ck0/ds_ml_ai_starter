# NOTE-ML-15-4: Chess Piece Values — Conventional Heuristic

**Date checked:** 2026-09-03

## Answer
The conventional piece-value heuristic widely taught to beginners is:
- **Pawn (P):** 1 point
- **Knight (N):** 3 points
- **Bishop (B):** 3 points
- **Rook (R):** 5 points
- **Queen (Q):** 9 points
- **King:** No point value (priceless; the objective)

These values are a **conventional heuristic**, not universal constants. They approximate relative trading strength and arise from piece mobility: a queen reaches more squares than a rook, a rook more than a bishop, and so on.

## Evidence

### Primary Source (Convention Reference)
- **Chess.com:** https://www.chess.com/terms/chess-piece-value — "The standard system — pawn = 1, knight = 3, bishop = 3, rook = 5, queen = 9 — has been taught to beginners for generations."
- **ChessBase:** https://learn.chessbase.com/en/page/the-value-of-the-pieces-in-chess — Confirms the same values and notes "In practice it also depends on the position. In the endgame the three pawns become more valuable. In the middlegame a piece which is attacking can be stronger than three pawns."
- **World Chess / House of Staunton:** https://worldchess.com/chess-terms/chess-point-values confirms standard values.

### Broader Context
- **Wikipedia / Chess Piece Relative Value:** https://en.wikipedia.org/wiki/Chess_piece_relative_value — Documents the history and variations (e.g., Larry Kaufman's refined scale with slightly adjusted bishop/knight values for engine evaluation).

## Caveats & Nuance

1. **Context-dependent:** The values are approximations. A bishop in an open endgame may be worth more than 3 pawns; a knight in a closed, crowded position may be undervalued at 3.

2. **Endgame variations:** In the endgame, pawns become more valuable (closer to promotion); rooks dominate more than their "5 points" might suggest.

3. **Engine refinements:** Modern chess engines (Stockfish, etc.) use more nuanced evaluations (e.g., bishop pair bonus, pawn structure details), but the 1–3–3–5–9 scale remains the pedagogical starting point.

4. **Not matched to game outcomes:** The values do not directly predict whether a position is winning (e.g., two rooks can dominate a queen + rook in some positions, despite queen being nominally stronger).

## Recommendation for Chapter

Present the piece values as a **conventional heuristic**:
> "The standard heuristic assigns each piece a point value: Pawn = 1, Knight = 3, Bishop = 3, Rook = 5, Queen = 9. These values approximate relative trading strength and arise from piece mobility. They are not universal laws but practical guidelines, especially useful in the middlegame. In our reward shaping for the RL agent, we will use these values to define the reward signal—capturing a rook is worth more than capturing a pawn."

This frames piece values as a **human-designed heuristic that an RL agent could eventually learn to refine or discover on its own** through self-play, which connects to the chapter's theme: moving from hand-tuned evaluators (piece values) to learned ones (the agent's Q-function).

### Citation in Spec References
The spec notes that `kaggle_chess/evaluation.py` uses Q=9, R=5, B=N=3, P=1, confirming this is the conventional scale the chapter should align with. Cite this NOTE and one of the sources above (e.g., Chess.com or ChessBase) in the chapter prose.
