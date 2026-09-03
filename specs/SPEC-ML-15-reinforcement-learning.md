# SPEC-ML-15: Reinforcement Learning — agents that learn a game from reward (chess as the through-line)

**Status:** approved
**Subject:** Machine Learning
**Section:** Worked Examples  (new sub-area `03-worked-examples/04-reinforcement-learning/`)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-1 (NN fundamentals — for the deep-RL section), SPEC-DS-14 (DS theory —
supervised vs. the "no labels, only reward" contrast). Related: SPEC-ML-2 (architectures, for the
policy/value network).

## Intent
Every model in the book so far learns from a **labelled target**: supervised learning. Reinforcement
learning is the other great branch — an **agent** takes **actions** in an **environment**, receives a
scalar **reward**, and must learn a **policy** that maximises long-run reward *with no labelled "right
move" ever provided*. Chess is the canonical use-case and the one the owner supplied two real code
bases for, so it is the chapter's through-line: it makes every abstract RL term concrete (a board is a
*state*, a legal move is an *action*, winning material or the game is *reward*, a bot that always
grabs the most valuable piece is a *greedy policy*, minimax is *model-based lookahead*, playing
yourself is *self-play*). A senior Java dev has written game engines and search (minimax/alpha-beta)
before; RL is framed as "what if the evaluation function *learned itself* from playing, instead of
being hand-tuned." The chapter teaches **all core RL concepts** and grounds them in **one small,
fully runnable** agent, then honestly explains how the same ideas scale to AlphaZero-strength chess.

The owner's two reference repos (study them as source material; do NOT import from them — the chapter's
code must be self-contained and runnable by a reader who has only this book):
- `C:\Users\b1ck0\Documents\python_projects\topcoder_chess` — a Horde-Chess engine: board
  representation, legal-move generation, and a **minimax** base-case. Use as the model of a game
  environment and the *model-based search* baseline RL is contrasted against.
- `C:\Users\b1ck0\Documents\python_projects\kaggle_chess` — a Kaggle chess bot: **material
  evaluation** (`evaluation.py`, piece values Q=9 R=5 B=N=3 P=1), **hand-coded heuristic policies**
  (`the_taker.py` = greedy capture, `the_defender.py`), a **self-play simulator** (via
  `kaggle-environments`), and a Stockfish reference. Use material-eval as the reward/value-shaping
  example, the heuristic bots as fixed baseline policies, and the simulator as the self-play picture.

## Learning objectives
After this chapter the reader can:
- LO1 — State the RL problem as a **Markov Decision Process**: state `s`, action `a`, transition,
  reward `r`, discount `γ`, and the **return** `G_t = Σ γ^k r_{t+k}`; and say precisely how RL differs
  from supervised learning (no labels; delayed, evaluative feedback; the agent's actions change the
  data it sees).
- LO2 — Define a **policy** `π(a|s)`, the **state-value** `V^π(s)` and **action-value** `Q^π(s,a)`,
  and the **Bellman equation** relating a state's value to its successors — using a chess position as
  the worked example (material eval as a first, hand-made value function).
- LO3 — Explain **exploration vs. exploitation** and the **ε-greedy** strategy; **on-policy vs.
  off-policy** and **model-free vs. model-based**; and place minimax (model-based lookahead),
  the greedy "taker" bot (a fixed policy), and a learning agent on that map.
- LO4 — Derive and implement **temporal-difference learning** — **Q-learning** (off-policy) and
  **SARSA** (on-policy) — the update `Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]`, and run a
  **tabular Q-learning agent** on a small chess-derived environment until it beats a random baseline,
  watching the reward-per-episode curve rise (iterate-visibly) and ε decay.
- LO5 — Explain why tables don't scale to real chess (state-space size), and how **function
  approximation** fixes it: **DQN** (a value network + replay + target net) and **policy-gradient/
  actor-critic** (REINFORCE at an intro level), then **self-play + MCTS = AlphaZero** — all grounded
  in the literature, with an honest note on the compute gap.

## Scope
In scope: the full conceptual arc (MDP → value/policy → Bellman → TD/Q-learning → deep RL → self-play/
MCTS), each anchored to chess; and ONE small, CPU-runnable tabular Q-learning example on a
deliberately tiny chess-derived environment (see below) that trains in minutes and whose Q-table and
learning curve are inspectable.
Out of scope (name + link, don't implement): training a full-strength chess engine; a from-scratch
AlphaZero (compute-prohibitive); a deep-dive on MCTS internals beyond the intuition; continuous-action
RL / robotics. The deep-RL and AlphaZero material is **explained and grounded, not executed**.

## The runnable environment (tractability is a hard requirement)
Tabular Q-learning needs a small, discrete state space. Use ONE of these (writer picks the most
robust after the Haiku note; the constraint is: trains on a laptop CPU in ≤ a few minutes, state
space small enough for a dict-based Q-table, and clearly chess-flavoured):
- a **minimal chess endgame** (e.g. King-and-Rook vs King, or a "lone piece must reach/capture"
  task) on a small board region using `python-chess` for legal moves and terminal detection, with a
  **shaped reward** built from the material/positional idea borrowed from `kaggle_chess/evaluation.py`; OR
- a **self-contained "capture grid"**: a tiny hand-rolled board (adapted from the Horde engine's move
  logic) where the agent piece is rewarded for capturing and penalised per move — small enough to
  enumerate states.
Whichever is chosen, the reward design must be explained (reward shaping, and its risks), the baseline
is a random and/or greedy-"taker" policy, and the agent must measurably beat it.

## Outline (section-by-section)
1. **Cold open** — a grounded origin story: TD-Gammon (1992) learning backgammon by self-play, then
   DeepMind's DQN on Atari (2015) and **AlphaZero teaching itself chess from nothing but the rules
   (2017/2018)**, superhuman in hours. Then the problem: your hand-tuned material evaluator (show the
   real `kaggle_chess` piece values) is *guessing* — what if the evaluation function learned itself?
2. **What & why** — RL vs. supervised (the "no labels, only reward" contrast), the agent–environment
   loop diagram, and the "you are here" map of the concept arc. Java analogy: minimax's static
   evaluator, but *learned*.
3. **The MDP** — state/action/reward/transition/γ/return on a real chess position; episodes; why
   delayed reward (you only find out you blundered ten moves later) is the whole difficulty.
4. **Policies & value** — π, V, Q, and the Bellman equation, with material-eval as a hand-made V and
   the greedy "taker" as a policy you can beat.
5. **Exploration & the learning update** — ε-greedy; Q-learning vs SARSA; the TD update derived and
   explained line by line; model-free vs model-based (minimax as the model-based foil).
6. **Worked example** — the tabular Q-learning agent on the tiny environment: code walked through,
   trained, reward curve + ε-decay plotted, learned Q-values inspected, evaluated vs. random/greedy.
   This is the iterate-visibly heart of the chapter.
7. **Scaling up** — why the table explodes for real chess; DQN (value net + experience replay +
   target network); policy gradients/actor-critic (REINFORCE intuition); **self-play + MCTS →
   AlphaZero/MuZero**, tying back to the cold open — grounded, with the compute-gap caveat.
8. **Pitfalls** — reward shaping gone wrong (reward hacking), non-stationarity in self-play, sparse
   reward, unstable deep-RL training, evaluating against a fixed weak baseline and fooling yourself.
9. **Recap & next** — the map filled in; pointers (AlphaZero paper, Sutton & Barto, Gymnasium).

## Assets to produce
- Prose: `02-machine-learning/03-worked-examples/04-reinforcement-learning/01-reinforcement-learning-and-self-play.md`
- Code: `.../04-reinforcement-learning/code/` — the environment + tabular Q-learning agent
  (self-contained, seed set, deps pinned), and the plotting script.
- Artefacts: a reward-per-episode learning curve (with ε-decay) and a small learned-Q / evaluation
  table, under `.../04-reinforcement-learning/artefacts/`.

## Claims to ground (Haiku research brief — do BEFORE writing)
- [ ] Package versions to pin: `python-chess` (a.k.a. `chess`), `gymnasium` (if used), `numpy`,
      `matplotlib` — current PyPI versions + dates. Confirm which gives the smallest, most robust
      runnable chess-endgame or grid setup for tabular Q-learning on CPU.
- [ ] Verify the RL formulas against an authoritative source (Sutton & Barto, *Reinforcement
      Learning: An Introduction*, 2nd ed.) — the return, Bellman equation, and the Q-learning and
      SARSA update rules — cite it.
- [ ] Verify the historical/grounding claims with real citations + dates: TD-Gammon (Tesauro 1995),
      DQN/Atari (Mnih et al., *Nature* 2015), **AlphaZero** (Silver et al., 2017 arXiv / *Science*
      2018) — including what "learned from self-play with only the rules" precisely means, and the
      compute used (so the gap claim is honest). MuZero (Schrittwieser et al. 2020) as a pointer.
- [ ] Confirm standard piece values (Q=9,R=5,B=3,N=3,P=1) are the conventional heuristic (they match
      `kaggle_chess/evaluation.py`) and cite them as a convention, not an assertion.

## Acceptance criteria (each maps to evidence)
- [ ] AC1 (LO1–LO3) — MDP, policy/value/Bellman, and the exploration/on-off-policy/model-based map all
      taught with chess anchors → evidence: the prose + diagrams.
- [ ] AC2 (LO4) — a tabular Q-learning agent trains and measurably beats a random/greedy baseline →
      evidence: runnable code + learning-curve artefact + a run log showing reward rising and ε decaying.
- [ ] AC3 (LO5) — deep RL (DQN, policy gradient) and self-play/MCTS/AlphaZero explained and grounded,
      with the honest compute-gap caveat → evidence: NOTE ids + inline citations.
- [ ] AC4 — every snippet runs (`check_snippets.py` + real run log); every formula/version/historical
      claim grounded (NOTE ids); NO code imported from the two private repos (self-contained).
- [ ] AC5 — audience-fit (minimax/game-engine Java framing, every term explained, artefacts shown).
- [ ] AC6 — renders on GitHub (`check_markdown_render.py` pass; every `$…$`/```mermaid eyeballed —
      watch `\text{}` escaping in the return/Bellman/Q-update formulas).

## Gates
Entry: this spec approved; research NOTEs landed. Exit: all ACs satisfied with evidence; snippets run;
links resolve; fresh-Sonnet review sign-off; architect merge. (See `docs/definition-of-done.md`.)
