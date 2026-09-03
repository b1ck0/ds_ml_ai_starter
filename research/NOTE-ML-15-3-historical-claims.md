# NOTE-ML-15-3: Historical Grounding — TD-Gammon, DQN, AlphaZero, MuZero

**Date checked:** 2026-09-03

## 1. TD-Gammon (Tesauro, 1995)

### Answer
Gerald Tesauro's TD-Gammon achieved superhuman backgammon play through **self-play temporal-difference learning** on a neural network. Published March 1995 in *Communications of the ACM*.

### Evidence
- **Publication:** Tesauro, G. (1995). "Temporal difference learning and TD-Gammon." *Communications of the ACM*, 38(3), 58–68.
- **Training:** Self-play of 200,000 games with a neural network (40 hidden units) using TD(λ) learning rule (λ=0.7).
- **Performance:** After training, achieved a level "equal to or slightly better than" the best human players of the era; discovered novel strategies humans had not yet explored.
- **Significance:** First demonstrated that temporal-difference learning + neural networks + self-play could scale to a complex game.

### Caveats
- TD-Gammon did not start from scratch; it used domain knowledge (board encoding) and was trained offline on games, not in real-time play.
- The training data was self-generated through self-play, which is the critical insight (no human move labels needed).

---

## 2. DQN / Atari (Mnih et al., Nature 2015)

### Answer
Google DeepMind's **Deep Q-Network (DQN)** achieved human-level performance on 49 Atari 2600 games by combining **deep neural networks with Q-learning** and two key innovations: **experience replay** and **target networks**. Published February 26, 2015 in *Nature*.

### Evidence
- **Publication:** Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529–533. DOI: 10.1038/nature14236.
- **Approach:** 
  - Convolutional neural network (CNN) to approximate Q(s,a) from raw pixel input
  - **Experience replay:** Store transitions (s, a, r, s') in a buffer; sample mini-batches to break temporal correlations and reduce instability
  - **Target network:** A separate, slower-updating copy of the Q-network to stabilize training
- **Performance:** Achieved human-level performance on 29 of 49 games; surpassed human experts on 6 games.
- **Significance:** First demonstrated that deep RL could learn from high-dimensional sensory input and scale to a diverse task suite.

### Caveats
- DQN requires significant engineering (replay buffer, target network, hyperparameter tuning) for stability; not a minimal algorithm.
- The 49 games were all Atari 2600 with discrete action spaces; generalization to continuous control came later (DDPG, etc.).

---

## 3. AlphaZero (Silver et al., arXiv 2017 / Science 2018)

### Answer
DeepMind's **AlphaZero** learned to play chess, shogi, and Go **at superhuman strength from scratch in a single unified algorithm**, learning only the rules and nothing else, via **self-play + MCTS + neural networks**. Preprint posted December 5, 2017; published Science December 7, 2018.

### Evidence
- **Preprint:** Silver, D., Hubert, T., Schrittwieser, J., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." arXiv:1712.01815 (submitted Dec. 5, 2017).
- **Published:** Silver, D., et al. (2018). "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science*, 362(6419), 1140–1144. DOI: 10.1126/science.aar6404 (published Dec. 7, 2018).
- **Key claim:** "Starting from random play, and given no domain knowledge except the game rules," AlphaZero reached superhuman performance in each game within 24 hours.
- **Algorithm:** 
  - **Self-play:** Agent plays itself using Monte Carlo Tree Search (MCTS) guided by a policy network and value network
  - **Training:** Alternates between self-play (to generate games) and network updates (on outcomes of finished games)
  - **Generality:** Same algorithm, same hyperparameters, same neural network architecture for all three games
- **Compute:** **5,000 v1 TPUs** for self-play games + **16 v2 TPUs** for neural network training, for **13 days total**. (Note: this is equivalent to ~100,000–200,000 GPU-hours, far beyond a laptop.)
- **Performance:** Defeated Stockfish 8 (chess), AlphaGo Zero (Go), and domain-specific engines with overwhelming margins in each game.

### Caveats
- The "no domain knowledge except rules" claim is precise: AlphaZero was given only the game state representation and the set of legal moves. It was **not** given piece values, heuristic evaluators, or opening books.
- The compute requirement is enormous (thousands of TPUs for days); the "24-hour" claim refers to wall-clock time with massive parallelism, not compute-efficient training.
- Self-play introduces non-stationarity (the opponent's policy changes as learning proceeds), which can cause training instability—the paper addresses this via careful exploration and replay buffering.

---

## 4. MuZero (Schrittwieser et al., 2019–2020)

### Answer
**MuZero** extends AlphaZero by learning a **latent model of the environment dynamics**, not just a policy and value function. The agent learns to predict value and reward in an abstract learned representation space, enabling MCTS planning without a true environment model. Preprint November 2019; Nature publication December 2020.

### Evidence
- **Preprint:** Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2019). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." arXiv:1911.08265 (posted Nov. 19, 2019; revised Feb. 21, 2020).
- **Published:** Schrittwieser, J., et al. (2020). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." *Nature*, 588(7839), 604–609. DOI: 10.1038/s41586-020-03051-4 (published Dec. 23, 2020).
- **Key innovation:** Instead of storing a true world model, MuZero learns a *representation* that supports planning and value prediction. The algorithm no longer needs access to a simulator or true reward—it learns reward prediction from bootstrapped estimates.
- **Generality:** Single architecture, single set of hyperparameters for Atari, Go, Chess, and Shogi—broader than AlphaZero (which did not tackle Atari).
- **Significance:** Demonstrated that model-based planning (via learned models) can be as effective as model-free approaches in high-dimensional tasks, and that learning in abstract representations is more sample-efficient than planning in pixel space.

### Caveats
- MuZero is significantly more complex than Q-learning or even AlphaZero; it is not a pedagogical example for introductory RL.
- The abstract learned model remains opaque (a "black box"), making interpretability harder than explicit world models.

---

## Interpretation for the chapter

The spec's "Cold open" should present the progression:
1. **TD-Gammon (1995):** Self-play + neural network learns a value function from scratch.
2. **DQN (2015):** Deep RL scales to high-dimensional input (pixels); introduces experience replay and target networks for stability.
3. **AlphaZero (2017/2018):** Combines RL (self-play), MCTS planning, and neural networks to achieve superhuman play from rules alone.
4. **MuZero (2020):** Learns the model of the environment implicitly (latent representation) alongside planning.

Each step is a progression toward **learning without human-provided evaluators or models**. The chapter should be **brutally honest about compute:** a laptop tabular Q-agent might train in 1 minute on 10–100 states; AlphaZero required 5,000 TPUs for days. This is the "compute gap" that justifies why the chapter does a toy example, then explains (not executes) deep RL and self-play.

---

## Summary of Evidence (URLs)

- TD-Gammon: https://www.bkgm.com/articles/tesauro/tdl.html | https://link.springer.com/chapter/10.1007/978-1-4757-2379-3_11
- DQN/Nature: https://www.nature.com/articles/nature14236
- AlphaZero/arXiv: https://arxiv.org/abs/1712.01815
- AlphaZero/Science: https://www.science.org/doi/10.1126/science.aar6404
- MuZero/arXiv: https://arxiv.org/abs/1911.08265
- MuZero/Nature: https://www.nature.com/articles/s41586-020-03051-4
