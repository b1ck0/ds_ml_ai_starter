# NOTE-ML-15-2: RL Formulas from Sutton & Barto (2nd Edition)

**Date checked:** 2026-09-03

## Answer
The return (discounted cumulative reward), Bellman equation, and Q-learning/SARSA updates are as follows, cited from the authoritative reference **Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.** A free draft is available at https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf.

### 1. **Return (Discounted Cumulative Reward)** – Chapter 3
The **return** G_t from timestep t is the cumulative reward, discounted by γ (0 ≤ γ ≤ 1):

```
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ... = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

Also written recursively: `G_t = R_{t+1} + γ G_{t+1}`. For undiscounted infinite episodes (γ=1), the sum must converge; finite-horizon episodes or γ<1 ensure finiteness.

### 2. **Bellman Equation (State-Value Function)** – Chapter 4
The **state-value function** V^π(s) under a policy π is the expected return starting from state s:

```
V^π(s) = E_π [G_t | S_t = s]
       = E_π [R_{t+1} + γ V^π(S_{t+1}) | S_t = s]
```

This is the **Bellman equation**: a state's value is the expected immediate reward plus the (discounted) value of the successor state.

### 3. **Bellman Equation (Action-Value Function)** – Chapter 4
The **action-value function** Q^π(s, a) is the expected return for taking action a in state s under policy π:

```
Q^π(s, a) = E_π [G_t | S_t = s, A_t = a]
          = E_π [R_{t+1} + γ Q^π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
```

### 4. **Q-Learning Update (Off-Policy)** – Chapter 6
Q-learning updates the Q-table toward the **maximum** action value in the successor state, regardless of the action actually taken:

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ max_a' Q(S_{t+1}, a') - Q(S_t, A_t)]
```

Where:
- α ∈ (0, 1] is the learning rate
- The bracketed term is the **temporal-difference (TD) error**
- **Off-policy** because the update uses max_a', even if the agent took a greedy action

### 5. **SARSA Update (On-Policy)** – Chapter 6
SARSA (State–Action–Reward–State–Action) updates the Q-table toward the **actual next action** taken (on-policy):

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

Where:
- A_{t+1} is the **actual next action** the agent takes (under the current policy, e.g., ε-greedy)
- **On-policy** because the update uses the actual action taken, not the best possible action

## Evidence
**Primary Source:** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. ISBN: 0262039249. Chapters 3–6 cover MDPs, returns, Bellman equations, and temporal-difference control.

**Free Draft (Stanford):** https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf

**Official Errata & Supplements:** http://incompleteideas.net/book/errata.html (Richard Sutton's website)

**Published Edition:** https://mitpress.mit.edu/9780262039246/reinforcement-learning/

## Caveats
- The formulas assume a discrete, finite state–action space and synchronous updates (value iteration or tabular methods). Deep RL (function approximation) generalizes these concepts.
- γ=1 (undiscounted) requires finite-horizon tasks or guaranteed episode termination to avoid divergence.
- SARSA and Q-learning are both **temporal-difference** methods; they differ in whether they look ahead greedily (Q-learning, off-policy) or use the actual next action (SARSA, on-policy).
- The book also covers policy-gradient methods (REINFORCE, actor–critic) in later chapters, not required for tabular Q-learning but mentioned in the spec's scope (LO5).

## Recommendation for chapter
Quote the return formula and Bellman equation from Sutton & Barto Chapter 3–4. Show both Q-learning and SARSA update rules side by side in the "Exploration & the learning update" section (outline §5) to illustrate on-policy vs. off-policy. The TD error (the bracketed quantity) is the learning signal—emphasize this as the "surprise" or "prediction error" the agent uses to update its Q-table.
