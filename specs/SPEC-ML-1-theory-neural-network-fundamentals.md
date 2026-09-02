# SPEC-ML-1: Theory — Neural Network Fundamentals

**Status:** approved
**Subject:** Machine Learning
**Section:** Theory
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-14 (DS theory), SPEC-DS-5 (regression as the linear baseline)

## Intent
Build the neural-network mental model from the ground up: a neuron as a weighted sum + nonlinearity,
a dense layer, how gradient descent + backprop learn the weights, and what dropout does. Runnable
micro-demos make the abstractions concrete.

## Learning objectives
- LO1 — Describe a neuron (weights, bias, activation) and a dense layer as a matrix multiply + nonlinearity.
- LO2 — Explain gradient descent and backpropagation intuitively (loss surface, following the slope) and the role of the learning rate.
- LO3 — Compare activation functions (sigmoid, tanh, ReLU, softmax) and know why ReLU dominates hidden layers.
- LO4 — Explain dropout as regularization and how it fights overfitting.

## Scope
In: neuron/layer math, forward pass, loss, gradient descent + backprop intuition, activations, dropout.
Out: specific architectures (→ ML-2), full autograd derivation (link a reference).

## Outline
1. From linear regression to a neuron — one extra nonlinearity.
2. A dense layer as `Wx+b` then activation; stacking layers.
3. Learning — loss surface, gradient descent, learning rate; backprop as the chain rule (intuition + a tiny numeric example).
4. Activations compared (plots) — sigmoid/tanh/ReLU/softmax; vanishing gradients.
5. Dropout — random unit-dropping as an ensemble-in-one-net; overfitting fought.
6. Pitfalls — bad learning rate, dead ReLUs, over/under-fitting.

## Assets to produce
- Prose: "Machine Learning/Theory/neural-network-fundamentals.md"
- Code: "Machine Learning/Theory/code/nn_fundamentals.py" (activation plots; a tiny gradient-descent-on-a-1D-loss demo in numpy — CPU, no torch needed)
- Artefacts: activation-function plots; a gradient-descent trajectory plot; a loss-vs-epoch curve.

## Claims to ground (Haiku, before writing)
- [ ] Verify the definitions/formulas of sigmoid, tanh, ReLU, softmax and the backprop chain-rule statement against an authoritative source (e.g. the Deep Learning book / official docs).
- [ ] Verify the standard explanation of dropout (Srivastava et al.) and the vanishing-gradient problem.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — nn_fundamentals.py runs (numpy-only) and produces the plots; snippet-check passes. AC3 — activation/backprop/dropout claims grounded. AC4 — built up from the regression the reader already saw; math explained, not dumped.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.
