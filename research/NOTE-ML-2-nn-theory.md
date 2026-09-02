# NOTE-ML-2: Neural Network Theory — Activation Functions, Backprop, Dropout, Vanishing Gradients

**Answer:** Sigmoid σ(x) = 1/(1+e^(-x)), Tanh = 2/(1+e^(-2x)) - 1, ReLU = max(0,x), Softmax = exp(x_i)/Σexp(x_j); backpropagation applies chain rule layer-by-layer to compute ∂L/∂w; Srivastava et al. (2014) dropout randomly drops units during training to prevent co-adaptation; sigmoid/tanh vanishing gradients (derivative ≤ 0.25) compound in deep nets; ReLU has constant gradient (0 or 1) solving this.

**Evidence:**

1. **Activation function formulas** (verified 2026-09-02):
   - **Sigmoid:** σ(x) = 1/(1 + exp(-x)), output range [0,1], saturates at extremes
   - **Tanh:** tanh(x) = (exp(2x) - 1)/(exp(2x) + 1) = 2/(1 + exp(-2x)) - 1, output range [-1,1], zero-centered
   - **ReLU:** ReLU(x) = max(0, x), piecewise linear, gradient = 0 (x<0) or 1 (x>0)
   - **Softmax:** softmax(x)_i = exp(x_i) / Σ_j exp(x_j), multiclass probability distribution
   - Sources: https://medium.com/@cmukesh8688/activation-functions-sigmoid-tanh-relu-leaky-relu-softmax-50d3778dcea5 and https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html

2. **Backpropagation chain rule** (verified 2026-09-02):
   - Core principle: ∂L/∂w = (∂L/∂a_n) × (∂a_n/∂a_(n-1)) × ... × (∂a_1/∂w)
   - Computed efficiently by iterating backward through network layers, applying chain rule at each step
   - Calculates error gradient of loss function w.r.t. each weight; computationally efficient via dynamic programming
   - Source: https://cs231n.github.io/optimization-2/ and https://machinelearningmastery.com/the-chain-rule-of-calculus-for-univariate-and-multivariate-functions/

3. **Dropout regularization** (Srivastava et al., 2014):
   - Full citation: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" by Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov. *Journal of Machine Learning Research*, Vol. 15, Pages 1929–1958, 2014.
   - Mechanism: randomly set activations to 0 (drop units) during training with probability p; prevents co-adaptation
   - Approximates Bayesian model averaging over exponential combinations of sub-networks
   - Source: https://jmlr.org/papers/v15/srivastava14a.html

4. **Vanishing gradient problem** (verified 2026-09-02):
   - **Root cause in sigmoid:** ∂σ/∂x ≤ 0.25 for all x; composing layer-by-layer produces (0.25)^L where L = number of layers
   - In deep nets, gradient shrinks exponentially: ∂L/∂w^(1) ≈ (0.25)^L × ... which approaches 0 as L increases
   - Prevents early layers from learning; weights do not update
   - **ReLU solution:** ∂ReLU/∂x = 0 for x<0, 1 for x>0 (no saturation); gradient flows freely in active regions
   - Sources: https://www.kdnuggets.com/2022/02/vanishing-gradient-problem.html and https://medium.com/@amanatulla1606/vanishing-gradient-problem-in-deep-learning-understanding-intuition-and-solutions-da90ef4ecb54

**Caveats / limits:**

- **Softmax naming:** Often used as *output activation* for multiclass; internally normalizes to probability. Sigmoid still used for binary classification.
- **ReLU "dead neuron" problem:** If weight updates push ReLU input permanently negative, gradient = 0 forever; neuron never activates again. Leaky ReLU (0.01×x for x<0) mitigates.
- **Vanishing gradient historical:** Sigmoid/Tanh dominated pre-2012; problem solved by ReLU + He initialization (2015). Deep learning's success post-2012 partly due to this fix.
- **Dropout during inference:** Must be disabled (set p=0); use `model.eval()` in PyTorch.
- **Backprop derivation complexity:** Full mathematical proof of chain rule not needed; intuitive "gradient flows backward" sufficient for ML-1 scope.

**Recommendation:**

1. **For activation plots:** Show sigmoid, tanh, ReLU, softmax with derivatives; highlight saturation vs non-saturation.
2. **For backprop explanation:** Use toy scalar chain rule first (dL/dw = dL/da × da/dw), then generalize to vectors.
3. **For dropout code:** Show conceptual illustration (random mask during training, no mask during eval).
4. **Cite Srivastava et al.** by full reference when introducing dropout; strongly grounded.
5. **ReLU emphasis:** Explain why it dominates modern nets; compare directly to sigmoid/tanh vanishing behavior.
