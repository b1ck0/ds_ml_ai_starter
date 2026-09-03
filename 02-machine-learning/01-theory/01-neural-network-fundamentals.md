# Neural Network Fundamentals

*Machine Learning · Theory · SPEC-ML-1*

## The wall the Perceptron hit

In July 1958, a psychologist named Frank Rosenblatt stood in front of reporters at the Cornell
Aeronautical Laboratory in Buffalo, New York, and unveiled a machine he called the Perceptron — a
room-sized tangle of wires and motorized potentiometers that learned to tell left-marked cards from
right-marked ones just by looking at examples. The *New York Times* ran the story under the
headline "NEW NAVY DEVICE LEARNS BY DOING," and Rosenblatt called his creation "the first machine
which is capable of having an original idea"
([source: Cornell Chronicle, "Professor's perceptron paved the way for AI — 60 years too
soon"](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon),
checked 2026-09-03). The Navy, quoted in the same coverage, expected a machine that would "walk,
talk, see, write, reproduce itself and be conscious of its existence."

The Perceptron itself was one neuron: a weighted sum of its inputs, thresholded into a yes/no
answer. Eleven years later, in 1969, MIT's Marvin Minsky and Seymour Papert published a book,
*Perceptrons*, that did the arithmetic Rosenblatt's era had skipped: a single perceptron can only
draw one straight line (or flat plane) through its input space, and there's a painfully simple
problem no straight line can ever solve — XOR, "exactly one of these two things, not both, not
neither." The book is widely credited with draining funding and interest out of neural-network
research for over a decade, the first "AI winter" — though, as historians later pointed out,
Minsky and Papert's proof was specifically about *single-layer* perceptrons, not about networks
with hidden layers stacked in between
([source: Wikipedia, "Perceptrons (book)"](<https://en.wikipedia.org/wiki/Perceptrons_(book)>),
checked 2026-09-03). Nobody yet had a working, well-understood way to *train* a network with
hidden layers, so that distinction didn't save the field in the 1970s.

That changed in 1986, when David Rumelhart, Geoffrey Hinton, and Ronald Williams published a
short paper in *Nature* titled "Learning representations by back-propagating errors," describing a
way to train networks with hidden layers by pushing an error signal backward through every layer,
one calculus step at a time
([source: Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors,"
*Nature* 323, 533–536 (1986)](https://www.nature.com/articles/323533a0), checked 2026-09-03). That
algorithm — **backpropagation** — is still, mechanically, the same one training every network in
this curriculum decades later.

Here's the whole chapter in one sentence you could repeat at dinner: **a single straight line
can't bend around a problem like XOR, so stack a nonlinear twist between several weighted sums, and
use backpropagation to train every layer of those sums at once.**

Walk through what that fix actually looks like, piece by piece — it's the whole chapter in
miniature:

```mermaid
flowchart LR
    A["Step 1<br/>start from LinearRegression<br/>y_hat = w·x + b"] --> B["Step 2<br/>watch it fail on XOR<br/>(no line separates it)"]
    B --> C["Step 3<br/>add a nonlinear twist<br/>a = f(w·x + b)"]
    C --> D["Step 4<br/>stack neurons into layers,<br/>layers into a network"]
    D --> E["Step 5<br/>learn the weights:<br/>gradient descent + backprop"]
    E --> F["Step 6<br/>keep it from memorizing:<br/>dropout"]
    F -.->|"this chapter rebuilds every step<br/>from scratch, in plain NumPy"| A
```

This chapter builds each of those six steps from the smallest possible piece — one neuron — up to
a tiny two-layer network you can watch learn, epoch by epoch, entirely in NumPy: no framework, no
autograd, no magic.

## 1. What & why — the line that can't bend

[DS-5 (Regression — NYC Taxi Fare Prediction)](../../01-data-science/03-worked-examples/05-regression-nyc-taxi.md)
fit a pricing function `f(features) -> dollars` by learning coefficients from data instead of
hand-writing them — `LinearRegression` found the best straight-line (or hyperplane) relationship
between features and fare. That's exactly Step 1's shape: `y_hat = w . x + b`.

**Now watch that shape hit the same wall Rosenblatt's Perceptron hit.** XOR is four points, each
labelled 0 or 1 — `(0,0)->0`, `(1,1)->0`, `(0,1)->1`, `(1,0)->1` (read it as "exactly one of the
two inputs is on"). Try to separate them with one straight decision boundary,
`z = w1*x1 + w2*x2 + b`, predicting label 1 whenever `z > 0`:

- For `(0,0)` to score negative (label 0), you need `b < 0`.
- For `(0,1)` and `(1,0)` to both score positive (label 1), you need `w2 + b > 0` and
  `w1 + b > 0`. Add those two: `w1 + w2 + 2b > 0`, i.e. `w1 + w2 > -2b`.
- For `(1,1)` to score negative (label 0), you need `w1 + w2 + b < 0`, i.e. `w1 + w2 < -b`.

Plug in a concrete number for `b`, say `b = -1`: the first requirement demands `w1 + w2 > 2`; the
second demands `w1 + w2 < 1`. **No number satisfies both at once.** That's not a fluke of this
particular `b` — the same contradiction, `-2b > -b` for every negative `b`, appears no matter which
`b` you pick. This is the arithmetic core of what Minsky and Papert proved in 1969: **one weighted
sum, however you tune it, cannot separate XOR.** No single straight line divides the two classes —
you'd need at least two lines, or one curved boundary.

A **neural network** solves this the way Steps 3–4 above sketched: composing several rounds of
"weighted sum, then a nonlinear twist" — and it turns out that stacking enough of those is enough
to approximate essentially any function, curved boundaries included. This chapter's own code demo
(§6) trains a network on exactly this XOR pattern and watches it succeed where one straight line
algebraically cannot. Everything below is building toward that one idea, one piece at a time: what
one neuron computes, how many neurons form a layer, how the network *learns* its weights (gradient
descent + backpropagation), which nonlinearity to use where (activation functions), and one of the
standard tools for stopping the network from memorising its training data instead of generalising
(dropout) — the same overfitting failure mode
[DS-14](../../01-data-science/01-theory/01-theory-overview.md#5-the-central-tension--overfitting-biasvariance-and-regularization)
covered for `LinearRegression` and tree ensembles, showing up again in a new model family.

**Environment for every snippet and artefact in this chapter:** the project's `.venv-ml`, running
**numpy 2.5.2** and **matplotlib 3.11.1** — both re-verified live against the interpreter
(`import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)`) and matching
[research/NOTE-2-package-versions.md](../../research/NOTE-2-package-versions.md) (checked
2026-09-02). No `torch` import appears anywhere in this chapter's code — per
[SPEC-ML-1](../../specs/SPEC-ML-1-theory-neural-network-fundamentals.md), the runnable demo is
NumPy-only, so every gradient in this chapter is one you can trace by hand. The full script is
[`code/nn_fundamentals.py`](code/nn_fundamentals.py).

## 2. A neuron: weighted sum + nonlinearity

Here's the one-line summary a Java engineer who just read DS-5 needs: **a neuron is a
`LinearRegression` unit with one extra step.**

$$z = w \cdot x + b \qquad \text{(exactly the linear-regression formula: weighted sum + bias)}$$

$$a = f(z) \qquad \text{(the new part: squash } z \text{ through a nonlinear "activation function")}$$

Plain-language glosses — every symbol here earns one the first time it shows up:

- **weight** (`w`) — how much each input matters; learned from data, exactly like
  `LinearRegression`'s coefficients.
- **bias** (`b`) — a constant offset; what the neuron would output if every input were 0.
- **activation function** (`f`) — a fixed, *non-learned* function that bends the weighted sum —
  the piece a plain linear model doesn't have.
- **activation** (`a`) — the neuron's actual output, after `f` has been applied to `z`.

```mermaid
flowchart LR
    x1["x1"] -->|"w1"| SUM(("Σ<br/>z = w·x + b"))
    x2["x2"] -->|"w2"| SUM
    x3["..."] -.-> SUM
    B["bias b"] --> SUM
    SUM --> F["activation f(z)"]
    F --> A["a (neuron output)"]
```

`w` (weights) and `b` (bias) are learned, exactly as in `LinearRegression`. `f` is a fixed,
non-learned function — sigmoid, tanh, or ReLU are the three this chapter compares in §5. Without
`f` (or with `f` being the identity function, `f(z) = z`), a neuron *is* linear regression: same
formula, same limitation — the same wall §1 just proved XOR sits behind. `f` is what buys a network
the ability to bend.

**Why the nonlinearity has to be there at all** isn't just a rule of thumb — it's forced by linear
algebra. Composing two linear functions is *still* linear: if `z1 = W1 x + b1` and
`z2 = W2 z1 + b2`, substituting gives `z2 = W2 W1 x + (W2 b1 + b2)`, which is just another
`weight . x + bias` — collapsible back into a single linear layer, no matter how many linear layers
you stack. Insert a nonlinear `f` between the layers and that collapse no longer happens; each layer
genuinely adds representational power. This is why every layer in a network except (usually) the
very last one is followed by an activation function — and it's the piece Rosenblatt's single
perceptron never got to use against a problem like XOR, because it never had a second layer stacked
behind its own activation.

## 3. A dense layer: `Wx + b`, then activation

One neuron produces one number. A **dense layer** (also called "fully connected") runs *several*
neurons side by side over the *same* input, each with its own weights:

$$Z = XW + b \qquad X:(n_{\text{samples}}, n_{\text{in}}) \quad W:(n_{\text{in}}, n_{\text{out}}) \quad b:(n_{\text{out}},) \quad Z:(n_{\text{samples}}, n_{\text{out}})$$

$$A = f(Z) \qquad \text{elementwise: same activation function applied to every entry of } Z$$

```mermaid
flowchart LR
    X["X<br/>(n_samples x n_in)"] --> MM["X @ W + b"]
    W["W<br/>(n_in x n_out)"] --> MM
    Bv["b<br/>(n_out,)"] --> MM
    MM --> Z["Z<br/>(n_samples x n_out)"]
    Z --> ACT["elementwise f(Z)"]
    ACT --> A["A<br/>(n_samples x n_out)"]
```

This is the exact matrix-multiply shape `TinyNet` in this chapter's code uses (see §6):

```python
z1 = X @ self.W1 + self.b1
a1 = relu(z1)
```

Read `W`'s shape as "one column of weights per output neuron, one row per input feature" — the same
`X` (rows = examples, columns = features) DS-14 defined for every supervised-learning problem in
this curriculum, now multiplied by a *matrix* of weights instead of a single weight vector, because
a layer has several neurons instead of one. Stack more dense layers (each one's output `A` becomes
the next one's input `X`) and you have a **multi-layer** or **"deep"** network — "deep learning" is
named for exactly this: enough stacked layers that the network is meaningfully deep, not just wide.
This is Step 4 from the roadmap above: neurons into a layer, layers into a network deep enough to
bend around XOR and much harder problems.

## 4. Learning: descending the loss surface

A freshly initialised network's weights are random, so its predictions are garbage. **Learning**
means adjusting every weight to reduce a **loss function** — a single number that says how wrong the
current predictions are (DS-14 already introduced RMSE as regression's version of this; this
chapter's own demo uses **binary cross-entropy**, the standard loss for a sigmoid-output binary
classifier).

Two more glosses, since they'll do a lot of work below:

- **gradient** — the direction (and size) of steepest *increase* of the loss, at the current
  weights; a compass needle that always points "more wrong."
- **epoch** — one full pass of the training data through the network, followed by a weight update;
  this chapter's own demo (§6) runs 4000 of them.

### 4.1 The loss surface and gradient descent

Picture the loss as a function of the weights — a landscape where every point is one possible
setting of every weight, and the height at that point is the loss you'd get from it. **Gradient
descent** is the strategy of always stepping in the direction that goes downhill fastest — the
negative gradient — and repeating, the same way you'd feel your way down a dark hillside by always
taking the next step in whichever direction slopes down the most:

$$w \leftarrow w - \text{lr} \cdot \frac{dL}{dw}$$

```mermaid
flowchart TD
    S["random start:<br/>high loss"] --> G1["compute the gradient dL/dw<br/>(the uphill direction)"]
    G1 --> ST["step downhill:<br/>w ← w - lr * dL/dw"]
    ST --> C{"loss still<br/>decreasing?"}
    C -->|"yes: take another step"| G1
    C -->|"converged"| MIN["near the minimum<br/>(low loss)"]
```

`dL/dw` (the **gradient** of the loss with respect to `w`) is the vector that points *uphill*
steepest; subtracting it (scaled by the **learning rate**, `lr`) steps downhill. `lr` controls how
big each step is: too small, and training crawls toward the minimum one tiny shuffle at a time; too
large, and a single step can overshoot the minimum entirely and land higher up the opposite slope
than where it started — exactly what the right-hand panel below shows. This chapter's code
(`bowl_loss`/`bowl_grad`/`run_gradient_descent` in
[`code/nn_fundamentals.py`](code/nn_fundamentals.py)) runs plain gradient descent on a synthetic,
deliberately elongated 2-D bowl, `L(w1, w2) = w1^2 + 5*w2^2`, from the same starting point, twice —
once with a learning rate that converges, once with one large enough to diverge:

```python
def bowl_loss(w: np.ndarray) -> float:
    return float(w[0] ** 2 + 5.0 * w[1] ** 2)


def bowl_grad(w: np.ndarray) -> np.ndarray:
    return np.array([2.0 * w[0], 10.0 * w[1]])


def run_gradient_descent(start: np.ndarray, lr: float, n_steps: int) -> np.ndarray:
    path = np.zeros((n_steps + 1, 2))
    w = start.copy()
    path[0] = w
    for i in range(1, n_steps + 1):
        w = w - lr * bowl_grad(w)
        path[i] = w
    return path
```

![Two side-by-side contour plots of the loss surface L(w1,w2) = w1^2 + 5*w2^2. Left panel, "lr=0.15 (converges)": a red trajectory of dots spirals inward from the orange starting square toward the black-star minimum at the origin, overshooting slightly on the steep w2 axis before settling. Right panel, "lr=0.42 (diverges)": a red trajectory shoots away from the origin along the w2 axis, growing explosively across a y-axis scaled in millions.](artefacts/gradient_descent_trajectory.png)

Running it (console output from `code/nn_fundamentals.py`):

```text
good lr=0.15: loss went 40.2500 -> 0.000000 over 25 steps
bad  lr=0.42: loss went 40.2500 -> 2.6585e+13 over 12 steps (diverging)
```

Two things worth reading directly off the left panel: the trajectory doesn't walk straight to the
minimum, it *spirals* — because the bowl is steeper along `w2` than `w1` (the `5*w2^2` term), so the
same learning rate produces a bigger corrective step on `w2` than `w1` every time, and the path
overshoots on that axis before settling. Bump the learning rate up on that same steep axis (right
panel) and the overshoot compounds instead of decaying: each step lands *further* from the minimum
than the last, and the loss explodes toward `2.66 x 10^13` in twelve steps. **A learning rate that's
merely too big for the steepest direction the loss surface has is enough to blow up training** —
this is §7's first pitfall, made concrete.

### 4.2 Backpropagation: the chain rule, applied layer by layer

A real network's loss depends on *every* weight in *every* layer, through a chain of function
compositions — layer 1's output feeds layer 2, whose output feeds the loss. **Backpropagation**
computes `dL/dw` for every weight by applying the calculus **chain rule** once per layer, working
backward from the loss to the input — the same algorithm Rumelhart, Hinton, and Williams described
in 1986, and the fix that finally let networks with hidden layers climb out of the wall this
chapter opened on:

$$\frac{dL}{dw} = \frac{dL}{da_n} \times \frac{da_n}{da_{n-1}} \times \cdots \times \frac{da_1}{dw}$$

— [research/NOTE-ML-2-nn-theory.md](../../research/NOTE-ML-2-nn-theory.md), evidence item 2,
grounded against [cs231n.github.io/optimization-2](https://cs231n.github.io/optimization-2/)
(checked 2026-09-02). Each layer only needs to know how to turn "the gradient of the loss with
respect to *my output*" into "the gradient of the loss with respect to *my input and my weights*" —
it doesn't need to know anything about any other layer. Chaining that one local computation
backward through every layer, reusing the values computed during the forward pass, is what makes
backprop efficient instead of re-deriving each weight's gradient from scratch.

```mermaid
flowchart LR
    X0["input x"] --> L1["layer 1<br/>z1 = w1·x + b1<br/>a1 = f(z1)"]
    L1 --> L2["layer 2<br/>z2 = w2·a1 + b2<br/>a2 = f(z2)"]
    L2 --> LOSS["loss L(a2, target)"]
    LOSS -.->|"backward: dL/da2"| B2["layer 2 gradient<br/>dL/dw2 = dL/da2 · da2/dz2 · dz2/dw2"]
    B2 -.->|"backward: dL/da1"| B1["layer 1 gradient<br/>dL/dw1 = dL/da2 · da2/dz2 · dz2/da1 · da1/dz1 · dz1/dw1"]
```

**A tiny numeric example**, small enough to trace by hand — two toy scalar "layers", a ReLU neuron
feeding a sigmoid neuron:

```python
x, target = 2.0, 1.0
w1, b1 = 0.5, 0.1
w2, b2 = -0.3, 0.2

def forward(w1_: float, w2_: float) -> tuple[float, float, float, float]:
    z1 = w1_ * x + b1
    a1 = max(0.0, z1)  # ReLU
    z2 = w2_ * a1 + b2
    a2 = 1.0 / (1.0 + np.exp(-z2))  # sigmoid
    loss = (a2 - target) ** 2
    return z1, a1, z2, a2, loss  # type: ignore[return-value]
```

Forward pass first (compute and *keep* every intermediate value — backprop needs them):

```text
z1 = 1.1000   a1 = 1.1000   z2 = -0.1300   a2 = 0.4675   loss = 0.2835
```

Backward pass — one chain-rule multiplication per layer, working from the loss back to `w1`:

```python
dloss_da2 = 2.0 * (a2 - target)              # dL/da2
da2_dz2 = a2 * (1.0 - a2)                    # sigmoid'(z2)
dz2_dw2 = a1                                 # d(w2*a1+b2)/dw2
dloss_dw2 = dloss_da2 * da2_dz2 * dz2_dw2    # chain rule, layer 2

dz2_da1 = w2                                 # d(w2*a1+b2)/da1
da1_dz1 = 1.0 if z1 > 0 else 0.0             # ReLU'(z1)
dz1_dw1 = x                                  # d(w1*x+b1)/dw1
dloss_dw1 = dloss_da2 * da2_dz2 * dz2_da1 * da1_dz1 * dz1_dw1  # chain rule, layer 1
```

That's the formula from above, written out one factor at a time: `dloss_dw2` is
`(dL/da2) x (da2/dz2) x (dz2/dw2)`, and `dloss_dw1` extends the same chain one layer further back.
To make sure the hand-derived chain rule is actually *right* and not just plausible-looking, the
code cross-checks both gradients against a **finite-difference** numerical gradient — nudge each
weight by a tiny `h` in each direction, re-run the forward pass, and see how much the loss moved
(the direct definition of a derivative, no calculus rules involved, just arithmetic):

```text
forward: z1=1.1000 a1=1.1000 z2=-0.1300 a2=0.4675 loss=0.2835
analytic  dloss/dw1=0.159063   numeric dloss/dw1=0.159063
analytic  dloss/dw2=-0.291616   numeric dloss/dw2=-0.291616
analytic chain-rule gradients match the finite-difference check (within 1e-4).
```

They match to five decimal places. `TinyNet.train_step` in §6 does exactly this same chain, just
with matrices instead of scalars and many neurons instead of one — vector/matrix calculus is
mechanically the same idea, applied to whole layers at once instead of one number at a time.

## 5. Activation functions compared

Four activation functions cover almost everything in this curriculum. Formulas below are
[research/NOTE-ML-2-nn-theory.md](../../research/NOTE-ML-2-nn-theory.md), evidence item 1, grounded
against [ml-cheatsheet.readthedocs.io/activation_functions](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)
(checked 2026-09-02):

| Function | Formula | Output range | Typical use |
|---|---|---|---|
| **Sigmoid** | $\sigma(x) = \dfrac{1}{1+e^{-x}}$ | $(0, 1)$ | Binary classification output |
| **Tanh** | $\tanh(x) = \dfrac{2}{1+e^{-2x}} - 1$ | $(-1, 1)$ | Older hidden-layer default; zero-centred |
| **ReLU** | $\mathrm{ReLU}(x) = \max(0, x)$ | $[0, \infty)$ | Modern hidden-layer default |
| **Softmax** | $\mathrm{softmax}(x)_i = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$ | $(0, 1)$, sums to 1 | Multi-class classification output |

```python
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh_fn(x: np.ndarray) -> np.ndarray:
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)   # numerical-stability trick, same result
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
```

![2x2 grid of plots. Top-left, sigmoid: an S-curve from 0 to 1 in solid blue, with a dashed red derivative curve peaking at 0.25 near x=0 and flattening to nearly 0 by x=plus-or-minus 4. Top-right, tanh: an S-curve from -1 to 1, derivative peaking at 1.0 near x=0 and also flattening to near 0 at the edges. Bottom-left, ReLU: a flat-then-linear blue line bending upward at x=0, with a dashed red derivative that is a flat step from 0 to 1 exactly at x=0. Bottom-right, a green bar chart titled "softmax(logits) -- sums to 1.000": four bars for logits 2, 1, 0.1, -1 with heights 0.64, 0.23, 0.10, 0.03.](artefacts/activation_functions.png)

Console output confirms the numbers on that plot:

```text
softmax([2.0, 1.0, 0.1, -1.0]) = [0.6381, 0.2347, 0.0954, 0.0318], sums to 1.000000
max sigmoid'(x) = 0.2500 (at x=0)
max tanh'(x)    = 1.0000 (at x=0)
sigmoid'(x) at x=6 (far from origin) = 0.002467  <- vanishing
ReLU'(x) at x=6 (active region)      = 1.000000  <- constant
```

### 5.1 Vanishing gradients — why ReLU dominates hidden layers

Look at the dashed derivative curves in the plot above: sigmoid's derivative tops out at **0.25**,
tanh's at **1.0**, but both collapse toward **0** the moment `x` moves a few units from the origin —
that's **saturation**. Backpropagation (§4.2) *multiplies* these derivatives together, one per
layer, to get the gradient for an early layer's weights. Chain `L` sigmoid layers together and the
early-layer gradient scales roughly like $0.25^{L}$
([research/NOTE-ML-2-nn-theory.md](../../research/NOTE-ML-2-nn-theory.md), evidence item 4,
grounded against [kdnuggets.com/2022/02/vanishing-gradient-problem](https://www.kdnuggets.com/2022/02/vanishing-gradient-problem.html),
checked 2026-09-02): at $L=10$ that's already $0.25^{10} \approx 9.5 \times 10^{-7}$. The gradient
reaching the earliest layers shrinks toward zero exponentially in depth — those layers stop
learning even though the loss is still high. That's the **vanishing gradient problem**, and it's
the historical reason sigmoid/tanh dominated hidden layers before roughly 2012, and ReLU (paired
with a matching initialisation scheme) after
([research/NOTE-ML-2-nn-theory.md](../../research/NOTE-ML-2-nn-theory.md) caveats).

ReLU's derivative is exactly **0 or 1** — never a small fraction — so chaining many ReLU layers
doesn't shrink the gradient the way chaining sigmoids does; a gradient that reaches an active ReLU
unit passes through unchanged. That's the entire reason ReLU is the modern default for hidden
layers: it's not more "correct" than sigmoid, it just doesn't strangle its own training signal in
deep networks. §7 covers the trade-off ReLU brings with it (the "dead ReLU" problem).

**Where each one still belongs:** sigmoid stays the standard choice for a *binary classification
output* (one number, needs to read as a probability of the positive class — exactly what `TinyNet`
in §6 uses). Softmax is the standard choice for a *multi-class output* — it turns a vector of raw
scores ("logits") into a proper probability distribution over more than two classes, which is
exactly what the bar chart above shows: four arbitrary logits, `[2.0, 1.0, 0.1, -1.0]`, become four
probabilities that sum to `1.000000`. Neither sigmoid nor softmax gets used in *hidden* layers in
modern practice, for the same saturation reason as tanh.

## 6. Dropout as regularization

`LinearRegression` fights overfitting with L1/L2 regularization — shrinking coefficients toward zero
([DS-14 §5.3](../../01-data-science/01-theory/01-theory-overview.md#53-regularization--fighting-variance-directly)).
A dense network has a different, network-specific tool: **dropout** — Step 6 from the roadmap that
opened this chapter.

**Dropout** ([Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, "Dropout: A Simple Way to
Prevent Neural Networks from Overfitting", *JMLR* Vol. 15, pp. 1929–1958, 2014](https://jmlr.org/papers/v15/srivastava14a.html),
grounded via [research/NOTE-ML-2-nn-theory.md](../../research/NOTE-ML-2-nn-theory.md) evidence item
3) randomly zeroes out a fraction `p` of a layer's activations on *every training step*, forcing the
surviving units to not rely on any one other unit always being present:

```mermaid
flowchart LR
    IN["hidden layer output<br/>(5 units, ReLU)"]
    IN --> H1["h1 (kept)"]
    IN --> H2["h2 (DROPPED this step)"]
    IN --> H3["h3 (kept)"]
    IN --> H4["h4 (DROPPED this step)"]
    IN --> H5["h5 (kept)"]
    H1 --> OUT["next layer<br/>(kept units rescaled by 1/(1-p))"]
    H3 --> OUT
    H5 --> OUT
    H2 -.->|"masked to 0"| OUT
    H4 -.->|"masked to 0"| OUT
```

Srivastava et al. frame the effect as approximating an ensemble of exponentially many "thinned"
sub-networks trained simultaneously, averaged together — a network-native cousin of the bagging
ensembles
([DS-14 §4](../../01-data-science/01-theory/01-theory-overview.md#4-models--regression-classification-and-ensembles))
already covered.

The implementation detail worth knowing before you meet it in a framework: **inverted dropout**.
Instead of zeroing units at train time and rescaling at inference, you zero *and* rescale the
survivors *at train time* (dividing by `1 - p`, so the expected activation stays the same), which
means inference needs no special-casing — it just runs the network with no mask at all. `TinyNet`
implements exactly this, and only inside its training path:

```python
def _forward(self, X: np.ndarray, training: bool) -> dict:
    z1 = X @ self.W1 + self.b1
    a1 = relu(z1)
    mask = np.ones_like(a1)
    if training and self.dropout_p > 0.0:
        mask = (self._rng.random(a1.shape) > self.dropout_p).astype(a1.dtype)
        a1 = a1 * mask / (1.0 - self.dropout_p)
    z2 = a1 @ self.W2 + self.b2
    a2 = sigmoid(z2)
    return {"z1": z1, "a1": a1, "mask": mask, "z2": z2, "a2": a2}
```

`.predict()` always calls `_forward(X, training=False)`, so dropout never touches an inference
call — the NumPy equivalent of the discipline PyTorch enforces with `model.eval()`
([research/NOTE-ML-2-nn-theory.md](../../research/NOTE-ML-2-nn-theory.md) caveats: "Dropout during
inference: Must be disabled").

**Seeing the effect.** [`code/nn_fundamentals.py`](code/nn_fundamentals.py) generates a small, noisy
XOR-pattern dataset — four Gaussian blobs at the corners of a unit square, labelled by the XOR rule
(`(0,0)->0 (1,1)->0 (0,1)->1 (1,0)->1`), with roughly 6% of labels deliberately flipped so there's
genuine noise for an over-parameterised network to memorise. This is the same XOR pattern §1
proved a single straight line can't separate — the whole point of this demo is watching a real,
trained network succeed where that algebra said one weighted sum never could:

```python
def make_xor_dataset(n_per_corner: int = 30, coord_noise: float = 0.28,
                      label_flip_rate: float = 0.06, seed: int = RNG_SEED
                      ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    corners = [((0.0, 0.0), 0.0), ((1.0, 1.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0)]
    xs, ys = [], []
    for (cx, cy), label in corners:
        pts = rng.normal(loc=[cx, cy], scale=coord_noise, size=(n_per_corner, 2))
        xs.append(pts)
        ys.append(np.full(n_per_corner, label))
    X = np.vstack(xs)
    y = np.concatenate(ys)
    flip_mask = rng.random(len(y)) < label_flip_rate
    y = np.where(flip_mask, 1.0 - y, y)
    order = rng.permutation(len(y))
    return X[order], y[order].reshape(-1, 1)
```

It then trains the *same* `TinyNet` architecture (2 inputs, 64 ReLU hidden units — deliberately
over-sized relative to the 80 training points, so it has room to overfit — one sigmoid output),
from the *same* random seed, twice: once with no dropout, once with dropout `p=0.2` on the hidden
layer, tracking training and validation loss every epoch for both (120 points total: 80 train, 40
held out for validation):

![Two side-by-side line plots of binary cross-entropy loss vs epoch, 0 to 4000. Left panel, "no dropout": blue train-loss line falls smoothly to about 0.21; orange validation-loss line falls then flattens and drifts slowly upward to about 0.38, opening a visible gap below it. Right panel, "dropout p=0.2": blue train-loss line falls to about 0.22, slightly above the left panel's; orange validation-loss line is visibly noisier (jittery) but ends around 0.38 too, with a marginally narrower final gap to the train-loss line than the left panel.](artefacts/loss_curve.png)

```text
no dropout: final train_loss=0.2135 val_loss=0.3764 gap=0.1628
dropout p=0.2: final train_loss=0.2237 val_loss=0.3783 gap=0.1546
```

Read this honestly rather than dramatically: the effect here is real but modest, which is itself the
right lesson for an 80-point toy training set. Dropout's train loss ends up slightly *higher*
(`0.2135 -> 0.2237`) — exactly what "the network can't lean as hard on any one unit" should produce,
since it's a little worse at memorising. Validation loss and the train/validation *gap* both end up
slightly better with dropout (`gap` narrows from `0.1628` to `0.1546`) — a small step toward closing
the overfitting gap DS-14 diagnosed the same way (§5.1). The dropout panel's orange line is also
visibly jumpier: that's the random mask itself showing up as noise in the loss, since a different
half of the hidden units survives on every step. At full scale — real datasets, real capacity — this
same mechanism is one of the standard tools for keeping a much larger network from memorising its
training set.

## 7. Pitfalls

- **Learning rate too high.** §4.1's right-hand panel isn't a contrived failure — it's what happens
  whenever the learning rate is too large for the steepest direction the loss surface has in that
  region. Symptoms: loss increasing instead of decreasing, or oscillating wildly rather than
  settling. Fix: lower the learning rate, or use an optimizer that adapts it automatically (out of
  this chapter's scope, but the reason such optimizers exist).
- **Dead ReLU units.** ReLU's derivative is exactly 0 for any negative input
  ([research/NOTE-ML-2-nn-theory.md](../../research/NOTE-ML-2-nn-theory.md) caveats). If a weight
  update pushes a unit's input permanently negative, its gradient is 0 forever — that unit stops
  learning and never activates again, silently reducing the network's effective capacity. Leaky ReLU
  (a small nonzero slope for negative inputs, e.g. `0.01 * x`) is the standard mitigation, at the
  cost of one more hyperparameter.
- **Confusing sigmoid and softmax outputs.** Sigmoid is for one independent yes/no decision;
  softmax is for "exactly one of N mutually exclusive classes." Using sigmoid where the problem is
  genuinely multi-class (or softmax where labels aren't mutually exclusive) pairs the output with
  the wrong loss function and quietly produces miscalibrated probabilities.
- **Over- and under-fitting look the same here as everywhere else in this curriculum.** §6's
  train/validation gap is this chapter's version of the diagnostic
  [DS-14 §5.1](../../01-data-science/01-theory/01-theory-overview.md#51-overfitting-and-underfitting)
  introduced: falling training loss with flat-or-rising validation loss is overfitting; both staying
  high is underfitting (usually fixed by more capacity — more units or layers — rather than less).
  Dropout is one lever against the former; it does nothing for the latter.

## 8. Recap & what's next

- **The wall (cold open):** Rosenblatt's 1958 Perceptron was one neuron; Minsky and Papert proved
  in 1969 that one weighted sum can never separate XOR — §1 reproduced that proof with real numbers.
  Rumelhart, Hinton, and Williams' 1986 backpropagation paper is what finally let networks with
  hidden layers train past that wall.
- **A neuron** (§2) is `LinearRegression`'s `w . x + b` plus one nonlinear activation function.
- **A dense layer** (§3) runs many neurons over the same input as one matrix multiply,
  `Z = X @ W + b`, followed by an elementwise activation; stacking layers needs a nonlinearity
  between them or the stack collapses back into one linear layer.
- **Learning** (§4) follows the negative gradient of a loss function downhill (gradient descent),
  with backpropagation computing every weight's gradient via one chain-rule multiplication per
  layer — verified by hand against a finite-difference check in this chapter's own tiny numeric
  example.
- **Activation functions** (§5): sigmoid and tanh saturate and cause vanishing gradients in deep
  networks; ReLU's constant gradient is why it dominates hidden layers today; softmax turns logits
  into a multi-class probability distribution.
- **Dropout** (§6) randomly drops hidden units during training only, cheaply approximating an
  ensemble of sub-networks and narrowing the train/validation gap — this chapter's own 80-point demo
  showed a modest version of exactly that effect, on the very XOR pattern §1 proved a straight line
  could never fit.

**What's next:** this chapter deliberately stayed at the level of one neuron and one dense layer.
The next Theory chapter in this subject, SPEC-ML-2 ("Network Architectures"), picks up specific
layer types built for specific data shapes — convolution layers for images, LSTM/GRU for sequences,
the transformer block for attention over sequences, and autoencoders for representation learning —
all of which are still, underneath, the same "weighted sum + nonlinearity, learned by gradient
descent + backprop" machinery this chapter just built from scratch.
