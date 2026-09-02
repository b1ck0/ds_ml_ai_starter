"""Neural-network fundamentals -- companion code for
Machine Learning/Theory/neural-network-fundamentals.md (SPEC-ML-1).

Everything here is plain NumPy: no torch, no autograd. The whole point of this chapter
is to make the mechanics that a framework normally hides -- the forward pass, the loss,
the chain rule, the weight update -- visible and steppable by hand. Five pieces:

  1. Activation functions (sigmoid, tanh, ReLU, softmax) and their derivatives, plotted
     side by side to make the vanishing-gradient problem visible: sigmoid/tanh
     derivatives cap out at 0.25/1.0 and flatten to ~0 away from the origin, while ReLU's
     derivative is a constant 0 or 1 wherever it's active.
     Formulas: research/NOTE-ML-2-nn-theory.md, evidence item 1 and item 4.
     Saves activation_functions.png.

  2. A tiny SCALAR chain-rule demo: forward pass through two toy layers (one ReLU
     neuron feeding one sigmoid neuron), then backward by hand, one multiplication per
     layer, matching NOTE-ML-2's stated form
     dL/dw = (dL/da_n) x (da_n/da_(n-1)) x ... x (da_1/dw) (evidence item 2). Every
     hand-computed gradient is cross-checked against a finite-difference numerical
     gradient so the chain-rule arithmetic is verifiably correct, not just asserted.

  3. Gradient descent on a 2-D bowl-shaped loss surface (a synthetic loss, not a
     trained model -- this is LO2's "loss surface + learning rate" intuition in
     isolation). Runs it twice from the same start: once with a learning rate that
     converges, once with one large enough to diverge, since "bad learning rate" is
     also this chapter's first pitfall.
     Saves gradient_descent_trajectory.png.

  4. A tiny two-layer dense net (2 -> hidden -> 1, ReLU hidden layer, sigmoid output,
     binary cross-entropy loss) trained by full-batch gradient descent with a
     hand-written backward pass -- the same chain rule from piece 2, generalised to
     vectors and matrices. Trained on a small, noisy, XOR-shaped synthetic dataset
     (XOR is the canonical "not linearly separable" toy problem -- exactly why DS-5's
     LinearRegression-style straight line can't solve it and a hidden layer + nonlinearity
     can, LO1's whole point).

  5. The same net trained twice on that dataset -- once with no dropout, once with
     inverted dropout (Srivastava et al. 2014, research/NOTE-ML-2-nn-theory.md evidence
     item 3) on the hidden layer -- with train and validation loss tracked per epoch for
     both, to make dropout's regularizing effect (LO4) visible as a shrunken train/val
     gap rather than just asserted.
     Saves loss_curve.png.

Grounded facts:
  - Activation formulas, backprop chain-rule statement, dropout mechanism, and the
    vanishing-gradient root cause (sigmoid/tanh derivative <= 0.25, composing across L
    layers as ~0.25^L) are all from research/NOTE-ML-2-nn-theory.md (checked 2026-09-02),
    itself citing https://cs231n.github.io/optimization-2/,
    https://jmlr.org/papers/v15/srivastava14a.html, and
    https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html.
  - "Inverted dropout" (scale kept units by 1/(1-p) at train time so no rescaling is
    needed at inference) is the standard implementation of the drop-and-rescale
    mechanism NOTE-ML-2 describes; NOTE-ML-2's own caveats section says inference must
    run with dropout disabled (p=0), which this script's TinyNet.predict enforces by
    never applying the dropout mask outside of .train_step().

Environment (research/NOTE-2-package-versions.md checked 2026-09-02, both versions
re-verified live against this project's .venv-ml on 2026-09-02):
    numpy==2.5.2, matplotlib==3.11.1, Python 3.11+ (this venv runs 3.13.7). No torch
    import anywhere in this file -- per SPEC-ML-1, the runnable demo is numpy-only.

Run:
    python nn_fundamentals.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np

ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"
RNG_SEED = 42


# ---------------------------------------------------------------------------
# 1. Activation functions and their derivatives
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """sigma(x) = 1 / (1 + exp(-x)).  NOTE-ML-2 evidence item 1."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """d(sigma)/dx = sigma(x) * (1 - sigma(x)); peaks at 0.25 when x=0."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_fn(x: np.ndarray) -> np.ndarray:
    """tanh(x) = 2 / (1 + exp(-2x)) - 1.  NOTE-ML-2 evidence item 1."""
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0


def tanh_grad(x: np.ndarray) -> np.ndarray:
    """d(tanh)/dx = 1 - tanh(x)^2; peaks at 1.0 when x=0."""
    t = tanh_fn(x)
    return 1.0 - t**2


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU(x) = max(0, x).  NOTE-ML-2 evidence item 1."""
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """d(ReLU)/dx = 0 for x<0, 1 for x>0 (subgradient 0 used at x=0 here)."""
    return (x > 0.0).astype(x.dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    """softmax(x)_i = exp(x_i) / sum_j exp(x_j), applied along the last axis.

    NOTE-ML-2 evidence item 1. Subtracts the row max first for numerical stability
    (a constant shift cancels in numerator/denominator, so it doesn't change the
    result -- standard trick, not a different formula).
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


def plot_activation_functions(path: Path) -> None:
    """2x2 grid: sigmoid, tanh, ReLU (each with its derivative) and a softmax bar
    example -- visualises LO3 (compare activations) and the vanishing-gradient half
    of LO3/pitfalls (sigmoid/tanh derivatives collapse toward 0 away from the origin;
    ReLU's stays a flat 1 wherever the unit is active)."""
    x = np.linspace(-6, 6, 400)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (name, fn, grad) in zip(
        axes.ravel()[:3],
        [("sigmoid", sigmoid, sigmoid_grad), ("tanh", tanh_fn, tanh_grad), ("ReLU", relu, relu_grad)],
    ):
        ax.plot(x, fn(x), label=f"{name}(x)", color="tab:blue", linewidth=2)
        ax.plot(x, grad(x), label=f"{name}'(x)", color="tab:red", linewidth=2, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(-1.2, 1.6)

    ax_softmax = axes.ravel()[3]
    example_logits = np.array([2.0, 1.0, 0.1, -1.0])
    probs = softmax(example_logits)
    labels = [f"class {i}\nlogit={v:g}" for i, v in enumerate(example_logits)]
    bars = ax_softmax.bar(labels, probs, color="tab:green")
    for bar, p in zip(bars, probs):
        ax_softmax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
    ax_softmax.set_ylim(0, 1.0)
    ax_softmax.set_ylabel("probability")
    ax_softmax.set_title(f"softmax(logits) -- sums to {probs.sum():.3f}")

    fig.suptitle("Activation functions and derivatives (sigmoid/tanh saturate; ReLU doesn't)")
    fig.tight_layout()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    print(f"softmax({example_logits.tolist()}) = {np.round(probs, 4).tolist()}, sums to {probs.sum():.6f}")
    print(f"max sigmoid'(x) = {sigmoid_grad(np.array([0.0]))[0]:.4f} (at x=0)")
    print(f"max tanh'(x)    = {tanh_grad(np.array([0.0]))[0]:.4f} (at x=0)")
    print(f"sigmoid'(x) at x=6 (far from origin) = {sigmoid_grad(np.array([6.0]))[0]:.6f}  <- vanishing")
    print(f"ReLU'(x) at x=6 (active region)      = {relu_grad(np.array([6.0]))[0]:.6f}  <- constant")


# ---------------------------------------------------------------------------
# 2. Tiny scalar chain-rule demo, cross-checked against a finite-difference gradient
# ---------------------------------------------------------------------------

def scalar_chain_rule_demo() -> None:
    """Two toy scalar layers: a1 = ReLU(w1*x + b1), a2 = sigmoid(w2*a1 + b2),
    loss = (a2 - target)^2. Computes dloss/dw1 and dloss/dw2 by hand, one
    multiplication per layer -- exactly the chain-rule form NOTE-ML-2 states:
    dL/dw = (dL/da_n) x (da_n/da_(n-1)) x ... x (da_1/dw). Then verifies both
    against a central-difference numerical gradient, which is just the calculus
    definition of a derivative applied directly to the loss function -- no library
    claim involved, so nothing here needs external grounding beyond that definition.
    """
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

    z1, a1, z2, a2, loss = forward(w1, w2)

    # --- backward pass, one chain-rule step at a time ---
    dloss_da2 = 2.0 * (a2 - target)              # dL/da2
    da2_dz2 = a2 * (1.0 - a2)                    # sigmoid'(z2)
    dz2_dw2 = a1                                 # d(w2*a1+b2)/dw2
    dloss_dw2 = dloss_da2 * da2_dz2 * dz2_dw2    # chain rule, layer 2

    dz2_da1 = w2                                 # d(w2*a1+b2)/da1
    da1_dz1 = 1.0 if z1 > 0 else 0.0             # ReLU'(z1)
    dz1_dw1 = x                                  # d(w1*x+b1)/dw1
    dloss_dw1 = dloss_da2 * da2_dz2 * dz2_da1 * da1_dz1 * dz1_dw1  # chain rule, layer 1

    # --- finite-difference check: perturb each weight by +-h, re-run forward ---
    h = 1e-5
    numeric_dw1 = (forward(w1 + h, w2)[4] - forward(w1 - h, w2)[4]) / (2 * h)
    numeric_dw2 = (forward(w1, w2 + h)[4] - forward(w1, w2 - h)[4]) / (2 * h)

    print("\n--- scalar chain-rule demo ---")
    print(f"forward: z1={z1:.4f} a1={a1:.4f} z2={z2:.4f} a2={a2:.4f} loss={loss:.4f}")
    print(f"analytic  dloss/dw1={dloss_dw1:.6f}   numeric dloss/dw1={numeric_dw1:.6f}")
    print(f"analytic  dloss/dw2={dloss_dw2:.6f}   numeric dloss/dw2={numeric_dw2:.6f}")
    assert abs(dloss_dw1 - numeric_dw1) < 1e-4, "chain-rule gradient for w1 disagrees with finite differences"
    assert abs(dloss_dw2 - numeric_dw2) < 1e-4, "chain-rule gradient for w2 disagrees with finite differences"
    print("analytic chain-rule gradients match the finite-difference check (within 1e-4).")


# ---------------------------------------------------------------------------
# 3. Gradient descent on a 2-D loss surface: good learning rate vs too-large one
# ---------------------------------------------------------------------------

def bowl_loss(w: np.ndarray) -> float:
    """A synthetic, elongated bowl: L(w1, w2) = w1^2 + 5*w2^2. Elongated on purpose
    -- it makes a single learning rate that is safe on one axis risky on the other,
    the classic reason gradient descent oscillates on ill-conditioned loss surfaces."""
    return float(w[0] ** 2 + 5.0 * w[1] ** 2)


def bowl_grad(w: np.ndarray) -> np.ndarray:
    """Analytic gradient of bowl_loss: [dL/dw1, dL/dw2] = [2*w1, 10*w2]."""
    return np.array([2.0 * w[0], 10.0 * w[1]])


def run_gradient_descent(start: np.ndarray, lr: float, n_steps: int) -> np.ndarray:
    """Plain gradient descent: w <- w - lr * grad(w), repeated n_steps times.
    Returns the full trajectory, shape (n_steps+1, 2), start included."""
    path = np.zeros((n_steps + 1, 2))
    w = start.copy()
    path[0] = w
    for i in range(1, n_steps + 1):
        w = w - lr * bowl_grad(w)
        path[i] = w
    return path


def plot_gradient_descent_trajectory(path_out: Path) -> None:
    """Contour of the bowl loss with two GD trajectories overlaid: a learning rate
    that converges, and one large enough to diverge -- LO2 (learning rate's role)
    and this chapter's first pitfall (bad learning rate), in one picture."""
    start = np.array([4.5, 2.0])
    good_lr, bad_lr = 0.15, 0.42
    n_steps = 25

    good_path = run_gradient_descent(start, good_lr, n_steps)
    bad_path = run_gradient_descent(start, bad_lr, min(n_steps, 12))  # fewer steps: it blows up fast

    w1_grid = np.linspace(-5, 5, 200)
    w2_grid = np.linspace(-3, 3, 200)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)
    Z = W1**2 + 5.0 * W2**2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, path, lr, title in [
        (axes[0], good_path, good_lr, "converges"),
        (axes[1], bad_path, bad_lr, "diverges"),
    ]:
        cs = ax.contour(W1, W2, Z, levels=15, cmap="viridis", alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=6)
        ax.plot(path[:, 0], path[:, 1], color="tab:red", marker="o", markersize=3, linewidth=1)
        ax.scatter([0], [0], color="black", marker="*", s=120, zorder=5, label="minimum")
        ax.scatter([start[0]], [start[1]], color="tab:orange", marker="s", s=50, zorder=5, label="start")
        ax.set_title(f"lr={lr} ({title})")
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Gradient descent on L(w1,w2) = w1^2 + 5*w2^2: learning rate controls the step size")
    fig.tight_layout()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
    print(f"wrote {path_out}")
    print(f"good lr={good_lr}: loss went {bowl_loss(good_path[0]):.4f} -> {bowl_loss(good_path[-1]):.6f} "
          f"over {n_steps} steps")
    print(f"bad  lr={bad_lr}: loss went {bowl_loss(bad_path[0]):.4f} -> {bowl_loss(bad_path[-1]):.4e} "
          f"over {len(bad_path) - 1} steps (diverging)")


# ---------------------------------------------------------------------------
# 4. A tiny two-layer dense net, trained by hand on a noisy XOR dataset
# ---------------------------------------------------------------------------

def make_xor_dataset(n_per_corner: int = 30, coord_noise: float = 0.28,
                      label_flip_rate: float = 0.06, seed: int = RNG_SEED
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Four Gaussian blobs centred on the corners of a unit square, labelled by the
    XOR pattern (0,0)->0  (1,1)->0  (0,1)->1  (1,0)->1 -- the canonical toy problem
    that is NOT linearly separable, so no straight line (DS-5's LinearRegression, or
    a single-layer perceptron) can solve it; a hidden layer + nonlinearity can. A
    small fraction of labels are flipped at random to inject genuine label noise --
    without it, an over-parameterised net has nothing to overfit *to* and the
    dropout comparison below has nothing to show.

    Returns X of shape (n, 2), y of shape (n, 1) with values in {0.0, 1.0}.
    """
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


class TinyNet:
    """2 -> hidden -> 1 dense net: z1 = X@W1+b1, a1 = ReLU(z1) [+ dropout while
    training], z2 = a1@W2+b2, a2 = sigmoid(z2) = predicted probability.
    Loss: binary cross-entropy. Weights updated by plain (full-batch) gradient
    descent -- the vector/matrix generalisation of the scalar chain-rule demo above.
    """

    def __init__(self, n_in: int, n_hidden: int, seed: int, dropout_p: float = 0.0):
        rng = np.random.default_rng(seed)
        # He-style scaling (variance ~ 2/fan_in) keeps ReLU activations from
        # exploding or collapsing at initialisation -- a practical detail, not a
        # claim this chapter grounds further; any small, variance-matched init works
        # for a toy net this size.
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), size=(n_hidden, 1))
        self.b2 = np.zeros(1)
        self.dropout_p = dropout_p
        self._rng = rng

    def _forward(self, X: np.ndarray, training: bool) -> dict:
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        mask = np.ones_like(a1)
        if training and self.dropout_p > 0.0:
            # inverted dropout: zero out units with probability p, rescale the
            # survivors by 1/(1-p) so E[a1] is unchanged -- no rescaling needed at
            # inference. NOTE-ML-2 caveat: dropout must be off at inference (p=0);
            # `training=False` (used by .predict()) skips this block entirely.
            mask = (self._rng.random(a1.shape) > self.dropout_p).astype(a1.dtype)
            a1 = a1 * mask / (1.0 - self.dropout_p)
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)
        return {"z1": z1, "a1": a1, "mask": mask, "z2": z2, "a2": a2}

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        a2 = self._forward(X, training=False)["a2"]
        eps = 1e-9  # avoid log(0)
        bce = -(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))
        return float(np.mean(bce))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X, training=False)["a2"]

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        n = X.shape[0]
        cache = self._forward(X, training=True)
        a1, mask, a2 = cache["a1"], cache["mask"], cache["a2"]

        eps = 1e-9
        bce = -(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))
        train_loss = float(np.mean(bce))

        # Backward pass -- same chain rule as scalar_chain_rule_demo(), generalised
        # to matrices. dz2 collapses (dL/da2 * sigmoid'(z2)) to (a2 - y): the
        # standard simplification for sigmoid output + binary cross-entropy loss.
        dz2 = (a2 - y) / n                     # (n, 1)
        dW2 = a1.T @ dz2                       # (hidden, 1)
        db2 = np.sum(dz2, axis=0)              # (1,)

        da1 = dz2 @ self.W2.T                  # (n, hidden)
        da1 = da1 * mask / (1.0 - self.dropout_p) if self.dropout_p > 0.0 else da1
        dz1 = da1 * relu_grad(cache["z1"])      # (n, hidden)
        dW1 = X.T @ dz1                        # (n_in, hidden)
        db1 = np.sum(dz1, axis=0)              # (hidden,)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return train_loss


def train_tiny_net(dropout_p: float, X_train, y_train, X_val, y_val,
                    n_hidden: int = 64, lr: float = 0.2, n_epochs: int = 4000,
                    seed: int = RNG_SEED) -> tuple[list[float], list[float]]:
    """Trains one TinyNet for n_epochs full-batch GD steps; returns (train_losses,
    val_losses), one entry per epoch. Same seed for both dropout settings so any
    difference in the resulting curves comes from dropout, not from a different
    random initialisation."""
    net = TinyNet(n_in=X_train.shape[1], n_hidden=n_hidden, seed=seed, dropout_p=dropout_p)
    train_losses, val_losses = [], []
    for _ in range(n_epochs):
        net.train_step(X_train, y_train, lr=lr)
        train_losses.append(net.loss(X_train, y_train))
        val_losses.append(net.loss(X_val, y_val))
    return train_losses, val_losses


def plot_loss_curves(path: Path) -> None:
    """Trains the same TinyNet architecture twice -- no dropout vs dropout(p=0.2) on
    the hidden layer -- on the same noisy XOR dataset, and plots train/validation
    loss per epoch for both side by side. LO4: dropout as regularisation, made
    visible as a narrower train/validation gap rather than merely asserted."""
    X, y = make_xor_dataset()
    n_val = 40
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    settings = [("no dropout", 0.0), ("dropout p=0.2", 0.2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    results = {}
    for ax, (title, p) in zip(axes, settings):
        train_losses, val_losses = train_tiny_net(p, X_train, y_train, X_val, y_val)
        results[title] = (train_losses, val_losses)
        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="train loss", color="tab:blue")
        ax.plot(epochs, val_losses, label="validation loss", color="tab:orange")
        final_gap = val_losses[-1] - train_losses[-1]
        ax.set_title(f"{title}\nfinal train={train_losses[-1]:.3f} val={val_losses[-1]:.3f} gap={final_gap:.3f}")
        ax.set_xlabel("epoch")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("binary cross-entropy loss")
    fig.suptitle("Loss vs epoch: same tiny net, with and without dropout (noisy XOR dataset)")
    fig.tight_layout()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    for title, (train_losses, val_losses) in results.items():
        print(f"{title}: final train_loss={train_losses[-1]:.4f} val_loss={val_losses[-1]:.4f} "
              f"gap={val_losses[-1] - train_losses[-1]:.4f}")


# ---------------------------------------------------------------------------
def main() -> None:
    plot_activation_functions(ARTEFACTS_DIR / "activation_functions.png")
    scalar_chain_rule_demo()
    plot_gradient_descent_trajectory(ARTEFACTS_DIR / "gradient_descent_trajectory.png")
    plot_loss_curves(ARTEFACTS_DIR / "loss_curve.png")


if __name__ == "__main__":
    main()
