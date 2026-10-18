"""Train a small MLP on a two-moons dataset using only our autograd engine.
Run from the repo root:  python -m experiments.train_moons
"""
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from autograd import Value, MLP

SEED = 42


def make_moons(n=100, noise=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, math.pi, n // 2)
    a = np.stack([np.cos(t), np.sin(t)], 1)
    b = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
    X = np.concatenate([a, b]) + rng.normal(0, noise, (n, 2))
    y = np.array([1] * (n // 2) + [-1] * (n // 2))
    return X, y


def loss_fn(model, X, y, alpha=1e-4):
    scores = [model(list(map(Value, xi))) for xi in X]
    # hinge loss (max-margin) + L2 regularisation
    losses = [(1 + -yi * s).relu() for yi, s in zip(y, scores)]
    data_loss = sum(losses, Value(0.0)) * (1.0 / len(losses))
    reg = alpha * sum((p * p for p in model.parameters()), Value(0.0))
    total = data_loss + reg
    acc = sum((yi > 0) == (s.data > 0) for yi, s in zip(y, scores)) / len(y)
    return total, acc


def main(steps=100):
    random.seed(SEED)
    X, y = make_moons()
    model = MLP(2, [16, 16, 1])
    for k in range(steps):
        loss, acc = loss_fn(model, X, y)
        model.zero_grad()
        loss.backward()
        lr = 1.0 - 0.9 * k / steps  # linear learning-rate decay
        for p in model.parameters():
            p.data -= lr * p.grad
        if k % 10 == 0 or k == steps - 1:
            print(f"step {k:3d}  loss {loss.data:.4f}  acc {acc * 100:.1f}%")

    # decision boundary plot
    xs, ys = np.meshgrid(np.linspace(-1.5, 2.5, 50), np.linspace(-1, 1.5, 50))
    grid = np.c_[xs.ravel(), ys.ravel()]
    zs = np.array([model(list(map(Value, p))).data > 0 for p in grid]).reshape(xs.shape)
    plt.contourf(xs, ys, zs, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title("MLP trained with from-scratch autograd")
    plt.savefig("experiments/moons_boundary.png", dpi=150)
    print("saved experiments/moons_boundary.png")


if __name__ == "__main__":
    main()
