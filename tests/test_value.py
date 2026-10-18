import random
import pytest
from autograd import Value, MLP


# ---------- numerical gradient checking ----------
def numerical_grad(f, x, eps=1e-6):
    """Central difference of f at x, where f maps a Value to a Value."""
    return (f(Value(x + eps)).data - f(Value(x - eps)).data) / (2 * eps)


def analytic_grad(f, x):
    v = Value(x)
    f(v).backward()
    return v.grad


UNARY_FUNCS = {
    "square": lambda v: v ** 2,
    "cubic_poly": lambda v: v ** 3 + 2 * v - 1,
    "tanh": lambda v: v.tanh(),
    "exp": lambda v: v.exp(),
    "log": lambda v: (v * v + 1).log(),
    "relu_chain": lambda v: (v * 3 - 1).relu() * v,
    "division": lambda v: 1 / (v * v + 2),
    "reuse": lambda v: v * v + v + v,  # same node used several times
    "sub_neg": lambda v: 5 - v - (-v) * 2,
}


@pytest.mark.parametrize("name", UNARY_FUNCS)
@pytest.mark.parametrize("x", [-1.7, 0.4, 2.3])
def test_numerical_gradients(name, x):
    f = UNARY_FUNCS[name]
    assert analytic_grad(f, x) == pytest.approx(numerical_grad(f, x), rel=1e-4, abs=1e-5)


# ---------- comparison with PyTorch ----------
def test_against_pytorch():
    torch = pytest.importorskip("torch")

    a, b = Value(-4.0), Value(2.0)
    c = a * b + b ** 3
    d = (c * 2 + b).relu()
    d.backward()

    x = torch.tensor(-4.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    z = x * y + y ** 3
    w = (z * 2 + y).relu()
    w.backward()

    assert d.data == pytest.approx(w.item())
    assert a.grad == pytest.approx(x.grad.item())
    assert b.grad == pytest.approx(y.grad.item())


def test_mlp_against_pytorch():
    torch = pytest.importorskip("torch")
    random.seed(0)
    model = MLP(3, [4, 4, 1])
    inp = [0.5, -1.2, 2.0]
    out = model(inp)
    out.backward()

    # rebuild the same network in torch using our weights
    params = []
    for layer in model.layers:
        W = torch.tensor([[w.data for w in n.w] for n in layer.neurons],
                         dtype=torch.float64, requires_grad=True)
        B = torch.tensor([n.b.data for n in layer.neurons],
                         dtype=torch.float64, requires_grad=True)
        params.append((W, B))
    h = torch.tensor(inp, dtype=torch.float64)
    for i, (W, B) in enumerate(params):
        h = W @ h + B
        if i != len(params) - 1:
            h = torch.tanh(h)
    h.sum().backward()

    assert out.data == pytest.approx(h.item())
    for layer, (W, B) in zip(model.layers, params):
        for j, n in enumerate(layer.neurons):
            assert n.b.grad == pytest.approx(B.grad[j].item(), abs=1e-9)
            for k, w in enumerate(n.w):
                assert w.grad == pytest.approx(W.grad[j, k].item(), abs=1e-9)


# ---------- edge cases ----------
def test_node_used_twice_accumulates():
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0


def test_deep_graph_no_recursion_error():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y + 0.0
    y.backward()
    assert x.grad == 1.0
