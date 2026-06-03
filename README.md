
# micrograd

A tiny scalar-valued automatic differentiation engine and a small neural network library built on top of it. Each `Value` tracks its data and gradient, and a full backward pass is computed by walking the computational graph in reverse topological order.

This is a learning-focused reimplementation in the spirit of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## What's inside

- **`Value`** — a scalar wrapper that records operations to build a computation graph and supports reverse-mode autodiff.
- **`Neuron` / `Layer` / `MLP`** — a minimal feed-forward neural network library where every parameter is a `Value`.
- A short training loop that fits a 3-input MLP to a toy dataset using gradient descent.

## Requirements

- Python 3.8+
- No external dependencies (only the standard library `math` and `random`).

## Project structure

```
.
├── micrograd.py   # the Value class (the autograd engine)
└── nn.py          # Neuron, Layer, MLP, and the training loop
```

## The autograd engine

A `Value` holds a number and a gradient. Operations on `Value` objects build up a graph, and calling `.backward()` populates the `.grad` of every node via the chain rule.

```python
from micrograd import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b + c          # forward pass
e = d.tanh()           # nonlinearity

e.backward()           # backward pass

print(a.grad)          # d(e)/d(a)
print(b.grad)          # d(e)/d(b)
```

Supported operations:

| Operation | Method        | Notes                                  |
|-----------|---------------|----------------------------------------|
| add       | `__add__`     | `a + b`                                |
| subtract  | `__sub__`     | `a - b`                                |
| multiply  | `__mul__`     | `a * b`                                |
| power     | `__pow__`     | `a ** k` (constant exponent)           |
| tanh      | `.tanh()`     | activation function                    |

`.backward()` builds a topological ordering of the graph, seeds the output gradient to `1.0`, and then calls each node's local `_backward` in reverse so gradients flow from output to inputs.

## The neural network library

The `nn` module composes `Value` objects into a multi-layer perceptron:

- **`Neuron(nin)`** — `nin` weights plus a bias, applying a `tanh` activation to the weighted sum.
- **`Layer(nin, nout)`** — a list of `nout` neurons.
- **`MLP(nin, nouts)`** — stacks layers; `nouts` is a list of layer sizes.

```python
from nn import MLP
from micrograd import Value

mlp = MLP(3, [4, 4, 1])          # 3 inputs -> 4 -> 4 -> 1 output
x = [Value(1.0), Value(2.0), Value(3.0)]
print(mlp(x))
```

## Training example

The included training loop fits the network to a small dataset using mean-squared-error loss and plain gradient descent:

```python
xs = [
    [Value(2.0), Value(3.0), Value(-1.0)],
    [Value(3.0), Value(-1.0), Value(0.5)],
    [Value(0.5), Value(1.0), Value(1.0)],
    [Value(1.0), Value(1.0), Value(-1.0)],
]
ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)]  # targets

for k in range(20):
    # forward pass
    ypred = [mlp(x)[0] for x in xs]
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0.0))

    # zero gradients, then backward pass
    for p in mlp.parameters():
        p.grad = 0.0
    loss.backward()

    # update weights
    for p in mlp.parameters():
        p.data -= 0.01 * p.grad

    print(f"step {k} loss {loss.data}")
```

Run it:

```bash
python nn.py
```

You should see the loss decrease step by step as the network learns the targets.

## How it works

1. **Forward pass** — operations on `Value` objects compute results and record parent/child relationships in the graph.
2. **Topological sort** — `.backward()` orders nodes so every node comes after its inputs.
3. **Reverse pass** — gradients are propagated from the loss back to every parameter using each operation's local derivative.
4. **Update** — each parameter is nudged in the direction that reduces the loss (`p.data -= lr * p.grad`).

## Acknowledgements

Inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd) and the accompanying "Building micrograd" lecture.

## License

MIT
