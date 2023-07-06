from engine import Tensor
import numpy as np


class Layer:
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nin, nout))
        self.b = Tensor(np.random.randn(nout))

        self.params = [self.W, self.b]

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        print(x.shape)
        print(x)
        print(self.W.shape)
        return x.matmul(self.W) + self.b


class Model:
    def __init__(self):
        self.layers = [
            Layer(3, 5),
            Layer(5, 1),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.params]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


if __name__ == "__main__":
    model = Model()

    xs = np.array([
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ])

    ys = np.array([1.0, -1.0, -1.0, 1.0])

    params = model.parameters

    epoch = 100
    for e in range(epoch):

        # forward pass
        ypred = [model(x) for x in xs]

        # calculate loss
        loss= sum((ygt - ypd)**2 for ygt, ypd in zip(ys, ypred))

        # zero gradient 
        model.zero_grad()

        # backward pass
        loss.backward()

        # update weights
        for p in params:
            p.data -= 0.01 * p.grad

        print(e, loss.data)
    print(f"\n\nExpected: {ys}")
    print(f"Predicted: {[mlp(x).data for x in xs]}")

