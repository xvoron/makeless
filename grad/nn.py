from engine import Value
from random import uniform


class Dataset:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)


class Dataloader:
    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[i:i+self.batch_size]


class Module:
    @property
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0



class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(uniform(-1, 1))

    def __call__(self, x):
        out = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return out.tanh()

    @property
    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    @property
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters]


class MLP(Module):
    def __init__(self, nin, nouts: list):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

        self._parameters = [p for l in self.layers for p in l.parameters]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def parameters(self):
        return self._parameters


if __name__ == '__main__':
    mlp = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    dataloader = Dataloader(Dataset(xs, ys), 1)

    params = mlp.parameters

    epoch = 100
    for e in range(epoch):

        # forward pass
        for x, y in dataloader:
            ypred = [mlp(x) for x in xs]

            # calculate loss
            loss= sum((ygt - ypd)**2 for ygt, ypd in zip(ys, ypred))

            # zero gradient 
            mlp.zero_grad()

            # backward pass
            loss.backward()

            # update weights
            for p in params:
                p.data -= 0.01 * p.grad

            print(e, loss.data)
    print(f"\n\nExpected: {ys}")
    print(f"Predicted: {[mlp(x).data for x in xs]}")

