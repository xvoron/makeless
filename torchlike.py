from abc import ABC, abstractmethod

import torch
from functools import wraps


def save_output(func):
    @wraps(func)
    def wrapper(self, x):

        out = func(self, x)

        if self.DEBUG:
            self.out = out

        return out
    return wrapper


class Module(ABC):
    DEBUG = True
    out = None

    @abstractmethod
    def parameters(self):
        ...


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        self.weight = torch.randn((in_features, out_features)) / in_features ** 0.5
        self.bias = torch.zeros(out_features) if bias else None

    @save_output
    def __call__(self, x: torch.Tensor):
        y = x @ self.weight
        if self.bias is not None:
            y += self.bias
        return y

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d(Module):
    """BatchNorm1d implementation

    Batch normalization technique uses mean and variance of the input batch to
    normalize the data.

    x_hat = (x - x_mean) / (var + eps) ** 0.5
    y = gamma * x + beta
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    @save_output
    def __call__(self, x: torch.Tensor):

        # Dimensions of x = (batch_size, features or channels, ...)
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            else:
                raise f"Error {x.ndim} must be 2 or 3."
            x_mean = x.mean(dim=dim, keepdim=True)
            x_var = x.var(dim=dim, keepdim=True)
        else:
            # Using estimated mean and var
            x_mean = self.running_mean
            x_var = self.running_var

        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)

        y = self.gamma * x_hat + self.beta

        # Estimation of mean and var
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + x_mean * self.momentum
                self.running_var = (1 - self.momentum) * self.running_var + x_var * self.momentum
        return y

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh(Module):
    @save_output
    def __call__(self, x: torch.Tensor):
        return torch.tanh(x)
    
    def parameters(self):
        return []


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    @save_output
    def __call__(self, IX: torch.Tensor):
        return self.weight[IX]
    
    def parameters(self):
        return [self.weight]


class FlattenConsecutive(Module):
    def __init__(self, n: int):
        self.n = n

    @save_output
    def __call__(self, x: torch.Tensor):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        return x.squeeze(1) if x.shape[1] == 1 else x

    def parameters(self):
        return []


class Sequential(Module):
    def __init__(self, layers: list[Module]):
        self.layers = layers

    @save_output
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def eval(self):
        for layer in self.layers:
            layer.training = False
        for p in self.parameters():
            p.requires_grad = False

    def train(self):
        for layer in self.layers:
            layer.training = True
        for p in self.parameters():
            p.requires_grad = True

        
if __name__ == '__main__':
    batch_size = 10
    data_size = 10
    n_neurons = 3

    x = torch.randn(batch_size, data_size)
    linear = Linear(data_size, n_neurons, True)
    batch_norm = BatchNorm1d(n_neurons)
    embedding = Embedding(10, 10)

    print(batch_norm(linear(x)))