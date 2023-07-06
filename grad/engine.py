from __future__ import annotations
import math
from typing import Union

import numpy as np


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0

        self._op = _op

        self._backward = lambda: None
        self._prev = set(_children)

    def __repr__(self):
        return f'Value(data={self.data};grad={self.grad})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        # Why don't use other.data?
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.data
        out._backward = _backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - out.data ** 2) * out.grad
        out._backward = _backward

        return out


    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rturediv__(self, other): # other / self
        return other * self**-1


class Tensor(Value):

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = np.zeros_like(data)

        self._op = _op

        self._backward = lambda: None
        self._prev = set(_children)

    def __repr__(self):
        return f'Tensor(data={self.data};grad={self.grad})'

    def __add__(self, other: Tensor):
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Tensor):
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]):
        assert isinstance(other, (int, float))
        # Why don't use other.data?
        out = Tensor(np.power(self.data, other), (self,), f'**{other}')

        def _backward():
            self.grad += (other * np.power(self.data, (other - 1))) * out.grad
        out._backward = _backward

        return out

    @property
    def shape(self):
        return self.data.shape

    def matmul(self, other: Tensor):
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.max(self.data, 0), (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.data
        out._backward = _backward

        return out

if __name__ == '__main__':
    tensor = Tensor(np.array([1, 2, 3, 4, 5]))
    print(tensor)
    x = tensor + tensor
    x.relu()
    print(x)
    x.backward()
    print(tensor)
    print(x)
