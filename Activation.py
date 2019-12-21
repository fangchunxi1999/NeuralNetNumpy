import numpy as np


class Activation():
    def __init__(self):
        super().__init__()


class Softmax(Activation):
    def cal(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

    def diff(self, dA):
        return dA * (1 - dA)


class Sigmoid(Activation):
    def cal(self, Z):
        return 1 / (1 + np.exp(-Z))

    def diff(self, dA):
        return dA * (1 - dA)


class Relu(Activation):
    def cal(self, Z):
        return np.maximum(0, Z)

    def diff(self, dA):
        return (dA > 0)
