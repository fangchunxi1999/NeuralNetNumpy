import numpy as np


class Activation():
    def __init__(self):
        super().__init__()


class Softmax(Activation):
    def cal(self, Z):
        self.Z = Z
        Z_ = Z - Z.max()
        e = np.exp(Z_)
        #return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
        return e / np.sum(e, axis=0, keepdims=True)

    def diff(self, dA):
        #return dA * (1 - dA)
        return dA * (self.Z * (1 - self.Z))


class Sigmoid(Activation):
    def cal(self, Z):
        return 1 / (1 + np.exp(-Z))

    def diff(self, dA):
        return dA * (1 - dA)


class Relu(Activation):
    def cal(self, Z):
        self.Z = Z
        #return np.maximum(0, Z)
        return np.where(Z >= 0, Z, 0)

    def diff(self, dA):
        #return (dA > 0)
        return dA * np.where(self.Z >= 0, 1, 0)
