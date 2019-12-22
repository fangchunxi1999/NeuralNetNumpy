import numpy as np


class Relu:
    def __init__(self):
        self.cache = {}
        self.hasW = False

    def hasWeight(self):
        return self.hasW

    def forward(self, Z, saveCache=False):
        if saveCache:
            self.cache['Z'] = Z
        return np.where(Z >= 0, Z, 0)

    def backward(self, dA):
        Z = self.cache['Z']
        return dA * np.where(Z >= 0, 1, 0)


class Softmax:
    def __init__(self):
        self.cache = {}
        self.hasW = False

    def hasWeight(self):
        return self.hasW

    def forward(self, Z, saveCache=False):
        if saveCache:
            self.cache['Z'] = Z
        Z = Z - Z.max()
        e = np.exp(Z)
        return e / np.sum(e, axis=0, keepdims=True)

    def backward(self, dA):
        Z = self.cache['Z']
        return dA * (Z * (1 - Z))
