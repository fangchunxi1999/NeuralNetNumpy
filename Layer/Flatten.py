import numpy as np


class Flatten:
    def __init__(self, transpose=True):
        self.shape = ()
        self.transpose = transpose
        self.hasW = False

    def hasWeight(self):
        return self.hasW

    def forward(self, Z, saveCache=False):
        shape = Z.shape

        if saveCache:
            self.shape = shape

        A = np.ravel(Z).reshape(shape[0], -1)

        if self.transpose:
            A = A.T

        return A

    def backward(self, dZ):
        if self.transpose:
            dZ = dZ.T
        dA0 = dZ.reshape(self.shape)
        return dA0
