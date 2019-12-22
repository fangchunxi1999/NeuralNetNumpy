import numpy as np

from NeuralNetNumpy.Util.Initalizer import he_normal

np.random.seed(0)


class Dense:
    def __init__(self, size: int):
        self.size = size
        self.paras = {}
        self.cache = {}
        self.grads = {}

        self.hasW = True

    def hasWeight(self):
        return self.hasW

    def forward(self, A0, saveCache=False):
        if 'W' not in self.paras:
            W, b = he_normal((A0.shape[0], self.size))
            self.paras['W'] = W
            self.paras['b'] = b

        Z = np.dot(self.paras['W'], A0) + self.paras['b']

        if saveCache:
            self.cache['A'] = A0

        return Z

    def backward(self, dZ):
        batchSize = dZ.shape[1]
        self.grads['dW'] = np.dot(dZ, self.cache['A'].T) / batchSize
        self.grads['db'] = np.sum(dZ, axis=1, keepdims=True)
        dA0 = np.dot(self.paras['W'].T, dZ)
        return dA0

    def applyGrads(self, learnRate=0.001, l2=1e-4):
        self.paras['W'] -= learnRate * \
            (self.grads['dW'] + l2 * self.paras['W'])
        self.paras['b'] -= learnRate * \
            (self.grads['db'] + l2 * self.paras['b'])
