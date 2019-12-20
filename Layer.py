import numpy as np
from NeuralNetNumpy.Activation import *


class Layer():
    def build(self, inputSize=None):
        if inputSize is not None:
            self.inputSize = inputSize

    def forward(self, A0):
        pass

    def backward(self, dA):
        pass

    def adjustParams(self, dW, dB, learningRate, count):
        pass


class Flatten(Layer):
    def __init__(self, inputSize=None):
        self.inputSize = inputSize

    def build(self, inputSize=None):
        if inputSize is not None:
            self.inputSize = inputSize
        if self.inputSize is None:
            raise ValueError("Input Size is not set!")

        self.outputSize = np.prod(self.inputSize)

    def forward(self, A0):
        A0 = np.concatenate([i[np.newaxis] for i in A0])
        self.A = np.empty((0, self.outputSize), int)
        for i in A0:
            i = i.reshape(1, -1)
            self.A = np.vstack((self.A, i))
        return self.A

    def backward(self, dA):
        dA = [i.reshape(self.inputSize) for i in np.vsplit(dA, dA.shape[0])]
        return dA, None, None

    def adjustParams(self, dW, dB, learningRate, count):
        pass


class Dense(Layer):
    def __init__(self, outputSize, Activation: Activation, inputSize=None):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = Activation

    def build(self, inputSize=None):
        if inputSize is not None:
            self.inputSize = inputSize
        if self.inputSize is None:
            raise ValueError("Input Size is not set!")

        self.weight = np.random.randn(
            self.inputSize, self.outputSize) / np.sqrt(self.outputSize)
        self.bias = np.random.randn(self.outputSize) / np.sqrt(self.outputSize)

    def forward(self, A0):
        self.Z = np.dot(A0, self.weight) + self.bias
        self.A = self.activation.cal(self.Z)
        return self.A

    def backward(self, dA):
        dA0 = np.dot(dA, self.weight.T)
        dW = np.dot(self.Z.T, dA0).T
        dB = np.sum(dA, axis=0, keepdims=True)
        return dA0, dW, dB

    def adjustParams(self, dW, dB, learningRate, count):
        self.weight = self.weight + (learningRate / count) * dW
        self.bias = self.bias + (learningRate / count) * dB


class Conv(Layer):
    pass


class MaxPool(Layer):
    def __init__(self, kernelSize=(2, 2), stride=2, inputSize=None):
        self.kernelSize = kernelSize
        self.stride = stride
        self.inputSize = inputSize

    def build(self, inputSize=None):
        if inputSize is not None:
            self.inputSize = inputSize
        if self.inputSize is None:
            raise ValueError("Input Size is not set!")

        old_H, old_W, old_C = self.inputSize
        kernel_H, kernel_W = self.kernelSize

        new_H = int(1 + (old_H - kernel_H) / self.stride)
        new_W = int(1 + (old_H - kernel_W) / self.stride)
        new_C = old_C

        self.outputSize = (new_H, new_W, new_C)

    def forward(self, A0):
        new_H, new_W, new_C = self.outputSize
        A = []

        for data in A0:
            tmp_A = np.zeros((new_H, new_W, new_C))
            for h in range(new_H):
                for w in range(new_W):

                    vStart = h * self.stride
                    vEnd = vStart + self.kernelSize[0]

                    hStart = w * self.stride
                    hEnd = hStart + self.kernelSize[1]

                    for c in range(new_C):
                        tmp_A[h, w, c] = np.max(
                            data[vStart:vEnd, hStart:hEnd, c])
            A = A + [tmp_A]

        self.Z = A0
        self.A = A

        return self.A

    def backward(self, dA):
        old_H, old_W, old_C = self.inputSize
        new_H, new_W, new_C = self.outputSize

        dA0 = []

        for i, data in enumerate(dA):
            a = self.Z[i]
            tmp_dA0 = np.zeros((old_H, old_W, old_C))

            for h in range(new_H):
                for w in range(new_W):

                    vStart = h * self.stride
                    vEnd = vStart + self.kernelSize[0]

                    hStart = w * self.stride
                    hEnd = hStart + self.kernelSize[1]

                    for c in range(new_C):
                        aSlice = a[vStart:vEnd, hStart:hEnd, c]
                        mask = self.createMask(aSlice)
                        tmp_dA0[vStart:vEnd, hStart:hEnd,
                                c] += data[h, w, c] * mask
            dA0 = dA0 + [tmp_dA0]

        return dA0

    def createMask(self, x):
        return x == np.max(x)
