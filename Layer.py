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


class Pooling(Layer):
    pass


##########################################################
""" 

def flatten(inputSize: tuple):
    outputSize = np.prod(inputSize)
    return (outputSize, "Flatten")


def flattenCal(A0, outputSize):
    A = np.empty((0, outputSize), int)
    for item in A0:
        item = item.reshape(1, -1)
        A = np.vstack((A, item))
    return A, A


def dense(activation: str, size: int):
    outputSize = size
    return (outputSize, activation, "Dense")


def denseCal(A0, weight, bias, activation: str):
    Z = np.dot(A0, weight) + bias
    A = Activation.activation(Z, activation)
    return Z, A


def layer(A0, layerIndex, weightList, biasList, layerList):
    TypeOfLayer = layerList[layerIndex][-1]
    if TypeOfLayer == "Flatten":
        outputSize = layerList[layerIndex][0]
        return flattenCal(A0, outputSize)
    elif TypeOfLayer == "Dense":
        weight = weightList[layerIndex - 1]
        bias = biasList[layerIndex - 1]
        activation = layerList[layerIndex][1]
        return denseCal(A0, weight, bias, activation)
    else:
        raise Exception("Not supported layer type")
 """
