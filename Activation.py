import numpy as np


class Activation():
    def __init__(self):
        super().__init__()


class Softmax(Activation):
    def cal(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

    def diff(self, Z, A):
        return Z * (1 - Z)


class Sigmoid(Activation):
    def cal(self, Z):
        return 1 / (1 + np.exp(-Z))

    def diff(self, Z, A):
        return A * (1 - A)


class Relu(Activation):
    def cal(self, Z):
        return np.maximum(0, Z)

    def diff(self, Z, A):
        return (Z > 0)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def activation(Z, TypeOfActivation):
    if TypeOfActivation == 'softmax':
        A = softmax(Z)
    elif TypeOfActivation == 'sigmoid':
        A = sigmoid(Z)
    elif TypeOfActivation == 'relu':
        A = relu(Z)
    else:
        raise Exception("Not supported activation type")
    return A


def softmaxDiff(Z, A):
    pass  # (╯°□°）╯︵ ┻━┻)


def sigmoidDiff(Z, A):
    return A * (1 - A)


def reluDiff(Z, A):
    return (Z > 0)


def activationDiff(Z, A, TypeOfActivation):
    if TypeOfActivation == 'sigmoid':
        diff = sigmoidDiff(Z, A)
    elif TypeOfActivation == 'relu':
        diff = reluDiff(Z, A)
    else:
        raise Exception("Not supported activation diff type")
    return diff
