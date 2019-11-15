class Activation:
    import numpy as np

    def softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def activation(self, Z, TypeOfActivation):
        if TypeOfActivation == 'softmax':
            A = softmax(Z)
        elif TypeOfActivation == 'sigmoid':
            A = sigmoid(Z)
        elif TypeOfActivation == 'relu':
            A = relu(Z)
        else:
            raise Exception("Activation function not found")

        return A
