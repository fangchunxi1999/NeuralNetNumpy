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
            raise Exception("Not supported activation type")

        return A

    def softmaxDiff(self, Z, A):
        pass  # (╯°□°）╯︵ ┻━┻)

    def sigmoidDiff(self, Z, A):
        return A * (1 - A)

    def reluDiff(self, Z, A):
        return (Z > 0)

    def activationDiff(self, Z, A, TypeOfActivation):
        if TypeOfActivation == 'sigmoid':
            diff = sigmoidDiff(Z, A)
        elif TypeOfActivation == 'relu':
            diff = reluDiff(Z, A)
        else:
            raise Exception("Not supported activation type")

        return diff
