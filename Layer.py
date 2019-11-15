class Layer:
    import numpy as np

    import Activation

    def flatten(self, inputSize: tuple):
        outputSize = np.prod(inputSize)
        return (outputSize, "Flatten")

    def flattenCal(self, A0, outputSize):
        A = np.empty((0, outputSize), int)
        for item in A0:
            item = item.reshape(1, -1)
            A = np.vstack((A, item))
        return A, A

    def dense(self, activation: str, size: int):
        outputSize = size
        return (outputSize, activation, "Dense")

    def denseCal(self, A0, weight, bias, activation: str):
        Z = np.dot(A0, weight) + bias
        A = Activation.activation(Z, activation)
        return Z, A

    def layer(self, A0, layerIndex, weightList, biasList, layerList):
        TypeOfLayer = layerList[layerIndex][-1]

        if TypeOfLayer == "Flatten":
            outputSize = layerList[layerIndex][0]
            return self.flattenCal(A0, outputSize)
        elif TypeOfLayer == "Dense":
            weight = weightList[layerIndex - 1]
            bias = biasList[layerIndex - 1]
            activation = layerList[layerIndex][1]
            return self.denseCal(A0, weight, bias, activation)

        else:
            raise Exception("Not supported layer type")
